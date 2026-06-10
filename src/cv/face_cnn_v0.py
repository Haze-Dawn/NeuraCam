import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import List, Optional, Dict
from src.cv.face_tracker import Face, BoundingBox, compute_iou


class FaceCNNV0Wrapper(nn.Module):
    def __init__(self, model_path="models/face_cnn_v0/face_cnn_v0.pth",
                 onnx_path="models/face_cnn_v0/face_cnn_v0.onnx",
                 score_threshold=0.50, nms_threshold=0.45, backend="pytorch"):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.backend = backend

        if backend == "pytorch":
            from src.training.architectures.face_cnn_v0 import FaceCNNV0
            self.model = FaceCNNV0()
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        elif backend == "onnx":
            import onnxruntime as ort
            self.sess = ort.InferenceSession(onnx_path)
            self.device = torch.device("cpu")

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def set_score_threshold(self, t):
        self.score_threshold = t

    def set_nms_threshold(self, t):
        self.nms_threshold = t

    @staticmethod
    def _decode_level(cls_map, obj_map, bbox_map, stride, conf_thresh,
                      sx, sy, isz, min_box=5):
        score = cls_map * obj_map
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(score, kernel)
        peaks = (score == dilated) & (score > conf_thresh)
        if not peaks.any():
            return []

        dets = []
        for cy, cx in zip(*np.where(peaks)):
            q = float(score[cy, cx])
            dx = float(bbox_map[0, cy, cx])
            dy = float(bbox_map[1, cy, cx])
            dw = float(np.clip(bbox_map[2, cy, cx], -2, 5))
            dh = float(np.clip(bbox_map[3, cy, cx], -2, 5))

            box_cx = (cx + 0.5 + dx) * stride
            box_cy = (cy + 0.5 + dy) * stride
            box_w = float(np.exp(dw)) * stride
            box_h = float(np.exp(dh)) * stride

            if box_w < min_box or box_h < min_box:
                continue
            x1 = int((box_cx - box_w / 2) * sx)
            y1 = int((box_cy - box_h / 2) * sy)
            bw = int(box_w * sx)
            bh = int(box_h * sy)
            x1 = max(0, x1)
            y1 = max(0, y1)
            if bw < min_box or bh < min_box:
                continue
            dets.append((q, BoundingBox(x=x1, y=y1, w=bw, h=bh)))
        return dets

    @torch.no_grad()
    def detect(self, frame: np.ndarray,
               conf_thresholds: Optional[Dict[str, float]] = None,
               nms_iou: Optional[float] = None,
               scales: Optional[List[float]] = None,
               input_size: int = 640) -> List[Face]:
        if conf_thresholds is None:
            conf_thresholds = {"p4": 0.30, "p2": 0.20}
        if nms_iou is None:
            nms_iou = self.nms_threshold
        if scales is None:
            scales = [1.0]

        thresh_main = conf_thresholds.get("p4", 0.30)
        thresh_low  = conf_thresholds.get("p2", 0.20)

        orig_h, orig_w = frame.shape[:2]
        all_dets = []

        for sf in scales:
            if sf != 1.0:
                h = int(round(orig_h * sf))
                w = int(round(orig_w * sf))
                scaled = cv2.resize(frame, (w, h))
                inv_sf = 1.0 / sf
            else:
                scaled = frame
                h, w = orig_h, orig_w
                inv_sf = 1.0

            isz = input_size
            img = cv2.resize(scaled, (isz, isz))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)

            if self.backend == "pytorch":
                tensor = tensor.to(self.device)
                out = self.model(tensor)
            else:
                out_np = self.sess.run(None, {"input": tensor.numpy()})
                keys = ["cls_8", "obj_8", "bbox_8", "cls_16", "obj_16", "bbox_16",
                        "cls_32", "obj_32", "bbox_32"]
                out = {k: torch.from_numpy(v) for k, v in zip(keys, out_np)}

            sx, sy = orig_w / isz, orig_h / isz

            found = False
            for lname, stride, th in [("8", 8, thresh_main),
                                       ("16", 16, thresh_low),
                                       ("32", 32, thresh_low)]:
                cls_p = torch.sigmoid(out[f"cls_{lname}"][0, 0]).cpu().numpy()
                obj_p = torch.sigmoid(out[f"obj_{lname}"][0, 0]).cpu().numpy()
                bbox_map = out[f"bbox_{lname}"][0].cpu().numpy()
                dets = self._decode_level(cls_p, obj_p, bbox_map, stride,
                                          th, sx, sy, isz)
                if dets:
                    found = True
                all_dets.extend(dets)

            if found and len(scales) > 1:
                break

        if not all_dets:
            return []

        all_dets.sort(key=lambda x: x[0], reverse=True)
        kept = []
        for det in all_dets:
            md = max((compute_iou(det[1], k[1]) for k in kept), default=0.0)
            if md > nms_iou:
                d = det[0] * (1.0 - md)
                if d > 0.01:
                    kept.append((d, det[1]))
            else:
                kept.append(det)

        return [Face(bbox=b, confidence=q) for q, b in kept]


if __name__ == "__main__":
    m = FaceCNNV0Wrapper(backend="pytorch")
    f = np.ones((480, 640, 3), dtype=np.uint8) * 127
    d = m.detect(f)
    print(f"FaceCNN V0: {len(d)} detections (blank frame)")
