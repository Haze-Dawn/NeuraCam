import os
import sys
import tempfile
import numpy as np
import yaml
import pytest
import warnings

from src.utils.config import load_config, Config


class TestConfigIntegration:
    def test_full_config_loads_with_all_keys(self):
        cfg = load_config("config/default.yaml")
        assert cfg.models.face_cnn is not None
        assert cfg.models.face_cnn_v5 is not None
        assert cfg.models.gesture_detector is not None
        assert cfg.face_detection.architecture is not None

    def test_unknown_config_keys_filtered(self):
        data = {
            "models": {
                "face_cnn": "models/face_cnn.pth",
                "face_cnn_v5": "models/face_cnn_v5_best.pth",
                "unknown_key": "should_be_ignored",
                "gesture_pca": "models/gesture_pca.pkl",
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = load_config(path)
            assert cfg.models.face_cnn == "models/face_cnn.pth"
            assert cfg.models.face_cnn_v5 == "models/face_cnn_v5_best.pth"
            assert cfg.models.gesture_pca == "models/gesture_pca.pkl"
            warning_msgs = [str(m.message) for m in w]
            assert any("unknown_key" in msg for msg in warning_msgs), \
                f"Expected warning about unknown_key, got: {warning_msgs}"
        os.unlink(path)

    def test_face_detection_v5_config(self):
        data = {
            "face_detection": {"architecture": "v5"},
            "face_detection_v5": {
                "confidence_threshold": 0.25,
                "nms_iou_threshold": 0.4,
            },
            "models": {
                "face_cnn": "models/face_cnn.pth",
                "face_cnn_v5": "models/face_cnn_v5_best.pth",
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        cfg = load_config(path)
        assert cfg.face_detection.architecture == "v5"
        assert cfg.face_detection_v5.confidence_threshold == 0.25
        assert cfg.face_detection_v5.nms_iou_threshold == 0.4
        os.unlink(path)


class TestModelForwardPass:
    @pytest.mark.skipif(not os.path.exists("models/face_cnn_v5_best.pth"),
                        reason="v5 checkpoint not found")
    def test_v5_forward_pass_no_nan(self):
        import torch
        from src.cv.face_detector_cnn import FaceFCNv5

        model = FaceFCNv5()
        ckpt = torch.load("models/face_cnn_v5_best.pth", map_location='cpu',
                           weights_only=True)
        if 'ema_state_dict' in ckpt:
            ckpt = ckpt['ema_state_dict']
        elif 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        model.load_state_dict(ckpt, strict=True)
        model.eval()

        test_input = torch.randn(1, 3, 480, 640)
        with torch.no_grad():
            output = model(test_input)

        for level in ['p2_obj', 'p3_obj', 'p4_obj']:
            assert torch.isfinite(output[level]).all(), \
                f"NaN/Inf in {level} output"

    @pytest.mark.skipif(not os.path.exists("models/face_cnn_v5_best.pth"),
                        reason="v5 checkpoint not found")
    def test_v5_output_health_check(self):
        import torch
        from src.cv.face_detector_cnn import FaceFCNv5

        model = FaceFCNv5()
        ckpt = torch.load("models/face_cnn_v5_best.pth", map_location='cpu',
                           weights_only=True)
        if 'ema_state_dict' in ckpt:
            ckpt = ckpt['ema_state_dict']
        elif 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        model.load_state_dict(ckpt, strict=True)
        model.train()

        test_input = torch.randn(1, 3, 480, 640)
        with torch.no_grad():
            output = model(test_input)

        p4_obj = torch.sigmoid(output['p4_obj'])
        p4_std = p4_obj.std().item()
        p4_min = p4_obj.min().item()
        p4_max = p4_obj.max().item()

        assert p4_max - p4_min > 1e-5, \
            f"P4 output range too small: [{p4_min:.6f}, {p4_max:.6f}]"
        assert p4_max > 0.001, \
            f"P4 max sigmoid too low: {p4_max}"
        assert p4_std > 1e-6, \
            f"P4 output std too low: {p4_std:.2e}"

    @pytest.mark.skipif(not os.path.exists("models/face_cnn_v5_best.pth"),
                        reason="v5 checkpoint not found")
    def test_v5_content_sensitivity(self):
        import torch
        from src.cv.face_detector_cnn import FaceFCNv5

        model = FaceFCNv5()
        ckpt = torch.load("models/face_cnn_v5_best.pth", map_location='cpu',
                           weights_only=True)
        if 'ema_state_dict' in ckpt:
            ckpt = ckpt['ema_state_dict']
        elif 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        model.load_state_dict(ckpt, strict=True)
        model.train()

        zero_in = torch.zeros(1, 3, 480, 640)
        one_in = torch.ones(1, 3, 480, 640)
        with torch.no_grad():
            out_zero = torch.sigmoid(model(zero_in)['p4_obj'])
            out_one = torch.sigmoid(model(one_in)['p4_obj'])

        delta = (out_one - out_zero).abs().max().item()
        assert delta > 1e-5, \
            f"No content sensitivity: delta={delta:.2e}"

    def test_v5_fresh_model_produces_output(self):
        import torch
        from src.cv.face_detector_cnn import FaceFCNv5

        model = FaceFCNv5()
        model.eval()

        test_input = torch.randn(1, 3, 480, 640)
        with torch.no_grad():
            output = model(test_input)

        assert 'p4_obj' in output
        assert 'p4_bbox' in output
        assert output['p4_obj'].shape == (1, 1, 60, 80)


class TestGestureClassifier:
    def test_hand_detector_rf_fallback(self):
        from src.cv.gesture_classifier import HandDetector

        det = HandDetector(rf_model_path="/nonexistent/rf_model.pkl")
        assert det.use_ml is False
        assert det.rf_model is None

    def test_gesture_classifier_on_synthetic(self):
        from src.cv.gesture_classifier import GestureClassifier, GestureResult
        from src.utils.config import load_config

        cfg = load_config("config/default.yaml")
        classifier = GestureClassifier(
            svm_path=cfg.models.gesture_svm,
            scaler_path=cfg.models.gesture_scaler,
            pca_path=cfg.models.gesture_pca,
            min_confidence=cfg.gesture.min_confidence,
        )
        hand_roi = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = classifier.predict(hand_roi)

        assert isinstance(result, GestureResult)
        assert isinstance(result.gesture, str)
        assert 0.0 <= result.confidence <= 1.0, \
            f"Invalid confidence: {result.confidence}"


class TestModelEMA:
    def test_ema_buffers_synced(self):
        import torch
        import torch.nn as nn
        import copy
        from src.training.train_face_cnn import ModelEMA

        class SimpleBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(4)

            def forward(self, x):
                return self.bn(x)

        model = SimpleBN()
        model.train()
        for _ in range(10):
            x = torch.randn(2, 4, 8, 8)
            model(x)

        ema = ModelEMA(model, decay=0.999)

        for _ in range(5):
            x = torch.randn(2, 4, 8, 8)
            model.train()
            model(x)
            ema.update(model)

        ema_model = ema.ema_model
        ema_bn_mean = ema_model.bn.running_mean
        live_bn_mean = model.bn.running_mean

        assert not torch.allclose(ema_bn_mean, torch.zeros_like(ema_bn_mean)), \
            "EMA BN running_mean is still at init (zeros) — buffer sync broken"
        assert torch.allclose(ema_bn_mean, live_bn_mean, atol=1e-2), \
            f"EMA BN running_mean diverged from live model. " \
            f"EMA: {ema_bn_mean[:2].tolist()}, Live: {live_bn_mean[:2].tolist()}"


class TestFaceCNNv5HealthCheck:
    def test_health_check_raises_on_dead_model(self):
        import torch
        import torch.nn as nn
        from src.cv.face_detector_cnn import FaceFCNv5, FaceCNNv5

        class DeadModel(FaceFCNv5):
            def forward(self, x):
                out = super().forward(x)
                out['p4_obj'] = torch.full_like(out['p4_obj'], -4.6)
                return out

        dead = DeadModel()
        dead.eval()

        detector = FaceCNNv5.__new__(FaceCNNv5)
        detector.device = torch.device('cpu')
        detector.model = dead

        with pytest.raises(RuntimeError, match="collapsed"):
            detector._sanity_check(dead)

    def test_health_check_warns_on_low_output(self):
        import torch
        import torch.nn as nn
        from src.cv.face_detector_cnn import FaceFCNv5, FaceCNNv5

        class WeakModel(FaceFCNv5):
            def forward(self, x):
                out = super().forward(x)
                out['p4_obj'] = torch.full((1, 1, 60, 80), -2.8) + torch.randn(
                    1, 1, 60, 80) * 0.01
                return out

        weak = WeakModel()
        weak.eval()

        detector = FaceCNNv5.__new__(FaceCNNv5)
        detector.device = torch.device('cpu')
        detector.model = weak

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            detector._sanity_check(weak)
            warning_msgs = [str(m.message) for m in w if m.category == UserWarning]
            assert any("max sigmoid" in msg.lower() for msg in warning_msgs), \
                f"Expected low-output warning, got: {warning_msgs}"


class TestHeadInit:
    def test_obj_pred_bias_init_value(self):
        import torch
        from src.cv.face_detector_cnn import AnchorFreeHead

        head = AnchorFreeHead(64)
        bias_val = head.obj_pred.bias.item()
        assert bias_val == -2.5, \
            f"obj_pred bias init is {bias_val}, expected -2.5"

    def test_head_uses_kaiming_init(self):
        import torch
        from src.cv.face_detector_cnn import AnchorFreeHead

        head = AnchorFreeHead(64)
        weight_std = head.obj_pred.weight.std().item()
        expected_kaiming = (2.0 / 64) ** 0.5
        assert weight_std > 0.03, \
            f"Weight std {weight_std:.4f} too small — still using normal(0.01)?"
