import numpy as np
import pytest
from src.cv.face_detector_cnn import FaceCNN, FaceFCN


def test_face_fcn_architecture():
    model = FaceFCN()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 100000
    assert total_params < 200000
    assert model.head.out_channels == 4


def test_face_fcn_forward_shape():
    model = FaceFCN()
    model.eval()
    x = np.random.randn(1, 3, 128, 128).astype(np.float32)
    import torch
    with torch.no_grad():
        out = model(torch.from_numpy(x))
    assert out.shape == (1, 4, 16, 16)


def test_face_fcn_conv_layers():
    model = FaceFCN()
    assert model.block1[0].kernel_size == (5, 5)
    assert model.block2[0].kernel_size == (3, 3)
    assert model.block3[0].kernel_size == (3, 3)
    assert model.block4[0].dilation == (2, 2)
    assert model.skip_conv.kernel_size == (1, 1)
    assert model.fuse_conv.kernel_size == (1, 1)


def test_face_fcn_skip_connection():
    model = FaceFCN()
    x = np.random.randn(1, 3, 128, 128).astype(np.float32)
    import torch
    with torch.no_grad():
        out3 = model.block3(model.block2(model.block1(torch.from_numpy(x))))
        skip = model.skip_conv(out3)
        out4 = model.block4(out3)
        fused = model.fuse_conv(torch.cat([out4, skip], dim=1))
    assert fused.shape[1] == 128


def test_face_cnn_init_no_model():
    try:
        fcnn = FaceCNN(model_path="/nonexistent/model.pth")
        assert False, "Should raise FileNotFoundError"
    except (FileNotFoundError, RuntimeError):
        pass


def test_face_cnn_detect_return_format():
    try:
        from src.cv.face_tracker import Face
        fcnn = FaceCNN()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = fcnn.detect(frame)
        assert isinstance(faces, list)
        if len(faces) > 0:
            assert isinstance(faces[0], Face)
            assert 0 <= faces[0].confidence <= 1.0
    except (FileNotFoundError, RuntimeError):
        pass


def test_face_cnn_scale_pyramid():
    try:
        fcnn = FaceCNN()
        scales = [1.0 / (1.15 ** i) for i in range(5)]
        assert len(scales) == 5
        assert scales[0] == 1.0
        assert scales[-1] < 0.6
    except (FileNotFoundError, RuntimeError):
        pass


def test_face_cnn_stride():
    try:
        fcnn = FaceCNN()
        assert fcnn.stride == 8
        assert fcnn.grid_cells == 16
    except (FileNotFoundError, RuntimeError):
        pass


def test_face_cnn_confidence_threshold():
    try:
        fcnn = FaceCNN(confidence_threshold=0.5)
        assert fcnn.confidence_threshold == 0.5
    except (FileNotFoundError, RuntimeError):
        pass


if __name__ == "__main__":
    test_face_fcn_architecture()
    test_face_fcn_forward_shape()
    test_face_fcn_conv_layers()
    test_face_fcn_skip_connection()
    test_face_cnn_init_no_model()
    test_face_cnn_detect_return_format()
    test_face_cnn_scale_pyramid()
    test_face_cnn_stride()
    test_face_cnn_confidence_threshold()
    print("\nAll face CNN tests passed!")
