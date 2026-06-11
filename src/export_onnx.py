"""Export final FERNv2 model to ONNX for production inference."""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_v2 import FERNv2


def export(
    checkpoint_path: str = r"..\models_final\fern_v2_latest.pth",
    output_path: str = r"..\models_final\fern_v2.onnx",
    num_joints: int = 10,
    num_classes: int = 8,
    cnn_out: int = 64,
    seq_len: int = 60,
    input_features: int = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_full = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt_full.get("args") if isinstance(ckpt_full, dict) else None
    if ckpt_args:
        if input_features is None:
            n_cam = ckpt_args.get("n_cameras", 1)
            input_features = (num_joints * 3) + (n_cam if n_cam > 1 else 0)
        cnn_out = ckpt_args.get("cnn_out", cnn_out)
        dropout = ckpt_args.get("dropout", 0.6)

    model = FERNv2(
        num_joints=num_joints,
        num_classes=num_classes,
        cnn_out=cnn_out,
        lstm_hidden=0,
        lstm_layers=1,
        dropout=dropout,
        input_features=input_features,
    ).to(device)

    if isinstance(ckpt_full, dict):
        state = ckpt_full.get("model_state") or ckpt_full.get("model_state_dict") or ckpt_full
    else:
        state = ckpt_full
    model.load_state_dict(state, strict=True)
    model.eval()

    feat_dim = input_features or num_joints * 3
    dummy = torch.randn(1, seq_len, feat_dim, device=device)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX exported -> {output_path}")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check: OK")

    session = ort.InferenceSession(output_path)
    ort_input = {session.get_inputs()[0].name: dummy.cpu().numpy()}
    ort_out = session.run(None, ort_input)[0]
    torch_out = model(dummy).detach().cpu().numpy()
    diff = np.abs(ort_out - torch_out).max()
    print(f"Max diff torch vs onnxruntime: {diff:.2e}")
    assert diff < 5e-3, f"ONNX mismatch: {diff:.2e}"
    print("ONNX inference: OK")


if __name__ == "__main__":
    import argparse
    ROOT = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Export FERNv2 checkpoint to ONNX")
    p.add_argument("--checkpoint_path", default=str(ROOT / "models_final_v2" / "fern_v2_latest.pth"))
    p.add_argument("--output_path", default=str(ROOT / "models_final_v2" / "fern_v2.onnx"))
    p.add_argument("--num_joints", type=int, default=10)
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--cnn_out", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--input_features", type=int, default=None)
    a = p.parse_args()
    export(
        checkpoint_path=a.checkpoint_path,
        output_path=a.output_path,
        num_joints=a.num_joints,
        num_classes=a.num_classes,
        cnn_out=a.cnn_out,
        seq_len=a.seq_len,
        input_features=a.input_features,
    )
