"""
ONNX export — instantiate a compiled model and export to ONNX format.

Usage:
    export_onnx(code_string, model_class_name, input_shape, onnx_path, device="cpu")
"""

import importlib.util
import os
import sys
import tempfile

import torch


def export_onnx(code_string, model_class_name, input_shape, onnx_path, device="cpu", batch_size=1):
    """
    Execute generated code, instantiate model, and export to ONNX.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code_string)
        tmp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("_onnx_model", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model_cls = getattr(mod, model_class_name)
        model = model_cls().to(device)
        model.eval()

        dummy = torch.randn(batch_size, *input_shape).to(device)

        os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        print(f"ONNX exported: {onnx_path}", file=sys.stderr)
    finally:
        os.unlink(tmp_path)
