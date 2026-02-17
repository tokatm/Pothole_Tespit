from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.modeling import DFineLikeDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('Export DFine-like model to ONNX for Jetson Orin')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--onnx', type=str, required=True)
    p.add_argument('--height', type=int, default=640)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--opset', type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = DFineLikeDetector(num_classes=2)
    state = torch.load(ckpt_path, map_location='cpu')
    weights = state['model'] if isinstance(state, dict) and 'model' in state else state
    model.load_state_dict(weights, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['pred_logits', 'pred_boxes'],
        dynamic_axes={
            'images': {0: 'batch'},
            'pred_logits': {0: 'batch'},
            'pred_boxes': {0: 'batch'},
        },
    )

    print(f'ONNX export tamamlandi: {out_path}')
    print('Jetson Orin TensorRT icin FP16/INT8 calibration asamasina gecilebilir.')


if __name__ == '__main__':
    main()
