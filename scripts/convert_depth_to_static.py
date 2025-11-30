#!/usr/bin/env python3
"""
動的シェイプのDepth Anything V2 ONNXモデルを静的シェイプに変換

使用法:
    python scripts/convert_depth_to_static.py \
        --input /path/to/depth_anything_v2_vits_dynamic.onnx \
        --output output/models/depth_anything_v2_static.onnx \
        --width 518 --height 518
"""

import argparse
from pathlib import Path
import onnx
from onnx import helper, TensorProto
import numpy as np


def convert_to_static_shape(
    input_path: str,
    output_path: str,
    width: int = 518,
    height: int = 518
):
    """
    動的シェイプのONNXモデルを静的シェイプに変換

    Args:
        input_path: 入力ONNXファイル
        output_path: 出力ONNXファイル
        width: 入力幅
        height: 入力高さ
    """
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)

    # 入力の形状を更新
    for inp in model.graph.input:
        if inp.type.tensor_type.shape.dim:
            dims = inp.type.tensor_type.shape.dim
            # batch=1, channels=3, height, width
            if len(dims) == 4:
                dims[0].dim_value = 1  # batch
                dims[1].dim_value = 3  # channels
                dims[2].dim_value = height
                dims[3].dim_value = width
                print(f"Updated input '{inp.name}' shape to [1, 3, {height}, {width}]")

    # 出力の形状を更新（動的からクリア）
    for out in model.graph.output:
        if out.type.tensor_type.shape.dim:
            dims = out.type.tensor_type.shape.dim
            # 出力も固定
            if len(dims) == 4:
                dims[0].dim_value = 1
                # dims[1] は出力チャンネル（通常1）
                dims[2].dim_value = height
                dims[3].dim_value = width
                print(f"Updated output '{out.name}' shape")

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    onnx.save(model, str(output_path))
    print(f"\n✓ Saved static model to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert dynamic ONNX model to static shape"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input ONNX file (dynamic shape)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/models/depth_anything_v2_static.onnx",
        help="Output ONNX file (static shape)"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=518,
        help="Input width (default: 518)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=518,
        help="Input height (default: 518)"
    )

    args = parser.parse_args()

    convert_to_static_shape(
        args.input,
        args.output,
        args.width,
        args.height
    )


if __name__ == "__main__":
    main()
