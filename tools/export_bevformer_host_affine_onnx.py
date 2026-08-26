#!/usr/bin/env python
"""Create the production host-affine ONNX candidate from the compat model."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import onnx

from deployment.host_affine_contract import rewrite_host_affine_input


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise RuntimeError('refusing to overwrite {}'.format(args.output))
    source = onnx.load(str(args.input))
    model, rewrite = rewrite_host_affine_input(source)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    counts = Counter((node.domain or 'ai.onnx', node.op_type)
                     for node in model.graph.node)
    report = {
        'input': str(args.input),
        'output': str(args.output),
        'size_bytes': args.output.stat().st_size,
        'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'node_count': len(model.graph.node),
        'graph_inputs': [value.name for value in model.graph.input],
        'graph_outputs': [value.name for value in model.graph.output],
        'msda_count': counts[('bevformer', 'MSDeformableAttention')],
        'onnx_checker': 'PASS',
        'shape_inference': 'PASS',
        'rewrite': rewrite,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
