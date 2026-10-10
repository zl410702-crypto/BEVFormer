"""Extract the deployed BEVFormer ResNet-50 backbone as a standalone ONNX.

The extraction boundary is discovered from the FPN lateral convolution rather
than from an unstable numeric node name.  The pass intentionally fails closed
if the expected BEVFormer-tiny C5 contract is not present.
"""

import argparse
import json
from pathlib import Path

import onnx
from onnx import shape_inference, utils


FPN_LATERAL_WEIGHT = 'model.img_neck.lateral_convs.0.conv.weight'
EXPECTED_INPUT_SHAPE = [1, 6, 3, 480, 800]
EXPECTED_C5_SHAPE = [6, 2048, 15, 25]


def _tensor_shape(value_info):
    tensor_type = value_info.type.tensor_type
    return [dimension.dim_value for dimension in tensor_type.shape.dim]


def find_resnet50_c5(model):
    """Return the tensor immediately before FPN, with strict validation."""
    lateral_nodes = [
        node for node in model.graph.node
        if node.op_type == 'Conv' and FPN_LATERAL_WEIGHT in node.input
    ]
    if len(lateral_nodes) != 1:
        raise ValueError(
            'expected exactly one FPN lateral Conv using {!r}, found {}'.format(
                FPN_LATERAL_WEIGHT, len(lateral_nodes)))
    lateral = lateral_nodes[0]
    if not lateral.input:
        raise ValueError('FPN lateral Conv has no data input')
    c5_name = lateral.input[0]
    producers = {
        output: node for node in model.graph.node for output in node.output
    }
    producer = producers.get(c5_name)
    if producer is None or producer.op_type != 'Relu':
        raise ValueError(
            'expected ResNet C5 to be produced by Relu, got {}'.format(
                None if producer is None else producer.op_type))

    inferred = shape_inference.infer_shapes(model)
    values = {
        item.name: item for item in
        list(inferred.graph.input) + list(inferred.graph.value_info) +
        list(inferred.graph.output)
    }
    if 'images' not in values:
        raise ValueError('expected graph input images')
    input_shape = _tensor_shape(values['images'])
    c5_shape = _tensor_shape(values[c5_name]) if c5_name in values else None
    if input_shape != EXPECTED_INPUT_SHAPE:
        raise ValueError('unexpected images shape: {}'.format(input_shape))
    if c5_shape != EXPECTED_C5_SHAPE:
        raise ValueError('unexpected ResNet C5 shape: {}'.format(c5_shape))
    return c5_name, producer.name, lateral.name


def extract_resnet50(input_path, output_path):
    """Extract and validate the exact ResNet-50 subgraph used by BEVFormer."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('output must not overwrite the source ONNX')
    model = onnx.load(str(input_path), load_external_data=False)
    c5_name, producer_name, consumer_name = find_resnet50_c5(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    utils.extract_model(
        str(input_path), str(output_path), ['images'], [c5_name])

    extracted = onnx.load(str(output_path))
    onnx.checker.check_model(extracted)
    inferred = shape_inference.infer_shapes(extracted)
    output_shape = _tensor_shape(inferred.graph.output[0])
    conv_count = sum(node.op_type == 'Conv' for node in extracted.graph.node)
    custom_count = sum(
        node.domain == 'bevformer' and
        node.op_type == 'MSDeformableAttention'
        for node in extracted.graph.node)
    if output_shape != EXPECTED_C5_SHAPE:
        raise ValueError('extracted output shape drifted: {}'.format(
            output_shape))
    if conv_count != 53:
        raise ValueError('expected ResNet-50 to contain 53 Conv nodes, got {}'.format(
            conv_count))
    if custom_count != 0:
        raise ValueError('backbone unexpectedly contains custom MSDA nodes')
    return {
        'source': str(input_path),
        'output': str(output_path),
        'input': {'name': 'images', 'shape': EXPECTED_INPUT_SHAPE,
                  'dtype': 'float32'},
        'output_tensor': {'name': c5_name, 'shape': output_shape,
                          'dtype': 'float32'},
        'boundary_producer': producer_name,
        'boundary_consumer': consumer_name,
        'node_count': len(extracted.graph.node),
        'conv_count': conv_count,
        'msda_count': custom_count,
        'checker': 'PASS',
        'shape_inference': 'PASS',
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract BEVFormer-tiny ResNet-50 C5 ONNX')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = extract_resnet50(args.input, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
