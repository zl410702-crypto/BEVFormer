"""Targeted TensorRT 8.4 compatibility rewrite for one audited ONNX Slice."""

import numpy as np
import onnx
from onnx import numpy_helper


def constant_value(name, producers, initializers, depth=0):
    """Evaluate the small Constant/Cast/Unsqueeze chain used by Slice axes."""
    if depth > 10:
        return None
    initializer = initializers.get(name)
    if initializer is not None:
        return numpy_helper.to_array(initializer)
    node = producers.get(name)
    if node is None:
        return None
    if node.op_type == 'Constant':
        attribute = next((item for item in node.attribute
                          if item.name == 'value'), None)
        return None if attribute is None else numpy_helper.to_array(attribute.t)
    if node.op_type == 'Cast':
        source = constant_value(node.input[0], producers, initializers,
                                depth + 1)
        target = next((item.i for item in node.attribute
                       if item.name == 'to'), None)
        if source is None or target not in (onnx.TensorProto.INT32,
                                            onnx.TensorProto.INT64):
            return None
        return source.astype(np.int32 if target == onnx.TensorProto.INT32
                             else np.int64)
    if node.op_type == 'Unsqueeze' and len(node.input) == 2:
        source = constant_value(node.input[0], producers, initializers,
                                depth + 1)
        axes = constant_value(node.input[1], producers, initializers,
                              depth + 1)
        if source is None or axes is None:
            return None
        result = source
        for axis in sorted(int(item) for item in axes.reshape(-1)):
            result = np.expand_dims(result, axis)
        return result
    if node.op_type == 'Concat':
        values = [constant_value(item, producers, initializers, depth + 1)
                  for item in node.input]
        axis = next((item.i for item in node.attribute
                     if item.name == 'axis'), None)
        if axis is None or any(item is None for item in values):
            return None
        return np.concatenate(values, axis=axis)
    return None


def canonicalize_static_slice_axes(model):
    """Canonicalize every dynamic-looking but provably constant Slice axis.

    Direct Initializer/Constant inputs are already parser-friendly and remain
    untouched. Runtime-derived axes are skipped. The evaluator intentionally
    accepts only a small whitelist of constant-preserving operators.
    """
    producers = {output: producer for producer in model.graph.node
                 for output in producer.output}
    initializers = {item.name: item for item in model.graph.initializer}
    report = {'canonicalized': [], 'skipped': []}
    for node in model.graph.node:
        if node.op_type != 'Slice':
            continue
        if len(node.input) < 4 or not node.input[3]:
            report['skipped'].append({
                'node': node.name, 'reason': 'axes input is absent'})
            continue
        axes_name = node.input[3]
        producer = producers.get(axes_name)
        if axes_name in initializers or (producer is not None and
                                         producer.op_type == 'Constant'):
            continue
        axes = constant_value(axes_name, producers, initializers)
        if axes is None:
            report['skipped'].append({
                'node': node.name,
                'reason': 'axes depends on a non-whitelisted/runtime value'})
            continue
        if axes.dtype != np.int64 or axes.ndim != 1 or axes.size == 0:
            raise ValueError('{} axes has invalid dtype/rank: {}'.format(
                node.name, axes))
        values = [int(item) for item in axes]
        if len(set(values)) != len(values):
            raise ValueError('{} axes contains duplicates: {}'.format(
                node.name, values))
        static_name = '{}_axes_static'.format(node.name)
        if static_name in initializers:
            raise ValueError('initializer already exists: {}'.format(
                static_name))
        model.graph.initializer.append(numpy_helper.from_array(
            axes.copy(), static_name))
        node.input[3] = static_name
        report['canonicalized'].append({
            'node': node.name, 'old_axes': axes_name,
            'new_axes': static_name, 'value': values})
    return report


def canonicalize_slice_axes(model, node_name, expected_axes):
    """Replace only the audited Slice's foldable axes chain by an initializer.

    Starts, ends, steps, data and all other Slice nodes remain untouched.
    The function fails closed unless the existing axes evaluates exactly to
    the caller-provided INT64 value and the Slice schema has five inputs.
    """
    nodes = [node for node in model.graph.node if node.name == node_name]
    if len(nodes) != 1:
        raise ValueError('expected one {}, found {}'.format(node_name,
                                                            len(nodes)))
    node = nodes[0]
    if node.op_type != 'Slice' or len(node.input) != 5:
        raise ValueError('{} is not a five-input Slice'.format(node_name))
    producers = {output: producer for producer in model.graph.node
                 for output in producer.output}
    initializers = {item.name: item for item in model.graph.initializer}
    expected_axes = np.asarray(expected_axes, dtype=np.int64)
    if expected_axes.ndim != 1 or expected_axes.size == 0:
        raise ValueError('expected_axes must be a non-empty rank-1 value')
    axes = constant_value(node.input[3], producers, initializers)
    if axes is None or axes.dtype != np.int64 or \
            axes.shape != expected_axes.shape or \
            not np.array_equal(axes, expected_axes):
        raise ValueError('{} axes is not proven INT64 {}: {}'.format(
            node_name, expected_axes.tolist(), axes))
    static_name = '{}_axes_static'.format(node_name)
    if static_name in initializers:
        raise ValueError('initializer already exists: {}'.format(static_name))
    model.graph.initializer.append(numpy_helper.from_array(
        expected_axes, static_name))
    old_axes = node.input[3]
    node.input[3] = static_name
    return {'node': node_name, 'old_axes': old_axes,
            'new_axes': static_name, 'value': expected_axes.tolist()}


def canonicalize_slice_axes_file(path, node_name, expected_axes):
    model = onnx.load(str(path))
    result = canonicalize_slice_axes(
        model, node_name=node_name, expected_axes=expected_axes)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return result


def canonicalize_static_slice_axes_file(path):
    model = onnx.load(str(path))
    report = canonicalize_static_slice_axes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return report
