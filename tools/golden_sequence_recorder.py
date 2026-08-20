"""Non-invasive, instance-local tensor recorder for BEVFormer inference."""

import types

import torch


class GoldenSequenceRecorder:
    def __init__(self, model, dumper):
        self.model = model
        self.dumper = dumper
        self.frame = None
        self._handles = []
        self._originals = []
        self._linear_outputs = {}

    def path(self, relative):
        return '{}/{}'.format(self.frame, relative)

    def dump(self, name, value, relative=None):
        if self.frame is not None and isinstance(value, torch.Tensor):
            return self.dumper.dump(name, value, self.path(relative or name))
        return None

    def _hook(self, module, callback):
        self._handles.append(module.register_forward_hook(callback))

    def _wrap(self, obj, method_name, callback):
        original = getattr(obj, method_name)
        self._originals.append((obj, method_name, original))

        def wrapped(this, *args, **kwargs):
            return callback(original, args, kwargs)
        setattr(obj, method_name, types.MethodType(wrapped, obj))

    def install(self):
        model = self.model
        head = model.pts_bbox_head
        transformer = head.transformer
        encoder = transformer.encoder

        def backbone_hook(module, inputs, output):
            values = list(output.values()) if isinstance(output, dict) else output
            values = values if isinstance(values, (list, tuple)) else [values]
            stages = list(getattr(model.img_backbone, 'out_indices', range(len(values))))
            for index, value in zip(stages, values):
                self.dump('backbone.c{}'.format(index + 2), value,
                          'backbone/c{}.npy'.format(index + 2))
        self._hook(model.img_backbone, backbone_hook)

        def fpn_hook(module, inputs, output):
            for index, value in enumerate(output):
                self.dump('fpn.level_{}'.format(index), value,
                          'fpn/level_{}.npy'.format(index))
        self._hook(model.img_neck, fpn_hook)

        def head_forward(original, args, kwargs):
            result = original(*args, **kwargs)
            if isinstance(result, dict):
                self.dump('bev.final_bev', result['bev_embed'], 'bev/final_bev.npy')
                self.dump('detection.cls_scores', result['all_cls_scores'],
                          'detection/cls_scores.npy')
                self.dump('detection.bbox_preds', result['all_bbox_preds'],
                          'detection/bbox_preds.npy')
            return result
        self._wrap(head, 'forward', head_forward)

        def positional_hook(module, inputs, output):
            self.dump('bev.bev_pos', output, 'bev/bev_pos.npy')
            self.dump('bev.bev_query', head.bev_embedding.weight,
                      'bev/bev_query.npy')
        self._hook(head.positional_encoding, positional_hook)

        def forward_test(original, args, kwargs):
            stored = model.prev_frame_info['prev_bev']
            if stored is not None:
                self.dump('temporal.prev_bev_stored', stored.clone(),
                          'temporal/prev_bev_stored.npy')
            return original(*args, **kwargs)
        self._wrap(model, 'forward_test', forward_test)

        def encoder_forward(original, args, kwargs):
            prev = kwargs.get('prev_bev')
            if prev is not None:
                self.dump('temporal.prev_bev_aligned', prev,
                          'temporal/prev_bev_aligned.npy')
            self.dump('bev.spatial_shapes', kwargs.get('spatial_shapes'),
                      'bev/spatial_shapes.npy')
            self.dump('bev.level_start_index', kwargs.get('level_start_index'),
                      'bev/level_start_index.npy')
            self.dump('temporal.shift', kwargs.get('shift'),
                      'temporal/shift.npy')
            return original(*args, **kwargs)
        self._wrap(encoder, 'forward', encoder_forward)

        def refs(original, args, kwargs):
            output = original(*args, **kwargs)
            dim = kwargs.get('dim', args[4] if len(args) > 4 else '3d')
            self.dump('bev.reference_points_{}'.format(dim), output,
                      'bev/reference_points_{}.npy'.format(dim))
            return output
        self._wrap(encoder, 'get_reference_points', refs)

        def sampling(original, args, kwargs):
            output = original(*args, **kwargs)
            self.dump('bev.reference_points_cam', output[0],
                      'bev/reference_points_cam.npy')
            self.dump('bev.bev_mask', output[1], 'bev/bev_mask.npy')
            return output
        self._wrap(encoder, 'point_sampling', sampling)

        for index, layer in enumerate(encoder.layers):
            temporal = layer.attentions[0]
            spatial = layer.attentions[1]
            deformable = spatial.deformable_attention

            def temporal_forward(original, args, kwargs, index=index,
                                 temporal=temporal):
                base = 'temporal/layer_{}'.format(index)
                labels = ('query', 'key', 'value', 'identity')
                for pos, label in enumerate(labels):
                    value = args[pos] if len(args) > pos else kwargs.get(label)
                    self.dump('temporal.layer_{}.{}'.format(index, label), value,
                              '{}/{}.npy'.format(base, label))
                self.dump('temporal.layer_{}.reference_points'.format(index),
                          kwargs.get('reference_points'),
                          '{}/reference_points.npy'.format(base))
                query = args[0] if args else kwargs['query']
                supplied_value = args[2] if len(args) > 2 else kwargs.get('value')
                if supplied_value is None:
                    bs, length, channels = query.shape
                    effective_value = torch.stack([query, query], 1).reshape(
                        bs * temporal.num_bev_queue, length, channels)
                else:
                    effective_value = supplied_value
                self.dump('temporal.layer_{}.value_effective'.format(index),
                          effective_value, '{}/value_effective.npy'.format(base))
                output = original(*args, **kwargs)
                offsets = self._linear_outputs.pop((index, 'temporal_offsets'))
                weights = self._linear_outputs.pop((index, 'temporal_weights'))
                bs, nq = query.shape[:2]
                offsets = offsets.view(bs, nq, temporal.num_heads,
                                       temporal.num_bev_queue,
                                       temporal.num_levels, temporal.num_points, 2)
                weights = weights.view(bs, nq, temporal.num_heads,
                                       temporal.num_bev_queue,
                                       temporal.num_levels * temporal.num_points)
                weights = weights.softmax(-1).view(
                    bs, nq, temporal.num_heads, temporal.num_bev_queue,
                    temporal.num_levels, temporal.num_points)
                offsets = offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
                    bs * temporal.num_bev_queue, nq, temporal.num_heads,
                    temporal.num_levels, temporal.num_points, 2)
                weights = weights.permute(0, 3, 1, 2, 4, 5).reshape(
                    bs * temporal.num_bev_queue, nq, temporal.num_heads,
                    temporal.num_levels, temporal.num_points)
                refs_value = kwargs.get('reference_points')
                shapes = kwargs.get('spatial_shapes')
                normalizer = torch.stack([shapes[..., 1], shapes[..., 0]], -1)
                locations = refs_value[:, :, None, :, None, :] + offsets / \
                    normalizer[None, None, None, :, None, :]
                for label, tensor in (('sampling_offsets', offsets),
                                      ('attention_weights', weights),
                                      ('sampling_locations', locations)):
                    self.dump('temporal.layer_{}.{}'.format(index, label), tensor,
                              '{}/{}.npy'.format(base, label))
                self.dump('temporal.layer_{}.output'.format(index), output,
                          '{}/output.npy'.format(base))
                return output
            self._wrap(temporal, 'forward', temporal_forward)
            self._hook(temporal.sampling_offsets,
                       lambda m, i, o, index=index: self._linear_outputs.__setitem__(
                           (index, 'temporal_offsets'), o))
            self._hook(temporal.attention_weights,
                       lambda m, i, o, index=index: self._linear_outputs.__setitem__(
                           (index, 'temporal_weights'), o))

            def spatial_forward(original, args, kwargs, index=index):
                base = 'spatial/layer_{}'.format(index)
                for pos, label in enumerate(('query', 'key', 'value')):
                    value = args[pos] if len(args) > pos else kwargs.get(label)
                    self.dump('spatial.layer_{}.{}'.format(index, label), value,
                              '{}/{}.npy'.format(base, label))
                for label in ('reference_points_cam', 'bev_mask',
                              'spatial_shapes', 'level_start_index'):
                    self.dump('spatial.layer_{}.{}'.format(index, label),
                              kwargs.get(label), '{}/{}.npy'.format(base, label))
                output = original(*args, **kwargs)
                self.dump('spatial.layer_{}.output'.format(index), output,
                          '{}/output.npy'.format(base))
                return output
            self._wrap(spatial, 'forward', spatial_forward)

            def deform_forward(original, args, kwargs, index=index, module=deformable):
                base = 'spatial/layer_{}/deformable'.format(index)
                query = args[0] if args else kwargs.get('query')
                value = kwargs.get('value', args[2] if len(args) > 2 else None)
                refs_value = kwargs.get('reference_points')
                shapes = kwargs.get('spatial_shapes')
                self.dump('spatial.layer_{}.deformable.query'.format(index), query,
                          '{}/query.npy'.format(base))
                self.dump('spatial.layer_{}.deformable.value'.format(index), value,
                          '{}/value.npy'.format(base))
                output = original(*args, **kwargs)
                offsets = self._linear_outputs.pop((index, 'offsets'))
                weights = self._linear_outputs.pop((index, 'weights'))
                bs, nq = query.shape[:2]
                offsets = offsets.view(bs, nq, module.num_heads,
                                       module.num_levels, module.num_points, 2)
                weights = weights.view(bs, nq, module.num_heads,
                                       module.num_levels, module.num_points).softmax(-1)
                normalizer = torch.stack([shapes[..., 1], shapes[..., 0]], -1)
                anchors = refs_value.shape[2]
                normalized = offsets / normalizer[None, None, None, :, None, :]
                normalized = normalized.view(bs, nq, module.num_heads,
                                             module.num_levels,
                                             module.num_points // anchors,
                                             anchors, 2)
                locations = refs_value[:, :, None, None, None, :, :] + normalized
                locations = locations.view(bs, nq, module.num_heads,
                                          module.num_levels, module.num_points, 2)
                for label, tensor in (('sampling_offsets', offsets),
                                      ('attention_weights', weights),
                                      ('sampling_locations', locations)):
                    self.dump('spatial.layer_{}.deformable.{}'.format(index, label),
                              tensor, '{}/{}.npy'.format(base, label))
                self.dump('spatial.layer_{}.deformable.output'.format(index), output,
                          '{}/output.npy'.format(base))
                return output
            self._wrap(deformable, 'forward', deform_forward)

            self._hook(deformable.sampling_offsets,
                       lambda m, i, o, index=index: self._linear_outputs.__setitem__(
                           (index, 'offsets'), o))
            self._hook(deformable.attention_weights,
                       lambda m, i, o, index=index: self._linear_outputs.__setitem__(
                           (index, 'weights'), o))

        def decoder_forward(original, args, kwargs):
            self.dump('decoder.reference_points_initial', kwargs.get('reference_points'),
                      'decoder/reference_points_initial.npy')
            output = original(*args, **kwargs)
            states, references = output
            for index, state in enumerate(states):
                self.dump('decoder.layer_{}.object_feature'.format(index), state,
                          'decoder/layer_{}/object_feature.npy'.format(index))
            self.dump('decoder.reference_points_after_refinement', references,
                      'decoder/reference_points_after_refinement.npy')
            return output
        self._wrap(transformer.decoder, 'forward', decoder_forward)
        return self

    def close(self):
        for handle in self._handles:
            handle.remove()
        for obj, name, original in reversed(self._originals):
            setattr(obj, name, original)
