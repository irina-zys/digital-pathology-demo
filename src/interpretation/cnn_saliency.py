"""Class-specific CNN Grad-CAM through a trained full-bag slide classifier."""

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def patch_gradcam(head, loaded, cached_features, index, rgb, *, class_index):
    """Explain a ResNet patch's contribution to a selected full-slide class logit.

    Grad-CAM weights layer4 feature channels by spatially averaged gradients of
    the selected logit, sums their activations, then retains positive support.
    Other patch embeddings stay fixed; learned MIL weights are recomputed over
    the complete bag. This is coarse class sensitivity, not attention or a mask.
    Reference: https://arxiv.org/abs/1610.02391
    """
    if loaded['metadata']['name'] != 'resnet50':
        raise ValueError('This Grad-CAM adapter supports the configured ResNet-50 encoder')
    model = loaded['model']
    if head.training or model.training:
        raise ValueError('Grad-CAM requires evaluation-mode encoder and slide head')
    if torch.is_inference_mode_enabled():
        raise ValueError('Grad-CAM needs autograd; call outside torch.inference_mode')
    if (cached_features.ndim != 2 or not len(cached_features)
            or type(index) is not int or not 0 <= index < len(cached_features)):
        raise ValueError('Expected a complete nonempty feature bag and a valid patch index')
    if type(class_index) is not int or not 0 <= class_index < head.classifier.out_features:
        raise ValueError('class_index must select a classifier output')
    rgb = np.asarray(rgb)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or not 0 < rgb.shape[0] * rgb.shape[1] <= 1_000_000):
        raise ValueError('Expected a bounded uint8 RGB patch')
    activations = []

    def capture(_module, _inputs, output):
        activations.append(output)

    handle = model.layer4.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            image = loaded['transform'](Image.fromarray(rgb)).unsqueeze(0).to(loaded['device']).requires_grad_(True)
            encoded = F.normalize(model(image), p=2, dim=1)
            if not torch.allclose(encoded[0], cached_features[index], atol=1e-4, rtol=1e-3):
                raise ValueError('Grad-CAM patch embedding differs from the cached bag')
            if len(activations) != 1 or activations[0].ndim != 4 or activations[0].shape[0] != 1:
                raise RuntimeError('Expected one spatial ResNet layer4 activation')
            logits, _ = head.forward_replaced(cached_features.detach(), index, encoded)
            target = logits[0, class_index]
            gradient = torch.autograd.grad(target, activations[0])[0]
            channel_weights = gradient.mean(dim=(-2, -1), keepdim=True)
            raw_cam = (channel_weights * activations[0]).sum(1)[0].detach()
    finally:
        handle.remove()
    if not torch.isfinite(raw_cam).all() or not torch.isfinite(target):
        raise RuntimeError('Nonfinite Grad-CAM or target logit')
    positive_cam = raw_cam.clamp_min(0)
    scale = float(positive_cam.max().cpu())
    normalized = positive_cam / scale if scale > 0 else torch.zeros_like(positive_cam)
    # Preserve the final convolution's coarse spatial cells instead of implying pixel precision.
    display_map = F.interpolate(normalized[None, None], size=image.shape[-2:], mode='nearest')[0, 0]
    return {
        'status': 'available', 'method': 'gradcam', 'target_layer': 'layer4',
        'raw_cam': raw_cam.cpu().numpy(), 'positive_cam': positive_cam.cpu().numpy(),
        'display_map': display_map.cpu().numpy(), 'feature_map_shape': list(activations[0].shape[1:]),
        'model_input_shape': list(image.shape[-2:]), 'target_class_index': class_index,
        'conditional_slide_logit': float(target.detach().cpu()), 'normalization_scale': scale,
        'has_positive_support': scale > 0, 'display_interpolation': 'nearest',
        'class_specific': True, 'other_patch_embeddings_fixed': True,
        'cached_embedding_verified': True,
        'embedding_max_abs_difference': float((encoded[0].detach() - cached_features[index]).abs().max().cpu()),
    }
