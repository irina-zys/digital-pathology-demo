"""Actual encoder CLS attention on a patch, kept distinct from MIL weights and gradients."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch
from torch.nn import functional as F


def extract_patch_attention(loaded, rgb, *, threshold_percentile=0.8):
    """Capture last-block CLS-to-image-token attention in the configured frozen ViT.

    Attention is averaged over heads, then ranked within this patch for display.
    Ties receive their average rank; the display threshold is a percentile, not
    retained attention mass. Nearest-neighbour upsampling preserves token
    boundaries rather than inventing pixel precision.
    This is class-agnostic encoder attention, not a slide diagnosis attribution.
    """
    name = loaded['metadata']['name']
    if name not in {'uni', 'dino'}:
        return {'status': 'unsupported', 'reason': f'{name} has no supported CLS-token self-attention map'}
    if not np.isfinite(threshold_percentile) or not 0 <= threshold_percentile <= 1:
        raise ValueError('threshold_percentile must be in [0, 1]')
    rgb = np.asarray(rgb)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or not 0 < rgb.shape[0] * rgb.shape[1] <= 1_000_000):
        raise ValueError('Expected a bounded uint8 RGB patch')
    model = loaded['model']
    attention = model.blocks[-1].attn
    if model.training or attention.training or model.cls_token is None:
        raise ValueError('Attention extraction requires an evaluation-mode ViT with a CLS token')
    image = loaded['transform'](Image.fromarray(rgb)).unsqueeze(0).to(loaded['device'])
    height, width = image.shape[-2:]
    patch_height, patch_width = model.patch_embed.patch_size
    if height % patch_height or width % patch_width:
        raise ValueError('Model input must divide exactly into spatial tokens')
    grid = (height // patch_height, width // patch_width)
    prefix_count = model.num_prefix_tokens
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output.detach().float().cpu())

    original_fused = attention.fused_attn
    handle = attention.attn_drop.register_forward_hook(capture)
    try:
        # The fused kernel does not expose probabilities; the explicit path applies the same Q/K norms and scale.
        attention.fused_attn = False
        with torch.inference_mode():
            embedding = F.normalize(model(image), p=2, dim=1).cpu().numpy()[0]
    finally:
        handle.remove()
        attention.fused_attn = original_fused
    if len(captured) != 1:
        raise RuntimeError('Expected one last-block attention capture')
    matrix = captured[0]
    tokens = prefix_count + grid[0] * grid[1]
    if matrix.ndim != 4 or matrix.shape[0] != 1 or matrix.shape[-2:] != (tokens, tokens):
        raise RuntimeError('Attention token layout does not match the model input')
    cls_attention = matrix[0, :, 0, :]
    if (not torch.isfinite(cls_attention).all() or torch.any(cls_attention < 0)
            or not torch.allclose(cls_attention.sum(-1), torch.ones(len(cls_attention)), atol=1e-5, rtol=1e-5)):
        raise RuntimeError('Invalid CLS attention probabilities')
    heads = cls_attention[:, prefix_count:].reshape(-1, *grid).numpy()
    token_attention = heads.mean(axis=0)
    spatial_mass = float(token_attention.sum())
    if spatial_mass <= 0 or not np.isfinite(embedding).all():
        raise RuntimeError('Invalid spatial attention mass or encoder embedding')
    percentile_attention = (rankdata(token_attention, method='average') / token_attention.size).reshape(grid)
    selected = percentile_attention >= threshold_percentile
    attention_map = F.interpolate(torch.from_numpy(percentile_attention.astype(np.float32))[None, None],
                                  size=(height, width), mode='nearest')[0, 0].numpy()
    high_attention_mask = F.interpolate(torch.from_numpy(selected.astype(np.float32))[None, None],
                                       size=(height, width), mode='nearest')[0, 0].numpy().astype(bool)
    return {'status': 'available', 'method': 'last_block_cls_to_spatial_tokens_mean_heads',
            'head_attention': heads, 'token_attention': token_attention,
            'percentile_attention': percentile_attention, 'attention_map': attention_map,
            'high_attention_mask': high_attention_mask, 'embedding': embedding,
            'grid_shape': list(grid), 'model_input_shape': [height, width],
            'head_count': int(len(heads)), 'prefix_tokens_excluded': int(prefix_count),
            'spatial_mass_before_normalization': spatial_mass,
            'threshold_percentile': float(threshold_percentile), 'selected_token_count': int(selected.sum()),
            'display_units': 'within_patch_attention_percentile',
            'display_interpolation': 'nearest', 'class_specific': False}



def export_patch_attention(loaded, patch_folder, patches, cache_path, output, *, threshold_percentile=0.8):
    """Export one shared encoder-attention result for at most three existing RGB patches."""
    from src.data.feature_cache import read_feature_cache

    if not np.isfinite(threshold_percentile) or not 0 <= threshold_percentile <= 1:
        raise ValueError('threshold_percentile must be in [0, 1]')
    if not 1 <= len(patches) <= 3 or len({p['index'] for p in patches}) != len(patches):
        raise ValueError('Export expects one to three distinct patch indices')
    output, patch_folder = Path(output).resolve(), Path(patch_folder).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata_file = output / 'metadata.json'
    name = loaded['metadata']['name']
    metadata = {'encoder': loaded['metadata'], 'metadata_file': str(metadata_file),
                'threshold_percentile': float(threshold_percentile), 'class_specific': False,
                'shared_across_slide_classifiers': True, 'patches': []}
    if name not in {'uni', 'dino'}:
        metadata.update(status='unsupported', reason=f'{name} has no supported CLS-token self-attention map')
        metadata['patches'] = [{'index': p['index'], 'xy': p['xy'], 'status': 'unsupported'} for p in patches]
    else:
        features, coords, identity = read_feature_cache(cache_path)
        if identity['encoder'] != loaded['metadata']:
            raise ValueError('Attention encoder differs from the feature-cache contract')
        for patch in patches:
            index = patch['index']
            if type(index) is not int or not 0 <= index < len(features):
                raise ValueError('Invalid selected patch index')
            source = patch_folder / f'patch_{index}.npz'
            with np.load(source, allow_pickle=False) as saved:
                rgb, xy = saved['rgb'], saved['xy']
            if not np.array_equal(xy, coords[index]) or not np.array_equal(xy, patch['xy']):
                raise ValueError('Selected RGB patch coordinates differ from the complete feature bag')
            result = extract_patch_attention(loaded, rgb, threshold_percentile=threshold_percentile)
            if not np.allclose(result['embedding'], features[index], atol=1e-4, rtol=1e-3):
                raise ValueError('Attention patch embedding differs from the cached feature')
            arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray) and key != 'embedding'}
            destination = output / f'patch_{index}.npz'
            np.savez_compressed(destination, **arrays, xy=xy)
            record = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
            record.update(index=index, xy=xy.tolist(), file=str(destination), source_patch_file=str(source),
                          source_rgb_sha256=hashlib.sha256(rgb.tobytes(order='C')).hexdigest(),
                          rgb_shape=list(rgb.shape), rgb_dtype=str(rgb.dtype),
                          cached_embedding_verified=True)
            metadata['patches'].append(record)
        metadata['status'] = 'available'
        metadata['source_feature_cache'] = str(Path(cache_path).resolve())
        metadata['source_rgb_hash_convention'] = 'SHA-256 of uint8 RGB array bytes in C order'
    metadata_file.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return metadata
