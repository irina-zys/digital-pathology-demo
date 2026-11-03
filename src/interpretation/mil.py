"""Whole-slide pooling/contribution maps and matched-patch input-gradient sensitivity."""

import json
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
import torch

from src.data.feature_cache import read_feature_cache
from src.wsi_preprocessing.tiling import read_tile


def patch_saliency(head, loaded, cached_features, index, rgb, *, class_index):
    """Differentiate the selected slide-class logit through one patch in the full bag.

    Other patch features stay fixed; the configured pool includes every patch.
    This is sensitivity to standardized model-input pixels, not a tumour mask,
    causal attribution, or transformer self-attention.
    """
    image = loaded['transform'](Image.fromarray(rgb)).unsqueeze(0).to(loaded['device']).requires_grad_(True)
    head.eval()
    loaded['model'].eval()
    encoded = torch.nn.functional.normalize(loaded['model'](image), p=2, dim=1)
    if not torch.allclose(encoded[0], cached_features[index], atol=1e-4, rtol=1e-3):
        raise ValueError('Recomputed patch embedding differs from the cached bag')
    logits, _ = head.forward_replaced(cached_features, index, encoded)
    gradient = torch.autograd.grad(logits[0, class_index], image)[0]
    saliency = gradient[0].abs().amax(0).detach().cpu().numpy()
    if not np.isfinite(saliency).all():
        raise RuntimeError('Nonfinite saliency')
    return saliency, float(logits[0, class_index].detach().cpu())


def plot_slide_scores(thumbnail, dimensions, coords, source_size, scores, *, title, score_kind, selected_indices=None, ax=None):
    """Draw eligible-tile scores; supply ax for a map within a matched comparison figure."""
    import matplotlib.pyplot as plt

    image = np.asarray(thumbnail)
    height, width = image.shape[:2]
    if len(coords) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected one finite score per eligible patch')
    if score_kind == 'attention':
        if not np.isclose(scores.sum(), 1, atol=1e-5) or np.any(scores < 0):
            raise ValueError('Expected normalized nonnegative attention weights')
        # Tie-aware within-slide percentile; raw softmax weights are saved separately.
        from scipy.stats import rankdata
        shown_scores = rankdata(scores, method='average') / len(scores)
        cmap, lower, upper = 'magma', 0, 1
        color_label = 'Within-slide MIL attention percentile'
        meaning = 'Pooling weights; not cell annotations or calibrated tumour probability'
    elif score_kind == 'contribution':
        shown_scores = scores
        bound = max(float(np.abs(scores).max()), np.finfo(np.float32).eps)
        cmap, lower, upper = 'RdBu_r', -bound, bound
        color_label = 'Signed target-logit contribution'
        meaning = 'Equal pooling weights; colour shows signed linear logit contribution, not attention'
    else:
        raise ValueError('score_kind must be attention or contribution')
    overlay, coverage = np.zeros((height, width)), np.zeros((height, width), np.uint32)
    for (x, y), value in zip(coords, shown_scores):
        x0, y0 = int(x*width/dimensions[0]), int(y*height/dimensions[1])
        x1 = max(x0+1, int(np.ceil((x+source_size[0])*width/dimensions[0])))
        y1 = max(y0+1, int(np.ceil((y+source_size[1])*height/dimensions[1])))
        overlay[y0:y1, x0:x1] += value
        coverage[y0:y1, x0:x1] += 1
    np.divide(overlay, coverage, out=overlay, where=coverage>0)
    masked = np.ma.masked_where(coverage == 0, overlay)
    standalone = ax is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Whole-slide tissue context')
        ax = axes[1]
    else:
        fig = ax.figure
    ax.imshow(image)
    ax.axis('off')
    view = ax.imshow(masked, cmap=cmap, vmin=lower, vmax=upper, alpha=.7, interpolation='nearest')
    ax.set_title(f'All {len(coords):,} eligible patches' if standalone else title)
    if selected_indices:
        from matplotlib.patches import Rectangle
        for index, label in zip(selected_indices, ('1', '2', '3')):
            x, y = coords[index]
            xy = (x*width/dimensions[0], y*height/dimensions[1])
            ax.add_patch(Rectangle(xy, source_size[0]*width/dimensions[0],
                source_size[1]*height/dimensions[1], fill=False, edgecolor='cyan', linewidth=1))
            ax.annotate(label, xy, color='cyan', fontsize=9, fontweight='bold')
    fig.colorbar(view, ax=ax, shrink=.75, label=color_label)
    if standalone:
        fig.suptitle(title + '\n' + meaning)
    return fig, shown_scores



def plot_slide_attention(thumbnail, dimensions, coords, source_size, attention, *, title, selected_indices=None):
    """Preserve the attention-only plotting API for existing callers."""
    return plot_slide_scores(thumbnail, dimensions, coords, source_size, attention,
                             title=title, score_kind='attention', selected_indices=selected_indices)


def interpret_slide(item, cache_path, head, loaded, output, *, class_names=('LUAD', 'LUSC'), class_index=None):
    """Explain a full held-out bag and the same three coordinate positions for every method."""
    import matplotlib.pyplot as plt

    if class_index is not None and (type(class_index) is not int or not 0 <= class_index < len(class_names)):
        raise ValueError('class_index must select a declared class')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    features, coords, identity = read_feature_cache(cache_path)
    if identity['encoder'] != loaded['metadata']:
        raise ValueError('Interpretation encoder differs from the feature-cache contract')
    tensor = torch.from_numpy(features).to(loaded['device'])
    head = head.to(loaded['device']).eval()
    with torch.inference_mode():
        logits, weights = head.forward_with_attention(tensor)
        pooling_weights = weights.cpu().numpy()
        predicted = int(logits.argmax(1).item())
        target = predicted if class_index is None else class_index
        probabilities = logits.softmax(1)[0].cpu().tolist()
        target_logit = float(logits[0, target].cpu())
        bias = float(head.classifier.bias[target].cpu())
        if head.pooling.contract['kind'] == 'mean':
            contributions = (tensor @ head.classifier.weight[target] / len(tensor)).cpu().numpy()
            if not np.isclose(float(contributions.sum()) + bias, target_logit, rtol=1e-4, atol=1e-5):
                raise RuntimeError('Mean-pooling patch contributions do not reconstruct the target logit')
            scores, score_kind, slide_filename = contributions, 'contribution', 'slide_contributions'
        else:
            scores, score_kind, slide_filename = pooling_weights, 'attention', 'slide_attention'
    row, selection = item['row'], item['selection']
    chosen = [0, len(coords)//2, len(coords)-1]
    with openslide.OpenSlide(row['wsi_path']) as slide:
        thumbnail = slide.get_thumbnail((1200, 1200)).convert('RGB')
        slide_fig, shown_scores = plot_slide_scores(thumbnail, selection['dimensions'], coords,
            selection['source_size'], scores, score_kind=score_kind,
            title=f"{loaded['metadata']['name']} / {head.pooling.contract['kind']} · {row['group_id']} · target {class_names[target]}")
        slide_fig.savefig(output/f'{slide_filename}.png')
        plt.close(slide_fig)
        labels = ['1: first coordinate', '2: middle coordinate', '3: last coordinate']
        fig, axes = plt.subplots(3, 3, figsize=(10, 10), layout='constrained')
        records = []
        for rank, (index, label) in enumerate(zip(chosen, labels)):
            rgb = read_tile(slide, coords[index], selection['source_size'], 256, max_read_pixels=1_000_000)
            saliency, score = patch_saliency(head, loaded, tensor, index, rgb, class_index=target)
            scale = float(np.quantile(saliency, .99))
            shown = np.clip(saliency/scale, 0, 1) if scale > 0 else np.zeros_like(saliency)
            model_rgb = np.asarray(Image.fromarray(rgb).resize((224, 224), resample=Image.Resampling.BILINEAR))
            axes[rank, 0].imshow(rgb)
            axes[rank, 0].set_title(f'{label}\npatch {index}, weight={pooling_weights[index]:.3g}')
            axes[rank, 1].imshow(shown, cmap='magma', vmin=0, vmax=1)
            axes[rank, 1].set_title(f'Absolute input-gradient sensitivity\n99th percentile = {scale:.2g}')
            axes[rank, 2].imshow(model_rgb)
            axes[rank, 2].imshow(shown, cmap='magma', vmin=0, vmax=1, alpha=.5)
            axes[rank, 2].set_title('Overlay on resized model input')
            for ax in axes[rank]:
                ax.axis('off')
            arrays = {'rgb': rgb, 'saliency': saliency, 'xy': coords[index]}
            gradcam_metadata = None
            if loaded['metadata']['name'] == 'resnet50':
                from src.interpretation.cnn_saliency import patch_gradcam
                gradcam = patch_gradcam(head, loaded, tensor, index, rgb, class_index=target)
                arrays.update(gradcam_raw=gradcam['raw_cam'], gradcam_positive=gradcam['positive_cam'],
                              gradcam_display=gradcam['display_map'])
                gradcam_metadata = {key: value for key, value in gradcam.items() if not isinstance(value, np.ndarray)}
            np.savez_compressed(output/f'patch_{index}.npz', **arrays)
            records.append({'index': index, 'xy': coords[index].tolist(), 'selection': label,
                'pooling_weight': float(pooling_weights[index]), 'map_score': float(scores[index]),
                'display_score': float(shown_scores[index]), 'score_kind': score_kind,
                'target_class': class_names[target], 'conditional_slide_logit': score,
                'gradcam': gradcam_metadata,
                'saliency_99pct_display_scale': scale, 'nonzero_saliency_pixels': int(np.count_nonzero(saliency))})
    fig.suptitle(f"{loaded['metadata']['name']} / {head.pooling.contract['kind']} · sensitivity of {class_names[target]} slide logit\n"
                 'Other patches fixed; each saliency display scaled at its own 99th percentile')
    fig.savefig(output/'selected_patch_saliency.png')
    plt.close(fig)
    np.savez_compressed(output/f'{slide_filename}.npz', coords=coords, weights=pooling_weights,
                        scores=scores, display_scores=shown_scores, target_class=target, bias=bias)
    result = {'slide_id': row['bag_id'], 'patient_id': row['group_id'], 'model': loaded['metadata']['name'],
              'class_names': list(class_names), 'probabilities': probabilities, 'target_class': class_names[target],
              'predicted_label': predicted, 'predicted_class': class_names[predicted],
              'pooling': head.pooling.contract['kind'], 'score_kind': score_kind, 'target_class_logit': target_logit,
              'classifier_bias': bias, 'patch_selection': 'first_middle_last_coordinate',
              'all_eligible_patches': len(features), 'pooling_weight_sum': float(pooling_weights.sum()),
              'pooling_entropy_normalized': float(-(pooling_weights*np.log(np.maximum(pooling_weights, 1e-30))).sum()/np.log(len(pooling_weights))) if len(pooling_weights)>1 else None,
              'patches': records,
              'interpretation': ('Uniform pooling with signed linear contributions' if head.pooling.contract['kind'] == 'mean'
                                 else 'Learned pooling attention') + '; input sensitivity is not causal or tissue-label validation'}
    (output/'interpretation.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result



def plot_patch_attention_zooms(item, coords, weights, indices):
    """Show calibrated 3x3-tile contexts with one discrete MIL weight per eligible tile."""
    import hashlib
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from scipy.stats import rankdata

    coords, weights = np.asarray(coords), np.asarray(weights)
    selection = item['selection']
    source_size = np.asarray(selection['source_size'])
    dimensions = np.asarray(selection['dimensions'])
    if (coords.ndim != 2 or coords.shape[1] != 2 or coords.dtype.kind not in 'iu'
            or weights.shape != (len(coords),) or not len(coords)
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, atol=1e-5, rtol=1e-5)):
        raise ValueError('Expected matching integer coordinates and finite nonnegative normalized MIL weights')
    if (source_size.shape != (2,) or source_size.dtype.kind not in 'iu' or np.any(source_size <= 0)
            or dimensions.shape != (2,) or dimensions.dtype.kind not in 'iu' or np.any(dimensions <= 0)
            or np.any(coords < 0) or np.any(coords + source_size > dimensions)):
        raise ValueError('Tile footprints must be calibrated and within the WSI')
    if (len(coords) != item['eligible_patches']
            or not np.array_equal(coords, np.asarray(selection['coords']))):
        raise ValueError('MIL weights must use the complete declared coordinate order')
    if (not 1 <= len(indices) <= 3 or len(set(indices)) != len(indices)
            or any(not isinstance(i, (int, np.integer)) or not 0 <= i < len(coords) for i in indices)):
        raise ValueError('Select one to three distinct valid patch indices')
    # Keep each zoom on the same global scale; local re-ranking would exaggerate weak neighbourhoods.
    percentiles = rankdata(weights, method='average') / len(weights)
    context_size = np.minimum(3 * source_size, dimensions)
    if int(np.prod(context_size)) > 4_000_000:
        raise ValueError('Three-by-three tile context exceeds the four-million source-pixel bound')
    output_size = np.maximum(1, np.rint(256 * context_size / source_size).astype(int))
    if np.any(output_size > 768):
        raise ValueError('Context exceeds the 768-pixel output bound')
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 4.6 * len(indices)),
                             layout='constrained', squeeze=False)
    cmap, norm = plt.get_cmap('viridis'), Normalize(vmin=0, vmax=1)
    records = []
    try:
        with openslide.OpenSlide(item['row']['wsi_path']) as slide:
            if not np.array_equal(slide.dimensions, dimensions):
                raise ValueError('WSI dimensions differ from the coordinate export')
            for row, index in enumerate(indices):
                origin = np.minimum(np.maximum(coords[index] - source_size, 0), dimensions - context_size)
                rgb = read_tile(slide, origin, context_size, output_size, max_read_pixels=4_000_000)
                ratios = output_size / context_size
                visible = np.flatnonzero(np.all(coords < origin + context_size, axis=1)
                                         & np.all(coords + source_size > origin, axis=1))
                for ax in axes[row]:
                    ax.imshow(rgb)
                    ax.set_xlim(-.5, output_size[0] - .5)
                    ax.set_ylim(output_size[1] - .5, -.5)
                    ax.axis('off')
                for tile_index in visible:
                    position = (coords[tile_index] - origin) * ratios - .5
                    size = source_size * ratios
                    colour = cmap(percentiles[tile_index])
                    axes[row, 1].add_patch(Rectangle(position, *size, facecolor=colour, edgecolor='white',
                                                     linewidth=1.2, alpha=.68, clip_on=True))
                    center = position + size / 2
                    if np.all(center >= 0) and np.all(center < output_size):
                        brightness = float(np.dot(colour[:3], [.2126, .7152, .0722]))
                        axes[row, 1].text(*center, f'{percentiles[tile_index]:.2f}', ha='center', va='center',
                                          fontsize=11, weight='bold', color='black' if brightness > .55 else 'white')
                selected_position = (coords[index] - origin) * ratios - .5
                for ax in axes[row]:
                    ax.add_patch(Rectangle(selected_position, *(source_size * ratios), fill=False,
                                            edgecolor='cyan', linewidth=2.5, clip_on=True))
                axes[row, 0].set_title(f'Original RGB context · selected tile {index}')
                axes[row, 1].set_title(f'MIL attention · full-slide percentile {percentiles[index]:.3f}\n'
                                       f'Selected raw weight {weights[index]:.6g}; cyan outline')
                records.append({'index': int(index), 'xy': coords[index].tolist(),
                    'raw_weight': float(weights[index]), 'full_slide_percentile': float(percentiles[index]),
                    'context_origin_level0': origin.tolist(), 'context_size_level0': context_size.tolist(),
                    'context_output_size': output_size.tolist(),
                    'context_rgb_sha256': hashlib.sha256(rgb.tobytes(order='C')).hexdigest(),
                    'visible_indices': visible.tolist(), 'visible_xy': coords[visible].tolist(),
                    'visible_raw_weights': weights[visible].tolist(),
                    'visible_full_slide_percentiles': percentiles[visible].tolist()})
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[:, 1].tolist(), shrink=.75,
                     label='Full-slide MIL attention percentile (0–1)')
        fig.suptitle(f"{item['row']['group_id']} · spatial attention from the MIL head\n"
                     'One scalar per 256 × 256 tile; no within-patch localization')
    except Exception:
        plt.close(fig)
        raise
    metadata = {'status': 'complete', 'kind': 'mil_pooling_weights_on_patch_footprints',
                'slide_id': item['row']['bag_id'], 'all_eligible_patches': len(coords),
                'weight_count': len(weights), 'weight_sum': float(weights.sum()),
                'coordinate_sha256': selection['coordinate_sha256'],
                'source_tile_size_level0': source_size.tolist(),
                'percentile_reference': 'all eligible patches in this slide; average ranks for ties',
                'colour_limits': [0, 1], 'colormap': 'viridis',
                'overlap_display': 'alpha-composited tile footprints with visible grid edges',
                'unscored_regions': 'original RGB, no assigned attention value', 'patches': records}
    return fig, metadata
