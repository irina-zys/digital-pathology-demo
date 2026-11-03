"""Whole-slide MIL attention and conditional input-gradient saliency on selected patches."""

import json
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
import torch

from src.data.feature_bags import read_feature_cache
from src.tiling import read_tile


def patch_saliency(head, loaded, cached_features, index, rgb, *, class_index):
    """Differentiate the selected slide-class logit through one patch in the full bag.

    Other patch features stay fixed; attention is recomputed over every patch.
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


def plot_slide_attention(thumbnail, dimensions, coords, source_size, attention, *, title, selected_indices=None):
    """Rasterize scored eligible tiles in level-0 coordinates over a whole-slide thumbnail."""
    import matplotlib.pyplot as plt

    image = np.asarray(thumbnail)
    height, width = image.shape[:2]
    if len(coords) != len(attention) or not np.isclose(attention.sum(), 1, atol=1e-5):
        raise ValueError('Expected one normalized attention weight per eligible patch')
    # Tie-aware within-slide percentile; raw softmax weights are saved separately.
    from scipy.stats import rankdata
    percentiles = rankdata(attention, method='average') / len(attention)
    overlay, coverage = np.zeros((height, width)), np.zeros((height, width), np.uint32)
    for (x, y), value in zip(coords, percentiles):
        x0, y0 = int(x*width/dimensions[0]), int(y*height/dimensions[1])
        x1 = max(x0+1, int(np.ceil((x+source_size[0])*width/dimensions[0])))
        y1 = max(y0+1, int(np.ceil((y+source_size[1])*height/dimensions[1])))
        overlay[y0:y1, x0:x1] += value
        coverage[y0:y1, x0:x1] += 1
    np.divide(overlay, coverage, out=overlay, where=coverage>0)
    masked = np.ma.masked_where(coverage == 0, overlay)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax in axes:
        ax.imshow(image)
        ax.axis('off')
    axes[0].set_title('Whole-slide tissue context')
    view = axes[1].imshow(masked, cmap='magma', vmin=0, vmax=1, alpha=.7, interpolation='nearest')
    axes[1].set_title(f'All {len(coords):,} eligible patches')
    if selected_indices:
        from matplotlib.patches import Rectangle
        for index, label in zip(selected_indices, ('H', 'M', 'L')):
            x, y = coords[index]
            xy = (x*width/dimensions[0], y*height/dimensions[1])
            axes[1].add_patch(Rectangle(xy, source_size[0]*width/dimensions[0],
                source_size[1]*height/dimensions[1], fill=False, edgecolor='cyan', linewidth=1))
            axes[1].annotate(label, xy, color='cyan', fontsize=9, fontweight='bold')
    fig.colorbar(view, ax=axes[1], shrink=.75, label='Within-slide MIL attention percentile')
    fig.suptitle(title + '\nPooling weights; not cell annotations or calibrated tumour probability')
    return fig, percentiles


def interpret_slide(item, cache_path, head, loaded, output, *, class_names=('LUAD', 'LUSC')):
    """Explain a full held-out bag and three deterministic attention-ranked patches."""
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    features, coords, identity = read_feature_cache(cache_path)
    if identity['encoder'] != loaded['metadata']:
        raise ValueError('Interpretation encoder differs from the feature-cache contract')
    tensor = torch.from_numpy(features).to(loaded['device'])
    head = head.to(loaded['device']).eval()
    with torch.inference_mode():
        logits, weights = head(tensor)
        attention = weights.cpu().numpy()
        target = int(logits.argmax(1).item())
        probabilities = logits.softmax(1)[0].cpu().tolist()
    row, selection = item['row'], item['selection']
    order = np.argsort(attention, kind='stable')
    chosen = [int(order[-1]), int(order[len(order)//2]), int(order[0])]
    with openslide.OpenSlide(row['wsi_path']) as slide:
        thumbnail = slide.get_thumbnail((1200, 1200)).convert('RGB')
        slide_fig, percentiles = plot_slide_attention(thumbnail, selection['dimensions'], coords,
            selection['source_size'], attention, title=f"{loaded['metadata']['name']} · {row['group_id']} · predicted {class_names[target]}", selected_indices=chosen)
        slide_fig.savefig(output/'slide_attention.png')
        plt.close(slide_fig)
        labels = ['Highest attention', 'Median attention', 'Lowest attention']
        fig, axes = plt.subplots(3, 3, figsize=(10, 10), layout='constrained')
        records = []
        for rank, (index, label) in enumerate(zip(chosen, labels)):
            rgb = read_tile(slide, coords[index], selection['source_size'], 256, max_read_pixels=1_000_000)
            saliency, score = patch_saliency(head, loaded, tensor, index, rgb, class_index=target)
            scale = float(np.quantile(saliency, .99))
            shown = np.clip(saliency/scale, 0, 1) if scale > 0 else np.zeros_like(saliency)
            model_rgb = np.asarray(Image.fromarray(rgb).resize((224, 224), resample=Image.Resampling.BILINEAR))
            axes[rank, 0].imshow(rgb)
            axes[rank, 0].set_title(f'{label}\npatch {index}, weight={attention[index]:.3g}')
            axes[rank, 1].imshow(shown, cmap='magma', vmin=0, vmax=1)
            axes[rank, 1].set_title(f'Absolute input-gradient sensitivity\n99th percentile = {scale:.2g}')
            axes[rank, 2].imshow(model_rgb)
            axes[rank, 2].imshow(shown, cmap='magma', vmin=0, vmax=1, alpha=.5)
            axes[rank, 2].set_title('Overlay on resized model input')
            for ax in axes[rank]:
                ax.axis('off')
            np.savez_compressed(output/f'patch_{index}.npz', rgb=rgb, saliency=saliency, xy=coords[index])
            records.append({'index': index, 'xy': coords[index].tolist(), 'selection': label,
                'attention': float(attention[index]), 'percentile': float(percentiles[index]),
                'target_class': class_names[target], 'conditional_slide_logit': score,
                'saliency_99pct_display_scale': scale, 'nonzero_saliency_pixels': int(np.count_nonzero(saliency))})
    fig.suptitle(f"{loaded['metadata']['name']} · sensitivity of {class_names[target]} slide logit\n"
                 'Other patches fixed; each saliency display scaled at its own 99th percentile')
    fig.savefig(output/'selected_patch_saliency.png')
    plt.close(fig)
    np.savez_compressed(output/'slide_attention.npz', coords=coords, weights=attention, percentiles=percentiles)
    result = {'slide_id': row['bag_id'], 'patient_id': row['group_id'], 'model': loaded['metadata']['name'],
              'class_names': list(class_names), 'probabilities': probabilities, 'target_class': class_names[target],
              'all_eligible_patches': len(features), 'attention_sum': float(attention.sum()),
              'attention_entropy_normalized': float(-(attention*np.log(np.maximum(attention, 1e-30))).sum()/np.log(len(attention))) if len(attention)>1 else None,
              'patches': records,
              'interpretation': 'Pooling attention and input sensitivity; not causal or tissue-label validation'}
    (output/'interpretation.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
