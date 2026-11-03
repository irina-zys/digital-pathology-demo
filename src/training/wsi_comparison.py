"""Frozen-encoder classification comparison over complete eligible WSI bags."""

import gc
import json
from pathlib import Path
import time

import torch

from src.inference.feature_extraction import cache_slide_features, load_comparison_encoder
from src.interpretation.mil import interpret_slide
from src.interpretation.patch_attention import export_patch_attention
from src.utils.files import source_hashes
from src.training.feature_mil import train_feature_mil
from src.training.history import plot_learning_curves


def run_comparison(items, project, output, cache_root, *, models=('uni', 'dino', 'resnet50'),
                   poolings=('mean', 'attention'), epochs=20, seed=42, device='cpu', batch_size=32, uni_batch_size=8,
                   lr=.001, weight_decay=.0001, attention_dim=128, class_names=('LUAD', 'LUSC'),
                   max_slide_seconds=1800, max_training_seconds=600, interpretation_class_index=1, patch_attention_percentile=0.8):
    """Matched coordinates, split and update budget for each encoder/pool pair."""
    import matplotlib.pyplot as plt

    if not models or len(set(models)) != len(models) or set(models)-{'uni', 'dino', 'resnet50'}:
        raise ValueError('Choose unique supported pretrained encoders')
    if not poolings or len(set(poolings)) != len(poolings) or set(poolings)-{'mean', 'attention'}:
        raise ValueError('Choose unique supported poolings: mean and/or attention')
    if type(interpretation_class_index) is not int or not 0 <= interpretation_class_index < len(class_names):
        raise ValueError('interpretation_class_index must select a declared class')
    if not 0 <= patch_attention_percentile <= 1:
        raise ValueError('patch_attention_percentile must be in [0, 1]')
    project, output = Path(project), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    results, encoder_attention = {}, {}
    try:
        for name in models:
            print('ENCODER', name, flush=True)
            encoder = load_comparison_encoder(name, project, device=device, allow_download=False)
            paths = [cache_slide_features(item, encoder, cache_root,
                batch_size=uni_batch_size if name == 'uni' else batch_size, max_seconds=max_slide_seconds) for item in items]
            encoder_attention[name] = {}
            for pooling in poolings:
                key = f'{name}_{pooling}'
                print('CLASSIFIER', key, flush=True)
                head, result = train_feature_mil(items, paths, output/key, project=project, class_names=class_names,
                    seed=seed, epochs=epochs, lr=lr, weight_decay=weight_decay, attention_dim=attention_dim,
                    device=device, max_seconds=max_training_seconds, pooling=pooling)
                fig = plot_learning_curves(result['history'], selected_epoch=result['selected_epoch'])
                fig.suptitle(f'{name}: frozen pretrained encoder + {pooling} pooling classifier')
                fig.savefig(output/key/'learning_curve.png')
                plt.close(fig)
                explanations = []
                for item, path in zip(items, paths, strict=True):
                    if item['row']['split'] == 'test':
                        slide_id = item['row']['bag_id']
                        patch_folder = output/key/'interpretation'/slide_id
                        explanation = interpret_slide(item, path, head, encoder, patch_folder,
                            class_names=class_names, class_index=interpretation_class_index)
                        explanations.append(explanation)
                        if pooling == poolings[0]:
                            encoder_attention[name][slide_id] = export_patch_attention(
                                encoder, patch_folder, explanation['patches'], path,
                                output/'encoder_attention'/name/slide_id,
                                threshold_percentile=patch_attention_percentile)
                result['interpretations'] = explanations
                results[key] = result
                (output/key/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
                (output/'partial_results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
                del head
            del encoder
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()
        signatures = [[(x['slide_id'], x['patient_id'], x['split'], x['patches']) for x in r['inputs']] for r in results.values()]
        if any(signature != signatures[0] for signature in signatures):
            raise ValueError('Encoder comparison has different bags or patient splits')
        receipt = {'status': 'complete', 'models': list(models), 'poolings': list(poolings),
                   'comparisons': list(results), 'results': results,
                   'interpretation_class_index': interpretation_class_index,
                   'patch_attention_percentile': patch_attention_percentile,
                   'encoder_attention': encoder_attention,
                   'all_eligible_patches_per_model': sum(i['eligible_patches'] for i in items),
                   'same_bags_and_splits': True, 'class_names': list(class_names), 'seed': seed,
                   'elapsed_seconds': time.monotonic()-start,
                   'source_code_sha256': source_hashes(project),
                   'scope': 'Ten-patient teaching comparison, frozen encoders with trained slide heads; not zero-shot'}
        (output/'comparison_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'status': 'failed', 'error': str(error),
            'completed_comparisons': list(results), 'partial_feature_caches_resumable': True}, indent=2))
        raise
    return receipt
