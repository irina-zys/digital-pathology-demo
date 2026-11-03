"""Frozen-encoder classification comparison over complete eligible WSI bags."""

import gc
import json
from pathlib import Path
import time

import torch

from src.inference.feature_extraction import cache_slide_features, load_comparison_encoder
from src.interpretation.mil import interpret_slide
from src.utils.files import source_hashes
from src.training.feature_mil import train_feature_mil
from src.training.history import plot_learning_curves


def run_comparison(items, project, output, cache_root, *, models=('uni', 'dino', 'resnet50'),
                   epochs=20, seed=42, device='cpu', batch_size=32, uni_batch_size=8,
                   lr=.001, weight_decay=.0001, attention_dim=128, class_names=('LUAD', 'LUSC'),
                   max_slide_seconds=1800, max_training_seconds=600):
    """Same coordinates, split, architecture and update budget; no backbone fine-tuning."""
    import matplotlib.pyplot as plt

    if not models or len(set(models)) != len(models) or set(models)-{'uni', 'dino', 'resnet50'}:
        raise ValueError('Choose unique supported pretrained encoders')
    project, output = Path(project), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    results = {}
    try:
        for name in models:
            print('ENCODER', name, flush=True)
            encoder = load_comparison_encoder(name, project, device=device, allow_download=False)
            paths = [cache_slide_features(item, encoder, cache_root,
                batch_size=uni_batch_size if name == 'uni' else batch_size, max_seconds=max_slide_seconds) for item in items]
            head, bags, result = train_feature_mil(items, paths, output/name, project=project, class_names=class_names,
                seed=seed, epochs=epochs, lr=lr, weight_decay=weight_decay, attention_dim=attention_dim,
                device=device, max_seconds=max_training_seconds)
            fig = plot_learning_curves(result['history'], selected_epoch=result['selected_epoch'])
            fig.suptitle(f'{name}: frozen pretrained encoder + trained attention classifier')
            fig.savefig(output/name/'learning_curve.png')
            plt.close(fig)
            explanations = []
            for item, path in zip(items, paths, strict=True):
                if item['row']['split'] == 'test':
                    explanations.append(interpret_slide(item, path, head, encoder,
                        output/name/'interpretation'/item['row']['bag_id'], class_names=class_names))
            result['interpretations'] = explanations
            results[name] = result
            (output/'partial_results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
            del encoder, head, bags
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()
        signatures = [[(x['slide_id'], x['patient_id'], x['split'], x['patches']) for x in r['inputs']] for r in results.values()]
        if any(signature != signatures[0] for signature in signatures):
            raise ValueError('Encoder comparison has different bags or patient splits')
        receipt = {'status': 'complete', 'models': list(models), 'results': results,
                   'all_eligible_patches_per_model': sum(i['eligible_patches'] for i in items),
                   'same_bags_and_splits': True, 'class_names': list(class_names), 'seed': seed,
                   'elapsed_seconds': time.monotonic()-start,
                   'source_code_sha256': source_hashes(project),
                   'scope': 'Ten-patient teaching comparison, frozen encoders with trained slide heads; not zero-shot'}
        (output/'comparison_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'status': 'failed', 'error': str(error),
            'completed_models': list(results), 'partial_feature_caches_resumable': True}, indent=2))
        raise
    return receipt
