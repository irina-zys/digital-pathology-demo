"""Lightning/Hydra orchestration for complete cached WSI feature bags."""

import csv
import json
from pathlib import Path

import numpy as np
from src.models.loading import restore_model

from src.data.feature_cache import read_feature_cache
from src.inference.feature_mil import predict_feature_bags
from src.utils.files import sha256
from src.training.history import read_epoch_history
from src.training.recipes import run_recipe


def train_feature_mil(items, cache_paths, output, *, project, class_names=('LUAD', 'LUSC'), seed=42,
                      epochs=20, lr=.001, weight_decay=.0001, attention_dim=128,
                      device='cpu', max_seconds=600, pooling='attention'):
    """Compose mil_features and call src.runners.train.run; no separate optimizer loop."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, identities = [], []
    for item, path in zip(items, cache_paths, strict=True):
        _, _, identity = read_feature_cache(path)
        identities.append(identity)
        row = item['row']
        rows.append({'bag_id': row['bag_id'], 'group_id': row['group_id'], 'label': row['label'],
            'split': row['split'], 'cache_path': str(Path(path).resolve()), 'cache_sha256': sha256(path),
            'eligible_patches': item['eligible_patches'], 'coordinate_sha256': item['selection']['coordinate_sha256']})
    if len({json.dumps(i['encoder'], sort_keys=True) for i in identities}) != 1:
        raise ValueError('Cannot mix encoder/preprocessing contracts')
    metadata = identities[0]['encoder']
    manifest = output/'feature_manifest.csv'
    with manifest.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if tuple(class_names) != ('LUAD', 'LUSC'):
        raise ValueError('The local teaching comparison uses the declared LUAD/LUSC label order')
    hours, remainder = divmod(max_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    cfg, fitted = run_recipe(project, 'mil_features', manifest, output/'fit', device=device, test=True, overrides=[
        f'seed={seed}', f'data.input_dim={metadata["features"]}', f'data.encoder_sha256={metadata["checkpoint_sha256"]}',
        f'model/pooling@model.net.pooling={pooling}',
        *([f'model.net.pooling.attention_dim={attention_dim}'] if pooling == 'attention' else []), f'model.optimizer.lr={lr}', f'model.optimizer.weight_decay={weight_decay}',
        f'trainer.max_epochs={epochs}', f'trainer.max_time=00:{hours:02d}:{minutes:02d}:{seconds:02d}'])
    files = list((output/'fit/csv').glob('version_*/metrics.csv'))
    if len(files) != 1:
        raise ValueError('Expected one isolated epoch log')
    history = read_epoch_history(files[0])
    if len(history) != epochs or fitted['optimizer_steps_this_run'] != epochs*sum(r['split']=='train' for r in rows):
        raise RuntimeError('Training did not complete the requested budget')
    checkpoint = Path(fitted['checkpoint'])
    model, state = restore_model(cfg.model, checkpoint, data_contract=fitted['data_contract'])
    head = model.net.to(device).eval()
    held_out = ((row, row['cache_path']) for row in rows if row['split'] == 'test')
    predictions = predict_feature_bags(head, held_out, device=device)
    selected_epoch = int(state['epoch'])+1
    selected = next(row for row in history if row['epoch'] == selected_epoch)
    if not np.isclose(selected['val/loss'], min(row['val/loss'] for row in history)):
        raise RuntimeError('Checkpoint selection disagrees with validation history')
    result = {'status': 'complete', 'epochs': epochs, 'optimizer_steps': fitted['optimizer_steps_this_run'],
        'seed': seed, 'lr': lr, 'weight_decay': weight_decay, 'attention_dim': attention_dim,
        'trainable_parameters': sum(p.numel() for p in head.parameters()), 'class_names': list(class_names),
        'encoder': metadata, 'encoder_name': metadata['name'], 'pooling': pooling, 'selected_epoch': selected_epoch, 'selected_val_loss': selected['val/loss'],
        'test_loss': fitted['metrics']['test/loss'], 'test_accuracy': fitted['metrics']['test/accuracy'],
        'test_predictions': predictions, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'history': history, 'history_sha256': sha256(files[0]),
        'inputs': [{'slide_id': r['bag_id'], 'patient_id': r['group_id'], 'split': r['split'],
                    'label': int(r['label']), 'patches': r['eligible_patches'], 'cache': r['cache_path'],
                    'cache_sha256': r['cache_sha256']} for r in rows],
        'all_eligible_patches_used': True, 'device': device, 'entrypoint': 'src.runners.train.run',
        'experiment': 'mil_features', 'interpretation': 'Ten-patient teaching comparison; not a model ranking'}
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return head, result
