"""Predict slide labels from complete cached feature bags."""

import numpy as np
import torch

from src.data.feature_cache import read_feature_cache
from src.utils.files import sha256


def predict_feature_bags(head, samples, *, device='cpu'):
    """Predict each ``(manifest_row, cache_path)`` pair without retaining device bags.

    The caller selects the evaluation split and supplies a restored classifier.
    Every feature in each verified cache participates in the slide prediction.
    """
    head.eval()
    predictions = []
    with torch.inference_mode():
        for row, path in samples:
            if row.get('cache_sha256') and sha256(path) != row['cache_sha256']:
                raise ValueError('Prediction feature cache checksum mismatch')
            features, _, identity = read_feature_cache(path)
            if (identity['slide'] != row['bag_id']
                    or ('coordinate_sha256' in row and identity['coordinates_sha256'] != row['coordinate_sha256'])
                    or ('eligible_patches' in row and len(features) != int(row['eligible_patches']))):
                raise ValueError('Prediction feature cache does not match the requested slide or coordinates')
            logits, _ = head.forward_with_attention(torch.from_numpy(features).to(device))
            scores = logits.softmax(1)[0].cpu().tolist()
            predictions.append({
                'slide_id': row['bag_id'], 'patient_id': row['group_id'],
                'label': int(row['label']), 'predicted_label': int(np.argmax(scores)),
                'probabilities': scores, 'patches': len(features),
            })
    return predictions
