"""Stream every eligible WSI coordinate into resumable, identity-bound feature caches."""

import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import openslide
import torch
from PIL import Image

from src.data.wsi import WSIBagDataModule, select_coordinates, file_sha256
from src.tiling import read_tile


def complete_coordinate_manifest(manifest, project, *, class_names=('LUAD', 'LUSC'), target_mpp=.5):
    """Reuse source/split guards, then select every exported coordinate without a bag cap."""
    data = WSIBagDataModule(manifest=str(manifest), data_root=str(project), task='mil', class_names=list(class_names),
        batch_size=1, num_workers=0, image_size=224, max_patches_per_bag=1, augment=False,
        max_pixels=1_000_000, target_mpp=target_mpp, max_slides=20)
    data.setup('fit')
    data.setup('test')
    result = []
    for row in data.rows:
        selection = select_coordinates(row, 200_000, target_mpp, 1_000_000)
        with h5py.File(row['coords_path'], 'r') as f:
            count = len(f['coords'])
            if count > 200_000 or selection['indices'] != list(range(count)):
                raise ValueError('All-coordinate coverage exceeds the declared bound or is incomplete')
        result.append({'row': row, 'selection': selection, 'eligible_patches': count})
    return result


def _identity(item, loaded):
    row, selection = item['row'], item['selection']
    return {'slide': row['bag_id'], 'source': row['wsi_path'], 'source_bytes': selection['source_bytes'],
            'source_mtime_ns': selection['source_mtime_ns'], 'coordinates_sha256': selection['coordinate_sha256'],
            'indices': selection['indices'], 'source_size': selection['source_size'],
            'encoder': loaded['metadata'], 'normalization_after_embedding': 'l2_per_patch'}


def read_feature_cache(path, *, fingerprint=None):
    """Read a completed finite cache, with exact coordinate/feature row correspondence."""
    with h5py.File(path, 'r') as f:
        if f.attrs.get('status') != 'complete' or (fingerprint and f.attrs['fingerprint'] != fingerprint):
            raise ValueError('Incomplete or incompatible feature cache')
        features, coords, indices = f['features'][:], f['coords'][:], f['indices'][:]
        if (features.ndim != 2 or len(features) != len(coords) or len(features) != len(indices)
                or not np.isfinite(features).all() or not np.array_equal(indices, np.arange(len(indices)))):
            raise ValueError('Invalid feature cache or incomplete eligible-patch coverage')
        metadata = json.loads(f.attrs['identity'])
    return features, coords, metadata


def cache_slide_features(item, loaded, cache_root, *, batch_size=32, max_seconds=1800):
    """Write bounded batches; resume only an owned partial with an identical input contract."""
    if type(batch_size) is not int or not 1 <= batch_size <= 64 or not 1 <= max_seconds <= 7200:
        raise ValueError('Use batch_size 1-64 and a per-slide time bound of 1-7200 seconds')
    row, selection = item['row'], item['selection']
    info = Path(row['wsi_path']).stat()
    if (info.st_size, info.st_mtime_ns) != (selection['source_bytes'], selection['source_mtime_ns']):
        raise ValueError('WSI changed after coordinate selection')
    if file_sha256(row['coords_path']) != selection['coordinate_sha256']:
        raise ValueError('Coordinate export changed')
    identity = _identity(item, loaded)
    serialized = json.dumps(identity, sort_keys=True)
    fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
    folder = Path(cache_root)/loaded['metadata']['name']
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f"{row['bag_id']}.h5"
    if path.exists():
        read_feature_cache(path, fingerprint=fingerprint)
        return path
    partial, lock = path.with_suffix('.h5.partial'), path.with_suffix('.lock')
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    start = time.monotonic()
    try:
        n, dim = len(selection['coords']), loaded['metadata']['features']
        with h5py.File(partial, 'a') as f, openslide.OpenSlide(row['wsi_path']) as slide:
            if 'fingerprint' in f.attrs:
                if f.attrs['fingerprint'] != fingerprint:
                    raise ValueError('Preserving a partial cache with a different input contract')
            else:
                f.attrs.update(fingerprint=fingerprint, identity=serialized, status='partial', completed_rows=0)
                f.create_dataset('features', (n, dim), dtype='float32', chunks=(min(batch_size, n), dim))
                f.create_dataset('coords', data=np.asarray(selection['coords'], dtype=np.int64))
                f.create_dataset('indices', data=np.asarray(selection['indices'], dtype=np.int64))
                f.flush()
            for batch_index, offset in enumerate(range(int(f.attrs['completed_rows']), n, batch_size)):
                if time.monotonic()-start > max_seconds:
                    raise TimeoutError('Feature extraction time bound reached; matching partial can be resumed')
                inputs = []
                for xy in selection['coords'][offset:offset+batch_size]:
                    rgb = read_tile(slide, xy, selection['source_size'], 256, max_read_pixels=1_000_000)
                    inputs.append(loaded['transform'](Image.fromarray(rgb)))
                batch = torch.stack(inputs).to(loaded['device'])
                with torch.inference_mode():
                    encoded = loaded['model'](batch)
                    encoded = torch.nn.functional.normalize(encoded, p=2, dim=1).cpu().float().numpy()
                if encoded.shape != (len(inputs), dim) or not np.isfinite(encoded).all():
                    raise RuntimeError('Unexpected encoder output')
                end = offset + len(inputs)
                f['features'][offset:end] = encoded
                f.flush()
                f.attrs['completed_rows'] = end
                f.flush()
                if end == n or batch_index % 20 == 0:
                    print(f"{loaded['metadata']['name']} {row['bag_id']} {end}/{n}", flush=True)
            info = Path(row['wsi_path']).stat()
            if ((info.st_size, info.st_mtime_ns) != (selection['source_bytes'], selection['source_mtime_ns'])
                    or file_sha256(row['coords_path']) != selection['coordinate_sha256']):
                raise ValueError('Source changed during extraction')
            f.attrs['status'] = 'complete'
            f.attrs['extraction_seconds'] = time.monotonic()-start
        if path.exists():
            raise FileExistsError('Another cache appeared during extraction')
        partial.rename(path)
        read_feature_cache(path, fingerprint=fingerprint)
    finally:
        lock.unlink(missing_ok=True)
    return path
