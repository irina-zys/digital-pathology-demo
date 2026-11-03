"""Bounded encoder inference into complete, resumable feature caches."""
from pathlib import Path
import time
import numpy as np
import openslide
import torch
from PIL import Image
from src.data.transforms import PREPROCESSING, matched_encoder_transform
from src.models.encoders.pretrained import load_pretrained_encoder
from src.data.feature_cache import cache_identity, cache_fingerprint, feature_cache_writer, read_feature_cache
from src.wsi_preprocessing.tiling import read_tile
from src.utils.files import sha256


def load_comparison_encoder(name, project, *, device='cpu', allow_download=False):
    """Bind frozen weights to the matched full-tile preprocessing/cache contract.

    The single-patch UNI demonstration keeps its distinct transform policy.
    """
    loaded = load_pretrained_encoder(name, project, device=device, allow_download=allow_download)
    loaded['transform'] = matched_encoder_transform()
    loaded['metadata']['preprocessing'] = dict(PREPROCESSING)
    return loaded


def cache_slide_features(item, loaded, cache_root, *, batch_size=32, max_seconds=1800):
    """Write bounded batches; resume only an owned partial with an identical input contract."""
    if type(batch_size) is not int or not 1 <= batch_size <= 64 or not 1 <= max_seconds <= 7200:
        raise ValueError('Use batch_size 1-64 and a per-slide time bound of 1-7200 seconds')
    row, selection = item['row'], item['selection']
    info = Path(row['wsi_path']).stat()
    if (info.st_size, info.st_mtime_ns) != (selection['source_bytes'], selection['source_mtime_ns']):
        raise ValueError('WSI changed after coordinate selection')
    if sha256(row['coords_path']) != selection['coordinate_sha256']:
        raise ValueError('Coordinate export changed')
    identity = cache_identity(item, loaded["metadata"])
    fingerprint = cache_fingerprint(identity)
    folder = Path(cache_root)/loaded['metadata']['name']
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f"{row['bag_id']}.h5"
    if path.exists():
        read_feature_cache(path, fingerprint=fingerprint)
        return path
    start = time.monotonic()
    n, dim = len(selection['coords']), loaded['metadata']['features']
    with feature_cache_writer(path, identity, selection, dim, batch_size) as f, openslide.OpenSlide(row['wsi_path']) as slide:
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
                or sha256(row['coords_path']) != selection['coordinate_sha256']):
            raise ValueError('Source changed during extraction')
        f.attrs['extraction_seconds'] = time.monotonic()-start
    return path
