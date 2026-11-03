"""Persisted complete feature bags and resumable, identity-bound writes."""
from contextlib import contextmanager
import hashlib
import json
import os
import h5py
import numpy as np


def cache_identity(item, encoder_metadata):
    row, selection = item['row'], item['selection']
    return {'slide': row['bag_id'], 'source': row['wsi_path'], 'source_bytes': selection['source_bytes'],
            'source_mtime_ns': selection['source_mtime_ns'], 'coordinates_sha256': selection['coordinate_sha256'],
            'indices': selection['indices'], 'source_size': selection['source_size'],
            'encoder': encoder_metadata, 'normalization_after_embedding': 'l2_per_patch'}

def read_feature_cache(path, *, fingerprint=None):
    """Read a completed finite cache, with exact coordinate/feature row correspondence."""
    with h5py.File(path, 'r') as f:
        if f.attrs.get('status') != 'complete' or (fingerprint and f.attrs['fingerprint'] != fingerprint):
            raise ValueError('Incomplete or incompatible feature cache')
        features, coords, indices = f['features'][:], f['coords'][:], f['indices'][:]
        if (features.ndim != 2 or not len(features) or not features.shape[1]
                or coords.shape != (len(features), 2) or coords.dtype.kind not in 'iu'
                or np.any(coords < 0) or len(features) != len(indices)
                or int(f.attrs.get('completed_rows', -1)) != len(features)
                or not np.isfinite(features).all() or not np.array_equal(indices, np.arange(len(indices)))):
            raise ValueError('Invalid feature cache or incomplete eligible-patch coverage')
        metadata = json.loads(f.attrs['identity'])
        if (cache_fingerprint(metadata) != f.attrs.get('fingerprint')
                or metadata.get('indices') != indices.tolist()
                or metadata.get('encoder', {}).get('features') != features.shape[1]):
            raise ValueError('Feature cache identity does not match its fingerprint or arrays')
    return features, coords, metadata


def cache_fingerprint(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@contextmanager
def feature_cache_writer(path, identity, selection, feature_dim, batch_size):
    partial, lock = path.with_suffix(".h5.partial"), path.with_suffix(".lock")
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        n = len(selection["coords"])
        fingerprint = cache_fingerprint(identity)
        with h5py.File(partial, "a") as handle:
            if "fingerprint" in handle.attrs:
                if handle.attrs["fingerprint"] != fingerprint:
                    raise ValueError("Preserving a partial cache with a different input contract")
            else:
                handle.attrs.update(fingerprint=fingerprint, identity=json.dumps(identity, sort_keys=True),
                                    status="partial", completed_rows=0)
                handle.create_dataset("features", (n, feature_dim), dtype="float32",
                                      chunks=(min(batch_size, n), feature_dim))
                handle.create_dataset("coords", data=np.asarray(selection["coords"], dtype=np.int64))
                handle.create_dataset("indices", data=np.asarray(selection["indices"], dtype=np.int64))
                handle.flush()
            yield handle
            if int(handle.attrs["completed_rows"]) != n:
                raise ValueError("Refusing to publish an incomplete feature cache")
            handle.attrs["status"] = "complete"
        if path.exists():
            raise FileExistsError("Another cache appeared during extraction")
        partial.rename(path)
        read_feature_cache(path, fingerprint=fingerprint)
    finally:
        lock.unlink(missing_ok=True)
