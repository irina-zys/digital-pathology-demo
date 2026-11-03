"""Local two-patch teaching workflow; model and metric components remain reusable."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image
import tifffile

from src.annotations import coco_samples, convert_annotations, rasterize, read_mask, mask_objects
from src.pretrained import CELLPOSE_SHA256, checkpoint_sha256, load_hovernext, predict_hovernext, run_cellpose_patch
from src.metrics import compare_instances, comparison_row
from src.postprocessing import filter_small_instances
from src.cellpose_worker import SIZE_CHECKPOINT_NAME, SIZE_CHECKPOINT_BYTES, SIZE_CHECKPOINT_SHA256


def _json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _new_directory(parent, prefix):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix + '_', dir=parent))


def prepare_demo(project, output_root=None):
    """Check data, pinned weights, both environments and caches before the walkthrough.

    No downloads or installations. The returned state owns one fresh run and the
    loaded CPU HoVer-NeXt model. Each later inference call gets a new attempt.
    """
    import torch

    project = Path(project).resolve(strict=True)
    out = _new_directory(output_root or project/'outputs/annotation_segmentation_demo',
                         datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    cache = out/'cache/numba'
    cache.mkdir(parents=True)
    with tempfile.TemporaryFile(dir=cache) as probe:
        probe.write(b'write-check')
    nu = project/'data/nuinsseg/extracted/human bladder'
    files = {'coco_json': project/'data/cytonuke/extracted/coco.json',
             'nuinsseg_rgb': nu/'tissue images/human_bladder_01.png',
             'nuinsseg_mask': nu/'label masks modify/human_bladder_01.tif',
             'nuinsseg_ambiguous': nu/'vague areas/mask binary/human_bladder_01.png'}
    hover_path = project/'data/models/hovernext/pannuke_convnextv2_tiny_1'
    cell_path = project/'data/models/cellpose/nucleitorch_0'
    size_path = cell_path.parent/SIZE_CHECKPOINT_NAME
    python = project/'environments/cellpose/.venv/bin/python'
    try:
        for path in [*files.values(), cell_path, size_path, python]:
            if not path.is_file():
                raise FileNotFoundError(f'Missing local input: {path}; see docs/pretrained-models.md and docs/datasets.md')
        if files['coco_json'].stat().st_size > 64_000_000:
            raise ValueError('COCO JSON exceeds the separate 64 MB parsing bound')
        if checkpoint_sha256(cell_path) != CELLPOSE_SHA256:
            raise ValueError('Cellpose checkpoint checksum differs from the pinned release')
        if size_path.stat().st_size != SIZE_CHECKPOINT_BYTES or checkpoint_sha256(size_path) != SIZE_CHECKPOINT_SHA256:
            raise ValueError('Cellpose size checkpoint differs from the pinned official asset')
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        env = {**os.environ, 'NUMBA_CACHE_DIR': str(cache)}
        command = [str(python), '-c', 'import cellpose.models; import importlib.metadata as m; '
                   'assert m.version("cellpose")=="3.1.1.2"; print(m.version("cellpose"))']
        checked = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, timeout=120)
        if checked.returncode:
            raise RuntimeError(f'Cellpose environment preflight failed: {checked.stderr[-1500:]}')
        hover = load_hovernext(hover_path, device='cpu')
        statuses = [{'component': 'Main Python / PyTorch', 'status': 'ready', 'detail': importlib.metadata.version('torch')},
                    {'component': 'HoVer-NeXt weights', 'status': 'verified', 'detail': 'PanNuke Tiny fold 1; CPU'},
                    {'component': 'Cellpose environment / weights', 'status': 'verified', 'detail': '3.1.1.2; nuclei and official size model; CPU'},
                    {'component': 'Output / Numba cache', 'status': 'writable', 'detail': str(cache)}]
        _json(out/'preflight.json', {'status': 'complete', 'components': statuses})
    except Exception as error:
        _json(out/'preflight.json', {'status': 'failed', 'error': str(error)})
        raise
    return {'project': project, 'out': out, 'cache': cache, 'files': files, 'hover': hover,
            'cellpose_checkpoint': cell_path, 'preflight': statuses, 'attempts': [], 'started': time.monotonic()}


def load_demo_patches(demo):
    """Load the same fixed patches as the original notebook, without exporting first."""
    files = demo['files']
    sample = next(coco_samples(files['coco_json'], 512*512, 1))
    root = files['coco_json'].parent/'images'
    rgb_path = (root/sample['source_image']).resolve(strict=True)
    if not rgb_path.is_relative_to(root.resolve()):
        raise ValueError('COCO image path escapes its image root')
    files['cytonuke_rgb'] = rgb_path
    objects = [obj for obj in sample['objects'] if obj['category_id'] == 0]
    if not objects or any(obj['category_name'] != 'NUCLEUS' for obj in objects):
        raise ValueError('Expected CytoNuke category 0 nuclei')
    cyto, _, _ = rasterize(objects, sample['height'], sample['width'], 'last')
    def rgb(path):
        with Image.open(path) as im:
            if im.width*im.height > 512*512:
                raise ValueError('Image exceeds patch loading budget')
            return np.array(im.convert('RGB'))
    cyto_rgb, nu_rgb = rgb(rgb_path), rgb(files['nuinsseg_rgb'])
    nu = read_mask(files['nuinsseg_mask'], 512*512)
    ambiguous = read_mask(files['nuinsseg_ambiguous'], 512*512)
    if cyto_rgb.shape != (256, 256, 3) or nu_rgb.shape != (512, 512, 3):
        raise ValueError('Unexpected source patch dimensions')
    if cyto.shape != (256, 256) or nu.shape != ambiguous.shape or nu.shape != (512, 512):
        raise ValueError('Source image and mask dimensions disagree')
    if not set(np.unique(ambiguous)) <= {0, 255}:
        raise ValueError('Unexpected ambiguity mask values')
    patches = {
        'cytonuke': {'rgb': cyto_rgb, 'reference': cyto, 'ambiguous': np.zeros(cyto.shape, bool),
                    'source': str(rgb_path), 'crop_xywh': [0, 0, 256, 256]},
        'nuinsseg': {'rgb': nu_rgb[128:384, 128:384].copy(), 'reference': nu[128:384, 128:384].copy(),
                    'ambiguous': (ambiguous[128:384, 128:384] > 0), 'source': str(files['nuinsseg_rgb']),
                    'crop_xywh': [128, 128, 256, 256]},
    }
    demo['sources'] = {name: {'path': str(path), 'sha256': checkpoint_sha256(path), 'bytes': path.stat().st_size}
                       for name, path in files.items()}
    demo['patches'] = patches
    demo['patch_provenance'] = {name: {'source': p['source'], 'crop_xywh': p['crop_xywh'],
                              'input_mpp_verified': False,
                              **{k+'_sha256': _array_hash(p[k]) for k in ('rgb', 'reference', 'ambiguous')}}
                              for name, p in patches.items()}
    _json(demo['out']/'patch_provenance.json', demo['patch_provenance'])
    return patches


def run_prediction(demo, name, model, *, invert=None, diameter=None, iou_threshold=0.5):
    """Run and retain a new attempt, so re-executing notebook cells never overwrites predictions."""
    if name not in demo['patches'] or model not in {'hovernext', 'cellpose'}:
        raise ValueError('Choose a loaded patch and hovernext/cellpose')
    if model == 'hovernext' and (invert not in (None, False) or diameter is not None):
        raise ValueError('Polarity and diameter controls apply only to Cellpose')
    patch = demo['patches'][name]
    directory = _new_directory(demo['out']/'attempts'/model, name)
    started = time.monotonic()
    try:
        if model == 'hovernext':
            result = predict_hovernext(patch['rgb'], demo['hover'], max_pixels=65_536)
        else:
            invert = True if invert is None else invert  # H&E nuclei are dark in this RGB input.
            result = run_cellpose_patch(patch['rgb'], directory/'worker', checkpoint=demo['cellpose_checkpoint'],
                invert=invert, diameter=diameter, channels=(0, 0), max_pixels=65_536,
                timeout_seconds=120, cache_dir=demo['cache'])
        labels = result['instance_mask']
        if labels.dtype != np.uint32 or labels.shape != (256, 256):
            raise ValueError('Unexpected model label shape or dtype')
        for key in ('rgb', 'reference', 'ambiguous'):
            if _array_hash(patch[key]) != demo['patch_provenance'][name][key+'_sha256']:
                raise RuntimeError('Patch data changed during inference')
        with (directory/'instances.tif').open('xb') as stream:
            tifffile.imwrite(stream, labels, photometric='minisblack', compression='deflate', metadata=None)
        assert np.array_equal(tifffile.imread(directory/'instances.tif'), labels)
        if 'classes' in result:
            assert set(np.unique(labels)) - {0} == result['classes'].keys()
            assert all(np.isfinite(a).all() for a in result['probabilities'].values())
            _json(directory/'classes.json', result['classes'])
        evaluation = compare_instances(patch['reference'], labels, patch['ambiguous'], iou_threshold=iou_threshold)
        metadata = {'status': 'complete', 'patch': demo['patch_provenance'][name], 'model': model,
                    'settings': result['metadata'], 'seconds': time.monotonic()-started,
                    'mask_sha256': checkpoint_sha256(directory/'instances.tif'), 'evaluation': evaluation}
        _json(directory/'metadata.json', metadata)
    except Exception as error:
        _json(directory/'failure.json', {'status': 'failed', 'error': str(error)})
        raise
    attempt = {'name': name, 'model': model, 'labels': labels, 'evaluation': evaluation,
               'directory': directory, 'metadata': metadata}
    if model == 'cellpose':
        preview = directory/'worker/normalized_input.npy'
        assert checkpoint_sha256(preview) == result['metadata']['normalized_input']['sha256']
        attempt['normalized_input'] = np.load(preview, allow_pickle=False)
    demo['attempts'].append(attempt)
    return attempt


def postprocess_prediction(demo, attempt, *, min_area=128):
    """Save area-only and edge-protected variants without changing a raw prediction."""
    patch = demo['patches'][attempt['name']]
    directory = _new_directory(demo['out']/'postprocessing', attempt['name'])
    original_hash = _array_hash(attempt['labels'])
    variants, records = {}, []
    for name, protect in (('area_only', False), ('edge_aware', True)):
        filtered = filter_small_instances(attempt['labels'], min_area, protect_edges=protect)
        labels = filtered.pop('instance_mask')
        evaluation = compare_instances(patch['reference'], labels, patch['ambiguous'],
                                       iou_threshold=attempt['evaluation']['iou_threshold'])
        mask_path = directory/f'{name}.tif'
        with mask_path.open('xb') as stream:
            tifffile.imwrite(stream, labels, photometric='minisblack', compression='deflate', metadata=None)
        assert labels.dtype == attempt['labels'].dtype
        assert np.array_equal(tifffile.imread(mask_path), labels)
        variants[name] = {'labels': labels, 'evaluation': evaluation, 'filter': filtered}
        records.append({'name': name, 'mask': mask_path.name, 'mask_sha256': checkpoint_sha256(mask_path),
                        **filtered, 'evaluation': evaluation})
    assert _array_hash(attempt['labels']) == original_hash
    metadata = {'raw_attempt': str(attempt['directory'].relative_to(demo['out'])),
                'raw_array_sha256': original_hash, 'raw_unchanged': True,
                'area_unit': 'input pixels', 'physical_scale_verified': False,
                'threshold_basis': 'illustrative prediction-size cutoff; annotations used only for evaluation',
                'variants': records}
    _json(directory/'metadata.json', metadata)
    demo.setdefault('postprocessing', []).append({'path': str(directory.relative_to(demo['out'])), **metadata})
    return variants


def export_demo(demo):
    """Appendix: exercise the shared converters and record inputs, attempts and source integrity."""
    report = _new_directory(demo['out']/'reports', 'export')
    cyto = convert_annotations(demo['files']['coco_json'], report/'cytonuke', input_format='coco',
                               output_format='both', max_images=1, max_pixels=512*512, overlap='last')
    nu = convert_annotations(demo['files']['nuinsseg_mask'], report/'nuinsseg', input_format='mask',
                             output_format='both', max_images=1, max_pixels=512*512)
    assert cyto['status'] == nu['status'] == 'complete'
    for category in cyto['samples'][0]['masks']:
        if category['category_id'] == 0:
            assert np.array_equal(read_mask(report/'cytonuke'/category['path'], 512*512), demo['patches']['cytonuke']['reference'])
    saved = read_mask(report/'nuinsseg'/nu['samples'][0]['masks'][0]['path'], 512*512)
    assert np.array_equal(saved, read_mask(demo['files']['nuinsseg_mask'], 512*512))
    for source in demo['sources'].values():
        assert checkpoint_sha256(source['path']) == source['sha256']
    for name, patch in demo['patches'].items():
        for key in ('rgb', 'reference', 'ambiguous'):
            assert _array_hash(patch[key]) == demo['patch_provenance'][name][key+'_sha256']
    receipt = {'status': 'complete', 'scope': 'two illustrative patches; not a held-out benchmark',
               'sources': demo['sources'], 'patches': demo['patch_provenance'], 'source_files_unchanged': True,
               'attempts': [{'path': str(a['directory'].relative_to(demo['out'])), **a['metadata']} for a in demo['attempts']],
               'postprocessing': demo.get('postprocessing', []),
               'uni': 'not part of the segmentation walkthrough; see encoder documentation',
               'elapsed_seconds': time.monotonic()-demo['started'],
               'source_code_sha256': {name: checkpoint_sha256(demo['project']/name) for name in
                   ('src/annotations.py', 'src/metrics/detection.py', 'src/metrics/segmentation.py',
                    'src/metrics/__init__.py', 'src/segmentation_demo.py', 'src/postprocessing.py',
                    'src/pretrained.py', 'src/hovernext.py', 'src/cellpose_worker.py')}}
    _json(report/'run_receipt.json', receipt)
    import pandas as pd
    pd.DataFrame([{'attempt': str(a['directory'].relative_to(demo['out'])),
                  **comparison_row(a['name'], a['model'], a['evaluation'])} for a in demo['attempts']]).to_csv(report/'comparison.csv', index=False)
    return report


def _outlines(ax, labels, ambiguous, color, linestyle='-'):
    for obj in mask_objects(np.where(ambiguous, 0, labels)):
        geometry = obj['geometry']
        polygons = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
        for polygon in polygons:
            for ring in (polygon.exterior, *polygon.interiors):
                ax.plot(*ring.xy, color=color, linewidth=1.1, linestyle=linestyle)


def plot_prediction(patch, attempt, title):
    """RGB and outline overlays with ignored regions and aggregate counts."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')
    labels, ev = attempt['labels'], attempt['evaluation']
    amb = patch['ambiguous']
    panels = [(None, 'RGB input'), (patch['reference'], 'Annotated ground truth'), (labels, title)]
    for ax, (mask, name) in zip(axes, panels):
        ax.imshow(patch['rgb'], extent=(0, 256, 256, 0))
        shade = np.zeros((256, 256, 4))
        shade[amb] = (0.6, 0.6, 0.6, 0.65)
        ax.imshow(shade, extent=(0, 256, 256, 0))
        if mask is not None:
            is_reference = ax is axes[1]
            _outlines(ax, mask, amb, '#00bcd4' if is_reference else '#ffad33', '-' if is_reference else '--')
        ax.set(title=name, xlim=(0, 256), ylim=(256, 0))
        ax.axis('off')
    if ev['prediction_all'] == 0:
        axes[2].text(.5, .5, '0 predicted nuclei', transform=axes[2].transAxes, ha='center',
                     bbox={'facecolor': 'white', 'alpha': .9, 'pad': 5})
    fig.suptitle(f"Matched {ev['matched']} · missed {ev['missed']} · extra {ev['extra']}\n"
                 'Local pixels; physical scale unverified · crop-edge objects retained')
    fig.legend(handles=[Line2D([], [], color='#00bcd4', label='Annotated ground truth', lw=2),
                        Line2D([], [], color='#ffad33', linestyle='--', label='Prediction', lw=2),
                        Patch(facecolor='0.6', label='Ignored ambiguity')], loc='outside lower center', ncol=3)
    return fig


def plot_polarity(patch, baseline, changed):
    """Compare normalized signals and predictions on one identical Cellpose patch."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    height, width = patch['reference'].shape
    attempts = (baseline, changed)
    if (baseline['name'] != changed['name'] or any(a['model'] != 'cellpose'
            or a['labels'].shape != (height, width)
            or a['normalized_input'].shape[:2] != (height, width) for a in attempts)):
        raise ValueError('Polarity comparison requires Cellpose results from the same patch')
    amb = patch['ambiguous']
    shade = np.zeros((height, width, 4))
    shade[amb] = (0.6, 0.6, 0.6, 0.65)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.8))
    fig.subplots_adjust(left=.015, right=.985, bottom=.15, top=.74, wspace=.04)
    for column, ax in enumerate(axes):
        attempt = attempts[column % 2]
        setting = f"invert={attempt['metadata']['settings']['settings']['invert']}"
        if column < 2:
            ax.imshow(attempt['normalized_input'], cmap='gray', vmin=0, vmax=1,
                      extent=(0, width, height, 0), interpolation='nearest')
            title = f'Normalized signal\n{setting}'
        else:
            ax.imshow(patch['rgb'], extent=(0, width, height, 0), interpolation='nearest')
            evaluation = attempt['evaluation']
            title = (f'Prediction · {setting}\n'
                     f"Matched {evaluation['matched']} · missed {evaluation['missed']} · extra {evaluation['extra']}")
        ax.imshow(shade, extent=(0, width, height, 0), interpolation='nearest')
        if column >= 2:
            _outlines(ax, patch['reference'], amb, '#00bcd4')
            _outlines(ax, attempt['labels'], amb, '#ffad33', '--')
            if evaluation['prediction_all'] == 0:
                ax.text(.5, .5, '0 predicted nuclei', transform=ax.transAxes, ha='center',
                        bbox={'facecolor': 'white', 'alpha': .9, 'pad': 5})
        ax.set(title=title, xlim=(0, width), ylim=(height, 0), aspect='equal')
        ax.axis('off')
    diameter = baseline['metadata']['settings']['settings']['diameter']
    fig.suptitle(f'One NuInsSeg crop · same RGB, diameter={diameter:.1f} px, thresholds and weights\n'
                 'Normalized signal before diameter rescaling; display clipped to [0, 1]', y=.985)
    fig.legend(handles=[Line2D([], [], color='#00bcd4', label='Annotated ground truth', lw=2),
                        Line2D([], [], color='#ffad33', linestyle='--', label='Prediction', lw=2),
                        Patch(facecolor='0.6', label='Ignored ambiguity')],
               loc='lower center', bbox_to_anchor=(.5, .01), ncol=3)
    return fig


def plot_model_comparison(patches, baseline):
    """Compare models in columns while keeping one identical patch in each row."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not patches:
        raise ValueError('Load at least one patch before plotting model comparisons')
    fig, axes = plt.subplots(len(patches), 4, figsize=(14, 4.5*len(patches)+.9), squeeze=False)
    fig.subplots_adjust(left=.015, right=.985, bottom=.07, top=.85, wspace=.04, hspace=.24)
    for row, (name, patch) in enumerate(patches.items()):
        height, width = patch['reference'].shape
        amb = patch['ambiguous']
        shade = np.zeros((height, width, 4))
        shade[amb] = (0.6, 0.6, 0.6, 0.65)
        attempts = [baseline[name, model] for model in ('hovernext', 'cellpose')]
        if any(a['name'] != name or a['labels'].shape != (height, width) for a in attempts):
            raise ValueError('Comparison predictions must use the displayed patch and crop')
        panels = [(None, f'{name} · RGB input'), (patch['reference'], 'Annotated ground truth'),
                  (attempts[0]['labels'], 'HoVer-NeXt'), (attempts[1]['labels'], 'Cellpose')]
        for column, (labels, title) in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(patch['rgb'], extent=(0, width, height, 0), interpolation='nearest')
            ax.imshow(shade, extent=(0, width, height, 0), interpolation='nearest')
            if labels is not None:
                reference = column == 1
                _outlines(ax, labels, amb, '#00bcd4' if reference else '#ffad33',
                          '-' if reference else '--')
            if column >= 2:
                evaluation = attempts[column-2]['evaluation']
                title += f"\nMatched {evaluation['matched']} · missed {evaluation['missed']} · extra {evaluation['extra']}"
                if evaluation['prediction_all'] == 0:
                    ax.text(.5, .5, '0 predicted nuclei', transform=ax.transAxes, ha='center',
                            bbox={'facecolor': 'white', 'alpha': .9, 'pad': 5})
            ax.set(title=title, xlim=(0, width), ylim=(height, 0), aspect='equal')
            ax.axis('off')
    fig.suptitle('Same patch and display coordinates across each row\n'
                 'Physical scale unverified · crop-edge objects retained · illustrative examples', y=.985)
    fig.legend(handles=[Line2D([], [], color='#00bcd4', label='Annotated ground truth', lw=2),
                        Line2D([], [], color='#ffad33', linestyle='--', label='Prediction', lw=2),
                        Patch(facecolor='0.6', label='Ignored ambiguity')],
               loc='lower center', bbox_to_anchor=(.5, .01), ncol=3)
    return fig


def plot_postprocessing(patch, raw, variants):
    """Show one patch with raw and area-filtered predictions, marking frame objects."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    height, width = patch['reference'].shape
    amb = patch['ambiguous']
    shade = np.zeros((height, width, 4))
    shade[amb] = (0.6, 0.6, 0.6, 0.65)
    minimum = variants['area_only']['filter']['min_area_pixels']
    panels = [(patch['reference'], raw['evaluation'], 'Annotated ground truth'),
              (raw['labels'], raw['evaluation'], 'Raw HoVer-NeXt'),
              (variants['area_only']['labels'], variants['area_only']['evaluation'], 'Minimum area only'),
              (variants['edge_aware']['labels'], variants['edge_aware']['evaluation'], 'Minimum area + edge protection')]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.8))
    fig.subplots_adjust(left=.015, right=.985, bottom=.15, top=.74, wspace=.04)
    for column, (ax, (labels, evaluation, title)) in enumerate(zip(axes, panels)):
        ax.imshow(patch['rgb'], extent=(0, width, height, 0), interpolation='nearest')
        ax.imshow(shade, extent=(0, width, height, 0), interpolation='nearest')
        reference = column == 0
        _outlines(ax, labels, amb, '#00bcd4' if reference else '#ffad33', '-' if reference else '--')
        if not reference:
            title += f"\nTP {evaluation['matched']} · FP {evaluation['extra']} · FN {evaluation['missed']}"
        ax.set(title=title, xlim=(0, width), ylim=(height, 0), aspect='equal')
        ax.axis('off')
    fig.suptitle(f'Minimum object area: {minimum} input pixels · physical scale unverified\n'
                 'Frame-touching objects can be protected; the raw prediction is retained', y=.985)
    fig.legend(handles=[Line2D([], [], color='#00bcd4', label='Annotated ground truth', lw=2),
                        Line2D([], [], color='#ffad33', linestyle='--', label='Prediction', lw=2),
                        Patch(facecolor='0.6', label='Ignored ambiguity')],
               loc='lower center', bbox_to_anchor=(.5, .01), ncol=3)
    return fig
