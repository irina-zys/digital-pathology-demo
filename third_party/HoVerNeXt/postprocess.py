"""Bounded array postprocessing adapted from pinned HoVer-NeXt (GPL-3.0).

The five upstream algorithms are unchanged except that the in-memory instance
array uses NumPy instead of Zarr. Model loading, tiling and file I/O are omitted.
See SOURCE.json for the exact source revision and original/local file hashes.
"""
import gc

import cv2
import numpy as np
from scipy.ndimage import find_objects
from skimage.segmentation import watershed


def faster_instance_seg(out_img, out_cls, best_fg_thresh_cl, best_seed_thresh_cl):
    _, rois = cv2.connectedComponents((out_img[0] > 0).astype(np.uint8), connectivity=8)
    bboxes = find_objects(rois)
    del rois
    gc.collect()
    skip = False
    labelling = np.zeros(out_cls.shape[1:], dtype=np.int32)
    if len(bboxes) == 0:
        skip = True
        return labelling, skip
    max_inst = 0
    for bb in bboxes:
        bg_pred = out_img[(slice(0, 1, None), *bb)].squeeze()
        if (
            (np.array(bg_pred.shape[-2:]) <= 2).any()
            | (np.array(bg_pred.shape).sum() <= 64)
            | (len(bg_pred.shape) < 2)
        ):
            continue
        fg_pred = out_img[(slice(1, 2, None), *bb)].squeeze()
        sem = out_cls[(slice(0, len(best_fg_thresh_cl), None), *bb)]
        ws_surface = 1.0 - fg_pred  # .astype(np.float32)
        fg = np.zeros_like(ws_surface, dtype="bool")
        seeds = np.zeros_like(ws_surface, dtype="bool")

        for cl, fg_t in enumerate(best_fg_thresh_cl):
            mask = sem[cl]
            fg[mask] |= (1.0 - bg_pred[mask]) > fg_t
            seeds[mask] |= fg_pred[mask] > best_seed_thresh_cl[cl]

        del fg_pred, bg_pred, sem, mask
        gc.collect()
        _, markers = cv2.connectedComponents((seeds).astype(np.uint8), connectivity=8)
        del seeds
        gc.collect()
        bb_ws = watershed(ws_surface, markers, mask=fg, connectivity=2)
        del ws_surface, markers, fg
        gc.collect()
        bb_ws[bb_ws != 0] += max_inst
        labelling[bb] = bb_ws
        max_inst = np.max(bb_ws)
        del bb_ws
        gc.collect()
    return labelling, skip


def post_proc_inst(
    pred_inst,
    hole_size=50,
):
    pshp = pred_inst.shape
    pred_inst = np.asarray(pred_inst)
    init = find_objects(pred_inst)
    init_large = []
    adj = 8
    for i, sl in enumerate(init):
        if sl:
            slx1 = sl[0].start - adj if (sl[0].start - adj) > 0 else 0
            slx2 = sl[0].stop + adj if (sl[0].stop + adj) < pshp[0] else pshp[0]
            sly1 = sl[1].start - adj if (sl[1].start - adj) > 0 else 0
            sly2 = sl[1].stop + adj if (sl[1].stop + adj) < pshp[1] else pshp[1]
            init_large.append(
                (i + 1, (slice(slx1, slx2, None), slice(sly1, sly2, None)))
            )
    out = np.zeros(pshp, dtype=np.int32)
    i = 1
    for sl in init_large:
        rm_small_hole = remove_small_holescv2(pred_inst[sl[1]] == (sl[0]), hole_size)
        out[sl[1]][rm_small_hole > 0] = i
        i += 1

    del pred_inst
    gc.collect()

    after_sh = find_objects(out)
    out_ = np.zeros(out.shape, dtype=np.int32)
    i_ = 1
    for i, sl in enumerate(after_sh):
        i += 1
        if sl:
            nr_objects, relabeled = cv2.connectedComponents(
                (out[sl] == i).astype(np.uint8), connectivity=8
            )
            for new_lab in range(1, nr_objects):
                out_[sl] += (relabeled == new_lab) * i_
                i_ += 1
    return out_


def make_ct(pred_class, instance_map, return_probs=False):
    if type(pred_class) != np.ndarray:
        pred_class = pred_class[:]
    slices = find_objects(instance_map)
    pred_class = np.rollaxis(pred_class, 0, 3)
    # pred_class = softmax(pred_class,0)
    out = []
    out.append((0, 0))
    probs = {} if return_probs else None
    for i, sl in enumerate(slices):
        i += 1
        if sl:
            inst = instance_map[sl] == i
            i_cls = pred_class[sl][inst]
            i_cls = np.sum(i_cls, axis=0)
            if return_probs:
                e = np.exp(i_cls - np.max(i_cls))
                probs[str(i)] = (e / e.sum()).astype(np.float32)
            out.append((i, int(i_cls.argmax()) + 1))
    out_ = np.array(out)
    pred_ct = {str(k): int(v) for k, v in out_ if v != 0}
    if return_probs:
        return pred_ct, probs
    return pred_ct


def remove_obj_cls(pred_inst, pred_cls_dict, best_min_threshs, best_max_threshs, probs=None):
    out_oi = np.zeros_like(pred_inst, dtype=np.int64)
    i_ = 1
    out_oc = []
    out_oc.append((0, 0))
    out_probs = {} if probs is not None else None
    slices = find_objects(pred_inst)

    for i, sl in enumerate(slices):
        i += 1
        px = np.sum([pred_inst[sl] == i])
        cls_ = pred_cls_dict[str(i)]
        if (px > best_min_threshs[cls_ - 1]) & (px < best_max_threshs[cls_ - 1]):
            out_oc.append((i_, cls_))
            if probs is not None:
                out_probs[str(i_)] = probs[str(i)]
            out_oi[sl][pred_inst[sl] == i] = i_
            i_ += 1
    out_oc = np.array(out_oc)
    out_dict = {str(k): int(v) for k, v in out_oc if v != 0}
    if probs is not None:
        return out_oi, out_dict, out_probs
    return out_oi, out_dict


def remove_small_holescv2(img, sz):
    # this is still pretty slow but at least its a bit faster than other approaches?
    img = np.logical_not(img).astype(np.uint8)

    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(img)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information.
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[1:, -1]
    nb_blobs -= 1
    im_result = np.zeros((img.shape), dtype=np.uint16)
    for blob in range(nb_blobs):
        if sizes[blob] >= sz:
            im_result[im_with_separated_blobs == blob + 1] = 1

    im_result = np.logical_not(im_result)
    return im_result
