"""
GPU-accelerated drop-in variant of compute_saliency_correlations.py.

Identical CLI, identical output filenames/CSV schemas/figures -- this is a
perf variant of that script, not a new tool. It works by importing
compute_saliency_correlations as a module and monkeypatching its
compute_pairwise_matrix/compute_aggregate_pairwise_matrix (comparison #1 --
see that module's docstring) with torch/CUDA implementations, then calling
its unmodified main(). Every other step (resampling, the sanity-check
comparison, cross-model derivation, plotting, --resume) runs exactly the
code in compute_saliency_correlations.py, unchanged.

WHAT'S ON GPU, AND WHY THIS IS THE RIGHT TARGET:
  - spearman/pearson: the CPU version already reduces these to one blocked
    matrix multiply per pair of blocks (prepare_vector mean-centers/L2-
    normalizes each vector -- see _center_unit -- so correlation becomes a
    dot product). Moving that matmul onto a GPU (cuBLAS) is a direct,
    numerically-equivalent speedup with no reimplementation risk.
  - ssd: the CPU version computes this with an explicit O(n_labels^2)
    Python double loop (no vectorized trick exists via score_pair -- see
    the base module's COST NOTE), which is genuinely slow once n_labels is
    in the thousands (--level subject). This file instead uses the
    algebraic identity sum((a-b)^2) = ||a||^2 - 2*a.b + ||b||^2, which
    IS a blocked matmul (same shape as the corr case) -- so SSD gets the
    same GPU speedup path as spearman/pearson, and the CPU loop is
    bypassed entirely rather than just moved onto the GPU.
  - mutual_information: also an O(n_labels^2) CPU loop, but each individual
    score_pair call (sklearn's normalized_mutual_info_score) has no matmul
    form -- it needs a joint histogram over each pair's ~7M-voxel vectors.
    This file reimplements that computation on GPU (torch.bincount over a
    joint-index tensor, then the same entropy-based normalized-MI formula
    sklearn's default average_method="arithmetic" uses -- see
    _nmi_pair_gpu) instead of calling sklearn (CPU-only, single-threaded).
    This is a REIMPLEMENTATION, not a call into sklearn -- same formula,
    independently computed, so treat its output as "should match to normal
    floating-point tolerance," not "byte-identical to the CPU script."
    Spot-check a subset of pairs against compute_saliency_correlations.py's
    own output if exact reproducibility matters for a given run.
  - ssim: unlike every other metric here, SSIM is LOCAL/windowed (see the
    base module's own "ssim" docstring entry), so its per-pair cost isn't
    one histogram or one sum -- it's several windowed filter passes over
    each pair's full 3D volume, the most expensive of the three non-matmul
    metrics per pair, and STILL genuinely slow at --level subject (thousands
    of labels -> hundreds of thousands of block-level convolution calls)
    even with everything below. This file reimplements skimage's
    uniform-filter SSIM (win_size=7, K1=0.01, K2=0.03 -- its own defaults)
    via 3D box-filter convolutions (torch.nn.functional.avg_pool3d) instead
    of calling skimage (CPU-only, scipy.ndimage filters). Batched per block:
    each block's row AND column volumes' own local mean/variance are
    computed ONCE (shared across every pair that volume appears in, exactly
    like _corr_block's row/col vectors are reused across a whole block)
    rather than recomputed per pair the way calling skimage per pair would;
    only the pair-specific local covariance needs a per-pair convolution,
    and even that is batched across the WHOLE row block in one avg_pool3d
    call per column volume (n_cols calls per block, not n_rows*n_cols -- see
    _ssim_block). Like _nmi_pair_gpu, this is a REIMPLEMENTATION of
    skimage's algorithm, not a call into it -- same "should match closely,
    not necessarily bit-identical" caveat applies; spot-check against
    compute_saliency_correlations.py's own CPU output if exact
    reproducibility matters. When --brain-mask is set, every convolution
    above is ALSO run only over the mask's own bounding box (plus a
    win_size//2 margin), not the full head-plus-background template grid
    -- see _ssim_bbox/_to_device_volumes -- since background outside that
    box is exactly 0 and never contributes to the mask-restricted average
    anyway (base.prepare_vector's "ssim" branch), there's no reason to
    convolve over it. compute_pairwise_matrix_gpu also prints a per-
    row-block timing/ETA line (every metric, but only ssim/mutual_
    information are slow enough for it to matter) so a run can be judged
    (and a --time budget sized) from partial progress instead of only
    from a "did it finish" all-or-nothing log.

WHAT'S DELIBERATELY NOT ON GPU:
  - Resampling (nilearn resample_to_img) -- CPU/IO-bound, and per the base
    module's own COST NOTE this is usually the dominant cost of the whole
    pipeline, not comparison #1. This file does not touch it. If your run
    is resampling-bound (e.g. --level aggregate, the default, where
    comparison #1 itself is tiny -- a handful of models), this GPU variant
    will show little to no speedup over the CPU script, because the part
    it accelerates isn't the bottleneck for that run. GPU acceleration
    matters most for --level subject (thousands of (model, subject)
    labels -> comparison #1's O(n_labels^2) cost dominates) and/or
    --metrics mutual_information/ssd (unvectorized on CPU at any level).
  - The per-subject sanity-check comparison (#2) and score_pair's use
    inside it -- O(n_subjects), not O(n_labels^2); comparatively cheap
    per the base module's docstring, not worth the added complexity here.

DEVICE SELECTION: automatic, matching this repo's existing convention
(calculate_gradcam.py/train_linear.py etc.) -- CUDA if torch.cuda.is_available()
else CPU. No --device flag; use CUDA_VISIBLE_DEVICES to pick a GPU, or run
on a CPU-only node to fall back to a torch (not numpy) CPU implementation of
the same code path (still correct, just not accelerated).

MEMORY: the CPU version's --block-size (default 200) bounds SYSTEM RAM to
block_size * n_voxels per block. GPU block matrices must instead fit in GPU
memory, typically far smaller. At the unmasked template grid (~7M
voxels/vector) and float64 (kept here for the same reason the CPU version
uses it -- see score_pair's docstring: float32 accumulation drifts ~7e-4 from
np.corrcoef/exact SSD over vectors this long), one row or column block is
block_size * 7e6 * 8 bytes -- block_size=200 alone is ~11GB, easily more than
one GPU's worth once both the row and column block, plus matmul workspace,
are resident. Pass a smaller --block-size (e.g. 32-64 on a 16-24GB GPU,
128+ on an 80GB A100/H100 -- see compute_correlations_gpu.sh) -- a
torch.cuda.OutOfMemoryError is caught below and re-raised as a SystemExit
naming the offending block size so this fails with an actionable message
instead of a bare CUDA traceback.

Usage: identical to compute_saliency_correlations.py, e.g.
    python compute_saliency_correlations_gpu.py --kind signed --level subject
    python compute_saliency_correlations_gpu.py --kind signed --metrics ssd mutual_information --block-size 64
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import compute_saliency_correlations as base

# skimage's own structural_similarity defaults (see that function's
# win_size/K1/K2 -- this file matches them so its output is comparable to
# the CPU script's, not just to itself).
_SSIM_WIN_SIZE = 7
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if _DEVICE.type == "cuda":
    print(f"[compute_saliency_correlations_gpu] using CUDA device: {torch.cuda.get_device_name(_DEVICE)}")
else:
    print("[compute_saliency_correlations_gpu] CUDA not available -- falling back to a CPU torch implementation "
          "(still correct, but no GPU speedup; see this module's docstring).")


def _to_device(vecs, dtype):
    """Stacks already-prepared 1-D vectors (see base.prepare_vector) into one (len(vecs), n_voxels) tensor on _DEVICE."""
    return torch.from_numpy(np.vstack(vecs)).to(device=_DEVICE, dtype=dtype)


_ssim_bbox_cache = None  # lazily built once (see _ssim_bbox) -- base._SSIM_BRAIN_MASK doesn't change mid-run


def _ssim_bbox():
    """
    (d0, d1, h0, h1, w0, w1) bounding box around base._SSIM_BRAIN_MASK's
    True voxels, expanded by _SSIM_WIN_SIZE // 2 on each side (clipped to
    the volume's own bounds) -- computed and cached once, printed once so a
    run's log shows exactly how much this is actually saving. The margin
    guarantees every true-mask voxel's full SSIM window still finds real
    neighbors inside the crop -- without it, a mask voxel sitting exactly
    at the tight bbox edge would lose neighbors its window needs. Returns
    None if this run has no brain mask -- there's no known region of
    interest to crop to, so _to_device_volumes falls back to the full
    (uncropped) volume.

    THIS is the actual GPU speedup for --brain-mask runs (see module
    docstring's "ssim" entry): --brain-mask already existed to restrict the
    SCORE to in-brain voxels (see base.score_pair's "ssim" branch /
    _ssim_mask_tensor below), but until this, every avg_pool3d convolution
    here still ran over the FULL unmasked template volume regardless --
    masking only restricted which OUTPUT voxels got averaged afterward, not
    what the convolutions actually had to process. Cropping the INPUT to
    the mask's own bounding box (typically a much smaller region than the
    full head-plus-background template grid) directly shrinks every
    convolution's cost, not just the final average.
    """
    global _ssim_bbox_cache
    if base._SSIM_BRAIN_MASK is None:
        return None
    if _ssim_bbox_cache is None:
        mask = base._SSIM_BRAIN_MASK
        pad = _SSIM_WIN_SIZE // 2
        d_idx = np.where(mask.any(axis=(1, 2)))[0]
        h_idx = np.where(mask.any(axis=(0, 2)))[0]
        w_idx = np.where(mask.any(axis=(0, 1)))[0]
        d0, d1 = max(int(d_idx[0]) - pad, 0), min(int(d_idx[-1]) + pad + 1, mask.shape[0])
        h0, h1 = max(int(h_idx[0]) - pad, 0), min(int(h_idx[-1]) + pad + 1, mask.shape[1])
        w0, w1 = max(int(w_idx[0]) - pad, 0), min(int(w_idx[-1]) + pad + 1, mask.shape[2])
        _ssim_bbox_cache = (d0, d1, h0, h1, w0, w1)
        crop_shape = (d1 - d0, h1 - h0, w1 - w0)
        frac = (crop_shape[0] * crop_shape[1] * crop_shape[2]) / mask.size
        print(f"[compute_saliency_correlations_gpu] ssim: cropping to brain-mask bounding box "
              f"{tuple(mask.shape)} -> {crop_shape} ({frac:.1%} of the full volume) before convolving.")
    return _ssim_bbox_cache


def _to_device_volumes(vols, dtype):
    """
    Like _to_device, but for ssim's "prepared" values -- (D, H, W) volumes
    (see base.prepare_vector's "ssim" branch), not flat vectors, so this
    uses np.stack (a genuine new leading axis) rather than _to_device's
    np.vstack (which would instead concatenate the volumes' first
    dimension, silently producing the wrong shape). Adds the channel
    dimension avg_pool3d/conv-style ops require -> (len(vols), 1, D, H, W).

    Cropped to the brain mask's own bounding box first when one is set (see
    _ssim_bbox) -- see that function's docstring for why this is the real
    GPU speedup, not just a memory-footprint reduction.
    """
    bbox = _ssim_bbox()
    if bbox is not None:
        d0, d1, h0, h1, w0, w1 = bbox
        vols = [v[d0:d1, h0:h1, w0:w1] for v in vols]
    return torch.from_numpy(np.stack(vols)).unsqueeze(1).to(device=_DEVICE, dtype=dtype)


def _corr_block(row_t, col_t):
    """row_t/col_t already mean-centered/L2-normalized (see base._center_unit) -- correlation is their dot product."""
    return (row_t @ col_t.T).cpu().numpy()


def _ssd_block(row_t, col_t):
    """
    sum((a-b)^2) = ||a||^2 - 2*a.b + ||b||^2 -- turns SSD into the same
    blocked-matmul shape as _corr_block instead of the CPU version's
    explicit per-pair loop (see module docstring). Clamped at 0: floating-
    point cancellation in the expansion can push an exact (self-vs-self)
    zero very slightly negative.
    """
    row_sq = (row_t * row_t).sum(dim=1, keepdim=True)
    col_sq = (col_t * col_t).sum(dim=1, keepdim=True)
    block = row_sq + col_sq.T - 2.0 * (row_t @ col_t.T)
    return block.clamp_(min=0).cpu().numpy()


_ssim_mask_cache = None  # lazily built once (see _ssim_mask_tensor) -- base._SSIM_BRAIN_MASK doesn't change mid-run


def _ssim_mask_tensor():
    """
    base._SSIM_BRAIN_MASK (see that module's set_ssim_context), cropped to
    the SAME bounding box _to_device_volumes crops the input volumes to
    (see _ssim_bbox) -- so it lines up with the volumes' now-smaller
    coordinate system -- and THEN by _ssim_block's own 'valid'-window
    border on top of that (see that function's docstring), moved to
    _DEVICE once, then cached. Every block's every column reuses the same
    cropped mask tensor rather than re-cropping/re-uploading it per call.
    Returns None if this run has no brain mask (--brain-mask not passed),
    matching base.score_pair's own None-means-unmasked convention.

    Correctness note: avg_pool3d's valid-mode output at LOCAL index k (0-
    based within the bbox-cropped volume) is the window centered at bbox-
    local input index k+pad, i.e. GLOBAL voxel bbox[0]+k+pad. This mask is
    bbox-cropped then pad-cropped the same way, so its local index 0 is
    ALSO global voxel bbox[0]+pad -- the two stay aligned to the same
    physical voxels despite neither being expressed in the full template's
    own coordinates anymore.
    """
    global _ssim_mask_cache
    if base._SSIM_BRAIN_MASK is None:
        return None
    if _ssim_mask_cache is None:
        pad = _SSIM_WIN_SIZE // 2
        d0, d1, h0, h1, w0, w1 = _ssim_bbox()
        mask = base._SSIM_BRAIN_MASK[d0:d1, h0:h1, w0:w1]
        cropped = mask[pad:-pad, pad:-pad, pad:-pad] if pad > 0 else mask
        _ssim_mask_cache = torch.from_numpy(cropped).to(device=_DEVICE, dtype=torch.bool)
    return _ssim_mask_cache


def _ssim_block(row_t, col_t):
    """
    GPU reimplementation of skimage.metrics.structural_similarity's
    uniform-filter SSIM (see module docstring's "ssim" entry for the full
    reasoning) between every row volume and every column volume in this
    block. row_t/col_t: (N, 1, D, H, W) float32 tensors (see
    _to_device_volumes) -- SSIM needs real spatial adjacency, unlike every
    other metric here, so these are volumes, not flattened vectors.

    Local mean/variance for EVERY row and EVERY column volume is computed
    ONCE up front (avg_pool3d over the whole batch in one call each) --
    shared across every pair that volume appears in, exactly like
    _corr_block's row/col tensors are reused across a whole block. Only
    the pair-specific local covariance genuinely needs a per-pair
    convolution (it depends on BOTH volumes' product) -- but even that is
    batched across the whole row block in ONE avg_pool3d call per column
    volume (broadcasting row_t * col_t[j] over the batch dim before
    filtering), so this loop runs n_cols times, not n_rows*n_cols.

    'valid' (unpadded) windowing throughout -- avg_pool3d's default has no
    padding, so the output is smaller than the input by (win_size - 1) in
    each spatial dim. This mirrors skimage's own crop-before-averaging step
    (excludes boundary voxels whose window fell partly outside the image
    from the mean SSIM score) rather than its zero-padded full-size S-map,
    so it needs no extra cropping step of its own here -- only
    _ssim_mask_tensor's cropped mask needs to match this same border when
    --brain-mask restricts the average to in-brain voxels (see
    base.score_pair's own "ssim" branch for the identical CPU-side
    reasoning/border).

    data_range is computed PER PAIR from each pair's own combined min/max
    (matches base.score_pair's "ssim" branch exactly) -- an exact 0 range
    (two identical, typically all-zero volumes) is mapped to SSIM=1
    directly rather than dividing by zero.
    """
    n_rows, n_cols = row_t.shape[0], col_t.shape[0]

    def local_stats(t):
        mu = F.avg_pool3d(t, kernel_size=_SSIM_WIN_SIZE, stride=1)
        sq_mu = F.avg_pool3d(t * t, kernel_size=_SSIM_WIN_SIZE, stride=1)
        var = (sq_mu - mu * mu).clamp_(min=0)
        return mu, var

    row_mu, row_var = local_stats(row_t)  # each (n_rows, 1, d', h', w')
    col_mu, col_var = local_stats(col_t)  # each (n_cols, 1, d', h', w')

    row_min = row_t.amin(dim=(1, 2, 3, 4))  # (n_rows,)
    row_max = row_t.amax(dim=(1, 2, 3, 4))
    col_min = col_t.amin(dim=(1, 2, 3, 4))  # (n_cols,)
    col_max = col_t.amax(dim=(1, 2, 3, 4))

    mask_t = _ssim_mask_tensor()
    n_valid = mask_t.sum() if mask_t is not None else None

    block = np.empty((n_rows, n_cols), dtype=np.float64)
    for j in range(n_cols):
        y = col_t[j : j + 1]  # (1, 1, D, H, W) -- broadcasts against row_t's batch dim below
        mu_xy = F.avg_pool3d(row_t * y, kernel_size=_SSIM_WIN_SIZE, stride=1)
        sigma_xy = mu_xy - row_mu * col_mu[j]

        data_range_j = torch.maximum(row_max, col_max[j]) - torch.minimum(row_min, col_min[j])
        c1 = (_SSIM_K1 * data_range_j).pow(2).view(-1, 1, 1, 1, 1)
        c2 = (_SSIM_K2 * data_range_j).pow(2).view(-1, 1, 1, 1, 1)

        numerator = (2 * row_mu * col_mu[j] + c1) * (2 * sigma_xy + c2)
        denominator = (row_mu.pow(2) + col_mu[j].pow(2) + c1) * (row_var + col_var[j] + c2)
        ssim_map = (numerator / denominator).squeeze(1)  # (n_rows, d', h', w')

        if mask_t is not None:
            col_scores = (ssim_map * mask_t).sum(dim=(1, 2, 3)) / n_valid
        else:
            col_scores = ssim_map.mean(dim=(1, 2, 3))

        col_scores = torch.where(data_range_j == 0, torch.ones_like(col_scores), col_scores)
        block[:, j] = col_scores.cpu().numpy()
    return block


def _nmi_pair(a, b, n_bins):
    """
    Normalized mutual information between two already-digitized (integer,
    0..n_bins-1) label tensors on _DEVICE, via a GPU joint histogram
    (torch.bincount) -- see module docstring for why this reimplements
    sklearn's normalized_mutual_info_score(..., average_method="arithmetic")
    (its default) rather than calling it.
    """
    n = a.numel()
    joint_idx = a * n_bins + b
    joint_counts = torch.bincount(joint_idx, minlength=n_bins * n_bins).double()
    pij = (joint_counts / n).view(n_bins, n_bins)
    pi = torch.bincount(a, minlength=n_bins).double() / n
    pj = torch.bincount(b, minlength=n_bins).double() / n

    hx = -torch.sum(pi[pi > 0] * torch.log(pi[pi > 0]))
    hy = -torch.sum(pj[pj > 0] * torch.log(pj[pj > 0]))
    if hx <= 0 or hy <= 0:
        # A constant label vector (single bin) carries no information to
        # share -- matches sklearn's own NMI=0 convention in this case.
        return 0.0

    pi_pj = pi.view(-1, 1) * pj.view(1, -1)
    mask = pij > 0
    mi = torch.sum(pij[mask] * (torch.log(pij[mask]) - torch.log(pi_pj[mask])))
    nmi = (mi / ((hx + hy) / 2.0)).item()
    return float(min(max(nmi, 0.0), 1.0))  # clip float noise at the [0, 1] boundary


def _mi_block(row_t, col_t):
    """
    Still one _nmi_pair call per (row, col) pair -- unlike corr/ssd, MI has
    no whole-block matmul form -- but each call's O(n_voxels) joint-histogram
    work runs on GPU instead of sklearn's single-threaded CPU
    confusion_matrix, which is where the real speedup comes from at ~7M
    voxels/vector (see module docstring).
    """
    n_bins = int(torch.max(row_t.max(), col_t.max()).item()) + 1
    n_rows, n_cols = row_t.shape[0], col_t.shape[0]
    block = np.empty((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        a = row_t[i]
        for j in range(n_cols):
            block[i, j] = _nmi_pair(a, col_t[j], n_bins)
    return block


def _score_block(row_t, col_t, metric):
    if metric in ("spearman", "pearson"):
        return _corr_block(row_t, col_t)
    if metric == "ssd":
        return _ssd_block(row_t, col_t)
    if metric == "mutual_information":
        return _mi_block(row_t, col_t)
    if metric == "ssim":
        return _ssim_block(row_t, col_t)
    raise ValueError(f"Unknown metric: {metric!r}")


def compute_pairwise_matrix_gpu(labels, baseline_lookup, load_prepared_baseline, tag, metric, block_size, n_jobs=1, prepared_dir=None, kind=None):
    """
    GPU replacement for compute_saliency_correlations.compute_pairwise_matrix
    (same block/memory strategy, see that function's docstring) -- monkeypatched
    in by name (see bottom of this file), so compute_correlations() calls this
    exactly as it would the CPU version. `n_jobs`/`prepared_dir`/`kind` are
    accepted only for call-signature compatibility -- n_jobs is unused: one
    GPU block already saturates the device, so no ProcessPoolExecutor is
    needed the way the CPU mutual_information/ssd path needs one.
    """
    n = len(labels)
    corr = np.empty((n, n), dtype=np.float64)
    if metric == "mutual_information":
        dtype = torch.int64
    elif metric == "ssim":
        dtype = torch.float32  # SSIM's smoothing filters don't need float64 precision -- see module docstring
    else:
        dtype = torch.float64
    # ssim's "prepared" values are (D, H, W) volumes, not flat vectors (see
    # base.prepare_vector's "ssim" branch) -- _to_device_volumes stacks
    # those correctly (np.stack, a new leading axis + channel dim), unlike
    # _to_device's np.vstack (built for 1-D vectors).
    stack_fn = _to_device_volumes if metric == "ssim" else _to_device
    n_blocks = -(-n // block_size)
    t_start = time.time()

    for bi in range(n_blocks):
        r0, r1 = bi * block_size, min((bi + 1) * block_size, n)
        row_vecs = [np.asarray(load_prepared_baseline(*baseline_lookup[label], tag)) for label in labels[r0:r1]]
        row_t = stack_fn(row_vecs, dtype)
        for bj in range(bi, n_blocks):
            c0, c1 = bj * block_size, min((bj + 1) * block_size, n)
            if bj == bi:
                col_t = row_t
            else:
                col_vecs = [np.asarray(load_prepared_baseline(*baseline_lookup[label], tag)) for label in labels[c0:c1]]
                col_t = stack_fn(col_vecs, dtype)
            try:
                block = _score_block(row_t, col_t, metric)
            except torch.cuda.OutOfMemoryError as e:
                # row_t.shape[1] is voxels/vector for every metric except
                # ssim, whose (N, 1, D, H, W) volumes carry that count
                # across 3 dims instead of 1 (shape[1] there is just the
                # channel dim, always 1) -- numel()/batch-size covers both.
                n_voxels = row_t.numel() // row_t.shape[0]
                raise SystemExit(
                    f"GPU out of memory scoring a {row_t.shape[0]}x{col_t.shape[0]} block "
                    f"(block-size={block_size}, {n_voxels} voxels/vector) -- pass a smaller --block-size "
                    "(see compute_correlations_gpu.sh's MEMORY note)."
                ) from e
            corr[r0:r1, c0:c1] = block
            corr[c0:c1, r0:r1] = block.T
            if bj != bi:
                del col_t
        del row_t
        if _DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        # Per-ROW-BLOCK (not per block-pair -- n_blocks lines, not
        # n_blocks*(n_blocks+1)/2) timing/ETA, so a killed/still-running job
        # shows real progress instead of the single "computing pairwise..."
        # line before this function returns. Matters most for ssim/
        # mutual_information (see module docstring) -- printed unconditionally
        # since it's cheap and harmless for the fast matmul metrics too.
        rows_done = bi + 1
        elapsed = time.time() - t_start
        avg_per_row = elapsed / rows_done
        eta = avg_per_row * (n_blocks - rows_done)
        print(
            f"  [{metric}] row-block {rows_done}/{n_blocks} done "
            f"({elapsed:.1f}s elapsed, {avg_per_row:.2f}s/row-block avg, "
            f"ETA {eta:.1f}s ({eta / 3600:.2f}h) remaining)"
        )
    return corr


def compute_aggregate_pairwise_matrix_gpu(aggregate_vectors, metric, mi_bins=base.DEFAULT_MI_BINS):
    """
    GPU counterpart of compute_saliency_correlations.compute_aggregate_pairwise_matrix.
    n_models is always small (single digits -- see MODELS), so this is just
    one block through the same GPU helpers above rather than a genuinely
    different code path.
    """
    labels = sorted(aggregate_vectors)
    if metric == "mutual_information":
        dtype = torch.int64
    elif metric == "ssim":
        dtype = torch.float32
    else:
        dtype = torch.float64
    prepared = [base.prepare_vector(aggregate_vectors[label], metric, mi_bins) for label in labels]
    t = (_to_device_volumes if metric == "ssim" else _to_device)(prepared, dtype)
    block = _score_block(t, t, metric)
    return pd.DataFrame(block, index=labels, columns=labels)


base.compute_pairwise_matrix = compute_pairwise_matrix_gpu
base.compute_aggregate_pairwise_matrix = compute_aggregate_pairwise_matrix_gpu


if __name__ == "__main__":
    base.main()
