#!/usr/bin/env bash
#SBATCH --job-name=viz_aggregate
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%a.out
#
# Same figures as visualize_subjects.sh, but using the across-subject
# aggregate maps in saliency_outputs/aggregate/ as the reference rows instead
# of any one subject's own map. The across-subject mean, standard deviation,
# and entropy are all computed (via aggregate_saliency.py, run below once per
# STAT), and a completely separate set of figures is produced per stat (one
# mean-only figure, one std-only figure, one entropy-only figure) so any one
# statistic can be viewed on its own without the others' rows in the way.
# Entropy (see aggregate_saliency.py's voxelwise_entropy, bin count
# ENTROPY_BINS below) is a disagreement metric distinct from std: it captures
# voxels where subjects split into distinct clusters, which a unimodal-spread
# std can't distinguish from a voxel with the same std but no such split.
# Both kinds are always
# computed (KINDS=(signed abs)), but the figures below currently only
# display "abs" (--kind abs) -- signed .nii.gz outputs still exist on disk
# for anyone plotting them separately.
#   1. One PNG per model per stat (len(SANITY_MODELS) x len(STATS)) with
#      that stat's aggregate row for that model, plus the sanity-check maps
#      (arch_{MODEL}_all_*, data_{MODEL}_*, input_{MODEL}_permute_*) in
#      saliency_affine/ itself -- included in both the mean-only and
#      std-only figures, since they're single reference maps, not tied to
#      either statistic. SANITY_MODELS is MODELS plus one extra entry,
#      optim_uncnn_avg (the average of optim_uncnn's raw/rank/sobel
#      channels, always computed below regardless of AVERAGE_UNCNN_CHANNELS
#      -- see that flag's comment) -- so optim_uncnn_avg_sanity_check_
#      {STAT}.png is produced ALONGSIDE, not instead of, the 3 individual
#      channels' own sanity-check figures. optim_uncnn_avg has no arch_/
#      data_/input_ sanity-check maps of its own (those only exist per real
#      channel), so its figure just has no such rows -- see
#      average_uncnn_channels.py's CAVEAT before reading too much into that
#      row's absolute values.
#   2. One more PNG per stat comparing all models' aggregate rows to each
#      other for that stat (len(STATS) total) -- all_models_{STAT}.png.
#   3. Same cross-model row set as #2, but each row shown as 3 orthogonal
#      views + an axial-slice montage strip (visualize_saliency.py's
#      --montage) instead of just the 3 orthogonal views -- see
#      MONTAGE_KINDS/MONTAGE_AXIAL_Z_MM below. One PNG per (STAT, kind in
#      MONTAGE_KINDS) -- by default that's both an abs-only and a
#      signed-only montage figure per stat --
#      all_models_montage_{MONTAGE_KIND}_{STAT}.png.
#
# AVERAGE_UNCNN_CHANNELS=1 (see below, off by default) collapses
# optim_uncnn's 3 per-channel rows (raw/rank/sobel) into one averaged
# "optim_uncnn_avg" row in figures #2 and #3 only -- see
# average_uncnn_channels.py's CAVEAT on what that averaging means before
# relying on it for anything beyond a compact overview. Figure #1 is
# unaffected by this flag either way -- it always gets its own separate
# optim_uncnn_avg_sanity_check_{STAT}.png (see SANITY_MODELS below) in
# addition to (never instead of) the 3 individual channels' figures, since
# average_uncnn_channels.py itself always runs now (figure #1's avg row
# needs those files regardless of whether figures #2/#3 collapse onto one
# row).
#
# Aggregate nii.gz outputs are named
# {model}_aggregate_{mean,std,entropy}_{signed,abs}_saliency.nii.gz (the stat
# and kind are both part of the filename so every combination coexists
# instead of overwriting the others).
#
# No per-subject true/predicted-age JSON is linked in here -- there isn't a
# single meaningful age for a mean-across-subjects map, so the suptitle just
# won't show one (visualize_saliency.py already handles that gracefully).
#
# DATASET env var (OPTIONAL): unset/empty (default) preserves this script's
# original behavior exactly -- aggregates/visualizes
# saliency_affine/<target>/ (this repo's own checkpoints, NKI only). Set to
# DATASET=nki|ppmi|hbn instead aggregates/visualizes
# saliency_affine_pretrained/<target>/<dataset>/ (the external externally
# pre-trained heads, from slurm/compute_saliency_pretrained_heads.sh/
# slurm/compute_sanity_checks_randomize_model.sh's own DATASET-set branch).
# MODELS includes fs_roi for DATASET unset/nki/hbn; dropped only for
# DATASET=ppmi, which has no compatible fs_roi checkpoint (see
# slurm/compute_saliency_pretrained_heads.sh's own SCOPE comment). No
# arch_/data_/input_ sanity-check files exist yet for the pretrained tree
# until compute_sanity_checks_randomize_model.sh's DATASET-set branch is
# actually run for that dataset -- until then, this script's per-model
# sanity-check figures just print "SKIP" for those rows, same
# graceful-degradation path fs_roi's own missing sanity-check files
# already demonstrate today, not a new failure mode.

set -uo pipefail
shopt -s nullglob

# SLURM copies the submitted script into a spool dir before running it, so
REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset -- submit this with sbatch from the repo root, e.g. cd untrained-cnn/saliency && sbatch slurm/visualize_aggregate.sh}"
source "${REPO_DIR}/slurm/env.sh"
cd "$REPO_DIR"

# --array=0-2 runs age/sex/bmi as 3 genuinely parallel SLURM array tasks
# (TASK_TARGETS=(age sex bmi) below, index = position in that list) instead
# of 3 sequential `TARGET=... sbatch` submissions -- selects which
# saliency_affine/<target>/ subtree to aggregate/visualize. Submit a single
# target's task directly with e.g. `sbatch --array=0 slurm/visualize_aggregate.sh`
# (age only), `--array=1` (sex only), `--array=2` (bmi only).
TASK_TARGETS=(age sex bmi)
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset -- submit this with --array=0-2 (all 3 targets) or e.g. --array=0 (age only), sbatch slurm/visualize_aggregate.sh}"
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASK_TARGETS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for TASK_TARGETS (${TASK_TARGETS[*]})" >&2
    exit 1
fi
TARGET="${TASK_TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "TARGET=$TARGET"

# DATASET (optional, see header comment above): empty/unset resolves to this
# script's original saliency_affine/<target>/ (NKI, own checkpoints)
# behavior exactly; nki/ppmi/hbn switches to the pretrained-heads tree.
DATASET="${DATASET:-hbn}"
case "$DATASET" in
    "") ;;
    nki|ppmi|hbn) ;;
    *)
        echo "ERROR: unknown DATASET=$DATASET (expected unset, nki, ppmi, or hbn)" >&2
        exit 1
        ;;
esac

# Both kinds are always computed here (aggregate_saliency.py needs both on
# disk regardless of which one any given figure below actually displays --
# the per-model and plain cross-model figures show "abs" only; the montage
# cross-model figure's MONTAGE_KIND can select either or --combine-kinds).
KINDS=(signed abs)
if [ -n "$DATASET" ]; then
    SALIENCY_DIR="${REPO_DIR}/saliency_affine_pretrained/${TARGET}/${DATASET}"
else
    SALIENCY_DIR="${REPO_DIR}/saliency_affine/${TARGET}"
fi
# fs_roi has no arch_/data_/input_ sanity-check files yet for ANY tree
# (calculate_gradcam_tabular.py doesn't implement --randomize-model/
# --randomize-data/--randomize-input -- see that script's own docstring) --
# it still flows through SANITY_MODELS below (derived from MODELS) same as
# every other model, so it still gets an aggregate row in its own
# sanity-check figure, just with no sanity-check rows alongside it. Same
# graceful-degradation pattern optim_uncnn_avg already relies on above (see
# that comment) for exactly the same reason. Excluded only for
# DATASET=ppmi: no fs_roi checkpoint exists for PPMI's feature schema (see
# src/list_pretrained_heads.py's module docstring /
# slurm/compute_saliency_pretrained_heads.sh's SCOPE comment) -- present
# for DATASET unset (this repo's own NKI checkpoint) and DATASET=nki/hbn
# (the external {DATASET}_FS_ROI_{target}.pt, wired in once
# artifacts/{nki,hbn}_freesurfer_roi.csv had matching 110-column schemas).
MODELS=(optim_uncnn_raw optim_uncnn_rank optim_uncnn_sobel brainiac swinbrain simclr anatcl_global anatcl_local)
if [ "$DATASET" != "ppmi" ]; then
    MODELS+=(fs_roi)
fi
AGGREGATE_DIR="${SALIENCY_DIR}/aggregate"

# "entropy" is the per-voxel Shannon entropy of the across-subject value
# histogram (see aggregate_saliency.py's voxelwise_entropy) -- a disagreement
# metric distinct from std: it can flag voxels where subjects split into
# distinct clusters even when that split has the same std as a voxel with a
# single unimodal spread. Rides through every loop below exactly like mean/
# std since they're already generically parametrized by $STAT; only
# ENTROPY_BINS (histogram bin count, default 10) is entropy-specific.
STATS=(mean std) #entropy
ENTROPY_BINS="${ENTROPY_BINS:-10}"

# Optional: collapse optim_uncnn's 3 per-channel rows (raw/rank/sobel) into
# one averaged "optim_uncnn_avg" row -- ONLY in the two cross-model
# comparison figures below (all_models_{STAT}.png and the new
# all_models_montage_*_{STAT}.png), not the per-model figures above (those
# keep their 3 separate channel rows, still the source of truth for
# optim_uncnn specifically -- see average_uncnn_channels.py's CAVEAT on what
# averaging raw/rank/sobel together actually means). Off by default; turn on
# with e.g. `AVERAGE_UNCNN_CHANNELS=1 sbatch slurm/visualize_aggregate.sh`.
AVERAGE_UNCNN_CHANNELS="${AVERAGE_UNCNN_CHANNELS:-1}"

# New montage-style cross-model figure (see visualize_saliency.py's
# --montage): which kind(s) it shows per row, and which fixed axial levels
# (mm) appear in the montage strip beyond the 3 orthogonal views. Both an
# abs-only and a signed-only montage figure are produced every run
# (MONTAGE_KINDS below) -- override to a single kind, or "combine" for one
# figure with signed+abs side by side, via e.g.
# `MONTAGE_KINDS=combine sbatch slurm/visualize_aggregate.sh` or
# `MONTAGE_KINDS=abs sbatch slurm/visualize_aggregate.sh`.
MONTAGE_KINDS=(${MONTAGE_KINDS:-signed abs})
# Fixed world-space z levels (mm), not an evenly-spaced count -- same 3
# anatomical levels shown consistently across every montage figure instead
# of levels that shift with each map's own brain extent (see
# visualize_saliency.py's axial_indices_for_mm / --montage-axial-z-mm).
MONTAGE_AXIAL_Z_MM=(${MONTAGE_AXIAL_Z_MM:--18 0 18})

# calculate_gradcam.py's 4 attribution methods. ATTRIBUTION_TAG_NAMES holds
# aggregate_saliency.py's own group-key suffix for each ("" for gradient --
# its group key is bare "$MODEL", no suffix -- "gxi"/"ig_zero"/"ig_reference"
# for the others, matching model_name_from_filename's output exactly since
# that's just the saliency filename with "_{kind}_saliency.nii.gz" stripped,
# tag included). FILENAME_TAG_PREFIXES is the same tags with a trailing "_",
# used below wherever the tag sits directly in front of another filename
# segment (e.g. "_agebin_"-style: "_aggregate_"/"_all_"/arch_,data_,input_'s
# own suffix) rather than at the end of a bare group-key string.
ATTRIBUTION_TAG_NAMES=("" "gxi" "ig_zero" "ig_reference")
FILENAME_TAG_PREFIXES=("" "gxi_" "ig_zero_" "ig_reference_")

# aggregate_saliency.py's own grouping is already tag-agnostic (it groups by
# whatever model_name_from_filename returns, which naturally includes any
# attribution tag baked into the source filename -- see that script's
# collect_baseline_files) -- the only reason non-gradient groups were being
# dropped before was --models here being restricted to the bare (gradient-
# only) model names. AGGREGATE_MODELS expands that to every (model,
# attribution method) combination so every tag actually gets aggregated.
AGGREGATE_MODELS=()
for MODEL in "${MODELS[@]}"; do
    for TAG_NAME in "${ATTRIBUTION_TAG_NAMES[@]}"; do
        if [ -z "$TAG_NAME" ]; then
            AGGREGATE_MODELS+=("$MODEL")
        else
            AGGREGATE_MODELS+=("${MODEL}_${TAG_NAME}")
        fi
    done
done

for STAT in "${STATS[@]}"; do
    python3 src/aggregate_saliency.py \
        --saliency-dir "$SALIENCY_DIR" --output-dir "$AGGREGATE_DIR" \
        --kinds "${KINDS[@]}" --stat "$STAT" --entropy-bins "$ENTROPY_BINS" --models "${AGGREGATE_MODELS[@]}"
done

# Always runs (independent of AVERAGE_UNCNN_CHANNELS -- see that flag's
# comment above): figure #1's optim_uncnn_avg_sanity_check_{STAT}.png needs
# these files regardless of whether AVERAGE_UNCNN_CHANNELS also collapses
# figures #2/#3 onto one row. Only writes optim_uncnn_avg_*.nii.gz (one
# averaged map per attribution tag/stat/kind, see average_uncnn_channels.py)
# -- never touches or replaces the 3 real per-channel aggregate files.
python3 src/average_uncnn_channels.py --aggregate-dir "$AGGREGATE_DIR" --stats "${STATS[@]}" --kinds "${KINDS[@]}"

# SANITY_MODELS is MODELS plus optim_uncnn_avg -- used ONLY by the per-model
# sanity-check loop below (figure #1), so that loop produces one extra
# optim_uncnn_avg_sanity_check_{STAT}.png alongside (not instead of) the 3
# individual channels' own figures. Kept separate from MODELS itself so
# AGGREGATE_MODELS above (built from MODELS) never tries to aggregate
# "optim_uncnn_avg" from per-subject files -- no such source files exist,
# only the 3 real channels do.
SANITY_MODELS=("${MODELS[@]}" optim_uncnn_avg)

# CROSS_MODELS is MODELS with optim_uncnn's 3 channel rows collapsed into
# one optim_uncnn_avg row when AVERAGE_UNCNN_CHANNELS=1 -- used by the two
# cross-model comparison figures below only (see the AVERAGE_UNCNN_CHANNELS
# comment above); the per-model figure loop above still iterates MODELS
# directly and is unaffected. The AVERAGE_UNCNN_CHANNELS=1 branch is a
# literal list (everything in MODELS except optim_uncnn's 3 raw channels,
# plus optim_uncnn_avg) rather than derived from MODELS -- fs_roi's presence
# is derived from $DATASET (same condition MODELS itself used above) instead
# of being hardcoded, so the two can't drift out of sync; still keep this in
# sync by hand if MODELS ever gains/loses any OTHER non-optim_uncnn entry.
if [ "$AVERAGE_UNCNN_CHANNELS" = "1" ]; then
    CROSS_MODELS=(optim_uncnn_avg brainiac swinbrain simclr anatcl_global anatcl_local)
    [ "$DATASET" != "ppmi" ] && CROSS_MODELS+=(fs_roi)
else
    CROSS_MODELS=("${MODELS[@]}")
fi

# Only arch variant any current script can (re)produce -- see
# slurm/compute_sanity_checks_randomize_model.sh's own header comment and
# src/visualize_paper_nilearn.py's sanity_prefix() docstring. An older
# full-model "all" variant exists on disk for NKI from a since-removed
# code path and cannot be regenerated for any dataset.
ARCH_VARIANT="${ARCH_VARIANT:-step01_head}"

for STAT in "${STATS[@]}"; do
    for MODEL in "${SANITY_MODELS[@]}"; do
        # See visualize_subjects.sh for why anatcl needs its own prefix shape.
        case "$MODEL" in
            anatcl_*)
                LEVEL="${MODEL#anatcl_}"
                ARCH_PREFIX="arch_anatcl_${ARCH_VARIANT}_${LEVEL}_"
                DATA_PREFIX="data_anatcl_${LEVEL}_"
                INPUT_PREFIX="input_anatcl_permute_${LEVEL}_"
                ;;
            optim_uncnn_*)
                CHANNEL="${MODEL#optim_uncnn_}"
                ARCH_PREFIX="arch_optim_uncnn_${ARCH_VARIANT}_${CHANNEL}_"
                DATA_PREFIX="data_optim_uncnn_${CHANNEL}_"
                INPUT_PREFIX="input_optim_uncnn_permute_${CHANNEL}_"
                ;;
            *)
                ARCH_PREFIX="arch_${MODEL}_${ARCH_VARIANT}_"
                DATA_PREFIX="data_${MODEL}_"
                INPUT_PREFIX="input_${MODEL}_permute_"
                ;;
        esac

        STAGE_DIR="$(mktemp -d)"

        # Glob (not an exact filename match) so every attribution-method
        # variant's aggregate for THIS stat gets staged, not just gradient's
        # -- same reasoning as visualize_subjects.sh's reference-row fix.
        # Both kinds are staged even though the figure below only displays
        # "abs" (--kind abs) -- signed .nii.gz files are still there on disk
        # for anyone plotting them separately; only this one STAT is
        # staged, so the mean-only and std-only figures can never
        # cross-contaminate rows.
        for KIND in "${KINDS[@]}"; do
            for f in "$AGGREGATE_DIR/${MODEL}_"*"aggregate_${STAT}_${KIND}_saliency.nii.gz"; do
                [ -e "$f" ] && ln -sf "$f" "$STAGE_DIR/"
            done
        done

        if ! compgen -G "$STAGE_DIR/${MODEL}_*aggregate_${STAT}_signed_saliency.nii.gz" > /dev/null; then
            echo "  SKIP ${MODEL} (${STAT}): no aggregate map found"
            rm -rf "$STAGE_DIR"
            continue
        fi

        # arch_/data_/input_ sanity-check maps are single reference maps
        # (not an aggregate stat), so they're included in both the
        # mean-only and std-only figures for context. Both kinds staged
        # here too, same reasoning as above.
        for KIND in "${KINDS[@]}"; do
            for f in "$SALIENCY_DIR/${ARCH_PREFIX}"*"${KIND}_saliency.nii.gz" \
                     "$SALIENCY_DIR/${DATA_PREFIX}"*"${KIND}_saliency.nii.gz" \
                     "$SALIENCY_DIR/${INPUT_PREFIX}"*"${KIND}_saliency.nii.gz"; do
                [ -e "$f" ] && ln -sf "$f" "$STAGE_DIR/"
            done
        done

        # Row order: grouped by attribution method first (each method's
        # stat+arch+data+input together), same convention as
        # visualize_subjects.sh's regrouped --models. Prefixes deliberately
        # omit the kind suffix -- visualize_saliency.py's --models matches by
        # filename prefix (startswith), so one prefix here matches this
        # figure's --kind abs file regardless of the (also-staged, unused
        # here) signed file sharing the same prefix.
        MODEL_FILTERS=()
        for i in "${!FILENAME_TAG_PREFIXES[@]}"; do
            TAG_PREFIX="${FILENAME_TAG_PREFIXES[$i]}"
            MODEL_FILTERS+=(
                "${MODEL}_${TAG_PREFIX}aggregate_${STAT}"
                "${ARCH_PREFIX}${TAG_PREFIX}"
                "${DATA_PREFIX}${TAG_PREFIX}"
                "${INPUT_PREFIX}${TAG_PREFIX}"
            )
        done

        OUT="$AGGREGATE_DIR/${MODEL}_sanity_check_${STAT}.png"
        if python3 src/visualize_saliency.py \
            --saliency-dir "$STAGE_DIR" --kind abs --norm \
            --models "${MODEL_FILTERS[@]}" \
            --title "${MODEL} sanity check -- across-subject aggregate (${STAT}), all attribution methods" \
            --output "$OUT" --no-brain-mask --mask-thresh-frac -0.1; then
            echo "  wrote $OUT"
        else
            echo "  WARN: visualize_saliency.py failed for ${MODEL} (${STAT})"
        fi
        rm -rf "$STAGE_DIR"
    done
done

# Cross-model comparison: all models' aggregate maps already live together
# in saliency_outputs/aggregate/ -- no staging needed. One figure per stat
# (mean-only, std-only), ordered model-then-attribution-method so every
# model's own rows (across the 4 attribution methods) stay together.
# CROSS_MODELS (not MODELS) so AVERAGE_UNCNN_CHANNELS=1 collapses
# optim_uncnn's 3 rows into 1 here, per the comment where CROSS_MODELS is
# built above.
for STAT in "${STATS[@]}"; do
    ALL_MODEL_PREFIXES=()
    for MODEL in "${CROSS_MODELS[@]}"; do
        for TAG_PREFIX in "${FILENAME_TAG_PREFIXES[@]}"; do
            ALL_MODEL_PREFIXES+=("${MODEL}_${TAG_PREFIX}aggregate_${STAT}")
        done
    done

    OUT="$AGGREGATE_DIR/all_models_${STAT}.png"
    if python3 src/visualize_saliency.py \
        --saliency-dir "$AGGREGATE_DIR" --kind abs --norm \
        --models "${ALL_MODEL_PREFIXES[@]}" \
        --title "Model comparison -- across-subject aggregate (${STAT}), all attribution methods" \
        --output "$OUT" --no-brain-mask --mask-thresh-frac -0.1; then
        echo "  wrote $OUT"
    else
        echo "  WARN: cross-model comparison failed for stat $STAT"
    fi
done

# Same cross-model row set (model x attribution method, aggregated across
# subjects), but laid out as 3 orthogonal views + a --montage-axial-z-mm
# strip of fixed axial levels per kind (see visualize_saliency.py's
# --montage and the module docstring's reference image), instead of just the
# 3 orthogonal views alone. One figure per (STAT, kind in MONTAGE_KINDS) --
# by default that's an abs-only AND a signed-only figure every run; "combine"
# instead means one figure with --combine-kinds (signed+abs side by side).
for MONTAGE_KIND in "${MONTAGE_KINDS[@]}"; do
    case "$MONTAGE_KIND" in
        combine) MONTAGE_KIND_FLAGS=(--combine-kinds) ;;
        signed) MONTAGE_KIND_FLAGS=(--kind signed) ;;
        abs) MONTAGE_KIND_FLAGS=(--kind abs) ;;
        *)
            echo "Unknown MONTAGE_KIND=$MONTAGE_KIND (expected signed, abs, or combine)" >&2
            exit 1
            ;;
    esac

    for STAT in "${STATS[@]}"; do
        ALL_MODEL_PREFIXES=()
        for MODEL in "${CROSS_MODELS[@]}"; do
            for TAG_PREFIX in "${FILENAME_TAG_PREFIXES[@]}"; do
                ALL_MODEL_PREFIXES+=("${MODEL}_${TAG_PREFIX}aggregate_${STAT}")
            done
        done

        OUT="$AGGREGATE_DIR/all_models_montage_${MONTAGE_KIND}_${STAT}.png"
        if python3 src/visualize_saliency.py --norm \
            --saliency-dir "$AGGREGATE_DIR" "${MONTAGE_KIND_FLAGS[@]}" \
            --montage --montage-axial-z-mm "${MONTAGE_AXIAL_Z_MM[@]}" \
            --models "${ALL_MODEL_PREFIXES[@]}" \
            --title "Model comparison -- across-subject aggregate (${STAT}), all attribution methods (montage, ${MONTAGE_KIND})" \
            --output "$OUT" --no-brain-mask --mask-thresh-frac -0.1; then
            echo "  wrote $OUT"
        else
            echo "  WARN: montage cross-model comparison failed for stat $STAT (kind $MONTAGE_KIND)"
        fi
    done
done

echo "=== Done: aggregate ==="
