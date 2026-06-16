"""
create_alt_test_file.py

Creates filtered alternate-sample ROOT files for all three splits in a single
pass over the input files.

Splits:
  test  — events whose eventNumber IS     in the reference file (ordered to match)
  train — events whose eventNumber is NOT in the reference file, first TRAIN_FRAC
  val   — events whose eventNumber is NOT in the reference file, last  (1-TRAIN_FRAC)

The train/val partition is deterministic: among all non-test events sorted by
eventNumber, index i where (i % period) != (period-1) → train, else → val.

Usage:
    python create_alt_test_file.py [--ref REF] [--glob GLOB] [--outdir OUTDIR] [--tree TREE]
"""

import argparse
import glob as _glob
import os
import sys

import awkward as ak
import numpy as np
import uproot
from tqdm import tqdm

TRAIN_FRAC = 0.9  # fraction of non-test events assigned to train

REFERENCE = (
    # "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/test_JZ1-9_hamza.root"
    # "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/hamza_samples/test_JZ1-9.root"
    # "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/matched_to_test_new/test_JZ1-9_af3_test.root"
    "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/test_JZ1-9_FullSim.root"
)
ALT_GLOB = (
    "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/user.edreyer.mc23_13p6TeV_jz*_truth_pflow.v4_EXT0/user.edreyer*.root"
    # "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/hamza_samples/test_JZ1-9.root"
)
# OUTDIR = "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3"
OUTDIR = "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/matched_to_test"
TREE = "evt_tree"

parser = argparse.ArgumentParser()
parser.add_argument("--ref",    default=REFERENCE, help="Reference ROOT file")
parser.add_argument("--glob",   default=ALT_GLOB,  help="Glob pattern for alternate files")
parser.add_argument("--outdir", default=OUTDIR,    help="Output directory")
parser.add_argument("--tree",   default=TREE,      help="TTree name")
parser.add_argument("--split",  default=None, choices=["test", "train", "val"],
                    help="Process only this split (default: all three)")
args = parser.parse_args()

period = round(1.0 / (1.0 - TRAIN_FRAC))  # = 10 for TRAIN_FRAC=0.9

def get_duplicate_event_numbers(event_numbers):
    """Return a set of event numbers that appear more than once."""
    unique, counts = np.unique(event_numbers, return_counts=True)
    duplicates = set(unique[counts > 1])
    return duplicates

# ── 1. Load reference event numbers ──────────────────────────────────────────
print(f"Reading reference file: {args.ref}")
with uproot.open(args.ref) as f:
    ref_evtnums = f[args.tree]["eventNumber"].array(library="np")
ref_duplicates = get_duplicate_event_numbers(ref_evtnums)
if ref_duplicates:
    print(f"WARNING: {len(ref_duplicates)} duplicate eventNumbers found in reference. Removing them...")
    ref_evtnums = ref_evtnums[~np.isin(ref_evtnums, list(ref_duplicates))]
ref_set = set(ref_evtnums.tolist())
print(f"  {len(ref_evtnums)} events in reference ({len(ref_set)} unique)")

# ── 2. Expand glob ────────────────────────────────────────────────────────────
candidate_files = sorted(_glob.glob(args.glob))
print(f"  {len(candidate_files)} candidate alternate files found")
if not candidate_files:
    sys.exit("ERROR: no files matched the glob pattern")

# ── 3. Single pass: collect test and trainval events ─────────────────────────
collected_test     = {}   # eventNumber -> {branch: value}
collected_trainval = {}   # eventNumber -> {branch: value}
branch_names = None
n_skipped = 0

print("Scanning alternate files (single pass for all splits)...")
alt_evtnums = set()  # to track duplicates across files
for path in tqdm(candidate_files):
    try:
        with uproot.open(path) as f:
            tree = f[args.tree]
            if branch_names is None:
                branch_names = list(tree.keys())
            arrays = tree.arrays(branch_names, library="ak")

        evtnums = np.asarray(arrays["eventNumber"])

        ### Remove duplicates and track which eventNumbers we've already seen
        duplicate_evtnums_self = get_duplicate_event_numbers(evtnums)
        duplicate_evtnums_other = set(evtnums) & alt_evtnums
        duplicate_evtnums = duplicate_evtnums_self | duplicate_evtnums_other
        if duplicate_evtnums:
            print(f"  WARNING: {len(duplicate_evtnums)} duplicate eventNumbers found in {path}. Removing them...")
            unique_mask = ~np.isin(evtnums, list(duplicate_evtnums))
            arrays = arrays[unique_mask]
            evtnums = evtnums[unique_mask]
        alt_evtnums.update(evtnums.tolist())       # track accepted events
        alt_evtnums.update(duplicate_evtnums_self)  # also block self-duplicates from leaking through later files

        is_test = np.isin(evtnums, list(ref_set))

        for flag, coll in ((is_test, collected_test), (~is_test, collected_trainval)):
            if not np.any(flag):
                continue
            sub = arrays[flag]
            sub_evtnums = evtnums[flag]
            for i, evn in enumerate(sub_evtnums):
                evn_int = int(evn)
                if evn_int not in coll:
                    coll[evn_int] = {b: sub[b][i] for b in branch_names}

    except Exception as e:
        print(f"  WARNING: skipping {path}: {e}")
        n_skipped += 1

print(f"  test:     {len(collected_test)} unique events")
print(f"  trainval: {len(collected_trainval)} unique events")
print(f"  {n_skipped} files skipped due to errors")

# ── 4. Deterministic train/val split ─────────────────────────────────────────
sorted_tv_evns = np.sort(np.array(list(collected_trainval.keys())))
collected_train = {
    int(e): collected_trainval[int(e)]
    for i, e in enumerate(sorted_tv_evns)
    if i % period != period - 1
}
collected_val = {
    int(e): collected_trainval[int(e)]
    for i, e in enumerate(sorted_tv_evns)
    if i % period == period - 1
}
print(f"  train: {len(collected_train)} events  |  val: {len(collected_val)} events")

# ── 5. Build ordered event lists ──────────────────────────────────────────────
missing = ref_set - set(collected_test.keys())
if missing:
    print(f"WARNING: {len(missing)} reference eventNumbers not found in alternate files")

all_splits = {
    "test":  [int(e) for e in ref_evtnums if int(e) in collected_test],
    "train": sorted(collected_train.keys()),
    "val":   sorted(collected_val.keys()),
}
all_collections = {
    "test":  collected_test,
    "train": collected_train,
    "val":   collected_val,
}
requested = [args.split] if args.split else list(all_splits.keys())
splits = {k: all_splits[k] for k in requested}
collections = {k: all_collections[k] for k in requested}

# ── 6. Write all three output files ──────────────────────────────────────────
basename = os.path.splitext(os.path.basename(args.ref))[0]  # e.g. test_JZ1-9_hamza
os.makedirs(args.outdir, exist_ok=True)

for split, ordered_evtnums in splits.items():
    coll = collections[split]
    out_path = os.path.join(args.outdir, f"{basename}_af3_{split}.root")
    print(f"Writing {len(ordered_evtnums)} events → {out_path}")

    ordered_data = {b: [] for b in branch_names}
    for evn in ordered_evtnums:
        event = coll[evn]
        for b in branch_names:
            ordered_data[b].append(event[b])

    out_tree = {b: ak.Array(ordered_data[b]) for b in branch_names}
    with uproot.recreate(out_path) as fout:
        fout[args.tree] = out_tree

print("Done.")

