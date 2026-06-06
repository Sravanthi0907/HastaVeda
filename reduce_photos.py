"""
=============================================================================
  reduce_photos.py  —  HASTAVEDA Mudra Dataset Photo Reducer
=============================================================================
  PURPOSE:
      Run this script AFTER training your mudra model is complete.
      It reduces the number of photos in each mudra folder to a target count
      (default: 20 per mudra) so the dataset is small enough to push to GitHub
      while still giving GitHub a representative set of sample images.

      At ~57 KB per image, 7 mudras × 20 photos ≈ 7.98 MB — well within
      GitHub's 100 MB per-file limit and 1 GB recommended repo size.

      It also trims mudra_samples.json to match the same count.

  HOW TO RUN:
      python reduce_photos.py

  OPTIONS:
      Edit the CONFIGURATION section below to change:
        - TARGET_PHOTOS_PER_MUDRA  →  how many photos to keep per mudra
        - DRY_RUN                  →  set True to preview without deleting

  WARNING:
      This PERMANENTLY DELETES photos from mudra_data/<mudra>/ folders.
      Make sure training is complete before running!
=============================================================================
"""

import os
import json
import glob

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit these values as needed)
# ──────────────────────────────────────────────────────────────────────────────

# Number of photos to keep per mudra folder
# At ~57 KB per image: 7 mudras × 10 photos ≈ 3.99 MB — fine for GitHub
TARGET_PHOTOS_PER_MUDRA = 10

# Set to True to preview what WOULD be deleted without actually deleting
DRY_RUN = False

# ──────────────────────────────────────────────────────────────────────────────

# Auto-detect script directory so it works from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MUDRA_DATA_DIR = os.path.join(SCRIPT_DIR, "mudra_data")
SAMPLES_JSON_PATH = os.path.join(MUDRA_DATA_DIR, "mudra_samples.json")


def get_mudra_folders():
    """Returns a list of all mudra class folder names inside mudra_data/."""
    folders = []
    if not os.path.isdir(MUDRA_DATA_DIR):
        print(f"[ERROR] mudra_data directory not found at: {MUDRA_DATA_DIR}")
        return folders
    for entry in os.listdir(MUDRA_DATA_DIR):
        full_path = os.path.join(MUDRA_DATA_DIR, entry)
        if os.path.isdir(full_path):
            folders.append(entry)
    return sorted(folders)


def reduce_photos_in_folder(mudra_name, target_count):
    """
    Keeps the first `target_count` .jpg photos in the mudra folder
    and deletes the rest (sorted numerically by filename index).
    Returns (kept, deleted) counts.
    """
    folder = os.path.join(MUDRA_DATA_DIR, mudra_name)
    all_jpgs = glob.glob(os.path.join(folder, "*.jpg"))

    # Sort by the numeric index embedded in filename, e.g. alapadma_23.jpg → 23
    def sort_key(filepath):
        basename = os.path.splitext(os.path.basename(filepath))[0]  # e.g. "alapadma_23"
        parts = basename.rsplit("_", 1)
        try:
            return int(parts[-1])
        except ValueError:
            return 0

    all_jpgs_sorted = sorted(all_jpgs, key=sort_key)

    keep = all_jpgs_sorted[:target_count]
    delete = all_jpgs_sorted[target_count:]

    kept_count = len(keep)
    deleted_count = 0

    for filepath in delete:
        if DRY_RUN:
            print(f"    [DRY RUN] Would delete: {os.path.basename(filepath)}")
        else:
            try:
                os.remove(filepath)
                deleted_count += 1
            except OSError as e:
                print(f"    [ERROR] Could not delete {os.path.basename(filepath)}: {e}")

    return kept_count, deleted_count


def trim_samples_json(target_count):
    """
    Trims mudra_samples.json so each mudra has at most `target_count` samples.
    """
    if not os.path.exists(SAMPLES_JSON_PATH):
        print(f"\n[WARNING] mudra_samples.json not found at: {SAMPLES_JSON_PATH}")
        print("          Skipping JSON trimming step.")
        return

    with open(SAMPLES_JSON_PATH, "r", encoding="utf-8") as f:
        samples_data = json.load(f)

    original_total = sum(len(v) for v in samples_data.values())
    trimmed_data = {}

    print("\n📄  Trimming mudra_samples.json...")
    for mudra, samples in samples_data.items():
        before = len(samples)
        trimmed_data[mudra] = samples[:target_count]
        after = len(trimmed_data[mudra])
        status = f"kept {after}" if before > target_count else f"unchanged ({after})"
        print(f"    [{mudra}]  {before} → {status} samples")

    new_total = sum(len(v) for v in trimmed_data.values())

    if not DRY_RUN:
        with open(SAMPLES_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(trimmed_data, f, indent=2)
        new_size_kb = os.path.getsize(SAMPLES_JSON_PATH) / 1024
        print(f"\n    ✅  Saved updated mudra_samples.json")
        print(f"    📦  New file size: {new_size_kb:.1f} KB")
        print(f"    📊  Total samples: {original_total} → {new_total}")
    else:
        print(f"\n    [DRY RUN] Would update mudra_samples.json")
        print(f"    📊  Total samples would go: {original_total} → {new_total}")


def main():
    print("=" * 65)
    print("  HASTAVEDA  —  Mudra Dataset Photo Reducer")
    print("=" * 65)
    print(f"  Target photos per mudra : {TARGET_PHOTOS_PER_MUDRA}")
    print(f"  Mudra data directory    : {MUDRA_DATA_DIR}")
    print(f"  Dry run mode            : {'YES (no files will be deleted)' if DRY_RUN else 'NO (files will be deleted)'}")
    print("=" * 65)

    if not DRY_RUN:
        confirm = input(
            "\n⚠️  This will PERMANENTLY DELETE photos beyond "
            f"index {TARGET_PHOTOS_PER_MUDRA - 1} in each mudra folder.\n"
            "   Make sure your model training is COMPLETE before continuing.\n"
            "   Type 'yes' to proceed: "
        ).strip().lower()
        if confirm != "yes":
            print("\n❌  Aborted. No files were deleted.")
            return

    mudra_folders = get_mudra_folders()
    if not mudra_folders:
        print("[ERROR] No mudra folders found. Exiting.")
        return

    print(f"\n📁  Found {len(mudra_folders)} mudra folders: {', '.join(mudra_folders)}\n")
    print("🗂️   Reducing photos per folder...")
    print("-" * 65)

    total_kept = 0
    total_deleted = 0

    for mudra in mudra_folders:
        kept, deleted = reduce_photos_in_folder(mudra, TARGET_PHOTOS_PER_MUDRA)
        total_kept += kept
        total_deleted += deleted
        action = "[DRY RUN]" if DRY_RUN else "✅"
        print(f"  {action}  [{mudra:15s}]  Kept: {kept:3d}  |  Deleted: {deleted:3d}")

    print("-" * 65)
    print(f"\n  📊  SUMMARY")
    print(f"       Total photos kept    : {total_kept}")
    print(f"       Total photos deleted : {total_deleted}")

    # Also trim samples JSON
    trim_samples_json(TARGET_PHOTOS_PER_MUDRA)

    print("\n" + "=" * 65)
    if DRY_RUN:
        print("  DRY RUN complete. Set DRY_RUN = False to apply changes.")
    else:
        print("  ✅  Reduction complete! Your dataset is now GitHub-ready.")
        print(f"  📦  Each mudra folder now has {TARGET_PHOTOS_PER_MUDRA} photos (~{TARGET_PHOTOS_PER_MUDRA * 57} KB per mudra)")
        print("  💡  You can now run:  git add . && git commit -m 'Reduce dataset'")
    print("=" * 65)


if __name__ == "__main__":
    main()
