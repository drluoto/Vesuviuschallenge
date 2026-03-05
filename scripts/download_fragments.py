"""Download fragment data for ink detection training.

Downloads surface volume layers, ink labels, and masks for fragments
that have ground truth ink annotations.

Usage:
  python scripts/download_fragments.py --fragment 3
  python scripts/download_fragments.py --fragment 3 --z-start 15 --z-end 45
  python scripts/download_fragments.py --fragment 3 --metadata-only
  python scripts/download_fragments.py --fragment 4
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

BASE_URL = "https://dl.ash2txt.org/fragments"

# Each fragment has different naming conventions on the server
FRAGMENTS = {
    1: {
        "name": "Frag1",
        "base": "Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface",
        "metadata": {"inklabels.png": "inklabels.png", "mask.png": "mask.png", "ir.png": "ir.png"},
        "surface_volume_dir": "surface_volume",
    },
    2: {
        "name": "Frag2",
        "base": "Frag2/PHercParis2Fr143.volpkg/working/54keV_exposed_surface",
        "metadata": {"inklabels.png": "inklabels.png", "mask.png": "mask.png", "ir.png": "ir.png"},
        "surface_volume_dir": "surface_volume",
    },
    3: {
        "name": "Frag3",
        "base": "Frag3/PHercParis1Fr34.volpkg/working/54keV_exposed_surface",
        "metadata": {"inklabels.png": "inklabels.png", "mask.png": "mask.png", "ir.png": "ir.png"},
        "surface_volume_dir": "surface_volume",
    },
    4: {
        "name": "Frag4",
        "base": "Frag4/PHercParis1Fr39.volpkg/working/54keV_exposed_surface",
        "metadata": {
            "inklabels.png": "PHercParis1Fr39_54keV_inklabels.png",
            "mask.png": "PHercParis1Fr39_54keV_mask.png",
        },
        "surface_volume_dir": "PHercParis1Fr39_54keV_surface_volume",
    },
}


def download_file(url: str, dest: Path) -> tuple[str, bool, str]:
    """Download a single file. Returns (filename, success, message)."""
    if dest.exists() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size / 1048576
        return (dest.name, True, f"already exists ({size_mb:.1f} MB)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, dest)
        size_mb = dest.stat().st_size / 1048576
        return (dest.name, True, f"{size_mb:.1f} MB")
    except (URLError, Exception) as e:
        if dest.exists():
            dest.unlink()
        return (dest.name, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Download Vesuvius fragment data")
    parser.add_argument("--fragment", type=int, default=3, choices=[1, 2, 3, 4],
                        help="Fragment number (default: 3)")
    parser.add_argument("--output-dir", type=str, default="data/fragments")
    parser.add_argument("--z-start", type=int, default=15, help="Start z-layer (default: 15)")
    parser.add_argument("--z-end", type=int, default=45, help="End z-layer exclusive (default: 45)")
    parser.add_argument("--metadata-only", action="store_true", help="Only download labels and masks")
    parser.add_argument("--workers", type=int, default=3, help="Parallel download threads")
    args = parser.parse_args()

    frag = FRAGMENTS[args.fragment]
    frag_url = f"{BASE_URL}/{frag['base']}"
    output_dir = Path(args.output_dir) / frag["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Downloading {frag['name']} ===")
    print(f"Source: {frag_url}")
    print(f"Target: {output_dir}")
    print()

    # Download metadata files
    print("Downloading metadata...")
    for local_name, remote_name in frag["metadata"].items():
        url = f"{frag_url}/{remote_name}"
        dest = output_dir / local_name
        name, ok, msg = download_file(url, dest)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {local_name}: {msg}")

    if args.metadata_only:
        print("\nMetadata download complete.")
        return

    # Download surface volume layers
    sv_dir = output_dir / "surface_volume"
    sv_dir.mkdir(exist_ok=True)
    sv_remote = frag["surface_volume_dir"]
    layer_range = range(args.z_start, args.z_end)
    print(f"\nDownloading surface volume layers {args.z_start}-{args.z_end - 1} ({len(layer_range)} layers)...")

    tasks = []
    for z in layer_range:
        url = f"{frag_url}/{sv_remote}/{z:02d}.tif"
        dest = sv_dir / f"{z:02d}.tif"
        tasks.append((url, dest))

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_file, url, dest): dest for url, dest in tasks}
        for future in as_completed(futures):
            name, ok, msg = future.result()
            completed += 1
            if ok:
                print(f"  [{completed}/{len(tasks)}] {name}: {msg}")
            else:
                failed += 1
                print(f"  [{completed}/{len(tasks)}] FAILED {name}: {msg}")

    print(f"\nDone. Downloaded {completed - failed}/{len(tasks)} layers.")
    if failed:
        print(f"  {failed} layers failed. Re-run to retry.")

    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1048576
    print(f"  Total size: {total_size:.0f} MB")


if __name__ == "__main__":
    main()
