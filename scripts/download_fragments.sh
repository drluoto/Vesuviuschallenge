#!/bin/bash
# Download fragment data for ink detection training.
#
# Fragments are small detached pieces of scroll with visible ink,
# providing ground truth for ML training via IR photography.
#
# Source: https://scrollprize.org/data
# License: CC-BY-NC 4.0

set -euo pipefail

DATA_DIR="${1:-data/fragments}"
BASE_URL="https://data.aws.ash2txt.org/samples"

mkdir -p "$DATA_DIR"

echo "=== Vesuvius Challenge Fragment Download ==="
echo "Target directory: $DATA_DIR"
echo ""

# Check for required tools
if ! command -v curl &> /dev/null; then
    echo "Error: curl is required but not installed."
    exit 1
fi

# Fragment 1 - has ink labels and surface volumes
# This is the primary training fragment used in the competition
FRAGMENT_IDS=("PHercParis1Fr39" "PHercParis1Fr88")

for FRAG_ID in "${FRAGMENT_IDS[@]}"; do
    echo "--- Downloading $FRAG_ID metadata ---"
    FRAG_DIR="$DATA_DIR/$FRAG_ID"
    mkdir -p "$FRAG_DIR"

    # Download the sample index to find available files
    curl -sS --fail "$BASE_URL/$FRAG_ID/" -o "$FRAG_DIR/index.html" 2>/dev/null || {
        echo "Warning: Could not fetch index for $FRAG_ID, skipping"
        continue
    }
    echo "  Index downloaded. Check $FRAG_DIR/index.html for available files."
done

echo ""
echo "=== Download Notes ==="
echo ""
echo "The full fragment datasets are large (multiple GB each)."
echo "For initial exploration, use the vesuvius Python library instead:"
echo ""
echo "  from vesuvius import Volume"
echo "  vol = Volume('PHercParis1Fr39')"
echo ""
echo "Or browse samples at: $BASE_URL"
echo ""
echo "For bulk download, consider using rclone or aws s3 cp:"
echo "  aws s3 cp --no-sign-request s3://vesuvius-challenge-open-data/ $DATA_DIR/ --recursive"
echo ""
echo "See: https://scrollprize.org/data for full details."
