#!/bin/bash
# Script to run the tokenset_combiner and tokenset_filter in sequence

# Set default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_INPUT_DIR="tokensets"
DEFAULT_COMBINED_OUTPUT="/tmp/tokenset_combined.bin"
DEFAULT_FILTERED_OUTPUT="/tmp/tokenset_filtered.bin"
BIN_DIR="./zig-out/bin"  # Adjust this path if your binaries are elsewhere

# Parse command line arguments
INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"
COMBINED_OUTPUT="${2:-$DEFAULT_COMBINED_OUTPUT}"
FILTERED_OUTPUT="${3:-$DEFAULT_FILTERED_OUTPUT}"

# Display usage information
function show_usage {
  echo "Usage: $0 [INPUT_DIR] [COMBINED_OUTPUT] [FILTERED_OUTPUT]"
  echo ""
  echo "Arguments:"
  echo "  INPUT_DIR         Directory containing tokenset files (default: $DEFAULT_INPUT_DIR)"
  echo "  COMBINED_OUTPUT   Path for combined tokenset (default: $DEFAULT_COMBINED_OUTPUT)"
  echo "  FILTERED_OUTPUT   Path for filtered tokenset (default: $DEFAULT_FILTERED_OUTPUT)"
  echo ""
  echo "This script builds and runs the tokenset combiner followed by the tokenset filter."
  echo "Example: $0 ./data/tokensets /tmp/combined.bin /tmp/filtered.bin"
}

# Show help if requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  show_usage
  exit 0
fi

# Build the project
echo "Building project..."
zig build
BUILD_EXIT_CODE=$?
if [ $BUILD_EXIT_CODE -ne 0 ]; then
  echo "Error: Build failed with exit code $BUILD_EXIT_CODE"
  exit $BUILD_EXIT_CODE
fi
echo "Build completed successfully."
echo ""

# Check if binaries exist (should exist after build, but checking as a safeguard)
if [ ! -f "$BIN_DIR/tokenset_combiner" ]; then
  echo "Error: tokenset_combiner binary not found at $BIN_DIR/tokenset_combiner"
  echo "Build was successful but binaries not found. Please check build configuration."
  exit 1
fi

if [ ! -f "$BIN_DIR/tokenset_filter" ]; then
  echo "Error: tokenset_filter binary not found at $BIN_DIR/tokenset_filter"
  echo "Build was successful but binaries not found. Please check build configuration."
  exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory not found: $INPUT_DIR"
  exit 1
fi

echo "=== Tokenset Processing Pipeline ==="
echo "Input directory:   $INPUT_DIR"
echo "Combined output:   $COMBINED_OUTPUT"
echo "Filtered output:   $FILTERED_OUTPUT"
echo ""

# Step 1: Run the tokenset_combiner
echo "Step 1: Combining tokensets from $INPUT_DIR..."
START_TIME=$(date +%s)

"$BIN_DIR/tokenset_combiner" "$INPUT_DIR" --output "$COMBINED_OUTPUT"
COMBINER_EXIT_CODE=$?

if [ $COMBINER_EXIT_CODE -ne 0 ]; then
  echo "Error: tokenset_combiner failed with exit code $COMBINER_EXIT_CODE"
  exit $COMBINER_EXIT_CODE
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Tokenset combining completed in ${DURATION}s"
echo ""

# Step 2: Run the tokenset_filter
echo "Step 2: Filtering combined tokenset..."
START_TIME=$(date +%s)

"$BIN_DIR/tokenset_filter" "$COMBINED_OUTPUT" --output "$FILTERED_OUTPUT"
FILTER_EXIT_CODE=$?

if [ $FILTER_EXIT_CODE -ne 0 ]; then
  echo "Error: tokenset_filter failed with exit code $FILTER_EXIT_CODE"
  exit $FILTER_EXIT_CODE
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Tokenset filtering completed in ${DURATION}s"
echo ""

# Calculate file sizes
COMBINED_SIZE=$(du -h "$COMBINED_OUTPUT" | cut -f1)
FILTERED_SIZE=$(du -h "$FILTERED_OUTPUT" | cut -f1)

echo "=== Summary ==="
echo "Combined tokenset size: $COMBINED_SIZE"
echo "Filtered tokenset size: $FILTERED_SIZE"
echo "Filtered tokenset saved to: $FILTERED_OUTPUT"
echo ""
echo "Pipeline completed successfully."