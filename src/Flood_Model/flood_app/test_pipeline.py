"""
Test script — verify the full prediction pipeline end-to-end.
"""

import sys
import os

# Ensure we can import from the flood_app package
sys.path.insert(0, os.path.dirname(__file__))

from basin_check import is_inside_basin
from raster_extract import extract_features
from preprocess import preprocess_features
from predict import predict


def test_point(name, lat, lon):
    print(f"\n{'='*60}")
    print(f"  Testing: {name} ({lat}, {lon})")
    print(f"{'='*60}")

    # Basin check
    inside = is_inside_basin(lat, lon)
    print(f"  Inside basin: {inside}")
    if not inside:
        print("  → Skipping prediction (outside basin)")
        return

    # Feature extraction
    print("  Extracting features...")
    features = extract_features(lat, lon)
    if features is None:
        print("  → Feature extraction returned None (nodata)")
        return

    print("  Extracted features:")
    for k, v in features.items():
        print(f"    {k:>22s}: {v:.4f}")

    # Preprocess
    print("  Preprocessing...")
    scaled = preprocess_features(features)
    print(f"  Scaled input shape: {scaled.shape}")

    # Predict
    print("  Running prediction...")
    result = predict(scaled)
    print(f"  Probability: {result['probability']:.4f}")
    print(f"  Risk class:  {result['risk_class']}")


if __name__ == "__main__":
    # Test 1: Cuttack (inside Mahanadi Basin)
    test_point("Cuttack", 20.4625, 85.8828)

    # Test 2: Delhi (outside Mahanadi Basin)
    test_point("Delhi", 28.6139, 77.2090)

    # Test 3: Sambalpur (inside Mahanadi Basin)
    test_point("Sambalpur", 21.4669, 83.9756)

    print(f"\n{'='*60}")
    print("  All tests complete!")
    print(f"{'='*60}")
