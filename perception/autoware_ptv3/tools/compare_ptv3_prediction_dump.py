#!/usr/bin/env python3

import argparse
import json
import re
import sys
from array import array
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PTv3 benchmark prediction dumps produced with --prediction-dump-prefix."
    )
    parser.add_argument("--baseline-prefix", required=True, help="Baseline dump prefix")
    parser.add_argument("--candidate-prefix", required=True, help="Candidate dump prefix")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Maximum allowed absolute difference for probabilities",
    )
    return parser.parse_args()


def collect_iterations(prefix_str):
    prefix = Path(prefix_str)
    parent = prefix.parent if prefix.parent != Path("") else Path(".")
    pattern = re.compile(rf"^{re.escape(prefix.name)}_iter(\d+)\.meta\.json$")

    iterations = {}
    for meta_path in sorted(parent.glob(f"{prefix.name}_iter*.meta.json")):
        match = pattern.match(meta_path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        base_path = parent / meta_path.name[: -len(".meta.json")]
        iterations[iteration] = {
            "meta": meta_path,
            "labels": Path(str(base_path) + ".labels.bin"),
            "probs": Path(str(base_path) + ".probs.bin"),
        }
    return iterations


def load_meta(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_array(path, typecode, expected_len):
    values = array(typecode)
    with path.open("rb") as stream:
        values.frombytes(stream.read())
    if len(values) != expected_len:
        raise ValueError(f"{path} has {len(values)} values, expected {expected_len}")
    return values


def compare_iteration(iteration, baseline_paths, candidate_paths, atol):
    baseline_meta = load_meta(baseline_paths["meta"])
    candidate_meta = load_meta(candidate_paths["meta"])

    for key in ("num_voxels", "num_classes"):
        if baseline_meta[key] != candidate_meta[key]:
            raise ValueError(
                f"Iteration {iteration}: metadata mismatch for {key}: "
                f"{baseline_meta[key]} != {candidate_meta[key]}"
            )

    num_voxels = baseline_meta["num_voxels"]
    num_classes = baseline_meta["num_classes"]
    prob_count = num_voxels * num_classes

    baseline_labels = load_array(baseline_paths["labels"], "q", num_voxels)
    candidate_labels = load_array(candidate_paths["labels"], "q", num_voxels)
    baseline_probs = load_array(baseline_paths["probs"], "f", prob_count)
    candidate_probs = load_array(candidate_paths["probs"], "f", prob_count)

    label_mismatches = 0
    first_label_mismatch = None
    for index, (baseline_value, candidate_value) in enumerate(zip(baseline_labels, candidate_labels)):
        if baseline_value != candidate_value:
            label_mismatches += 1
            if first_label_mismatch is None:
                first_label_mismatch = (index, baseline_value, candidate_value)

    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    max_abs_diff_index = 0
    for index, (baseline_value, candidate_value) in enumerate(zip(baseline_probs, candidate_probs)):
        diff = abs(float(baseline_value) - float(candidate_value))
        sum_abs_diff += diff
        if diff > max_abs_diff:
            max_abs_diff = diff
            max_abs_diff_index = index

    mean_abs_diff = sum_abs_diff / prob_count if prob_count > 0 else 0.0

    print(
        f"iter={iteration:03d} voxels={num_voxels} labels_mismatch={label_mismatches} "
        f"prob_max_abs_diff={max_abs_diff:.8g} prob_mean_abs_diff={mean_abs_diff:.8g}"
    )

    if first_label_mismatch is not None:
        index, baseline_value, candidate_value = first_label_mismatch
        print(
            f"  first label mismatch at index {index}: baseline={baseline_value} "
            f"candidate={candidate_value}"
        )

    if max_abs_diff > atol:
        print(
            f"  probability max abs diff exceeded tolerance at flat index {max_abs_diff_index}: "
            f"{max_abs_diff:.8g} > {atol:.8g}"
        )

    return label_mismatches == 0 and max_abs_diff <= atol


def main():
    args = parse_args()

    baseline_iterations = collect_iterations(args.baseline_prefix)
    candidate_iterations = collect_iterations(args.candidate_prefix)

    if not baseline_iterations:
        raise ValueError("No baseline prediction dumps found.")
    if not candidate_iterations:
        raise ValueError("No candidate prediction dumps found.")

    if set(baseline_iterations) != set(candidate_iterations):
        raise ValueError(
            "Iteration sets do not match: "
            f"baseline={sorted(baseline_iterations)} candidate={sorted(candidate_iterations)}"
        )

    all_ok = True
    for iteration in sorted(baseline_iterations):
        all_ok &= compare_iteration(
            iteration, baseline_iterations[iteration], candidate_iterations[iteration], args.atol
        )

    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
