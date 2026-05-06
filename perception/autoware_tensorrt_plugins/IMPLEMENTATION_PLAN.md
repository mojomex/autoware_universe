# TensorRT Plugin Optimization Plan

This package will be split into stacked PRs on top of a clean `autoware_universe` branch.
Each PR should stay limited to one work package so performance and correctness regressions are easy
to isolate.

## PR 1: Remove Thrust from `Argsort`, `CustomUnique`, and `SegmentCSR`

Scope:
- Replace the `thrust::sequence` path in `src/argsort_ops/argsort.cu`.
- Replace the `thrust::{sequence,adjacent_difference,inclusive_scan,scatter,unique_by_key}` path
  in `src/unique_ops/unique.cu`.
- Confirm whether `src/scatter_ops/segment_csr.cu` contains any Thrust usage.

Implementation direction:
- `Argsort`: keep `cub::DeviceRadixSort`, replace the Thrust-generated index buffer with a small
  CUDA iota kernel, and keep the sorted-key scratch buffer in the TensorRT workspace.
- `CustomUnique`: keep radix sort, replace the rest with CUB primitives plus small CUDA kernels:
  run-start marking, prefix-scan, inverse-index scatter, unique-offset extraction, and count
  materialization.
- `SegmentCSR`: no Thrust path exists today, so this item is expected to be a verification-only
  no-op unless deeper inspection finds hidden Thrust includes elsewhere.

Validation:
- Add kernel-level tests for `Argsort`.
- Keep the existing `CustomUnique` and `SegmentCSR` reference tests passing.

## PR 2: Remove Runtime Dynamic Allocations

Scope:
- Remove `cudaMalloc(Async)` and `cudaFree(Async)` from these three plugins and their helper kernels.
- Replace them with fixed-size buffers allocated during plugin startup or reserved through the
  TensorRT workspace contract.

Current allocation inventory:
- `SegmentCSR` currently allocates `indptr_size_dev` per enqueue in
  `src/scatter_ops/segment_csr.cu`.
- `Argsort` and `CustomUnique` already use the TensorRT workspace for their hot-path scratch, but
  we still need to make the layout explicit and fixed against the maximum supported shape.

Open sizing questions to answer before implementing:
- What is the maximum `num_elements` for `Argsort` and `CustomUnique` from the TensorRT optimization
  profiles that Autoware actually builds?
- What is the maximum `indptr` rank and element count that `SegmentCSR` must support?
- Do we want these caps serialized in the plugin instance, inferred from `inputs[...].max`, or both?

Implementation direction:
- Derive fixed maxima from `DynamicPluginTensorDesc::max` during build/configure time.
- Precompute exact workspace/buffer slices from those maxima and reject unsupported runtime shapes
  early instead of falling back to dynamic allocation.

## PR 3: Remove Unnecessary Device/Stream Synchronization

Scope:
- Remove `cudaDeviceSynchronize` and `cudaStreamSynchronize` calls that are not required for plugin
  correctness.
- Confirm whether any synchronizations are currently hidden behind Thrust/CUB or host-side scalar
  materialization.

Current state to verify in code and profiling:
- These three target files do not currently contain explicit sync calls, but we still need to
  inspect the resulting kernel traces for hidden host synchronization around CUB/Thrust operations
  and plugin outputs.

Validation:
- Use Nsight Systems or equivalent profiling around representative PTv3 execution to confirm the
  optimized path stays stream-ordered without host-side gaps.

## PR 4: Remove Avoidable Sort-Related Synchronization and Retune CUB Usage

Scope:
- Revisit `cub::DeviceRadixSort` usage in `Argsort` and `CustomUnique`.
- Eliminate any remaining `DeviceRadixSortOneSweepKernels` plus synchronization patterns caused by
  workspace queries, shape-dependent setup, or host-visible scalar outputs.

Implementation direction:
- Precompute everything that can be precomputed from the max profile shape.
- Keep sort scratch fully resident in preallocated buffers.
- Prefer device-visible outputs over host-visible scalar round-trips where TensorRT already expects
  a device tensor output.

Validation:
- Compare kernel timelines before and after.
- Confirm that the sort path no longer alternates with host sync points during steady-state
  inference.
