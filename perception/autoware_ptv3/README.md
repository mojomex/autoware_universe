# autoware_ptv3

## Purpose

The `autoware_ptv3` package is used for 3D lidar segmentation.

## Inner-workings / Algorithms

This package implements a TensorRT powered inference node for Point Transformers V3 (PTv3) [1].
The sparse convolution backend corresponds to [spconv](https://github.com/traveller59/spconv).
Autoware installs it automatically in its setup script. If needed, the user can also build it and install it following the [following instructions](https://github.com/autowarefoundation/spconv_cpp).

## Inputs / Outputs

### Input

| Name                 | Type                            | Description             |
| -------------------- | ------------------------------- | ----------------------- |
| `~/input/pointcloud` | `sensor_msgs::msg::PointCloud2` | Input pointcloud topic. |

### Output

| Name                                   | Type                                                | Description                                             |
| -------------------------------------- | --------------------------------------------------- | ------------------------------------------------------- |
| `~/output/pointcloud/segmentation`     | `sensor_msgs::msg::PointCloud2`                     | XYZ cloud with class ID and probability fields.         |
| `~/output/pointcloud/visualization`    | `sensor_msgs::msg::PointCloud2`                     | XYZ cloud with RGB field.                               |
| `~/output/pointcloud/filtered`         | `sensor_msgs::msg::PointCloud2`                     | Filtered cloud in the requested `filter.output_format`. |
| `debug/cyclic_time_ms`                 | `autoware_internal_debug_msgs::msg::Float64Stamped` | Cyclic time (ms).                                       |
| `debug/pipeline_latency_ms`            | `autoware_internal_debug_msgs::msg::Float64Stamped` | Pipeline latency time (ms).                             |
| `debug/processing_time/preprocess_ms`  | `autoware_internal_debug_msgs::msg::Float64Stamped` | Preprocess (ms).                                        |
| `debug/processing_time/inference_ms`   | `autoware_internal_debug_msgs::msg::Float64Stamped` | Inference time (ms).                                    |
| `debug/processing_time/postprocess_ms` | `autoware_internal_debug_msgs::msg::Float64Stamped` | Postprocess time (ms).                                  |
| `debug/processing_time/total_ms`       | `autoware_internal_debug_msgs::msg::Float64Stamped` | Total processing time (ms).                             |

## Parameters

### PTv3Node node

{{ json_to_markdown("perception/autoware_ptv3/schema/ptv3.schema.json") }}

### PTv3Node model

{{ json_to_markdown("perception/autoware_ptv3/schema/ml_package_ptv3.schema.json") }}

`filter.*` parameters are configured in `config/ptv3.param.yaml`, while class metadata and the
visualization `palette` are configured in `config/ml_package_ptv3.param.yaml`.

### The `build_only` option

The `autoware_ptv3` node has a `build_only` option to build the TensorRT engine file from the specified ONNX file, after which the program exits.

```bash
ros2 launch autoware_ptv3 ptv3.launch.xml build_only:=true
```

### The `log_level` option

The default logging severity level for `autoware_ptv3` is `info`. For debugging purposes, the developer may decrease severity level using `log_level` parameter:

```bash
ros2 launch autoware_ptv3 ptv3.launch.xml log_level:=debug
```

### Benchmark prediction dumps

`autoware_ptv3_benchmark` supports writing raw predictions for regression checks with:

```bash
ros2 run autoware_ptv3 autoware_ptv3_benchmark \
  --warmup 0 \
  --iterations 1 \
  --prediction-dump-prefix /tmp/ptv3-baseline/pred
```

This writes one metadata file plus raw `int64` labels and `float32` probabilities per measured
iteration:

- `/tmp/ptv3-baseline/pred_iter000.meta.json`
- `/tmp/ptv3-baseline/pred_iter000.labels.bin`
- `/tmp/ptv3-baseline/pred_iter000.probs.bin`

The dump is intended for plugin/runtime validation with the same:

- bag input
- params file
- ONNX / engine artifacts
- plugin shared library path

To compare a baseline dump against a candidate build, run:

```bash
python3 \
  src/universe/autoware_universe/perception/autoware_ptv3/tools/compare_ptv3_prediction_dump.py \
  --baseline-prefix /tmp/ptv3-baseline/pred \
  --candidate-prefix /tmp/ptv3-candidate/pred \
  --atol 1e-6
```

For code-only plugin optimizations, the expected result is:

- exact `pred_labels` match
- exact `num_voxels` match
- `pred_probs` max absolute difference within a small floating-point tolerance

This check is complementary to direct plugin reference tests in
`autoware_tensorrt_plugins`.

If a direct plugin reference test exposes a pre-existing correctness bug in the baseline plugin
implementation, prefer the direct reference result and treat the historical PTv3 dump as a
change-detection signal rather than the final oracle.

## Assumptions / Known limits

This node detects the input pointcloud format automatically on the first received message and
supports:

- `XYZIRCAEDT` (10 fields)
- `XYZIRADRT` (9 fields)
- `XYZIRC` (6 fields)
- `XYZI` (4 fields)

The filtered output cloud format is controlled by `filter.output_format`. When it is set to an
empty string, the filtered output preserves the same format as the input cloud.

## Trained Models

The model was trained on the T4Dataset using approximately 4,000 frames and is available in the Autoware artifacts.

## Troubleshooting

### `Fail to create host memory`

This error may occur when the system runs out of memory during TensorRT engine building. To mitigate this issue, consider reducing the `voxels_num` maximum limit in the model parameters, which controls the maximum number of voxels processed by the model. For the instance, using GPU with 8 GB of memory, setting maximum `voxels_num` to 192'000 may help to avoid this error.

## References/External links

[1] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. "Point Transformer V3: Simpler, Faster, Stronger." 2024 Conference on Computer Vision and Pattern Recognition. <!-- cspell:disable-line -->
