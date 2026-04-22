// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/ptv3/ptv3_config.hpp"
#include "autoware/ptv3/ptv3_trt.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoware/tensorrt_common/tensorrt_common.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_transport/reader_writer_factory.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::ptv3
{

struct BenchmarkOptions
{
  std::string bag_path =
    "/home/maxschmeller/.webauto/data/data/downsampled-and-filtered-middle-pointcloud-only/"
    "downsampled-and-filtered-middle-pointcloud-only_0.db3";
  std::string params_file =
    ament_index_cpp::get_package_share_directory("autoware_ptv3") +
    "/config/ml_package_ptv3.param.yaml";
  std::string model_path =
    std::string(std::getenv("HOME") != nullptr ? std::getenv("HOME") : "") + "/autoware_data/ptv3";
  std::string plugins_path =
    ament_index_cpp::get_package_share_directory("autoware_tensorrt_plugins") +
    "/plugins/libautoware_tensorrt_plugins.so";
  std::string csv_output_path{};
  std::string trt_precision{"fp16"};
  std::string filter_output_format{};
  std::vector<std::string> filter_classes{"drivable_surface"};
  int warmup_iterations{20};
  int measured_iterations{100};
  std::int64_t cloud_capacity{2000000};
  float filter_class_probability_threshold{0.05F};
  bool publish_segmented{true};
  bool publish_visualization{false};
  bool publish_filtered{false};
  bool capture_nsys{true};
};

struct SummaryStats
{
  double min_ms{0.0};
  double mean_ms{0.0};
  double p50_ms{0.0};
  double p90_ms{0.0};
  double p95_ms{0.0};
  double max_ms{0.0};
};

std::string usage(const char * program_name)
{
  std::ostringstream oss;
  oss << "Usage: " << program_name << " [options]\n"
      << "  --bag-path <path>         Input rosbag file or bag directory\n"
      << "  --params-file <path>      ML package params yaml\n"
      << "  --model-path <path>       Directory containing ptv3.onnx and ptv3.engine\n"
      << "  --plugins-path <path>     TensorRT plugin shared library\n"
      << "  --trt-precision <name>    TensorRT precision (default: fp16)\n"
      << "  --warmup <count>          Warmup iterations (default: 20)\n"
      << "  --iterations <count>      Measured iterations (default: 100)\n"
      << "  --csv <path>              Optional CSV output path\n"
      << "  --no-segmented            Disable segmented output\n"
      << "  --visualization           Enable visualization output\n"
      << "  --filtered                Enable filtered output\n"
      << "  --no-nsys-capture         Do not call cudaProfilerStart/Stop\n"
      << "  --help                    Show this help\n";
  return oss.str();
}

BenchmarkOptions parseArguments(int argc, char ** argv)
{
  BenchmarkOptions options;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto require_value = [&](const char * flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + flag);
      }
      return argv[++i];
    };

    if (arg == "--bag-path") {
      options.bag_path = require_value("--bag-path");
    } else if (arg == "--params-file") {
      options.params_file = require_value("--params-file");
    } else if (arg == "--model-path") {
      options.model_path = require_value("--model-path");
    } else if (arg == "--plugins-path") {
      options.plugins_path = require_value("--plugins-path");
    } else if (arg == "--trt-precision") {
      options.trt_precision = require_value("--trt-precision");
    } else if (arg == "--warmup") {
      options.warmup_iterations = std::stoi(require_value("--warmup"));
    } else if (arg == "--iterations") {
      options.measured_iterations = std::stoi(require_value("--iterations"));
    } else if (arg == "--csv") {
      options.csv_output_path = require_value("--csv");
    } else if (arg == "--no-segmented") {
      options.publish_segmented = false;
    } else if (arg == "--visualization") {
      options.publish_visualization = true;
    } else if (arg == "--filtered") {
      options.publish_filtered = true;
    } else if (arg == "--no-nsys-capture") {
      options.capture_nsys = false;
    } else if (arg == "--help") {
      std::cout << usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(arg));
    }
  }

  if (options.warmup_iterations < 0) {
    throw std::runtime_error("Warmup iterations must be >= 0.");
  }
  if (options.measured_iterations <= 0) {
    throw std::runtime_error("Measured iterations must be > 0.");
  }
  if (
    !options.publish_segmented && !options.publish_visualization && !options.publish_filtered) {
    throw std::runtime_error("At least one output must be enabled.");
  }

  return options;
}

std::filesystem::path resolveBagPath(const std::string & input_path)
{
  const std::filesystem::path path(input_path);
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("Bag path does not exist: " + path.string());
  }

  if (std::filesystem::is_regular_file(path)) {
    return path;
  }

  if (!std::filesystem::is_directory(path)) {
    throw std::runtime_error("Bag path must be a bag file or a directory: " + path.string());
  }

  for (const auto & entry : std::filesystem::directory_iterator(path)) {
    if (
      entry.is_regular_file() &&
      (entry.path().extension() == ".db3" || entry.path().extension() == ".mcap")) {
      return entry.path();
    }
  }

  throw std::runtime_error("No .db3 or .mcap file found under: " + path.string());
}

sensor_msgs::msg::PointCloud2 loadPointCloudFromBag(const std::filesystem::path & bag_path)
{
  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = bag_path.string();

  if (bag_path.extension() == ".mcap") {
    storage_options.storage_id = "mcap";
  } else if (bag_path.extension() == ".db3") {
    storage_options.storage_id = "sqlite3";
  } else {
    throw std::runtime_error("Unsupported bag format: " + bag_path.extension().string());
  }

  auto reader = rosbag2_transport::ReaderWriterFactory::make_reader(storage_options);
  reader->open(storage_options);
  const auto topics = reader->get_all_topics_and_types();

  std::string pointcloud_topic;
  for (const auto & topic : topics) {
    if (topic.type == "sensor_msgs/msg/PointCloud2") {
      pointcloud_topic = topic.name;
      break;
    }
  }

  if (pointcloud_topic.empty()) {
    throw std::runtime_error("No sensor_msgs/msg/PointCloud2 topic found in bag.");
  }

  rosbag2_storage::StorageFilter filter;
  filter.topics = {pointcloud_topic};
  reader->set_filter(filter);

  if (!reader->has_next()) {
    throw std::runtime_error("No PointCloud2 message found in bag after applying topic filter.");
  }

  return reader->read_next<sensor_msgs::msg::PointCloud2>();
}

PTv3Config loadConfig(const std::shared_ptr<rclcpp::Node> & node)
{
  auto descriptor = rcl_interfaces::msg::ParameterDescriptor{}.set__read_only(true);
  auto to_float_vector = [](const auto & values) {
    return std::vector<float>(values.begin(), values.end());
  };

  const std::string plugins_path = node->declare_parameter<std::string>("plugins_path", descriptor);
  const auto cloud_capacity = node->declare_parameter<std::int64_t>("cloud_capacity", descriptor);
  const auto voxels_num =
    node->declare_parameter<std::vector<std::int64_t>>("voxels_num", descriptor);
  const auto point_cloud_range =
    to_float_vector(node->declare_parameter<std::vector<double>>("point_cloud_range", descriptor));
  const auto voxel_size =
    to_float_vector(node->declare_parameter<std::vector<double>>("voxel_size", descriptor));
  const auto class_names =
    node->declare_parameter<std::vector<std::string>>("class_names", descriptor);
  const auto palette = node->declare_parameter<std::vector<std::int64_t>>("palette", descriptor);
  const auto filter_class_probability_threshold =
    node->declare_parameter<float>("filter.class_probability_threshold", descriptor);
  const auto filter_classes =
    node->declare_parameter<std::vector<std::string>>("filter.classes", descriptor);
  const auto filter_output_format =
    node->declare_parameter<std::string>("filter.output_format", descriptor);

  return PTv3Config(
    plugins_path, cloud_capacity, voxels_num, point_cloud_range, voxel_size, class_names, palette,
    filter_class_probability_threshold, filter_classes, filter_output_format);
}

tensorrt_common::TrtCommonConfig loadTrtConfig(const std::shared_ptr<rclcpp::Node> & node)
{
  auto descriptor = rcl_interfaces::msg::ParameterDescriptor{}.set__read_only(true);
  const std::string onnx_path = node->declare_parameter<std::string>("onnx_path", descriptor);
  const std::string engine_path = node->declare_parameter<std::string>("engine_path", descriptor);
  const std::string trt_precision =
    node->declare_parameter<std::string>("trt_precision", descriptor);

  return tensorrt_common::TrtCommonConfig(onnx_path, trt_precision, engine_path, 1ULL << 33U);
}

SummaryStats summarize(const std::vector<double> & values)
{
  if (values.empty()) {
    return {};
  }

  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());

  const auto percentile = [&](const double p) {
    const double rank = p * static_cast<double>(sorted.size() - 1);
    const auto lower_index = static_cast<size_t>(std::floor(rank));
    const auto upper_index = static_cast<size_t>(std::ceil(rank));
    if (lower_index == upper_index) {
      return sorted[lower_index];
    }

    const double fraction = rank - static_cast<double>(lower_index);
    return sorted[lower_index] * (1.0 - fraction) + sorted[upper_index] * fraction;
  };

  const double mean =
    std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());

  return SummaryStats{
    sorted.front(), mean, percentile(0.50), percentile(0.90), percentile(0.95), sorted.back()};
}

void printSummaryLine(const std::string & name, const SummaryStats & stats)
{
  std::cout << std::fixed << std::setprecision(3) << std::left << std::setw(20) << name
            << " min=" << std::setw(8) << stats.min_ms << " mean=" << std::setw(8) << stats.mean_ms
            << " p50=" << std::setw(8) << stats.p50_ms << " p90=" << std::setw(8) << stats.p90_ms
            << " p95=" << std::setw(8) << stats.p95_ms << " max=" << stats.max_ms << '\n';
}

void writeCsv(
  const std::string & csv_path, const std::vector<PTv3BenchmarkMetrics> & metrics_per_iteration)
{
  if (csv_path.empty()) {
    return;
  }

  std::ofstream stream(csv_path);
  if (!stream.is_open()) {
    throw std::runtime_error("Failed to open CSV output path: " + csv_path);
  }

  stream << "iteration,cpu_preprocess_ms,cpu_inference_enqueue_ms,cpu_postprocess_ms,"
            "cpu_total_ms,gpu_inference_ms,gpu_postprocess_ms,gpu_total_ms,num_voxels\n";

  for (size_t i = 0; i < metrics_per_iteration.size(); ++i) {
    const auto & metrics = metrics_per_iteration[i];
    stream << i << ',' << metrics.cpu_preprocess_ms << ',' << metrics.cpu_inference_enqueue_ms
           << ',' << metrics.cpu_postprocess_ms << ',' << metrics.cpu_total_ms << ','
           << metrics.gpu_inference_ms << ',' << metrics.gpu_postprocess_ms << ','
           << metrics.gpu_total_ms << ',' << metrics.num_voxels << '\n';
  }
}

}  // namespace autoware::ptv3

int main(int argc, char ** argv)
{
  using autoware::ptv3::BenchmarkOptions;
  using autoware::ptv3::PTv3BenchmarkMetrics;
  using autoware::ptv3::PTv3Config;
  using autoware::ptv3::PTv3TRT;
  using autoware::ptv3::loadConfig;
  using autoware::ptv3::loadPointCloudFromBag;
  using autoware::ptv3::loadTrtConfig;
  using autoware::ptv3::parseArguments;
  using autoware::ptv3::printSummaryLine;
  using autoware::ptv3::resolveBagPath;
  using autoware::ptv3::summarize;
  using autoware::ptv3::writeCsv;

  try {
    const BenchmarkOptions options = parseArguments(argc, argv);

    int rclcpp_argc = 0;
    rclcpp::init(rclcpp_argc, nullptr);

    rclcpp::NodeOptions node_options;
    node_options.arguments({"--ros-args", "--params-file", options.params_file});
    node_options.parameter_overrides({
      rclcpp::Parameter("plugins_path", options.plugins_path),
      rclcpp::Parameter("trt_precision", options.trt_precision),
      rclcpp::Parameter("cloud_capacity", options.cloud_capacity),
      rclcpp::Parameter("onnx_path", options.model_path + "/ptv3.onnx"),
      rclcpp::Parameter("engine_path", options.model_path + "/ptv3.engine"),
      rclcpp::Parameter(
        "filter.class_probability_threshold", options.filter_class_probability_threshold),
      rclcpp::Parameter("filter.classes", options.filter_classes),
      rclcpp::Parameter("filter.output_format", options.filter_output_format),
    });
    const auto node = std::make_shared<rclcpp::Node>("ptv3_benchmark", node_options);

    const PTv3Config config = loadConfig(node);
    auto trt_config = loadTrtConfig(node);
    PTv3TRT model(trt_config, config);

    std::uint64_t segmented_publish_count = 0;
    std::uint64_t visualization_publish_count = 0;
    std::uint64_t filtered_publish_count = 0;

    model.setPublishSegmentedPointcloud(
      [&segmented_publish_count](std::unique_ptr<const cuda_blackboard::CudaPointCloud2>) {
        ++segmented_publish_count;
      });
    model.setPublishVisualizationPointcloud(
      [&visualization_publish_count](std::unique_ptr<const cuda_blackboard::CudaPointCloud2>) {
        ++visualization_publish_count;
      });
    model.setPublishFilteredPointcloud(
      [&filtered_publish_count](std::unique_ptr<const cuda_blackboard::CudaPointCloud2>) {
        ++filtered_publish_count;
      });

    const auto bag_path = resolveBagPath(options.bag_path);
    const auto pointcloud = loadPointCloudFromBag(bag_path);
    const auto cuda_pointcloud =
      std::make_shared<const cuda_blackboard::CudaPointCloud2>(pointcloud);

    std::cout << "Loaded pointcloud with " << (pointcloud.width * pointcloud.height) << " points"
              << " from " << bag_path << '\n';
    std::cout << "Warmup iterations: " << options.warmup_iterations
              << ", measured iterations: " << options.measured_iterations << '\n';
    std::cout << "Outputs: segmented=" << std::boolalpha << options.publish_segmented
              << " visualization=" << options.publish_visualization
              << " filtered=" << options.publish_filtered << '\n';

    PTv3BenchmarkMetrics metrics;
    for (int i = 0; i < options.warmup_iterations; ++i) {
      if (!model.benchmarkSegment(
            cuda_pointcloud, options.publish_segmented, options.publish_visualization,
            options.publish_filtered, metrics)) {
        throw std::runtime_error("Warmup iteration failed.");
      }
    }

    segmented_publish_count = 0;
    visualization_publish_count = 0;
    filtered_publish_count = 0;

    std::vector<PTv3BenchmarkMetrics> metrics_per_iteration;
    metrics_per_iteration.reserve(static_cast<size_t>(options.measured_iterations));

    if (options.capture_nsys) {
      cudaDeviceSynchronize();
      cudaProfilerStart();
    }

    for (int i = 0; i < options.measured_iterations; ++i) {
      if (!model.benchmarkSegment(
            cuda_pointcloud, options.publish_segmented, options.publish_visualization,
            options.publish_filtered, metrics)) {
        throw std::runtime_error("Measured iteration failed.");
      }
      metrics_per_iteration.push_back(metrics);
    }

    if (options.capture_nsys) {
      cudaDeviceSynchronize();
      cudaProfilerStop();
    }

    writeCsv(options.csv_output_path, metrics_per_iteration);

    std::vector<double> cpu_preprocess_ms;
    std::vector<double> cpu_inference_enqueue_ms;
    std::vector<double> cpu_postprocess_ms;
    std::vector<double> cpu_total_ms;
    std::vector<double> gpu_inference_ms;
    std::vector<double> gpu_postprocess_ms;
    std::vector<double> gpu_total_ms;
    std::vector<double> num_voxels;

    cpu_preprocess_ms.reserve(metrics_per_iteration.size());
    cpu_inference_enqueue_ms.reserve(metrics_per_iteration.size());
    cpu_postprocess_ms.reserve(metrics_per_iteration.size());
    cpu_total_ms.reserve(metrics_per_iteration.size());
    gpu_inference_ms.reserve(metrics_per_iteration.size());
    gpu_postprocess_ms.reserve(metrics_per_iteration.size());
    gpu_total_ms.reserve(metrics_per_iteration.size());
    num_voxels.reserve(metrics_per_iteration.size());

    for (const auto & iteration_metrics : metrics_per_iteration) {
      cpu_preprocess_ms.push_back(iteration_metrics.cpu_preprocess_ms);
      cpu_inference_enqueue_ms.push_back(iteration_metrics.cpu_inference_enqueue_ms);
      cpu_postprocess_ms.push_back(iteration_metrics.cpu_postprocess_ms);
      cpu_total_ms.push_back(iteration_metrics.cpu_total_ms);
      gpu_inference_ms.push_back(iteration_metrics.gpu_inference_ms);
      gpu_postprocess_ms.push_back(iteration_metrics.gpu_postprocess_ms);
      gpu_total_ms.push_back(iteration_metrics.gpu_total_ms);
      num_voxels.push_back(static_cast<double>(iteration_metrics.num_voxels));
    }

    std::cout << '\n' << "Summary (ms, except num_voxels)" << '\n';
    printSummaryLine("cpu_preprocess", summarize(cpu_preprocess_ms));
    printSummaryLine("cpu_inference", summarize(cpu_inference_enqueue_ms));
    printSummaryLine("cpu_postprocess", summarize(cpu_postprocess_ms));
    printSummaryLine("cpu_total", summarize(cpu_total_ms));
    printSummaryLine("gpu_inference", summarize(gpu_inference_ms));
    printSummaryLine("gpu_postprocess", summarize(gpu_postprocess_ms));
    printSummaryLine("gpu_total", summarize(gpu_total_ms));
    printSummaryLine("num_voxels", summarize(num_voxels));

    std::cout << '\n'
              << "Published messages during measured phase:"
              << " segmented=" << segmented_publish_count
              << " visualization=" << visualization_publish_count
              << " filtered=" << filtered_publish_count << '\n';

    if (!options.csv_output_path.empty()) {
      std::cout << "Wrote CSV metrics to " << options.csv_output_path << '\n';
    }

    rclcpp::shutdown();
    return 0;
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return 1;
  }
}
