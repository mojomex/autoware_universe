// Copyright 2025 TIER IV, Inc.
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

// cSpell:ignore Indice INDICE Matmul indice
#include "autoware/tensorrt_plugins/indice_conv_plugin.hpp"

#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <nvtx3/nvtx3.hpp>

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>  // cSpell:ignore spconvlib
#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
#include <spconvlib/spconv/csrc/sparse/convops/gemmops/GemmTunerSimple.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvinfer1::plugin
{
namespace
{

class AsyncDeviceBuffer
{
public:
  AsyncDeviceBuffer(std::size_t bytes, cudaStream_t stream) : stream_{stream}
  {
    if (bytes == 0) {
      return;
    }
    status_ = cudaMallocAsync(&ptr_, bytes, stream_);
    if (status_ == cudaSuccess) {
      status_ = cudaMemsetAsync(ptr_, 0, bytes, stream_);
    }
  }

  ~AsyncDeviceBuffer()
  {
    if (ptr_) {
      static_cast<void>(cudaFreeAsync(ptr_, stream_));
    }
  }

  AsyncDeviceBuffer(AsyncDeviceBuffer const &) = delete;
  AsyncDeviceBuffer & operator=(AsyncDeviceBuffer const &) = delete;

  void * get() const { return ptr_; }
  cudaError_t status() const { return status_; }

private:
  void * ptr_{nullptr};
  cudaStream_t stream_{nullptr};
  cudaError_t status_{cudaSuccess};
};

std::size_t tensorBytes(std::initializer_list<std::int64_t> dims, tv::DType dtype)
{
  std::size_t elements = 1;
  for (const auto dim : dims) {
    elements *= static_cast<std::size_t>(std::max<std::int64_t>(dim, 1));
  }
  return elements * tv::detail::sizeof_dtype(dtype);
}

}  // namespace

IndiceConvPlugin::IndiceConvPlugin(const std::string & name, IndiceConvParameters const & params)
: layer_name_{name}, params_{params}
{
  using ConvGemmOps = spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;
  using GemmMain = spconvlib::cumm::gemm::main::GemmMainUnitTest;

  initFieldsToSerialize();

  arch_ = ConvGemmOps::get_compute_capability();
  tuner_fp16_ptr_ =
    std::make_unique<GemmTunerSimple>(GemmMain::get_all_algo_desp());  // cSpell:ignore desp
  tuner_fp32_ptr_ = std::make_unique<GemmTunerSimple>(GemmMain::get_all_algo_desp());
}

void IndiceConvPlugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();
  data_to_serialize_.emplace_back(
    "is_subm", &params_.is_subm, PluginFieldType::kINT32, 1);  // cSpell:ignore subm

  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

IPluginCapability * IndiceConvPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
  try {
    if (type == PluginCapabilityType::kBUILD) {
      return static_cast<IPluginV3OneBuild *>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME) {
      return static_cast<IPluginV3OneRuntime *>(this);
    }
    PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore *>(this);
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV3 * IndiceConvPlugin::clone() noexcept
{
  try {
    IPluginV3 * const plugin{new IndiceConvPlugin{layer_name_, params_}};
    return plugin;
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * IndiceConvPlugin::getPluginName() const noexcept
{
  return kINDICE_CONV_PLUGIN_NAME;
}

char const * IndiceConvPlugin::getPluginVersion() const noexcept
{
  return kINDICE_CONV_PLUGIN_VERSION;
}

char const * IndiceConvPlugin::getPluginNamespace() const noexcept
{
  return kINDICE_CONV_PLUGIN_NAMESPACE;
}

std::int32_t IndiceConvPlugin::getNbOutputs() const noexcept
{
  return 1;
}

std::int32_t IndiceConvPlugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs, DynamicPluginTensorDesc const * out,
  std::int32_t num_outputs) noexcept
{
  // Validate input arguments.
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(in[INOUT_IN_FEATURES_INDEX].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(in[INOUT_FILTERS_INDEX].desc.dims.nbDims == 5);
  PLUGIN_ASSERT(in[INOUT_INDICE_PAIRS_INDEX].desc.dims.nbDims == 3);
  PLUGIN_ASSERT(in[INOUT_INDICE_PAIRS_NUM_INDEX].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(in[INOUT_NUM_ACTIVATE_OUT_INDEX].desc.dims.nbDims == 0);
  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(
    in[INOUT_FILTERS_INDEX].desc.dims.d[4] == in[INOUT_IN_FEATURES_INDEX].desc.dims.d[1]);

  PLUGIN_ASSERT(in[INOUT_INDICE_PAIRS_INDEX].desc.dims.d[0] == 2);

  PLUGIN_ASSERT(in[INOUT_IN_FEATURES_INDEX].desc.type == in[INOUT_FILTERS_INDEX].desc.type);
  PLUGIN_ASSERT(in[INOUT_IN_FEATURES_INDEX].desc.type == out[0].desc.type);
  PLUGIN_ASSERT(
    in[INOUT_INDICE_PAIRS_INDEX].desc.type == in[INOUT_INDICE_PAIRS_NUM_INDEX].desc.type);
  return 0;
}

bool IndiceConvPlugin::supportsFormatCombination(
  std::int32_t pos, DynamicPluginTensorDesc const * in_out, std::int32_t num_inputs,
  std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);

  bool supported = in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;

  switch (pos) {
    case INOUT_IN_FEATURES_INDEX:
      supported &=
        (in_out[pos].desc.type == nvinfer1::DataType::kFLOAT ||
         in_out[pos].desc.type == nvinfer1::DataType::kHALF);
      break;
    case INOUT_FILTERS_INDEX:
    case INOUT_OUT_FEATURES_INDEX:
      supported &= in_out[pos].desc.type == in_out[INOUT_IN_FEATURES_INDEX].desc.type;
      break;
    case INOUT_INDICE_PAIRS_INDEX:
    case INOUT_INDICE_PAIRS_NUM_INDEX:
    case INOUT_NUM_ACTIVATE_OUT_INDEX:
      supported &= in_out[pos].desc.type == nvinfer1::DataType::kINT32;
      break;
    default:
      supported = false;
      break;
  }

  return supported;
}

std::int32_t IndiceConvPlugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
  std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);

  output_types[0] = input_types[INOUT_IN_FEATURES_INDEX];

  return 0;
}

std::int32_t IndiceConvPlugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs, [[maybe_unused]] std::int32_t num_shape_inputs,
  DimsExprs * outputs, std::int32_t num_outputs,
  [[maybe_unused]] IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(inputs[0].nbDims == 2);

  outputs[0].nbDims = 2;
  outputs[0].d[0] = inputs[INOUT_INDICE_PAIRS_INDEX].d[2];
  outputs[0].d[1] = inputs[INOUT_FILTERS_INDEX].d[0];

  return 0;
}

std::int32_t IndiceConvPlugin::enqueue(
  PluginTensorDesc const * input_desc, [[maybe_unused]] PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, [[maybe_unused]] void * workspace,
  cudaStream_t stream) noexcept
{
  using StaticAllocator = spconvlib::spconv::csrc::sparse::alloc::StaticAllocator;
  using ConvGemmOps = spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;
  using SimpleExternalSpconvMatmul =
    spconvlib::spconv::csrc::sparse::convops::SimpleExternalSpconvMatmul;

  std::int64_t num_act_in = input_desc[INOUT_IN_FEATURES_INDEX].dims.d[0];
  std::int64_t num_in_features = input_desc[INOUT_IN_FEATURES_INDEX].dims.d[1];
  std::int64_t num_act_out = input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[2];
  std::int64_t num_out_features = input_desc[INOUT_FILTERS_INDEX].dims.d[0];

  auto in_features_type = input_desc[INOUT_IN_FEATURES_INDEX].type;
  [[maybe_unused]] auto filters_type = input_desc[INOUT_FILTERS_INDEX].type;
  [[maybe_unused]] auto out_features_type = input_desc[INOUT_OUT_FEATURES_INDEX].type;

  assert(in_features_type == filters_type);
  assert(in_features_type == out_features_type);

  auto dtype = in_features_type == DataType::kFLOAT ? tv::float32 : tv::float16;

  tv::Tensor input_features =
    tv::from_blob(inputs[INOUT_IN_FEATURES_INDEX], {num_act_in, num_in_features}, dtype, 0);

  tv::Tensor input_features_fp32 =
    tv::from_blob(inputs[INOUT_IN_FEATURES_INDEX], {num_act_in, num_in_features}, tv::float32, 0);

  tv::Tensor weights = tv::from_blob(
    inputs[INOUT_FILTERS_INDEX],
    {input_desc[INOUT_FILTERS_INDEX].dims.d[0], input_desc[INOUT_FILTERS_INDEX].dims.d[1],
     input_desc[INOUT_FILTERS_INDEX].dims.d[2], input_desc[INOUT_FILTERS_INDEX].dims.d[3],
     input_desc[INOUT_FILTERS_INDEX].dims.d[4]},
    dtype, 0);

  tv::Tensor pairs = tv::from_blob(
    inputs[INOUT_INDICE_PAIRS_INDEX],
    {input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[0], input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[1],
     input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[2]},
    tv::int32, 0);

  tv::Tensor pairs_num = tv::from_blob(
    inputs[INOUT_INDICE_PAIRS_NUM_INDEX], {input_desc[INOUT_INDICE_PAIRS_NUM_INDEX].dims.d[0]},
    tv::int32, 0);

  tv::Tensor out_features = tv::from_blob(outputs[0], {num_act_out, num_out_features}, dtype, 0);

  auto & tuner_ptr = dtype == tv::float32 ? tuner_fp32_ptr_ : tuner_fp16_ptr_;

  std::unordered_map<std::string, tv::Tensor> tensor_dict{
    {SPCONV_ALLOC_FEATURES, input_features},
    {SPCONV_ALLOC_FILTERS, weights},
    {SPCONV_ALLOC_OUT_FEATURES, out_features}};
  StaticAllocator alloc2(tensor_dict);

  SimpleExternalSpconvMatmul ext_mm(alloc2);

  {
    nvtx3::scoped_range nvtx_spconv{"ConvGemmOps::indice_conv"};
    ConvGemmOps::indice_conv(
      alloc2, ext_mm, *tuner_ptr, true, false, input_features, weights, pairs, pairs_num, arch_,
      out_features.dim(0), false, params_.is_subm,
      static_cast<int>(tv::gemm::SparseConvAlgo::kNative), reinterpret_cast<std::uintptr_t>(stream),
      tv::Tensor(), 0.f, 0.f, tv::gemm::Activation::kNone, false);
  }

  return 0;
}

std::int32_t IndiceConvPlugin::prewarmTuner(
  PluginTensorDesc const * input_desc, PluginTensorDesc const * output_desc, cudaStream_t stream)
{
  constexpr int kShuffleAC = static_cast<int>(tv::gemm::ShuffleStrideType::kShuffleAC);

  const auto in_features_type = input_desc[INOUT_IN_FEATURES_INDEX].type;
  const auto dtype = in_features_type == DataType::kFLOAT ? tv::float32 : tv::float16;
  auto & tuner_ptr = dtype == tv::float32 ? tuner_fp32_ptr_ : tuner_fp16_ptr_;

  const std::int64_t num_act_in =
    std::max<std::int64_t>(input_desc[INOUT_IN_FEATURES_INDEX].dims.d[0], 1);
  const std::int64_t num_in_features = input_desc[INOUT_IN_FEATURES_INDEX].dims.d[1];
  const std::int64_t num_act_out = std::max<std::int64_t>(output_desc[0].dims.d[0], 1);
  const std::int64_t num_out_features = input_desc[INOUT_FILTERS_INDEX].dims.d[0];
  const std::int64_t nhot = std::max<std::int64_t>(
    1, std::min<std::int64_t>(input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[2], num_act_out));

  const std::vector<std::int64_t> input_shape{num_act_in, num_in_features};
  const std::vector<std::int64_t> filter_shape{num_out_features, num_in_features};
  const std::vector<std::int64_t> output_shape{num_act_out, num_out_features};
  const std::vector<std::int64_t> indices_shape{nhot};

  auto tuned_res_exist = tuner_ptr->get_tuned_algo(
    static_cast<int>(dtype), static_cast<int>(dtype), static_cast<int>(dtype), input_shape,
    filter_shape, output_shape, false, true, false, arch_, kShuffleAC, indices_shape, {},
    indices_shape, 1);
  if (std::get<1>(tuned_res_exist)) {
    return 0;
  }

  AsyncDeviceBuffer input_features_storage(
    tensorBytes({num_act_in, num_in_features}, dtype), stream);
  AsyncDeviceBuffer weights_storage(
    tensorBytes({num_out_features, num_in_features}, dtype), stream);
  AsyncDeviceBuffer out_features_storage(
    tensorBytes({num_act_out, num_out_features}, dtype), stream);
  AsyncDeviceBuffer input_indices_storage(tensorBytes({nhot}, tv::int32), stream);
  AsyncDeviceBuffer output_indices_storage(tensorBytes({nhot}, tv::int32), stream);

  const cudaError_t status = cudaGetLastError();
  if (
    input_features_storage.status() != cudaSuccess || weights_storage.status() != cudaSuccess ||
    out_features_storage.status() != cudaSuccess || input_indices_storage.status() != cudaSuccess ||
    output_indices_storage.status() != cudaSuccess || status != cudaSuccess) {
    return status == cudaSuccess ? cudaErrorMemoryAllocation : status;
  }

  tv::Tensor input_features =
    tv::from_blob(input_features_storage.get(), {num_act_in, num_in_features}, dtype, 0);
  tv::Tensor weights =
    tv::from_blob(weights_storage.get(), {num_out_features, num_in_features}, dtype, 0);
  tv::Tensor out_features =
    tv::from_blob(out_features_storage.get(), {num_act_out, num_out_features}, dtype, 0);
  tv::Tensor input_indices = tv::from_blob(input_indices_storage.get(), {nhot}, tv::int32, 0);
  tv::Tensor output_indices = tv::from_blob(output_indices_storage.get(), {nhot}, tv::int32, 0);

  static_cast<void>(tuner_ptr->tune_and_cache(
    input_features, weights, out_features, false, true, false, arch_, kShuffleAC, input_indices,
    tv::Tensor(), output_indices, 1, 1.0, 0.0, reinterpret_cast<std::uintptr_t>(stream), 5, false));

  return cudaGetLastError();
}

std::int32_t IndiceConvPlugin::onShapeChange(
  PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  PluginTensorDesc const * out, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  try {
    PLUGIN_ASSERT(num_inputs == 5);
    PLUGIN_ASSERT(num_outputs == 1);
    return prewarmTuner(in, out, nullptr);
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return -1;
}

IPluginV3 * IndiceConvPlugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * IndiceConvPlugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t IndiceConvPlugin::getWorkspaceSize(
  [[maybe_unused]] DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  return 0;
}

}  // namespace nvinfer1::plugin
