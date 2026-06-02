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

#include "autoware/tensorrt_plugins/implicit_gemm_plugin.hpp"

#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <nvtx3/nvtx3.hpp>

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>  // cSpell:ignore spconvlib
#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

ImplicitGemmPlugin::ImplicitGemmPlugin(
  const std::string & name, ImplicitGemmParameters const & params)
: layer_name_{name}, params_{params}
{
  using ConvGemmOps = spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;
  using ConvMain = spconvlib::cumm::conv::main::ConvMainUnitTest;

  initFieldsToSerialize();

  arch_ = ConvGemmOps::get_compute_capability();
  tuner_fp16_ptr_ =
    std::make_unique<ConvTunerSimple>(ConvMain::get_all_conv_algo_desp());  // cSpell:ignore desp
  tuner_fp32_ptr_ = std::make_unique<ConvTunerSimple>(ConvMain::get_all_conv_algo_desp());

  // Pre-allocate CPU mask tensor to avoid heap allocation during CUDA graph capture.
  mask_tensor_ = tv::zeros({1}, tv::uint32, -1);
  mask_tensor_.data_ptr<uint32_t>()[0] = 0xffffffff;
}

void ImplicitGemmPlugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();
  data_to_serialize_.emplace_back("act_alpha", &params_.act_alpha, PluginFieldType::kFLOAT32, 1);
  data_to_serialize_.emplace_back("act_alpha", &params_.act_beta, PluginFieldType::kFLOAT32, 1);

  data_to_serialize_.emplace_back(
    "is_subm", &params_.is_subm, PluginFieldType::kINT32, 1);  // cSpell:ignore subm
  data_to_serialize_.emplace_back("is_train", &params_.is_train, PluginFieldType::kINT32, 1);

  data_to_serialize_.emplace_back(
    "output_add_scale", &params_.output_add_scale, PluginFieldType::kFLOAT32, 1);
  data_to_serialize_.emplace_back(
    "output_scale", &params_.output_scale, PluginFieldType::kFLOAT32, 1);

  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

IPluginCapability * ImplicitGemmPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3 * ImplicitGemmPlugin::clone() noexcept
{
  try {
    IPluginV3 * const plugin{new ImplicitGemmPlugin{layer_name_, params_}};
    return plugin;
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * ImplicitGemmPlugin::getPluginName() const noexcept
{
  return kIMPLICIT_GEMM_PLUGIN_NAME;
}

char const * ImplicitGemmPlugin::getPluginVersion() const noexcept
{
  return kIMPLICIT_GEMM_PLUGIN_VERSION;
}

char const * ImplicitGemmPlugin::getPluginNamespace() const noexcept
{
  return kIMPLICIT_GEMM_PLUGIN_NAMESPACE;
}

std::int32_t ImplicitGemmPlugin::getNbOutputs() const noexcept
{
  return 1;
}

std::int32_t ImplicitGemmPlugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs, DynamicPluginTensorDesc const * out,
  std::int32_t num_outputs) noexcept
{
  // Validate input arguments.
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(in[INOUT_IN_FEATURES_INDEX].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(in[INOUT_FILTERS_INDEX].desc.dims.nbDims == 5);
  PLUGIN_ASSERT(in[INOUT_PAIR_FWD_INDEX].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(in[INOUT_PAIR_MASK_FWD_SPLITS_INDEX].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(in[INOUT_MASK_ARGSORT_FWD_SPLITS_INDEX].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(
    in[INOUT_FILTERS_INDEX].desc.dims.d[4] == in[INOUT_IN_FEATURES_INDEX].desc.dims.d[1]);
  PLUGIN_ASSERT(
    in[INOUT_PAIR_MASK_FWD_SPLITS_INDEX].desc.dims.d[0] == in[INOUT_PAIR_FWD_INDEX].desc.dims.d[1]);
  PLUGIN_ASSERT(in[INOUT_PAIR_MASK_FWD_SPLITS_INDEX].desc.dims.d[1] == 1);
  PLUGIN_ASSERT(
    in[INOUT_MASK_ARGSORT_FWD_SPLITS_INDEX].desc.dims.d[0] ==
    in[INOUT_PAIR_FWD_INDEX].desc.dims.d[1]);
  PLUGIN_ASSERT(in[INOUT_IN_FEATURES_INDEX].desc.type == in[INOUT_FILTERS_INDEX].desc.type);
  PLUGIN_ASSERT(in[INOUT_IN_FEATURES_INDEX].desc.type == out[0].desc.type);
  PLUGIN_ASSERT(
    in[INOUT_PAIR_FWD_INDEX].desc.type == in[INOUT_PAIR_MASK_FWD_SPLITS_INDEX].desc.type);
  PLUGIN_ASSERT(
    in[INOUT_PAIR_FWD_INDEX].desc.type == in[INOUT_MASK_ARGSORT_FWD_SPLITS_INDEX].desc.type);

  return 0;
}

bool ImplicitGemmPlugin::supportsFormatCombination(
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
    case INOUT_PAIR_FWD_INDEX:
    case INOUT_PAIR_MASK_FWD_SPLITS_INDEX:
    case INOUT_MASK_ARGSORT_FWD_SPLITS_INDEX:
      supported &= in_out[pos].desc.type == nvinfer1::DataType::kINT32;
      break;
    default:
      supported = false;
      break;
  }

  return supported;
}

std::int32_t ImplicitGemmPlugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
  std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);

  output_types[0] = input_types[INOUT_IN_FEATURES_INDEX];

  return 0;
}

std::int32_t ImplicitGemmPlugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs, [[maybe_unused]] std::int32_t num_shape_inputs,
  DimsExprs * outputs, std::int32_t num_outputs,
  [[maybe_unused]] IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(inputs[0].nbDims == 2);

  outputs[0].nbDims = 2;
  outputs[0].d[0] = inputs[3].d[0];
  outputs[0].d[1] = inputs[1].d[0];

  return 0;
}

std::int32_t ImplicitGemmPlugin::enqueue(
  PluginTensorDesc const * input_desc, [[maybe_unused]] PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, [[maybe_unused]] void * workspace,
  cudaStream_t stream) noexcept
{
  using StaticAllocator = spconvlib::spconv::csrc::sparse::alloc::StaticAllocator;
  using ConvGemmOps = spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;

  std::int64_t num_act_in = input_desc[INOUT_IN_FEATURES_INDEX].dims.d[0];
  std::int64_t num_in_features = input_desc[INOUT_IN_FEATURES_INDEX].dims.d[1];
  // std::int64_t kernel_volume = input_desc[INOUT_PAIR_FWD_INDEX].dims.d[0];
  std::int64_t num_act_out = input_desc[INOUT_PAIR_FWD_INDEX].dims.d[1];
  std::int64_t num_out_features = input_desc[INOUT_FILTERS_INDEX].dims.d[0];

  auto in_features_type = input_desc[INOUT_IN_FEATURES_INDEX].type;
  [[maybe_unused]] auto filters_type = input_desc[INOUT_FILTERS_INDEX].type;
  [[maybe_unused]] auto out_features_type = input_desc[INOUT_OUT_FEATURES_INDEX].type;

  assert(in_features_type == filters_type);
  assert(in_features_type == out_features_type);

  auto dtype = in_features_type == DataType::kFLOAT ? tv::float32 : tv::float16;

  tv::Tensor input_features =
    tv::from_blob(inputs[INOUT_IN_FEATURES_INDEX], {num_act_in, num_in_features}, dtype, 0);

  tv::Tensor weights = tv::from_blob(
    inputs[INOUT_FILTERS_INDEX],
    {input_desc[INOUT_FILTERS_INDEX].dims.d[0], input_desc[INOUT_FILTERS_INDEX].dims.d[1],
     input_desc[INOUT_FILTERS_INDEX].dims.d[2], input_desc[INOUT_FILTERS_INDEX].dims.d[3],
     input_desc[INOUT_FILTERS_INDEX].dims.d[4]},
    dtype, 0);

  tv::Tensor pair_fwd = tv::from_blob(
    inputs[INOUT_PAIR_FWD_INDEX],
    {input_desc[INOUT_PAIR_FWD_INDEX].dims.d[0], input_desc[INOUT_PAIR_FWD_INDEX].dims.d[1]},
    tv::int32, 0);

  tv::Tensor pair_mask_fwd_splits = tv::from_blob(
    inputs[INOUT_PAIR_MASK_FWD_SPLITS_INDEX],
    {1, input_desc[INOUT_PAIR_MASK_FWD_SPLITS_INDEX].dims.d[0]}, tv::int32, 0);

  tv::Tensor mask_argsort_fwd_splits = tv::from_blob(
    inputs[INOUT_MASK_ARGSORT_FWD_SPLITS_INDEX],
    {
      1,
      input_desc[INOUT_MASK_ARGSORT_FWD_SPLITS_INDEX].dims.d[0],
    },
    tv::int32, 0);

  PLUGIN_ASSERT(
    input_desc[INOUT_PAIR_FWD_INDEX].dims.d[1] ==
    input_desc[INOUT_MASK_ARGSORT_FWD_SPLITS_INDEX].dims.d[0]);
  PLUGIN_ASSERT(input_desc[INOUT_MASK_ARGSORT_FWD_SPLITS_INDEX].dims.nbDims == 1);

  tv::Tensor out_features = tv::from_blob(outputs[0], {num_act_out, num_out_features}, dtype, 0);

  std::vector<tv::Tensor> pair_mask_splits;
  std::vector<tv::Tensor> mask_argsort_splits;

  pair_mask_splits.push_back(pair_mask_fwd_splits);
  mask_argsort_splits.push_back(mask_argsort_fwd_splits);

  std::unordered_map<std::string, tv::Tensor> tensor_dict{
    {SPCONV_ALLOC_FEATURES, input_features},
    {SPCONV_ALLOC_FILTERS, weights},
    {SPCONV_ALLOC_OUT_FEATURES, out_features}};
  StaticAllocator alloc2(tensor_dict);

  auto & tuner_ptr = dtype == tv::float32 ? tuner_fp32_ptr_ : tuner_fp16_ptr_;

  auto conv_run_status = [&]() {
    nvtx3::scoped_range nvtx_spconv{"ConvGemmOps::implicit_gemm"};
    return ConvGemmOps::implicit_gemm(
      alloc2, *tuner_ptr, input_features, weights, pair_fwd, pair_mask_splits, mask_argsort_splits,
      num_act_out, mask_tensor_, arch_, false, params_.is_subm,
      reinterpret_cast<std::uintptr_t>(stream), tv::CUDAKernelTimer(false), true, false,
      tv::Tensor(), 0.0, 0.0, tv::gemm::Activation::kNone, false, 1.0, tv::Tensor(), tv::Tensor(),
      0.0, -1);
  }();

  return 0;
}

std::int32_t ImplicitGemmPlugin::prewarmTuner(
  PluginTensorDesc const * input_desc, cudaStream_t stream)
{
  constexpr auto kForwardInt = static_cast<int>(tv::gemm::ConvOpType::kForward);
  constexpr auto kChannelLastInt = static_cast<int>(tv::gemm::ConvLayoutType::kChannelLast);

  const auto in_features_type = input_desc[INOUT_IN_FEATURES_INDEX].type;
  const auto dtype = in_features_type == DataType::kFLOAT ? tv::float32 : tv::float16;
  auto & tuner_ptr = dtype == tv::float32 ? tuner_fp32_ptr_ : tuner_fp16_ptr_;

  const std::int64_t num_act_in =
    std::max<std::int64_t>(input_desc[INOUT_IN_FEATURES_INDEX].dims.d[0], 1);
  const std::int64_t num_in_features = input_desc[INOUT_IN_FEATURES_INDEX].dims.d[1];
  const std::int64_t num_act_out =
    std::max<std::int64_t>(input_desc[INOUT_PAIR_FWD_INDEX].dims.d[1], 1);
  const std::int64_t kernel_volume = input_desc[INOUT_PAIR_FWD_INDEX].dims.d[0];
  const std::int64_t num_out_features = input_desc[INOUT_FILTERS_INDEX].dims.d[0];
  const bool need_dynamic_mask = kernel_volume > 32;

  auto tuned_res_exist = tuner_ptr->get_tuned_algo(
    kForwardInt, static_cast<int>(dtype), static_cast<int>(dtype), static_cast<int>(dtype),
    static_cast<int>(num_out_features), static_cast<int>(num_in_features), arch_, -1,
    need_dynamic_mask);
  if (std::get<1>(tuned_res_exist)) {
    return 0;
  }

  AsyncDeviceBuffer input_features_storage(
    tensorBytes({num_act_in, num_in_features}, dtype), stream);
  AsyncDeviceBuffer weights_storage(
    tensorBytes(
      {input_desc[INOUT_FILTERS_INDEX].dims.d[0], input_desc[INOUT_FILTERS_INDEX].dims.d[1],
       input_desc[INOUT_FILTERS_INDEX].dims.d[2], input_desc[INOUT_FILTERS_INDEX].dims.d[3],
       input_desc[INOUT_FILTERS_INDEX].dims.d[4]},
      dtype),
    stream);
  AsyncDeviceBuffer out_features_storage(
    tensorBytes({num_act_out, num_out_features}, dtype), stream);
  AsyncDeviceBuffer pair_fwd_storage(tensorBytes({kernel_volume, num_act_out}, tv::int32), stream);
  AsyncDeviceBuffer pair_mask_storage(tensorBytes({1, num_act_out}, tv::int32), stream);
  AsyncDeviceBuffer mask_argsort_storage(tensorBytes({num_act_out}, tv::int32), stream);

  const cudaError_t status = cudaGetLastError();
  if (
    input_features_storage.status() != cudaSuccess || weights_storage.status() != cudaSuccess ||
    out_features_storage.status() != cudaSuccess || pair_fwd_storage.status() != cudaSuccess ||
    pair_mask_storage.status() != cudaSuccess || mask_argsort_storage.status() != cudaSuccess ||
    status != cudaSuccess) {
    return status == cudaSuccess ? cudaErrorMemoryAllocation : status;
  }

  tv::Tensor input_features =
    tv::from_blob(input_features_storage.get(), {num_act_in, num_in_features}, dtype, 0);
  tv::Tensor weights = tv::from_blob(
    weights_storage.get(),
    {input_desc[INOUT_FILTERS_INDEX].dims.d[0], input_desc[INOUT_FILTERS_INDEX].dims.d[1],
     input_desc[INOUT_FILTERS_INDEX].dims.d[2], input_desc[INOUT_FILTERS_INDEX].dims.d[3],
     input_desc[INOUT_FILTERS_INDEX].dims.d[4]},
    dtype, 0);
  tv::Tensor out_features =
    tv::from_blob(out_features_storage.get(), {num_act_out, num_out_features}, dtype, 0);
  tv::Tensor pair_fwd =
    tv::from_blob(pair_fwd_storage.get(), {kernel_volume, num_act_out}, tv::int32, 0);
  tv::Tensor pair_mask = tv::from_blob(pair_mask_storage.get(), {1, num_act_out}, tv::int32, 0);
  tv::Tensor mask_argsort = tv::from_blob(mask_argsort_storage.get(), {num_act_out}, tv::int32, 0);

  static_cast<void>(tuner_ptr->tune_and_cache(
    kForwardInt, input_features, weights, out_features, kChannelLastInt, kChannelLastInt,
    kChannelLastInt, 1, 1, 1, arch_, pair_mask.type_view(tv::uint32), mask_argsort, pair_fwd, false,
    0xffffffff, -1, tv::Tensor(), 1.0, 0.0, reinterpret_cast<std::uintptr_t>(stream), true, false,
    5, false, tv::Tensor(), tv::Tensor()));

  return cudaGetLastError();
}

std::int32_t ImplicitGemmPlugin::onShapeChange(
  PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] PluginTensorDesc const * out, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  try {
    PLUGIN_ASSERT(num_inputs == 5);
    return prewarmTuner(in, nullptr);
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return -1;
}

IPluginV3 * ImplicitGemmPlugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * ImplicitGemmPlugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t ImplicitGemmPlugin::getWorkspaceSize(
  [[maybe_unused]] DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  return 0;
}

}  // namespace nvinfer1::plugin
