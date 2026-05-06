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

#include "autoware/tensorrt_plugins/argsort_plugin_creator.hpp"

#include "autoware/tensorrt_plugins//argsort_plugin.hpp"
#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <NvInferRuntimePlugin.h>

#include <cstdint>
#include <string>

namespace nvinfer1::plugin
{

REGISTER_TENSORRT_PLUGIN(ArgsortPluginCreator);

ArgsortPluginCreator::ArgsortPluginCreator()
{
  plugin_attributes_.clear();
  fc_.nbFields = plugin_attributes_.size();
  fc_.fields = plugin_attributes_.data();
}

nvinfer1::PluginFieldCollection const * ArgsortPluginCreator::getFieldNames() noexcept
{
  return &fc_;
}

IPluginV3 * ArgsortPluginCreator::createPlugin(
  char const * name, PluginFieldCollection const * fc, TensorRTPhase phase) noexcept
{
  if (phase != TensorRTPhase::kBUILD && phase != TensorRTPhase::kRUNTIME) {
    return nullptr;
  }

  try {
    std::int64_t max_num_elements = 0;
    for (std::int32_t i = 0; i < fc->nbFields; ++i) {
      const auto & field = fc->fields[i];
      const std::string attr_name = field.name;
      if (attr_name == "max_num_elements") {
        PLUGIN_VALIDATE(field.type == PluginFieldType::kINT64);
        PLUGIN_VALIDATE(field.length == 1);
        max_num_elements = static_cast<std::int64_t const *>(field.data)[0];
      }
    }
    return new (std::nothrow) ArgsortPlugin(std::string(name), max_num_elements);
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

}  // namespace nvinfer1::plugin
