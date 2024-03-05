//***************************************************************************
// Copyright 2022 Intel Corporation
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
//*****************************************************************************
#include "metric_module.hpp"

#include "metric_registry.hpp"
#include "status.hpp"

namespace ovms {

MetricModule::~MetricModule() = default;

MetricModule::MetricModule() :
    registry(std::make_unique<MetricRegistry>()) {}

#define IGNORE_UNUSED_WARNING(A) (void)(A)
Status MetricModule::start(__attribute__ ((unused)) const Config& config) {
        return StatusCode::OK; }

void MetricModule::shutdown() {}

MetricRegistry& MetricModule::getRegistry() const { return *this->registry; }
}  // namespace ovms
