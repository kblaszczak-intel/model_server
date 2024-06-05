//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <spdlog/spdlog.h>

#include <rapidjson/document.h>

namespace ovms {
struct HttpPayload {
    public:
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;                 // always
    rapidjson::Document* parsedJson;  // pre-parsed body             = null
};
}  // namespace ovms
