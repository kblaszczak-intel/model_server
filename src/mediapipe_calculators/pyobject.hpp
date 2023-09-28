//*****************************************************************************
// Copyright 2023 Intel Corporation
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

// Declaration common for C++ and Python
// C++ definition part in separate location
// pybind11 binding part also in separate location
#pragma once
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <pybind11/embed.h>  // everything needed for embedding

// Temporary added file
namespace py = pybind11;
class PYOBJECT {
public:
    // Try to create using other Python object that exposes buffer protocol (like numpy array, pytorch tensor etc.)
    PYOBJECT(py::object object);

private:
};
