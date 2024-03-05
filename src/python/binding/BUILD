#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")
load("//:common_settings.bzl",
     "COMMON_LOCAL_DEFINES", "create_config_settings", "ovms_copts", "ovms_linkopts", "ovms_test_copts", "PYBIND_DEPS")
load("@pip_deps//:requirements.bzl", "all_requirements")

pybind_extension(
    name = "pyovms",
    srcs = ["ovms_py_tensor.cpp",], # TODO consider naming convention if exist
    deps = ["//src/python:ovmspytensor",],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = ovms_copts(),
    linkopts = ovms_copts(),
)

py_test(
    name = "test_python_binding",
    srcs = glob(["tests/*.py"]),
    main = "tests/run.py",
    imports=["."],
    data = [":pyovms.so"],
    deps = all_requirements
)

