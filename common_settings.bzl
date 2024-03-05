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
###############################
# bazel config settings
###############################
#To build without mediapipe use flags - bazel build --define MEDIAPIPE_DISABLE=1 --cxxopt=-DMEDIAPIPE_DISABLE=1 //src:ovms
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@mediapipe//mediapipe/framework:more_selects.bzl", "more_selects")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("//:distro.bzl", "distro_flag")
def create_config_settings():
    distro_flag()
    native.config_setting(
        name = "disable_mediapipe",
        define_values = {
            "MEDIAPIPE_DISABLE": "1",
        },
        visibility = ["//visibility:public"],
    )
    more_selects.config_setting_negation(
        name = "not_disable_mediapipe",
        negate = ":disable_mediapipe",
    )
    #To build without python use flags - bazel build --define PYTHON_DISABLE=1 //src:ovms
    native.config_setting(
        name = "disable_python",
        define_values = {
            "PYTHON_DISABLE": "1",
        },
        visibility = ["//visibility:public"],
    )
    more_selects.config_setting_negation(
        name = "not_disable_python",
        negate = ":disable_python",
    )
    native.config_setting(
        name = "fuzzer_build",
        define_values = {
            "FUZZER_BUILD": "1",
        },
        visibility = ["//visibility:public"],
    )
    more_selects.config_setting_negation(
        name = "not_fuzzer_build",
        negate = ":fuzzer_build",
    )
###############################
# compilation settings
###############################
def ovms_copts():
    return COMMON_STATIC_LIBS_COPTS + select({
        "//conditions:default": [],
        "//:fuzzer_build" : COMMON_FUZZER_COPTS,
    }) + select({
        "//conditions:default": ["-DPYTHON_DISABLE=1"],
        "//:not_disable_python" : ["-DPYTHON_DISABLE=0"],
    }) + select({
        "//conditions:default": ["-DMEDIAPIPE_DISABLE=1"],
        "//:not_disable_mediapipe" : ["-DMEDIAPIPE_DISABLE=0"],
    })

def ovms_test_copts():
    return [
        "-Wall",
        "-Wno-unknown-pragmas",
        "-Werror",
        "-Isrc",
        "-fconcepts", # for gmock related utils
        "-fvisibility=hidden",# Needed for pybind targets
    ] + select({
            "//conditions:default": ["-DPYTHON_DISABLE=1"],
            "//:not_disable_python" : ["-DPYTHON_DISABLE=0"],
    }) + select({
            "//conditions:default": ["-DMEDIAPIPE_DISABLE=1"],
            "//:not_disable_mediapipe" : ["-DMEDIAPIPE_DISABLE=0"],
    })

def ovms_linkopts():
    return COMMON_STATIC_LIBS_LINKOPTS + select({
        "//conditions:default": [],
        "//:fuzzer_build" : COMMON_FUZZER_LINKOPTS,
    })

COMMON_STATIC_LIBS_COPTS = [
    "-Wall",
#    "-Wextra", a lot of unuseful warnings anable later TODO
    "-Werror", # TODO make this in release build only
    # string formatting
    "-Wformat",
    "-Wformat-security",
    "-Werror=format-security",
    # other
    "-Wno-unknown-pragmas",
    "-Wno-sign-compare",
    "-z noexecstack", # TODO release only
    # PIC & PIE
    "-fPIE",
    "-fPIC",
    "-fno-strict-overflow",
    "-fno-delete-null-pointer-checks",
    "-fwrapv",
    "-fstack-protector",
    "-fstack-clash-protection",
    "-fvisibility=hidden", # Needed for pybind targets # TODO make it only in release
    "-fcf-protection=full", #incompatible with minirect-branch
    #"-D_FORTIFY_SOURCE=2", # bazel by default = 1 and we have error bc of redefinition TODO
    # following are incompatible with fcf-protection=full
    #"-mfunction-return=thunk",
    #"-mindirect-branch=thunk",
    #"-mindirect-branch-register",

]
COMMON_STATIC_LIBS_LINKOPTS = [
    # Read-only relocation, Stack and Heap Overlap Protection
    "-Wl,-z,relro,-z,now",
    "-pie",
    "-lxml2",
    "-luuid",
    "-lstdc++fs",
    "-lcrypto",
]
COMMON_FUZZER_COPTS = [
    "-fsanitize=address",
    "-fprofile-generate",
    "-ftest-coverage",
]
COMMON_FUZZER_LINKOPTS = [
    "-fprofile-generate",
    "-fsanitize=address",
    "-fsanitize-coverage=trace-pc",
    "-static-libasan",
]
COMMON_LOCAL_DEFINES = ["SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE"]
PYBIND_DEPS = [
    "@python3_linux//:python3-lib",
    "@pybind11//:pybind11_embed",
]
