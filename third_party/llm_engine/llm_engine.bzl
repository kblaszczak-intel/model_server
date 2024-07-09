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

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
def llm_engine():
    llm_engine_repository(name="_llm_engine")
    new_git_repository(
        name = "llm_engine",
        remote = "https://github.com/dkalinowski/openvino.genai",
        #branch = "simple_metrics",
        # parallel tokenizer, 32->256 queue, plugin config, trace perf
        commit = "3f0e214bb2afa22fbe87a8243d0c889645106b0c", 
        build_file = "@_llm_engine//:BUILD",
        init_submodules = True,
        recursive_init_submodules = True,
    )
    # when using local repository manually run: git submodule update --recursive 
    #native.new_local_repository(
    #    name = "llm_engine",
    #    path = "/openvino.genai",
    #    build_file = "@_llm_engine//:BUILD",
    #)

def _impl(repository_ctx):
    http_proxy = repository_ctx.os.environ.get("http_proxy", "")
    https_proxy = repository_ctx.os.environ.get("https_proxy", "")
    OpenVINO_DIR = repository_ctx.os.environ.get("OpenVINO_DIR", "")
    result = repository_ctx.execute(["cat","/etc/os-release"],quiet=False)
    ubuntu20_count = result.stdout.count("PRETTY_NAME=\"Ubuntu 20")
    ubuntu22_count = result.stdout.count("PRETTY_NAME=\"Ubuntu 22")

    if ubuntu20_count == 1 or ubuntu22_count == 1:
        lib_path = "lib"
    else: # for redhat
        lib_path = "lib64"

    # Note we need to escape '{/}' by doubling them due to call to format
    build_file_content = """
load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
visibility = ["//visibility:public"]

config_setting(
    name = "dbg",
    values = {{"compilation_mode": "dbg"}},
)

config_setting(
    name = "opt",
    values = {{"compilation_mode": "opt"}},
)

filegroup(
    name = "all_srcs",
    srcs = glob(["text_generation/causal_lm/cpp/continuous_batching/library/**"]),
    visibility = ["//visibility:public"],
)

build_release = {{"CMAKE_BUILD_TYPE": "Release"}}
build_debug = {{"CMAKE_BUILD_TYPE": "Debug"}}
cmake(
    name = "llm_engine_cmake",
    build_args = [
        "--verbose",
        "--",  # <- Pass remaining options to the native tool.
        # https://github.com/bazelbuild/rules_foreign_cc/issues/329
        # there is no elegant paralell compilation support
        "VERBOSE=1",
        "-j 4",
    ],
    cache_entries = {{
        "BUILD_SHARED_LIBS": "OFF",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_CXX_FLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1 -Wno-error=deprecated-declarations -Wuninitialized\",
        "CMAKE_ARCHIVE_OUTPUT_DIRECTORY": "lib"
    }} | select({{
           "//conditions:default": dict(
               build_release
            ),
            ":dbg":  dict(
               build_debug
            ),
        }}),
    env = {{
        "OpenVINO_DIR": "{OpenVINO_DIR}",
        "HTTP_PROXY": "{http_proxy}",
        "HTTPS_PROXY": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    out_lib_dir = "{lib_path}",
   
    # linking order
    out_static_libs = [
            "libopenvino_continuous_batching.a",
        ],
    tags = ["requires-network"],
    alwayslink = False,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "llm_engine",
    deps = [
        ":llm_engine_cmake",
    ],
    visibility = ["//visibility:public"],
    alwayslink = False,
)
"""
    repository_ctx.file("BUILD", build_file_content.format(OpenVINO_DIR=OpenVINO_DIR, http_proxy=http_proxy, https_proxy=https_proxy, lib_path=lib_path))

llm_engine_repository = repository_rule(
    implementation = _impl,
    local=False,
)
