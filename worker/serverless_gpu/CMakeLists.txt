cmake_minimum_required(VERSION 3.13)

project(svgpu-manager C CXX)

list(APPEND CMAKE_MODULE_PATH)

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)

find_program(_PROTOBUF_PROTOC protoc)

#find_package(absl CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(_GRPC_GRPCPP gRPC::grpc++)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

#gen grpc stuff
get_filename_component(rm_proto "pb/resmngr.proto" ABSOLUTE)
get_filename_component(rm_proto_path "${rm_proto}" PATH)

set(rm_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/resmngr.pb.cc")
set(rm_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/resmngr.pb.h")
set(rm_grpc_srcs  "${CMAKE_CURRENT_BINARY_DIR}/resmngr.grpc.pb.cc")
set(rm_grpc_hdrs  "${CMAKE_CURRENT_BINARY_DIR}/resmngr.grpc.pb.h")

add_custom_command(
  OUTPUT "${rm_proto_srcs}" "${rm_proto_hdrs}" "${rm_grpc_srcs}" "${rm_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
    --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
    -I "${rm_proto_path}"
    --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
    "${rm_proto}"
  DEPENDS "${rm_proto}")

include_directories(${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  "/usr/local/cuda/include"
  ${CMAKE_SOURCE_DIR}
)

add_library(grpc_proto
  ${rm_proto_srcs}
  ${rm_proto_hdrs}
  ${rm_grpc_srcs}
  ${rm_grpc_hdrs})

target_link_libraries(grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

add_executable(svgpu-manager
  manager.cpp
  svgpu_manager.cpp
  central_manager.cpp
  ${manager_service_proto_srcs}
  ${manager_service_ava_srcs}
)

#find_package(CUDA 10.2 REQUIRED)

target_link_libraries(svgpu-manager
  ${manager_service_libraries}
  ${absl_flags_libraries}
  absl::symbolize
  absl::failure_signal_handler
  grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  ${CUDA_LIBRARIES}
  nvidia-ml
  zmq
)

set_target_properties(svgpu-manager
  PROPERTIES OUTPUT_NAME manager
)
add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-write-strings)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/manager
  TYPE BIN
  RENAME svgpu_manager
)
