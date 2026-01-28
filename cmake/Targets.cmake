include_guard(GLOBAL)

include(CMakePushCheckState)
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

add_executable(BindlessHeadless
  "src/main.cpp"
  "src/ArgumentParse.cxx"
  "src/BindlessHeadless.cxx"
  "src/RenderContext.cxx"
  "src/Types.cxx"
  "src/Profiler.cxx"
  "src/Pipelines.cxx"
  "src/ResizeableGraph.cxx"
  "src/Swapchain.cxx"
  "src/ImageOperations.cxx"
  "src/Buffer.cxx"
  "src/Logger.cxx"
  "src/Camera.cxx"
  "src/Compiler.cxx"
  "src/Mesh.cxx"
  "src/GlobalCommandContext.cxx"
)


add_library(BindlessHeadlessAllocator STATIC "src/allocator.cpp")
add_library(ThirdPartySTB STATIC "3PP/stb.c")

target_precompile_headers(BindlessHeadless PRIVATE PCH.hxx)

target_include_directories(BindlessHeadless PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

# Link base deps
target_link_libraries(BindlessHeadlessAllocator PRIVATE
  volk
  volk::volk_headers
  VulkanMemoryAllocator
)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(BindlessHeadlessAllocator PRIVATE -Wno-nullability-completeness)
endif()

target_compile_definitions(ThirdPartySTB PRIVATE STB_IMAGE_IMPLEMENTATION)
target_include_directories(ThirdPartySTB PUBLIC "3PP")

cmake_push_check_state()
set(ASAN_FLAG "-fsanitize=address")
set(CMAKE_REQUIRED_FLAGS ${ASAN_FLAG})
check_c_compiler_flag(${ASAN_FLAG} C__fsanitize_address_VALID)
check_cxx_compiler_flag(${ASAN_FLAG} CXX__fsanitize_address_VALID)
if(NOT C__fsanitize_address_VALID OR NOT CXX__fsanitize_address_VALID)
  message(STATUS "ENABLE_ASAN was requested, but not supported!")
endif()
cmake_pop_check_state()

if (MSVC)
  add_compile_options(/bigobj)
endif()


target_link_libraries(BindlessHeadless PRIVATE
  volk
  volk::volk_headers
  BindlessHeadlessAllocator
  spdlog::spdlog
  efsw-static
  ThirdPartySTB
  CLI11::CLI11
  glm::glm
  glfw
  expected
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(BindlessHeadless PRIVATE -ftime-trace)
endif()

# Slang runtime deps only when runtime path
if (ENGINE_OFFLINE_SHADERS)
  target_compile_definitions(BindlessHeadless PRIVATE ENGINE_OFFLINE_SHADERS=1)
else()
  target_compile_definitions(BindlessHeadless PRIVATE ENGINE_RUNTIME_SHADERS=1)
  target_sources(BindlessHeadless PRIVATE "src/Reflection.cxx")
  target_include_directories(BindlessHeadless PRIVATE ${SLANG_INCLUDE_DIR})
  target_link_libraries(BindlessHeadless PRIVATE slang::slang slang-compiler slang-rt)
endif()

set_property(TARGET BindlessHeadless PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

if (HAS_TRACY)
  target_link_libraries(BindlessHeadless PRIVATE Tracy::TracyClient)
  target_compile_definitions(BindlessHeadless PRIVATE TRACY_ENABLE)
endif ()
