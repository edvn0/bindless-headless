include_guard(GLOBAL)

include(cmake/CompileOptions.cmake)

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
  "src/Buffer.cxx"
  "src/Logger.cxx"
  "src/Camera.cxx"
  "src/Compiler.cxx"
  "src/Mesh.cxx"
  "src/GlobalCommandContext.cxx"
  "src/ThreadPool.cxx"
)


add_library(BindlessHeadlessAllocator STATIC "src/allocator.cpp")
add_library(ThirdPartySTB STATIC "3PP/stb.c")

target_precompile_headers(BindlessHeadless PRIVATE PCH.hxx)

target_include_directories(BindlessHeadless PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

# Link base deps
target_link_libraries(BindlessHeadlessAllocator PUBLIC
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




find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_library(ImageOperationsOpenMP
        "src/ImageOperations_OpenMP.cxx"
    )
    target_include_directories(ImageOperationsOpenMP PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(ImageOperationsOpenMP PUBLIC
        OpenMP::OpenMP_CXX
        BindlessHeadlessAllocator
    )
    add_library(ImageOperations ALIAS ImageOperationsOpenMP)
DEFAULT_COMPILE_OPTIONS(ImageOperationsOpenMP)
    
    message(STATUS "Building ImageOperations with OpenMP support")
else()
add_library(ImageOperationsThreadPool
    "src/ImageOperations.cxx"
)
target_include_directories(ImageOperationsThreadPool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ImageOperationsThreadPool PUBLIC
    BindlessHeadlessAllocator
)
DEFAULT_COMPILE_OPTIONS(ImageOperationsThreadPool)
    add_library(ImageOperations ALIAS ImageOperationsThreadPool)
    message(STATUS "OpenMP not found, only thread pool version will be built")
endif()

target_link_libraries(BindlessHeadless PRIVATE
  volk
  BindlessHeadlessAllocator
  spdlog::spdlog
  efsw-static
  ThirdPartySTB
  CLI11::CLI11
  glm::glm
  glfw
  expected
  ImageOperations
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

DEFAULT_COMPILE_OPTIONS(BindlessHeadless)

target_compile_definitions(BindlessHeadlessAllocator PUBLIC VK_NO_PROTOTYPES)

target_compile_definitions(BindlessHeadless PRIVATE
  SLANG_DISABLE_EXCEPTIONS=1
  GLM_FORCE_DEPTH_ZERO_TO_ONE
  GLM_ENABLE_EXPERIMENTAL
  ${VOLK_PLATFORM_DEFINE}
)