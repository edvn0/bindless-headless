include_guard(GLOBAL)

CPMAddPackage(
  URI "gh:g-truc/glm#1.0.3"
  OPTIONS
    "GLM_ENABLE_CXX_20 ON"
    "GLM_ENABLE_SIMD_AVX2 ON"
    "GLM_ENABLE_SIMD_AVX ON"
    "GLM_ENABLE_SIMD_SSE4_2 ON"
    "GLM_ENABLE_SIMD_SSE2 ON"
  GIT_SHALLOW YES
)

CPMAddPackage(
  URI "gh:SpartanJ/efsw#1.5.1"
  OPTIONS "BUILD_SHARED_LIBS OFF"
  GIT_SHALLOW YES
)

CPMAddPackage(
  NAME CLI11
  GITHUB_REPOSITORY CLIUtils/CLI11
  VERSION 2.6.1
  GIT_SHALLOW YES
)

CPMAddPackage(
  URI "gh:glfw/glfw#3.4"
  GIT_SHALLOW YES
)

CPMAddPackage(
  URI "gh:TartanLlama/expected@1.3.1"
  OPTIONS "BUILD_TESTING OFF"
  GIT_SHALLOW YES
)

CPMAddPackage(
  NAME Vulkan-Headers
  GITHUB_REPOSITORY KhronosGroup/Vulkan-Headers
  GIT_TAG vulkan-sdk-1.4.335.0
  GIT_SHALLOW YES
)

get_target_property(VULKAN_HEADERS_INCLUDE Vulkan::Headers INTERFACE_INCLUDE_DIRECTORIES)
set(VULKAN_HEADERS_INSTALL_DIR "${VULKAN_HEADERS_INCLUDE}" CACHE PATH "" FORCE)

set(VOLK_PULL_IN_VULKAN OFF CACHE BOOL "" FORCE)
set(VOLK_HEADERS_ONLY OFF CACHE BOOL "" FORCE)

CPMAddPackage(
  NAME volk
  GITHUB_REPOSITORY zeux/volk
  GIT_TAG vulkan-sdk-1.4.335.0
  GIT_SHALLOW YES
  OPTIONS
    "VOLK_STATIC_DEFINES=${VOLK_PLATFORM_DEFINE}"
)

target_include_directories(volk_headers INTERFACE ${VULKAN_HEADERS_INCLUDE})
target_include_directories(volk PRIVATE ${VULKAN_HEADERS_INCLUDE})

CPMAddPackage(
  NAME VulkanMemoryAllocator
  GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
  VERSION 3.3.0
  GIT_SHALLOW YES
)

CPMAddPackage(
  URI "gh:tinyobjloader/tinyobjloader#d56555b026c1c7cec0f93f3ec7f1de2ff005c5ad"
  GIT_SHALLOW YES
)

CPMAddPackage(
  URI "gh:gabime/spdlog@1.17.0"
  GIT_SHALLOW YES
)

CPMAddPackage(
  URI "gh:KhronosGroup/KTX-Software@4.4.2"
  GIT_SHALLOW YES
  OPTIONS "KTX_FEATURE_TESTS OFF KTX_FEATURE_JS OFF BUILD_SHARED_LIBS OFF KTX_FEATURE_TOOLS OFF KTX_FEATURE_TESTS OFF KTX_FEATURE_LOADTEST_APPS OFF"
)

set(CMAKE_SKIP_INSTALL_RULES ON CACHE BOOL "" FORCE)
# --- zlib ---
CPMAddPackage(
  NAME zlib
  GITHUB_REPOSITORY madler/zlib
  GIT_TAG v1.3.1
  GIT_SHALLOW YES
)

# Provide the canonical target name libpng expects.
if(TARGET zlibstatic AND NOT TARGET ZLIB::ZLIB)
  add_library(ZLIB::ZLIB ALIAS zlibstatic)
elseif(TARGET zlib AND NOT TARGET ZLIB::ZLIB)
  add_library(ZLIB::ZLIB ALIAS zlib)
endif()

# --- libpng ---
set(PNG_SHARED OFF CACHE BOOL "" FORCE)
set(PNG_TESTS OFF CACHE BOOL "" FORCE)
set(PNG_TOOLS OFF CACHE BOOL "" FORCE)

CPMAddPackage(
  NAME libpng
  GITHUB_REPOSITORY glennrp/libpng
  GIT_TAG v1.6.43
  GIT_SHALLOW YES
)

# Optional: normalize a PNG::PNG target name for convenience
if(TARGET png_static AND NOT TARGET PNG::PNG)
  add_library(PNG::PNG ALIAS png_static)
elseif(TARGET png AND NOT TARGET PNG::PNG)
  add_library(PNG::PNG ALIAS png)
elseif(TARGET png_shared AND NOT TARGET PNG::PNG)
  add_library(PNG::PNG ALIAS png_shared)
endif()

if(libpng_ADDED)
  if(TARGET png_static)
    target_include_directories(png_static PUBLIC
      $<BUILD_INTERFACE:${libpng_SOURCE_DIR}>
      $<BUILD_INTERFACE:${libpng_BINARY_DIR}>
    )
  elseif(TARGET png)
    target_include_directories(png PUBLIC
      $<BUILD_INTERFACE:${libpng_SOURCE_DIR}>
      $<BUILD_INTERFACE:${libpng_BINARY_DIR}>
    )
  endif()
endif()


if (HAS_TRACY)
    set(TRACY_ENABLE ON CACHE BOOL "Enable Tracy profiler" FORCE)
    CPMAddPackage(
    URI "gh:wolfpld/tracy@0.13.1"
    GIT_SHALLOW YES
  )
endif()

if (NOT ENGINE_OFFLINE_SHADERS)
    find_package(Slang CONFIG REQUIRED)
    set(SLANG_LIB_DIR "${SLANG_ROOT}/lib")
    set(SLANG_INCLUDE_DIR "${SLANG_ROOT}/include")

    add_library(slang-compiler STATIC IMPORTED)
    set_target_properties(slang-compiler PROPERTIES
    IMPORTED_LOCATION "${SLANG_LIB_DIR}/slang-compiler.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${SLANG_INCLUDE_DIR}"
  )

    add_library(slang-rt STATIC IMPORTED)
    set_target_properties(slang-rt PROPERTIES
    IMPORTED_LOCATION "${SLANG_LIB_DIR}/slang-rt.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${SLANG_INCLUDE_DIR}"
  )
endif()
