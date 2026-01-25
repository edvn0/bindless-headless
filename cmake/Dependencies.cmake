include_guard(GLOBAL)

# --- CPM dependencies (yours, reorganized) ---

CPMAddPackage(
  URI "gh:g-truc/glm#1.0.3"
  OPTIONS
    "GLM_ENABLE_CXX_20 ON"
    "GLM_ENABLE_SIMD_AVX2 ON"
    "GLM_ENABLE_SIMD_AVX ON"
    "GLM_ENABLE_SIMD_SSE4_2 ON"
    "GLM_ENABLE_SIMD_SSE2 ON"
)

CPMAddPackage(
  URI "gh:SpartanJ/efsw#1.5.1"
  OPTIONS "BUILD_SHARED_LIBS OFF"
)

CPMAddPackage(
  NAME CLI11
  GITHUB_REPOSITORY CLIUtils/CLI11
  VERSION 2.6.1
)

CPMAddPackage(
  URI "gh:glfw/glfw#3.4"
)

CPMAddPackage(
  URI "gh:TartanLlama/expected@1.3.1"
  OPTIONS "BUILD_TESTING OFF"
)

CPMAddPackage(
  NAME Vulkan-Headers
  GITHUB_REPOSITORY KhronosGroup/Vulkan-Headers
  GIT_TAG vulkan-sdk-1.4.335.0
)

get_target_property(VULKAN_HEADERS_INCLUDE Vulkan::Headers INTERFACE_INCLUDE_DIRECTORIES)
set(VULKAN_HEADERS_INSTALL_DIR "${VULKAN_HEADERS_INCLUDE}" CACHE PATH "" FORCE)

set(VOLK_PULL_IN_VULKAN OFF CACHE BOOL "" FORCE)
set(VOLK_HEADERS_ONLY OFF CACHE BOOL "" FORCE)

CPMAddPackage(
  NAME volk
  GITHUB_REPOSITORY zeux/volk
  GIT_TAG vulkan-sdk-1.4.335.0
  OPTIONS
    "VOLK_STATIC_DEFINES=${VOLK_PLATFORM_DEFINE}"
)

target_include_directories(volk_headers INTERFACE ${VULKAN_HEADERS_INCLUDE})
target_include_directories(volk PRIVATE ${VULKAN_HEADERS_INCLUDE})

CPMAddPackage(
  NAME VulkanMemoryAllocator
  GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
  VERSION 3.3.0
)

# --- Non-CPM deps via find_package ---
find_package(spdlog REQUIRED)

# Tracy (optional)
if (HAS_TRACY)
  set(TRACY_ENABLE ON CACHE BOOL "Enable Tracy profiler" FORCE)
  CPMAddPackage(URI "gh:wolfpld/tracy#07147111b26ddaf43fb46fabbab42de4451fa567")
endif()

# Slang only needed for runtime path
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
