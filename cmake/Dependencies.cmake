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

set(GLFW_CPM_OPTIONS "")
if(UNIX AND NOT APPLE)
  list(APPEND GLFW_CPM_OPTIONS
    "GLFW_BUILD_WAYLAND OFF"
    "GLFW_BUILD_X11 ON"
  )
endif()

CPMAddPackage(
  URI "gh:glfw/glfw#3.4"
  GIT_SHALLOW YES
  OPTIONS ${GLFW_CPM_OPTIONS}
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
  URI "gh:tinyobjloader/tinyobjloader#8a3f8b92d18309e00911bf5cf5e465cfe4eaa1b1"
  GIT_SHALLOW YES
)

set(SPDLOG_BUILD_SHARED OFF)
CPMAddPackage(
  URI "gh:gabime/spdlog@1.17.0"
  GIT_SHALLOW YES
)

CPMAddPackage(
  URI "gh:KhronosGroup/KTX-Software@4.4.2"
  GIT_SHALLOW YES
  OPTIONS "KTX_FEATURE_TESTS OFF KTX_FEATURE_JS OFF BUILD_SHARED_LIBS OFF KTX_FEATURE_TOOLS OFF KTX_FEATURE_TESTS OFF KTX_FEATURE_LOADTEST_APPS OFF"
)

CPMAddPackage(
  URI "gh:spnda/fastgltf@0.9.0"
  GIT_SHALLOW YES
  OPTIONS "FASTGLTF_COMPILE_AS_CPP20 ON FASTGLTF_USE_CUSTOM_SMALLVECTOR ON"
)

# git@github.com:zeux/meshoptimizer.git
CPMAddPackage(
  URI "gh:zeux/meshoptimizer@1.0.1"
  GIT_SHALLOW YES
)

CPMAddPackage(
  URI "gh:skypjack/entt"
  GIT_TAG "9fdc43f6f8189581ccc81dace2ece1d5a981ace0"
  GIT_SHALLOW YES
  OPTIONS "ENTT_INCLUDE_NATVIS ON ENTT_INCLUDE_HEADERS ON"
)

if(HAS_IMAGE_WRITERS)
  CPMAddPackage(
    URI "gh:madler/zlib@1.3.1"
    GIT_SHALLOW YES
  )

  if(TARGET zlibstatic AND NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
  elseif(TARGET zlib AND NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlib)
  endif()

  find_package(PNG QUIET)

  if(NOT PNG_FOUND)
    message(STATUS "System libpng not found, using CPM fallback")

    # ---- zlib (dependency of libpng) ----
    CPMAddPackage(
      URI "gh:madler/zlib@1.3.1"
      GIT_SHALLOW YES
    )

    if(TARGET zlibstatic AND NOT TARGET ZLIB::ZLIB)
      add_library(ZLIB::ZLIB ALIAS zlibstatic)
    elseif(TARGET zlib AND NOT TARGET ZLIB::ZLIB)
      add_library(ZLIB::ZLIB ALIAS zlib)
    endif()

    # ---- libpng (vendored) ----
    set(PNG_STATIC ON CACHE BOOL "" FORCE)
    set(PNG_SHARED OFF CACHE BOOL "" FORCE)
    set(PNG_TESTS OFF CACHE BOOL "" FORCE)
    set(PNG_TOOLS OFF CACHE BOOL "" FORCE)

    # 🔑 Prevent install/export issues
    set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)

    CPMAddPackage(
      URI "gh:glennrp/libpng@1.6.43"
      GIT_SHALLOW YES
    )

    # Normalize target name
    if(TARGET png_static AND NOT TARGET PNG::PNG)
      add_library(PNG::PNG ALIAS png_static)
    elseif(TARGET png_shared AND NOT TARGET PNG::PNG)
      add_library(PNG::PNG ALIAS png_shared)
    endif()

    if(TARGET png_static)
      target_include_directories(png_static
        PUBLIC
        $<BUILD_INTERFACE:${libpng_SOURCE_DIR}>
        $<BUILD_INTERFACE:${libpng_BINARY_DIR}>
      )
    elseif(TARGET png_shared)
      target_include_directories(png_shared
        PUBLIC
        $<BUILD_INTERFACE:${libpng_SOURCE_DIR}>
        $<BUILD_INTERFACE:${libpng_BINARY_DIR}>
      )
    endif()
  endif()
endif()

if(HAS_TRACY)
  set(TRACY_ENABLE ON CACHE BOOL "Enable Tracy profiler" FORCE)
  CPMAddPackage(
    URI "gh:wolfpld/tracy@0.13.1"
    GIT_SHALLOW YES
  )
endif()

function(FIND_SLANG)
find_package(Slang CONFIG REQUIRED)

# Ensure SLANG_ROOT is set if find_package didn't set it automatically
set(SLANG_LIB_DIR "${SLANG_ROOT}/lib")
set(SLANG_INCLUDE_DIR "${SLANG_ROOT}/include")

add_library(slang-compiler SHARED IMPORTED)
add_library(slang-rt SHARED IMPORTED)

set_target_properties(slang-compiler PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${SLANG_INCLUDE_DIR}"
  INTERFACE_COMPILE_DEFINITIONS "SLANG_DYNAMIC"
)
set_target_properties(slang-rt PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${SLANG_INCLUDE_DIR}"
  INTERFACE_COMPILE_DEFINITIONS "SLANG_DYNAMIC"
)

if(WIN32)
  # Typical layout: import libs in lib/, DLLs sometimes in bin/ (depends on the package)
  # Adjust BIN dir if your package puts DLLs elsewhere.
  set(SLANG_BIN_DIR "${SLANG_ROOT}/bin")

  set_target_properties(slang-compiler PROPERTIES
    IMPORTED_IMPLIB "${SLANG_LIB_DIR}/slang-compiler.lib"
    IMPORTED_LOCATION "${SLANG_BIN_DIR}/slang-compiler.dll"
  )
  set_target_properties(slang-rt PROPERTIES
    IMPORTED_IMPLIB "${SLANG_LIB_DIR}/slang-rt.lib"
    IMPORTED_LOCATION "${SLANG_BIN_DIR}/slang-rt.dll"
  )

elseif(APPLE)
  set_target_properties(slang-compiler PROPERTIES
    IMPORTED_LOCATION "${SLANG_LIB_DIR}/libslang-compiler.dylib"
  )
  set_target_properties(slang-rt PROPERTIES
    IMPORTED_LOCATION "${SLANG_LIB_DIR}/libslang-rt.dylib"
  )

else() # Linux
  set_target_properties(slang-compiler PROPERTIES
    IMPORTED_LOCATION "${SLANG_LIB_DIR}/libslang-compiler.so"
  )
  set_target_properties(slang-rt PROPERTIES
    IMPORTED_LOCATION "${SLANG_LIB_DIR}/libslang-rt.so"
  )
endif()

message(STATUS "Slang root: ${SLANG_ROOT}")
message(STATUS "Slang lib dir: ${SLANG_LIB_DIR}")
endfunction()

FIND_SLANG()


CPMAddPackage(
  NAME ImGui
  GITHUB_REPOSITORY ocornut/imgui
  GIT_TAG v1.92.5-docking
  GIT_SHALLOW YES
  DOWNLOAD_ONLY YES
)
CPMAddPackage(
  NAME ImPlot
  GITHUB_REPOSITORY epezent/implot
  GIT_TAG 93c801b4bb801c5c11031d880b6af1d1f70bd79d
  GIT_SHALLOW YES
  DOWNLOAD_ONLY YES
)

CPMAddPackage(
  NAME ImGuizmo
  GITHUB_REPOSITORY CedricGuillemet/ImGuizmo
  GIT_TAG a15acd87a3f3241a29ea1363ceafc680dca3a96b
  GIT_SHALLOW YES
  DOWNLOAD_ONLY YES
)

if(ImGui_SOURCE_DIR AND ImPlot_SOURCE_DIR AND ImGuizmo_SOURCE_DIR)
  set(IMGUI_SRCS
    ${ImGui_SOURCE_DIR}/imgui.cpp
    ${ImGui_SOURCE_DIR}/imgui_demo.cpp
    ${ImGui_SOURCE_DIR}/imgui_draw.cpp
    ${ImGui_SOURCE_DIR}/imgui_tables.cpp
    ${ImGui_SOURCE_DIR}/imgui_widgets.cpp
    ${ImGui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${ImGui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    ${ImGui_SOURCE_DIR}/misc/freetype/imgui_freetype.cpp
  )

  set(IMPLOT_SRCS
    ${ImPlot_SOURCE_DIR}/implot.cpp
    ${ImPlot_SOURCE_DIR}/implot_items.cpp
  )

  set(IMGUIZMO_SRCS
    ${ImGuizmo_SOURCE_DIR}/ImGuizmo.cpp
  )
  list(PREPEND CMAKE_PREFIX_PATH "C:/D/Builds")

  find_package(harfbuzz CONFIG REQUIRED)

  if(NOT TARGET HarfBuzz::HarfBuzz)
    if(TARGET harfbuzz::harfbuzz)
      add_library(HarfBuzz::HarfBuzz ALIAS harfbuzz::harfbuzz)
    elseif(TARGET harfbuzz)
      add_library(HarfBuzz::HarfBuzz ALIAS harfbuzz)
    else()
      message(FATAL_ERROR "Found harfbuzz package, but no known harfbuzz target was exported.")
    endif()
  endif()
  find_package(Freetype CONFIG REQUIRED)

  add_library(imgui STATIC ${IMGUI_SRCS} ${IMPLOT_SRCS} ${IMGUIZMO_SRCS})

  target_include_directories(imgui PUBLIC
    ${ImGui_SOURCE_DIR}
    ${ImGui_SOURCE_DIR}/backends
    ${ImPlot_SOURCE_DIR}
    ${ImGuizmo_SOURCE_DIR}
  )

  target_link_libraries(imgui PUBLIC
    glfw
    volk
    volk::volk_headers
    platform_wsi
    Freetype::Freetype
  )

  target_compile_definitions(imgui
    PUBLIC
    GLFW_INCLUDE_NONE
    IMGUI_IMPL_VULKAN_USE_VOLK
    IMGUI_ENABLE_FREETYPE
  )
endif()
