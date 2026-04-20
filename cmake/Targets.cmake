include_guard(GLOBAL)

include(cmake/CompileOptions.cmake)
include(CMakePushCheckState)
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

# ------------------------------------------------------------
# Scene compression
# ------------------------------------------------------------
if(SCENE_COMPRESSION STREQUAL "bzip2")
  find_package(BZip2 REQUIRED)
elseif(SCENE_COMPRESSION STREQUAL "zstd")
  find_package(zstd CONFIG REQUIRED)
elseif(SCENE_COMPRESSION STREQUAL "lz4")
  find_package(lz4 CONFIG REQUIRED)
else()
  message(FATAL_ERROR
    "Unknown SCENE_COMPRESSION value: '${SCENE_COMPRESSION}'. Must be bzip2, zstd, or lz4.")
endif()

# ------------------------------------------------------------
# WSI platform interface
# ------------------------------------------------------------
add_library(platform_wsi INTERFACE)

if(WIN32)
  target_compile_definitions(platform_wsi INTERFACE
    VK_USE_PLATFORM_WIN32_KHR
    GLFW_EXPOSE_NATIVE_WIN32
  )
elseif(UNIX AND NOT APPLE)
  find_package(X11 QUIET)

  find_path(WAYLAND_CLIENT_INCLUDE_DIR  NAMES wayland-client.h)
  find_path(WAYLAND_SERVER_INCLUDE_DIR  NAMES wayland-server.h)
  find_library(WAYLAND_CLIENT_LIBRARY   NAMES wayland-client libwayland-client)
  find_library(WAYLAND_SERVER_LIBRARY   NAMES wayland-server libwayland-server)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(WAYLAND_CLIENT
    REQUIRED_VARS WAYLAND_CLIENT_LIBRARY WAYLAND_CLIENT_INCLUDE_DIR)
  find_package_handle_standard_args(WAYLAND_SERVER
    REQUIRED_VARS WAYLAND_SERVER_LIBRARY WAYLAND_SERVER_INCLUDE_DIR)
  mark_as_advanced(
    WAYLAND_CLIENT_INCLUDE_DIR WAYLAND_CLIENT_LIBRARY
    WAYLAND_SERVER_INCLUDE_DIR WAYLAND_SERVER_LIBRARY)

  if(WAYLAND_CLIENT_INCLUDE_DIR AND WAYLAND_CLIENT_LIBRARY AND NOT TARGET wayland::client)
    add_library(wayland::client UNKNOWN IMPORTED)
    set_target_properties(wayland::client PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES   "${WAYLAND_CLIENT_INCLUDE_DIR}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      IMPORTED_LOCATION               "${WAYLAND_CLIENT_LIBRARY}")
  endif()

  if(WAYLAND_SERVER_INCLUDE_DIR AND WAYLAND_SERVER_LIBRARY AND NOT TARGET wayland::server)
    add_library(wayland::server UNKNOWN IMPORTED)
    set_target_properties(wayland::server PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES   "${WAYLAND_SERVER_INCLUDE_DIR}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      IMPORTED_LOCATION               "${WAYLAND_SERVER_LIBRARY}")
  endif()

  if(X11_FOUND)
    target_link_libraries(platform_wsi INTERFACE X11::X11)
    target_compile_definitions(platform_wsi INTERFACE
      VK_USE_PLATFORM_XCB_KHR
      GLFW_HAS_X11=1)
  endif()

  if(WAYLAND_CLIENT_INCLUDE_DIR AND WAYLAND_CLIENT_LIBRARY)
    target_link_libraries(platform_wsi INTERFACE "${WAYLAND_CLIENT_LIBRARY}")
    target_compile_definitions(platform_wsi INTERFACE
      VK_USE_PLATFORM_WAYLAND_KHR
      GLFW_HAS_WAYLAND=1)
  endif()
endif()

# ------------------------------------------------------------
# ThirdPartySTB
# ------------------------------------------------------------
add_library(ThirdPartySTB STATIC "3PP/stb.c")
target_compile_definitions(ThirdPartySTB PRIVATE
  STB_IMAGE_IMPLEMENTATION
  STB_IMAGE_RESIZE_IMPLEMENTATION)
target_include_directories(ThirdPartySTB PUBLIC "3PP")

# ------------------------------------------------------------
# BindlessHeadlessAllocator
# ------------------------------------------------------------
add_library(BindlessHeadlessAllocator STATIC "src/allocator.cpp")
target_link_libraries(BindlessHeadlessAllocator PUBLIC
  volk::volk_headers
  VulkanMemoryAllocator)
target_compile_definitions(BindlessHeadlessAllocator PUBLIC VK_NO_PROTOTYPES)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(BindlessHeadlessAllocator PUBLIC
    -Wno-nullability-completeness
    -Wno-nullability-extension)
endif()

add_library(BindlessCommon STATIC
  "src/Logger.cxx"
  "src/StringPool.cxx"
)
target_include_directories(BindlessCommon PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(BindlessCommon PUBLIC 
  spdlog::spdlog 
  volk::volk_headers
)
DEFAULT_COMPILE_OPTIONS(BindlessCommon)

if(HAS_IMAGE_WRITERS)
  find_package(OpenMP)

  set(_image_writer_sources
    "src/image/BMPWriter.cxx"
    "src/image/PNGWriter.cxx"
    "src/image/ImageWriterFactory.cxx"
  )

  if(OpenMP_CXX_FOUND)
    add_library(ImageOperationsOpenMP
      "src/ImageOperations_OpenMP.cxx"
      ${_image_writer_sources})
    target_include_directories(ImageOperationsOpenMP PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(ImageOperationsOpenMP PUBLIC
      OpenMP::OpenMP_CXX
      BindlessHeadlessAllocator
      glm::glm
      PNG::PNG)
    DEFAULT_COMPILE_OPTIONS(ImageOperationsOpenMP)
    add_library(ImageOperations ALIAS ImageOperationsOpenMP)
  else()
    add_library(ImageOperationsThreadPool
      "src/ImageOperations.cxx"
      ${_image_writer_sources})
    target_include_directories(ImageOperationsThreadPool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(ImageOperationsThreadPool PUBLIC
      BindlessHeadlessAllocator
      glm::glm)
    DEFAULT_COMPILE_OPTIONS(ImageOperationsThreadPool)
    add_library(ImageOperations ALIAS ImageOperationsThreadPool)
  endif()
endif()

# ------------------------------------------------------------
# SceneLoader (defined before BindlessHeadless which links it)
# ------------------------------------------------------------
add_library(SceneLoader STATIC "src/SceneLoader.cxx")
target_include_directories(SceneLoader PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(SceneLoader PUBLIC
  glm::glm
  ktx
  expected
  fastgltf::fastgltf
  ThirdPartySTB
  volk::volk_headers
  meshoptimizer)
DEFAULT_COMPILE_OPTIONS(SceneLoader)

# ------------------------------------------------------------
# BindlessEngine
# ------------------------------------------------------------
add_library(BindlessEngine STATIC
  "src/BindlessHeadless.cxx"
  "src/BindlessSet.cxx"
  "src/RenderContext.cxx"
  "src/Types.cxx"
  "src/TypesString.cxx"
  "src/Profiler.cxx"
  "src/Pipelines.cxx"
  "src/ResizeableGraph.cxx"
  "src/Swapchain.cxx"
  "src/Buffer.cxx"
  "src/AABB.cxx"
  "src/Camera.cxx"
  "src/Compiler.cxx"
  "src/Mesh.cxx"
  "src/GlobalCommandContext.cxx"
  "src/ThreadPool.cxx"
  "src/ImGuiRenderer.cxx"
  "src/RenderSubmission.cxx"
  "src/DeviceThreadPool.cxx"
  "src/Compression.cxx"
  "src/Stream.cxx"
  "src/RenderDoc.cxx"
  "src/Reflection.cxx"
  "src/scene/Scene.cxx"
)
target_precompile_headers(BindlessEngine PRIVATE PCH.hxx)
target_include_directories(BindlessEngine
  PUBLIC  ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${SLANG_INCLUDE_DIR})
set_target_properties(BindlessEngine PROPERTIES
  BUILD_RPATH   "${SLANG_ROOT}/lib;$ENV{VULKAN_SDK}/lib"
  INSTALL_RPATH "${SLANG_ROOT}/lib;$ENV{VULKAN_SDK}/lib")
target_compile_definitions(BindlessEngine PUBLIC
  SLANG_DISABLE_EXCEPTIONS=1
  GLM_FORCE_DEPTH_ZERO_TO_ONE
  GLFW_INCLUDE_NONE
  GLM_ENABLE_EXPERIMENTAL
  ${VOLK_PLATFORM_DEFINE}
  $<$<BOOL:${HAS_IMAGE_WRITERS}>:HAS_IMAGE_WRITERS>)
target_link_libraries(BindlessEngine
  PUBLIC
    BindlessCommon
    volk
    platform_wsi
    bfg::lyra
    BindlessHeadlessAllocator
    ktx
    efsw-static
    ThirdPartySTB
    glm::glm
    glfw
    expected
    tinyobjloader
    imgui
    EnTT::EnTT
    meshoptimizer
    $<$<BOOL:${HAS_IMAGE_WRITERS}>:ImageOperations>
  PRIVATE
    slang::slang
    slang-compiler
    slang-rt)
if(HAS_TRACY)
  target_link_libraries(BindlessEngine PRIVATE Tracy::TracyClient)
  target_compile_definitions(BindlessEngine PRIVATE
    TRACY_ENABLE
    TRACY_VK_USE_SYMBOL_TABLE)
endif()
if(RENDERDOC_INCLUDE_PATH)
  if(NOT EXISTS "${RENDERDOC_INCLUDE_PATH}/renderdoc_app.h")
    message(FATAL_ERROR
      "RENDERDOC_INCLUDE_PATH is '${RENDERDOC_INCLUDE_PATH}' but renderdoc_app.h was not found there.")
  endif()
  target_include_directories(BindlessEngine PRIVATE "${RENDERDOC_INCLUDE_PATH}")
  message(STATUS "RenderDoc: using header from ${RENDERDOC_INCLUDE_PATH}")
else()
  message(STATUS "RenderDoc: using renderdoc_app.h from default include paths")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(BindlessEngine PRIVATE -ftime-trace)
endif()
DEFAULT_COMPILE_OPTIONS(BindlessEngine)

# ------------------------------------------------------------
# BaseApplication
# ------------------------------------------------------------
add_library(BaseApplication STATIC
  "src/framework/BaseApplication.cxx"
  "src/framework/LogWidget.cxx"
  "src/ArgumentParse.cxx"
)
target_precompile_headers(BaseApplication PRIVATE PCH.hxx)
target_include_directories(BaseApplication PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(BaseApplication
  PUBLIC  BindlessEngine
  PRIVATE platform_wsi)
DEFAULT_COMPILE_OPTIONS(BaseApplication)

# ------------------------------------------------------------
# BindlessApp
# ------------------------------------------------------------
add_library(BindlessApp STATIC
  "src/app/listeners.cxx"
  "src/app/math.cxx"
  "src/app/frame.cxx"
  "src/app/ui.cxx"
  "src/app/render.cxx"
  "src/app/render_passes.cxx"
  "src/app/app.cxx"
)
target_precompile_headers(BindlessApp PRIVATE PCH.hxx)
target_include_directories(BindlessApp PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(BindlessApp
  PUBLIC  BindlessEngine
  PRIVATE platform_wsi)
if(HAS_TRACY)
  target_link_libraries(BindlessApp PRIVATE Tracy::TracyClient)
  target_compile_definitions(BindlessApp PRIVATE
    TRACY_ENABLE
    TRACY_VK_USE_SYMBOL_TABLE
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_ENABLE_EXPERIMENTAL)
endif()
DEFAULT_COMPILE_OPTIONS(BindlessApp)

# ------------------------------------------------------------
# BindlessHeadless executable
# ------------------------------------------------------------
add_executable(BindlessHeadless "src/main.cpp")
target_precompile_headers(BindlessHeadless PRIVATE PCH.hxx)
target_include_directories(BindlessHeadless PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(BindlessHeadless PRIVATE
  BindlessApp
  BaseApplication
  platform_wsi
  SceneLoader)
set_target_properties(BindlessHeadless PROPERTIES
  BUILD_RPATH          "${SLANG_ROOT}/lib"
  INSTALL_RPATH        "${SLANG_ROOT}/lib"
  BUILD_RPATH_USE_ORIGIN ON)
target_link_options(BindlessHeadless PRIVATE "-Wl,--disable-new-dtags")
DEFAULT_COMPILE_OPTIONS(BindlessHeadless)

# ------------------------------------------------------------
# scene_inspect
# ------------------------------------------------------------
add_executable(scene_inspect "src/SceneInspector.cxx")
target_include_directories(scene_inspect PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(scene_inspect PRIVATE BindlessEngine)
DEFAULT_COMPILE_OPTIONS(scene_inspect)

# ------------------------------------------------------------
# ImageViewer
# ------------------------------------------------------------
add_executable(ImageViewer "src/tools/ImageViewer.cxx")
target_link_libraries(ImageViewer PRIVATE BaseApplication SceneLoader)
DEFAULT_COMPILE_OPTIONS(ImageViewer)

# ------------------------------------------------------------
# tex_convert (requires ImageMagick)
# ------------------------------------------------------------
find_package(ImageMagick7)

if(ImageMagick7_FOUND)
  add_executable(tex_convert "src/tools/TextureConvert.cxx")
  target_include_directories(tex_convert PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(tex_convert PRIVATE
    BindlessCommon
    ktx
    ThirdPartySTB
    ImageMagick::Magick++
    bfg::lyra
    volk::volk_headers
    volk)
  target_compile_definitions(tex_convert PRIVATE
    MAGICKCORE_QUANTUM_DEPTH=16
    MAGICKCORE_HDRI_ENABLE=1)
  DEFAULT_COMPILE_OPTIONS(tex_convert)
endif()

message(STATUS "tex_convert: ${ImageMagick7_FOUND}")
message(STATUS "  include:    ${ImageMagick7_INCLUDE_DIR}")
message(STATUS "  Magick++:   ${ImageMagick7_MagickPP_LIBRARY}")
message(STATUS "  MagickCore: ${ImageMagick7_MagickCore_LIBRARY}")
message(STATUS "  MagickWand: ${ImageMagick7_MagickWand_LIBRARY}")

# ------------------------------------------------------------
# ASan check
# ------------------------------------------------------------
cmake_push_check_state()
set(ASAN_FLAG "-fsanitize=address")
set(CMAKE_REQUIRED_FLAGS "${ASAN_FLAG}")
check_c_compiler_flag("${ASAN_FLAG}" C__fsanitize_address_VALID)
check_cxx_compiler_flag("${ASAN_FLAG}" CXX__fsanitize_address_VALID)
if(NOT C__fsanitize_address_VALID OR NOT CXX__fsanitize_address_VALID)
  message(STATUS "ENABLE_ASAN was requested, but is not supported by this compiler")
endif()
cmake_pop_check_state()

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
enable_testing()

add_executable(BindlessTests
  "tests/test_main.cxx"
  "tests/test_watermarked_queue.cxx"
)
target_include_directories(BindlessTests PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(BindlessTests PRIVATE
  doctest::doctest
  BindlessEngine)
add_test(NAME BindlessUnitTests COMMAND BindlessTests)