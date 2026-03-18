include_guard(GLOBAL)

include(cmake/CompileOptions.cmake)

include(CMakePushCheckState)
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

if(SCENE_COMPRESSION STREQUAL "bzip2")
    find_package(BZip2 REQUIRED)
elseif(SCENE_COMPRESSION STREQUAL "zstd")
    find_package(zstd CONFIG REQUIRED)
elseif(SCENE_COMPRESSION STREQUAL "lz4")
    find_package(lz4 CONFIG REQUIRED)
else()
    message(FATAL_ERROR "Unknown SCENE_COMPRESSION value: ${SCENE_COMPRESSION}. Must be bzip2, zstd or lz4.")
endif()

set(HAS_X11 OFF)
find_package(X11 QUIET)
if(X11_FOUND)
  set(HAS_X11 ON)
endif()

add_library(platform_wsi INTERFACE)

find_path(
  WAYLAND_CLIENT_INCLUDE_DIR
  NAMES wayland-client.h
)

find_library(
  WAYLAND_CLIENT_LIBRARY
  NAMES wayland-client libwayland-client
)

if(WAYLAND_CLIENT_INCLUDE_DIR AND WAYLAND_CLIENT_LIBRARY)
  add_library(wayland::client UNKNOWN IMPORTED)

  set_target_properties(
    wayland::client PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${WAYLAND_CLIENT_INCLUDE_DIR}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${WAYLAND_CLIENT_LIBRARY}"
  )
endif()

find_path(
  WAYLAND_SERVER_INCLUDE_DIR
  NAMES wayland-server.h
)

find_library(
  WAYLAND_SERVER_LIBRARY
  NAMES wayland-server libwayland-server
)

if(WAYLAND_SERVER_INCLUDE_DIR AND WAYLAND_SERVER_LIBRARY)
  add_library(wayland::server UNKNOWN IMPORTED)

  set_target_properties(
    wayland::server PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${WAYLAND_SERVER_INCLUDE_DIR}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${WAYLAND_SERVER_LIBRARY}"
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  WAYLAND_CLIENT
  REQUIRED_VARS WAYLAND_CLIENT_LIBRARY WAYLAND_CLIENT_INCLUDE_DIR
)

find_package_handle_standard_args(
  WAYLAND_SERVER
  REQUIRED_VARS WAYLAND_SERVER_LIBRARY WAYLAND_SERVER_INCLUDE_DIR
)

mark_as_advanced(
  WAYLAND_CLIENT_INCLUDE_DIR
  WAYLAND_CLIENT_LIBRARY
  WAYLAND_SERVER_INCLUDE_DIR
  WAYLAND_SERVER_LIBRARY
)

target_link_libraries(platform_wsi INTERFACE ${WAYLAND_CLIENT_LIBRARY} X11::X11)
target_compile_definitions(platform_wsi INTERFACE
  VK_USE_PLATFORM_XCB_KHR
  GLFW_HAS_X11=1
  GLFW_HAS_WAYLAND=1
)

# ------------------------------------------------------------
# Core libs you already have
# ------------------------------------------------------------
add_library(BindlessHeadlessAllocator STATIC "src/allocator.cpp")
add_library(ThirdPartySTB STATIC "3PP/stb.c")

target_link_libraries(BindlessHeadlessAllocator PUBLIC
  volk::volk_headers
  VulkanMemoryAllocator
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(BindlessHeadlessAllocator PUBLIC -Wno-nullability-completeness -Wno-nullability-extension)
endif()

target_compile_definitions(ThirdPartySTB PRIVATE STB_IMAGE_IMPLEMENTATION STB_IMAGE_RESIZE_IMPLEMENTATION)
target_include_directories(ThirdPartySTB PUBLIC "3PP")

target_compile_definitions(BindlessHeadlessAllocator PUBLIC VK_NO_PROTOTYPES)

# ------------------------------------------------------------
# Optional ImageOperations stays as-is (just no longer tied to exe)
# ------------------------------------------------------------
if(HAS_IMAGE_WRITERS)
  find_package(OpenMP)

  set(IMAGE_WRITER_SOURCES
    "src/image/BMPWriter.cxx"
    "src/image/PNGWriter.cxx"
    "src/image/ImageWriterFactory.cxx"
  )

  if(OpenMP_CXX_FOUND)
    add_library(ImageOperationsOpenMP
      "src/ImageOperations_OpenMP.cxx"
      ${IMAGE_WRITER_SOURCES}
    )
    target_include_directories(ImageOperationsOpenMP PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(ImageOperationsOpenMP PUBLIC
      OpenMP::OpenMP_CXX
      BindlessHeadlessAllocator
      glm::glm
      PNG::PNG
    )
    DEFAULT_COMPILE_OPTIONS(ImageOperationsOpenMP)
    add_library(ImageOperations ALIAS ImageOperationsOpenMP)
  else()
    add_library(ImageOperationsThreadPool
      "src/ImageOperations.cxx"
      ${IMAGE_WRITER_SOURCES}
    )
    target_include_directories(ImageOperationsThreadPool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(ImageOperationsThreadPool PUBLIC
      BindlessHeadlessAllocator
      glm::glm
    )
    DEFAULT_COMPILE_OPTIONS(ImageOperationsThreadPool)
    add_library(ImageOperations ALIAS ImageOperationsThreadPool)
  endif()
endif()

# ------------------------------------------------------------
# Engine library: reusable stuff (no main)
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
  "src/Logger.cxx"
  "src/Camera.cxx"
  "src/Compiler.cxx"
  "src/Mesh.cxx"
  "src/GlobalCommandContext.cxx"
  "src/ThreadPool.cxx"
  "src/ImGuiRenderer.cxx"
  "src/StringPool.cxx"
  "src/RenderSubmission.cxx"
  "src/DeviceThreadPool.cxx"
  "src/Compression.cxx"

  "src/RenderDoc.cxx"

  "src/scene/Scene.cxx"
)

target_precompile_headers(BindlessEngine PRIVATE PCH.hxx)

target_include_directories(BindlessEngine PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(BindlessEngine PROPERTIES
  BUILD_RPATH   "${SLANG_ROOT}/lib;$ENV{VULKAN_SDK}/lib"
  INSTALL_RPATH "${SLANG_ROOT}/lib;$ENV{VULKAN_SDK}/lib"
)

target_link_libraries(BindlessEngine PUBLIC
  volk
  platform_wsi
  BindlessHeadlessAllocator
  ktx
  spdlog::spdlog
  efsw-static
  ThirdPartySTB
  glm::glm
  glfw
  expected
  tinyobjloader
  imgui
  $<$<BOOL:${HAS_IMAGE_WRITERS}>:ImageOperations>
  EnTT::EnTT
  meshoptimizer
)

target_sources(BindlessEngine PRIVATE "src/Reflection.cxx")
target_include_directories(BindlessEngine PRIVATE ${SLANG_INCLUDE_DIR})
target_link_libraries(BindlessEngine PRIVATE slang::slang slang-compiler slang-rt)

if(HAS_IMAGE_WRITERS)
  target_compile_definitions(BindlessEngine PUBLIC HAS_IMAGE_WRITERS)
endif()

if(HAS_TRACY)
  target_link_libraries(BindlessEngine PRIVATE Tracy::TracyClient)
  target_compile_definitions(BindlessEngine PRIVATE TRACY_ENABLE TRACY_VK_USE_SYMBOL_TABLE)
endif()

if(RENDERDOC_INCLUDE_PATH)
    if(NOT EXISTS "${RENDERDOC_INCLUDE_PATH}/renderdoc_app.h")
        message(FATAL_ERROR
            "RENDERDOC_INCLUDE_PATH is set to '${RENDERDOC_INCLUDE_PATH}' "
            "but renderdoc_app.h was not found there.")
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

target_compile_definitions(BindlessEngine PUBLIC
  SLANG_DISABLE_EXCEPTIONS=1
  GLM_FORCE_DEPTH_ZERO_TO_ONE
  GLFW_INCLUDE_NONE
  GLM_ENABLE_EXPERIMENTAL
  ${VOLK_PLATFORM_DEFINE}
)

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
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(BindlessApp PUBLIC
  BindlessEngine
  PRIVATE
    platform_wsi
)

if(HAS_TRACY)
  target_link_libraries(BindlessApp PRIVATE Tracy::TracyClient)
  target_compile_definitions(BindlessApp PRIVATE TRACY_ENABLE TRACY_VK_USE_SYMBOL_TABLE GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_ENABLE_EXPERIMENTAL)
endif()

DEFAULT_COMPILE_OPTIONS(BindlessApp)

add_executable(BindlessHeadless
  "src/main.cpp"
)

target_precompile_headers(BindlessHeadless PRIVATE PCH.hxx)

target_include_directories(BindlessHeadless PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(BindlessHeadless PRIVATE
  BindlessApp
  BaseApplication
  platform_wsi
  SceneLoader
)

DEFAULT_COMPILE_OPTIONS(BindlessHeadless)

# Tooling - SceneLoader.
add_library(SceneLoader STATIC
  "src/SceneLoader.cxx"
)

set_target_properties(BindlessHeadless PROPERTIES
  BUILD_RPATH   "${SLANG_ROOT}/lib"
  INSTALL_RPATH "${SLANG_ROOT}/lib"
  BUILD_RPATH_USE_ORIGIN ON   # emits $ORIGIN-relative paths, more portable
)
target_link_options(BindlessHeadless PRIVATE "-Wl,--disable-new-dtags")


target_include_directories(SceneLoader PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(SceneLoader PUBLIC
  glm::glm
  ktx
  expected
  fastgltf::fastgltf
  ThirdPartySTB
  volk::volk_headers
  meshoptimizer
)
DEFAULT_COMPILE_OPTIONS(SceneLoader)

add_executable(scene_inspect src/SceneInspector.cxx)
target_link_libraries(scene_inspect PRIVATE BindlessEngine)
target_include_directories(scene_inspect PRIVATE ${CMAKE_SOURCE_DIR}/include)

DEFAULT_COMPILE_OPTIONS(scene_inspect)

add_library(BaseApplication STATIC
  "src/framework/BaseApplication.cxx"
  "src/ArgumentParse.cxx"         # moved here from BindlessHeadless
)

target_precompile_headers(BaseApplication PRIVATE PCH.hxx)

target_include_directories(BaseApplication PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(BaseApplication PUBLIC
  BindlessEngine
  CLI11::CLI11               # moved here from BindlessHeadless
  platform_wsi
)

DEFAULT_COMPILE_OPTIONS(BaseApplication)

add_executable(ImageViewer "src/tools/ImageViewer.cxx")
target_link_libraries(ImageViewer PRIVATE BaseApplication SceneLoader)
DEFAULT_COMPILE_OPTIONS(ImageViewer)

cmake_push_check_state()
set(ASAN_FLAG "-fsanitize=address")
set(CMAKE_REQUIRED_FLAGS ${ASAN_FLAG})
check_c_compiler_flag(${ASAN_FLAG} C__fsanitize_address_VALID)
check_cxx_compiler_flag(${ASAN_FLAG} CXX__fsanitize_address_VALID)

if(NOT C__fsanitize_address_VALID OR NOT CXX__fsanitize_address_VALID)
  message(STATUS "ENABLE_ASAN was requested, but not supported!")
endif()

cmake_pop_check_state()

find_path(MagickPP_INCLUDE_DIR
  NAMES Magick++.h
  PATH_SUFFIXES ImageMagick-7 ImageMagick
)
 
find_library(MagickPP_LIBRARY
  NAMES Magick++-7.Q16HDRI
)
 
find_library(MagickCore_LIBRARY
  NAMES MagickCore-7.Q16HDRI
)
 
find_library(MagickWand_LIBRARY
  NAMES MagickWand-7.Q16HDRI
)
 
if(MagickPP_INCLUDE_DIR AND MagickPP_LIBRARY AND MagickCore_LIBRARY AND MagickWand_LIBRARY)
  add_library(ImageMagick::MagickCore IMPORTED SHARED)
  set_target_properties(ImageMagick::MagickCore PROPERTIES
    IMPORTED_LOCATION             "${MagickCore_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MagickPP_INCLUDE_DIR}"
  )
 
  add_library(ImageMagick::MagickWand IMPORTED SHARED)
  set_target_properties(ImageMagick::MagickWand PROPERTIES
    IMPORTED_LOCATION             "${MagickWand_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MagickPP_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "ImageMagick::MagickCore"
  )
 
  add_library(ImageMagick::Magick++ IMPORTED SHARED)
  set_target_properties(ImageMagick::Magick++ PROPERTIES
    IMPORTED_LOCATION             "${MagickPP_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MagickPP_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "ImageMagick::MagickWand;ImageMagick::MagickCore"
  )
 
  add_executable(tex_convert "src/tools/TextureConvert.cxx")
 
  target_include_directories(tex_convert PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
  )
 
  # ktx is already found by the rest of the build (used by BindlessEngine/SceneLoader)
  target_link_libraries(tex_convert PRIVATE
    ktx
    ImageMagick::Magick++
    CLI11::CLI11
    volk_headers
    volk
  )
 
  target_compile_definitions(tex_convert PRIVATE
    MAGICKCORE_QUANTUM_DEPTH=16
    MAGICKCORE_HDRI_ENABLE=1
  )
 
  DEFAULT_COMPILE_OPTIONS(tex_convert)
 
  message(STATUS "tex_convert enabled")
  message(STATUS "  include:    ${MagickPP_INCLUDE_DIR}")
  message(STATUS "  Magick++:   ${MagickPP_LIBRARY}")
  message(STATUS "  MagickCore: ${MagickCore_LIBRARY}")
  message(STATUS "  MagickWand: ${MagickWand_LIBRARY}")
else()
  message(STATUS "tex_convert disabled - set -DMAGICK_ROOT=/path/to/imagemagick")
  message(STATUS "  include:    ${MagickPP_INCLUDE_DIR}")
  message(STATUS "  Magick++:   ${MagickPP_LIBRARY}")
  message(STATUS "  MagickCore: ${MagickCore_LIBRARY}")
  message(STATUS "  MagickWand: ${MagickWand_LIBRARY}")
endif()

