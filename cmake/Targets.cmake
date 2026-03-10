include_guard(GLOBAL)

include(cmake/CompileOptions.cmake)

include(CMakePushCheckState)
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

find_package(BZip2 REQUIRED)

set(HAS_X11 OFF)
if(UNIX AND NOT APPLE)
  find_package(X11 QUIET)
  if(X11_FOUND)
    set(HAS_X11 ON)
  endif()
endif()

add_library(platform_wsi INTERFACE)

if(HAS_X11)
  target_link_libraries(platform_wsi INTERFACE X11::X11)
  target_compile_definitions(platform_wsi INTERFACE
    VK_USE_PLATFORM_XCB_KHR
    GLFW_HAS_X11=1
    GLFW_HAS_WAYLAND=0
  )
elseif(WIN32)
  target_compile_definitions(platform_wsi INTERFACE
    VK_USE_PLATFORM_WIN32_KHR
  )
endif()

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

  "src/scene/Scene.cxx"
)

target_precompile_headers(BindlessEngine PRIVATE PCH.hxx)

target_include_directories(BindlessEngine PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
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
  BZip2::BZip2
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
  BZip2::BZip2
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
