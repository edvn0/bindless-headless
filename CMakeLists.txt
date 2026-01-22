cmake_minimum_required(VERSION 3.31)

set(CMAKE_CXX_STANDARD 23)


# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141 AND HAS_HOT_RELOADING)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif ()

project("BindlessHeadless")

option(HAS_RENDERDOC "Enable RenderDoc integration" OFF)
option(HAS_TRACY "Enable Tracy integration" OFF)
option(HAS_HOT_RELOADING "Enable MSVC hot reloading" OFF)

include(cmake/CPM.cmake)

# CPMAddPackage("gh:nlohmann/json@3.10.5")
CPMAddPackage(
        URI "gh:g-truc/glm#1.0.3"
        OPTIONS "GLM_ENABLE_CXX_20 ON" "GLM_ENABLE_SIMD_AVX2 ON" "GLM_ENABLE_SIMD_AVX ON" "GLM_ENABLE_SIMD_SSE4_2 ON" "GLM_ENABLE_SIMD_SSE2 ON"
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
  NAME Vulkan-Headers
  GITHUB_REPOSITORY KhronosGroup/Vulkan-Headers
  GIT_TAG vulkan-sdk-1.4.335.0
)
get_target_property(VULKAN_HEADERS_INCLUDE Vulkan::Headers INTERFACE_INCLUDE_DIRECTORIES)

set(VULKAN_HEADERS_INSTALL_DIR "${VULKAN_HEADERS_INCLUDE}" CACHE PATH "" FORCE)
set(VOLK_PLATFORM_DEFINE)

if (WIN32)
  list(APPEND VOLK_PLATFORM_DEFINE VK_USE_PLATFORM_WIN32_KHR)
elseif(UNIX)
  list(APPEND VOLK_PLATFORM_DEFINE VK_USE_PLATFORM_XCB_KHR)
endif()

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

if (HAS_TRACY)
    set(TRACY_ENABLE ON CACHE BOOL "Enable Tracy profiler" FORCE)

    CPMAddPackage(
        URI "gh:wolfpld/tracy#07147111b26ddaf43fb46fabbab42de4451fa567"
    )
endif ()

message(STATUS "HAS_TRACY = ${HAS_TRACY}")
message(STATUS "TRACY_ENABLE = ${TRACY_ENABLE}")

add_executable(BindlessHeadless "src/main.cpp" "src/BindlessHeadless.cpp"
        "src/Pool.cxx" "src/Types.cxx"
        "src/Profiler.cxx" "src/Pipelines.cxx"
        "src/ResizeableGraph.cxx"
        "src/Swapchain.cxx" "src/ImageOperations.cxx" "src/Buffer.cxx" "src/Logger.cxx" "src/Compiler.cxx" "src/GlobalCommandContext.cxx")
add_library(BindlessHeadlessAllocator STATIC "src/allocator.cpp")

target_precompile_headers(BindlessHeadless PRIVATE PCH.hxx)

find_package(spdlog REQUIRED)
find_package(Slang CONFIG REQUIRED)
set(SLANG_LIB_DIR "${SLANG_ROOT}/lib")
set(SLANG_INCLUDE_DIR "${SLANG_ROOT}/include")

# Create imported static library target for the compiler
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


target_include_directories(BindlessHeadless PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include ${SLANG_INCLUDE_DIR})
target_link_libraries(BindlessHeadlessAllocator PRIVATE volk volk::volk_headers VulkanMemoryAllocator)
target_link_libraries(BindlessHeadless PRIVATE
        volk 
        volk::volk_headers
        BindlessHeadlessAllocator
        slang::slang
  slang-compiler
  slang-rt
        spdlog::spdlog
        PUBLIC
        glm::glm
        CLI11::CLI11
        glfw
)

set_property(TARGET BindlessHeadless PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)


if (HAS_RENDERDOC)
    target_compile_definitions(BindlessHeadless PRIVATE HAS_RENDERDOC=1)
endif ()

if (HAS_TRACY)
    target_link_libraries(BindlessHeadless PRIVATE Tracy::TracyClient)
    target_compile_definitions(BindlessHeadless PRIVATE TRACY_ENABLE)
endif ()

target_compile_definitions(BindlessHeadlessAllocator PRIVATE VK_NO_PROTOTYPES)
target_compile_definitions(BindlessHeadless PRIVATE VK_NO_PROTOTYPES WIN32_LEAN_AND_MEAN NOMINMAX _HAS_EXCEPTIONS=0 SLANG_DISABLE_EXCEPTIONS=1 GLM_FORCE_DEPTH_ZERO_TO_ONE ${VOLK_PLATFORM_DEFINE})
if(MSVC)
    target_compile_options(BindlessHeadless PRIVATE /arch:AVX2)
endif()
target_compile_options(BindlessHeadless PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

target_compile_definitions(BindlessHeadless PRIVATE
        IS_RELEASE=$<IF:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,1,0>
)

add_custom_command(TARGET BindlessHeadless POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:BindlessHeadless>/shaders
        COMMENT "Copying shaders directory to output"
)
