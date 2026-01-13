cmake_minimum_required (VERSION 3.31)
# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("BindlessHeadless")

add_executable (BindlessHeadless "BindlessHeadless.cpp" "Pool.cxx" "ImageOperations.cxx")
add_library(BindlessHeadlessAllocator STATIC "allocator.cpp")

target_precompile_headers(BindlessHeadless PRIVATE PCH.hxx)

find_package(Vulkan REQUIRED)

# Add Slang libraries manually
set(SLANG_LIB_DIR "${Vulkan_INCLUDE_DIR}/../Lib")
set(SLANG_INCLUDE_DIR "${Vulkan_INCLUDE_DIR}/../Include")

# Find the Slang libraries
find_library(SLANG_LIBRARY NAMES slang PATHS ${SLANG_LIB_DIR} REQUIRED)
find_library(SLANG_COMPILER_LIBRARY NAMES slang-compiler PATHS ${SLANG_LIB_DIR} REQUIRED)
find_library(SLANG_RT_LIBRARY NAMES slang-rt PATHS ${SLANG_LIB_DIR} REQUIRED)

find_package(FastNoise2 CONFIG REQUIRED)

target_link_libraries(BindlessHeadlessAllocator PRIVATE Vulkan::Vulkan PUBLIC FastNoise2::FastNoise)
target_link_libraries(BindlessHeadless PRIVATE
  Vulkan::Vulkan
  BindlessHeadlessAllocator
  ${SLANG_LIBRARY}
  ${SLANG_COMPILER_LIBRARY}
  ${SLANG_RT_LIBRARY}
)

# Add Slang include directory
target_include_directories(BindlessHeadless PRIVATE ${SLANG_INCLUDE_DIR})

target_compile_definitions(BindlessHeadless PRIVATE _HAS_EXCEPTIONS=0 SLANG_DISABLE_EXCEPTIONS=1)
target_compile_options(BindlessHeadless PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

target_compile_definitions(BindlessHeadless PRIVATE
  IS_RELEASE=$<IF:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,1,0>
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_property(TARGET BindlessHeadless PROPERTY CXX_STANDARD 23)
endif()

add_custom_command(TARGET BindlessHeadless POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:BindlessHeadless>/shaders
  COMMENT "Copying shaders directory to output"
)
