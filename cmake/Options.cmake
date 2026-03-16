# Options
option(HAS_TRACY "Enable Tracy integration" OFF)
option(HAS_HOT_RELOADING "Enable MSVC hot reloading" OFF)
option(HAS_IMAGE_WRITERS "Enable writing images to output" ON)
set(RENDERDOC_INCLUDE_PATH "" CACHE PATH "Path to directory containing renderdoc_app.h")

message(STATUS "HAS_TRACY = ${HAS_TRACY}")
message(STATUS "HAS_HOT_RELOADING = ${HAS_HOT_RELOADING}")
message(STATUS "HAS_IMAGE_WRITERS = ${HAS_IMAGE_WRITERS}")

# Enable Hot Reload for MSVC compilers if supported.
if(POLICY CMP0141 AND HAS_HOT_RELOADING)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
    "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,\
      $<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,\
      $<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()
