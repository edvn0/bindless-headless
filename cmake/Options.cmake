# Options
option(HAS_TRACY "Enable Tracy integration" OFF)
option(HAS_HOT_RELOADING "Enable MSVC hot reloading" OFF)
option(ENGINE_OFFLINE_SHADERS "Compile shaders at build time (slangc) and load SPIR-V at runtime" OFF)

message(STATUS "HAS_TRACY = ${HAS_TRACY}")
message(STATUS "HAS_HOT_RELOADING = ${HAS_HOT_RELOADING}")
message(STATUS "ENGINE_OFFLINE_SHADERS = ${ENGINE_OFFLINE_SHADERS}")

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141 AND HAS_HOT_RELOADING)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
      "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,\
      $<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,\
      $<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif ()
