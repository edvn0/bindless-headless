include_guard(GLOBAL)

target_compile_definitions(BindlessHeadlessAllocator PRIVATE VK_NO_PROTOTYPES)

target_compile_definitions(BindlessHeadless PRIVATE
  VK_NO_PROTOTYPES
  WIN32_LEAN_AND_MEAN
  NOMINMAX
  _HAS_EXCEPTIONS=0
  SLANG_DISABLE_EXCEPTIONS=1
  GLM_FORCE_DEPTH_ZERO_TO_ONE
  ${VOLK_PLATFORM_DEFINE}
)

if (MSVC)
  target_compile_options(BindlessHeadless PRIVATE /arch:AVX2)
endif()

target_compile_options(BindlessHeadless PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

target_compile_definitions(BindlessHeadless PRIVATE
  IS_RELEASE=$<IF:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,1,0>
)
