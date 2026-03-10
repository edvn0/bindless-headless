include_guard(GLOBAL)

function(DEFAULT_COMPILE_OPTIONS TARGET)
  if (MSVC)
    target_compile_options(${TARGET} PRIVATE /arch:AVX2)
    target_compile_definitions(${TARGET} PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _HAS_EXCEPTIONS=0
    )
  endif()

  target_compile_options(${TARGET} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
  )

  target_compile_definitions(${TARGET} PRIVATE
    IS_RELEASE=$<IF:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>,1,0>
  )
endfunction()
