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
  target_link_scene_compression(${TARGET} PUBLIC)
endfunction()

function(target_link_scene_compression target scope)
    if(SCENE_COMPRESSION STREQUAL "bzip2")
        target_link_libraries(${target} ${scope} BZip2::BZip2)
        target_compile_definitions(${target} ${scope} SCENE_COMPRESSION_BZIP2)
    elseif(SCENE_COMPRESSION STREQUAL "zstd")
        target_link_libraries(${target} ${scope}
            $<IF:$<TARGET_EXISTS:zstd::libzstd_static>,zstd::libzstd_static,zstd::libzstd_shared>)
        target_compile_definitions(${target} ${scope} SCENE_COMPRESSION_ZSTD)
    elseif(SCENE_COMPRESSION STREQUAL "lz4")
        target_link_libraries(${target} ${scope} LZ4::lz4_static)
        target_compile_definitions(${target} ${scope} SCENE_COMPRESSION_LZ4)
    else()
        message(FATAL_ERROR "Unknown SCENE_COMPRESSION value: ${SCENE_COMPRESSION}. Must be bzip2, zstd or lz4.")
    endif()
endfunction()