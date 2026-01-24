include_guard(GLOBAL)

if (ENGINE_OFFLINE_SHADERS)
  find_program(SLANGC_EXECUTABLE slangc REQUIRED)

  set(SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders")
  set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders_spv")
  file(MAKE_DIRECTORY "${SHADER_OUT_DIR}")

  # Explicit shader list
  set(SHADER_LIGHT_CULL   "${SHADER_SRC_DIR}/light_cull_compact_modern.slang")
  set(SHADER_POINT_LIGHT  "${SHADER_SRC_DIR}/point_light.slang")
  set(SHADER_PREDEPTH     "${SHADER_SRC_DIR}/predepth.slang")
  set(SHADER_TONEMAP      "${SHADER_SRC_DIR}/tonemap.slang")

  # Entry points (must match your C++ expectations)
  set(LIGHT_CULL_ENTRIES    LightFlagsCS LightCompactCS)
  set(POINT_LIGHT_ENTRIES   main_vs main_fs)
  set(PREDEPTH_ENTRIES      main_vs main_fs)
  set(TONEMAP_ENTRIES       vs_main fs_main)

  function(add_slang_spv slang_file)
    set(options)
    set(oneValueArgs OUT_VAR)
    set(multiValueArgs ENTRIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    get_filename_component(base "${slang_file}" NAME_WE)
    set(outputs "")

    foreach(entry IN LISTS ARG_ENTRIES)
      set(out_spv "${SHADER_OUT_DIR}/${base}__${entry}.spv")
      list(APPEND outputs "${out_spv}")

      add_custom_command(
        OUTPUT "${out_spv}"
        COMMAND "${SLANGC_EXECUTABLE}"
                "${slang_file}"
                -target spirv
                -profile glsl_460
                -fvk-use-entrypoint-name
                -entry "${entry}"
                -o "${out_spv}"
        DEPENDS "${slang_file}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        VERBATIM
      )
    endforeach()

    set(${ARG_OUT_VAR} "${outputs}" PARENT_SCOPE)
  endfunction()

  add_slang_spv("${SHADER_LIGHT_CULL}"  OUT_VAR LIGHT_CULL_SPV  ENTRIES ${LIGHT_CULL_ENTRIES})
  add_slang_spv("${SHADER_POINT_LIGHT}" OUT_VAR POINT_LIGHT_SPV ENTRIES ${POINT_LIGHT_ENTRIES})
  add_slang_spv("${SHADER_PREDEPTH}"    OUT_VAR PREDEPTH_SPV    ENTRIES ${PREDEPTH_ENTRIES})
  add_slang_spv("${SHADER_TONEMAP}"     OUT_VAR TONEMAP_SPV     ENTRIES ${TONEMAP_ENTRIES})

  add_custom_target(BindlessHeadlessShaders ALL
    DEPENDS ${LIGHT_CULL_SPV} ${POINT_LIGHT_SPV} ${PREDEPTH_SPV} ${TONEMAP_SPV}
  )
  add_dependencies(BindlessHeadless BindlessHeadlessShaders)

  add_custom_command(TARGET BindlessHeadless POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:BindlessHeadless>/shaders_spv"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${SHADER_OUT_DIR}"
            "$<TARGET_FILE_DIR:BindlessHeadless>/shaders_spv"
    COMMENT "Copying compiled SPIR-V shaders to output"
  )

else()
  add_custom_command(TARGET BindlessHeadless POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/shaders
            $<TARGET_FILE_DIR:BindlessHeadless>/shaders
    COMMENT "Copying shaders directory to output"
  )
endif()
