include_guard(GLOBAL)

add_executable(BindlessHeadless
  "src/main.cpp"
  "src/ArgumentParse.cxx"
  "src/BindlessHeadless.cpp"
  "src/RenderContext.cxx"
  "src/Types.cxx"
  "src/Profiler.cxx"
  "src/Pipelines.cxx"
  "src/ResizeableGraph.cxx"
  "src/Swapchain.cxx"
  "src/ImageOperations.cxx"
  "src/Buffer.cxx"
  "src/Logger.cxx"
  "src/Compiler.cxx"
  "src/GlobalCommandContext.cxx"
)


add_library(BindlessHeadlessAllocator STATIC "src/allocator.cpp")

target_precompile_headers(BindlessHeadless PRIVATE PCH.hxx)

target_include_directories(BindlessHeadless PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

# Link base deps
target_link_libraries(BindlessHeadlessAllocator PRIVATE
  volk
  volk::volk_headers
  VulkanMemoryAllocator
)

target_link_libraries(BindlessHeadless PRIVATE
  volk
  volk::volk_headers
  BindlessHeadlessAllocator
  spdlog::spdlog
  efsw-static
  PUBLIC
  glm::glm
  CLI11::CLI11
  glfw
  expected 
)

# Slang runtime deps only when runtime path
if (ENGINE_OFFLINE_SHADERS)
  target_compile_definitions(BindlessHeadless PRIVATE ENGINE_OFFLINE_SHADERS=1)
else()
  target_compile_definitions(BindlessHeadless PRIVATE ENGINE_RUNTIME_SHADERS=1)
  target_sources(BindlessHeadless PRIVATE "src/Reflection.cxx")
  target_include_directories(BindlessHeadless PRIVATE ${SLANG_INCLUDE_DIR})
  target_link_libraries(BindlessHeadless PRIVATE slang::slang slang-compiler slang-rt)
endif()

set_property(TARGET BindlessHeadless PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

if (HAS_TRACY)
  target_link_libraries(BindlessHeadless PRIVATE Tracy::TracyClient)
  target_compile_definitions(BindlessHeadless PRIVATE TRACY_ENABLE)
endif ()
