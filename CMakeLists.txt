cmake_minimum_required(VERSION 3.31)
project(BindlessHeadless LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Your existing CPM bootstrap lives here
include(cmake/CPM.cmake)

include(cmake/Options.cmake)
include(cmake/ToolchainHints.cmake)     # optional sanity / messages
include(cmake/Dependencies.cmake)
include(cmake/Targets.cmake)
include(cmake/CompileOptions.cmake)
include(cmake/Shaders.cmake)
