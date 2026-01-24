# Toolchain/platform hints used by deps and compile definitions

set(VOLK_PLATFORM_DEFINE)

if (WIN32)
  list(APPEND VOLK_PLATFORM_DEFINE VK_USE_PLATFORM_WIN32_KHR)
elseif(UNIX)
  # Keep your existing choice; adjust if you use Wayland/X11/XLIB etc
  list(APPEND VOLK_PLATFORM_DEFINE VK_USE_PLATFORM_XCB_KHR)
endif()

message(STATUS "VOLK platform defines: ${VOLK_PLATFORM_DEFINE}")
