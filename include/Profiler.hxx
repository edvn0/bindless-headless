#pragma once

#include <volk.h>
#include "Types.hxx"

#if defined(TRACY_ENABLE)
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>
#else
// CPU
#define ZoneScoped
#define ZoneScopedN(x)
#define ZoneScopedNC(x, y)
#define FrameMark
#define TracyMessageL(x)
// GPU (compile out cleanly)
using TracyVkCtx = void *;
#define TracyVkContext(...) nullptr
#define TracyVkContextCalibrated(...) nullptr
#define TracyVkContextHostCalibrated(...) nullptr
#define TracyVkDestroy(x)                                                                                              \
    do {                                                                                                               \
    } while (0)
#define TracyVkContextName(ctx, name, size)                                                                            \
    do {                                                                                                               \
    } while (0)
#define TracyVkZone(ctx, cmdbuf, name)                                                                                 \
    do {                                                                                                               \
    } while (0)
#define TracyVkZoneC(ctx, cmdbuf, name, color)                                                                         \
    do {                                                                                                               \
    } while (0)
#define TracyVkCollect(ctx, cmdbuf)                                                                                    \
    do {                                                                                                               \
    } while (0)
#define TracyVkCollectHost(ctx)                                                                                        \
    do {                                                                                                               \
    } while (0)
#endif

#if defined(TRACY_ENABLE)
#define TRACY_GPU_ZONE(ctx_, cmd_, name_literal_) TracyVkZone((ctx_), (cmd_), name_literal_)
#define TRACY_GPU_COLLECT(ctx_, cmd_) TracyVkCollect((ctx_), (cmd_))
#else
#define TRACY_GPU_ZONE(ctx_, cmd_, name_literal_)                                                                      \
    do {                                                                                                               \
    } while (false)
#define TRACY_GPU_COLLECT(ctx_, cmd_)                                                                                  \
    do {                                                                                                               \
    } while (false)
#endif

struct TracyGpuContext {
#if defined(TRACY_ENABLE)
    VkDevice device{VK_NULL_HANDLE};
    VkCommandPool pool{VK_NULL_HANDLE};
    VkCommandBuffer init_cmd{VK_NULL_HANDLE};
    TracyVkCtx ctx{nullptr};
#endif

    auto init_calibrated(VkInstance, VkPhysicalDevice, VkDevice, VkQueue, u32, const char *) -> void;

    auto shutdown() -> void;
};
