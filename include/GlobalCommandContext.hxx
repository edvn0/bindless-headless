#pragma once

#include "Types.hxx"

#include <vulkan/vulkan.h>

struct GlobalCommandContext {
    VkDevice device{};
    VkCommandPool pool{};
    VkQueue queue{};
    VkSemaphore timeline{};
    u64 current_value{};
    u64 completed_value{};

    static constexpr u64 max_pending = 16;

    auto destroy() -> void {
        if (timeline) vkDestroySemaphore(device, timeline, nullptr);
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
        *this = {};
    }
};

inline auto create_global_cmd_context(
    VkDevice device,
    VkQueue queue,
    u32 family_index) {
    GlobalCommandContext ctx{};
    ctx.device = device;
    ctx.queue = queue;
    ctx.current_value = 0;
    ctx.completed_value = 0;

    VkSemaphoreTypeCreateInfo type_ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0
    };

    VkSemaphoreCreateInfo sci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &type_ci,
        .flags = 0
    };

    vk_check(vkCreateSemaphore(device, &sci, nullptr, &ctx.timeline));

    VkCommandPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = family_index,
    };
    vk_check(vkCreateCommandPool(device, &pci, nullptr, &ctx.pool));

    return ctx;
}

template<typename RecordFn>
auto submit_one_time_cmd(
    GlobalCommandContext &ctx,
    RecordFn &&record,
    bool wait_immediately = false) -> u64 {
    // Throttle if too many pending submissions
    if (ctx.current_value > ctx.completed_value + GlobalCommandContext::max_pending) {
        u64 wait_val = ctx.current_value - GlobalCommandContext::max_pending;
        VkSemaphoreWaitInfo wi{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext = nullptr,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &ctx.timeline,
            .pValues = &wait_val
        };
        vk_check(vkWaitSemaphores(ctx.device, &wi, UINT64_MAX));
        ctx.completed_value = wait_val;
    }

    VkCommandBufferAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = ctx.pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VkCommandBuffer cmd{};
    vk_check(vkAllocateCommandBuffers(ctx.device, &ai, &cmd));

    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr
    };
    vk_check(vkBeginCommandBuffer(cmd, &bi));

    record(cmd);

    vk_check(vkEndCommandBuffer(cmd));

    u64 signal_val = ctx.current_value + 1;

    VkTimelineSemaphoreSubmitInfo timeline_info{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount = 0,
        .pWaitSemaphoreValues = nullptr,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &signal_val
    };

    VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timeline_info,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &ctx.timeline
    };

    vk_check(vkQueueSubmit(ctx.queue, 1, &si, VK_NULL_HANDLE));
    ctx.current_value = signal_val;

    if (wait_immediately) {
        VkSemaphoreWaitInfo wi{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext = nullptr,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &ctx.timeline,
            .pValues = &signal_val
        };
        vk_check(vkWaitSemaphores(ctx.device, &wi, UINT64_MAX));
        ctx.completed_value = signal_val;
    }

    vkFreeCommandBuffers(ctx.device, ctx.pool, 1, &cmd);

    return signal_val;
}

inline auto wait_global_cmd_idle(GlobalCommandContext &ctx) -> void {
    if (ctx.current_value == 0) {
        return;
    }

    VkSemaphoreWaitInfo wi{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext = nullptr,
        .flags = 0,
        .semaphoreCount = 1,
        .pSemaphores = &ctx.timeline,
        .pValues = &ctx.current_value
    };
    vk_check(vkWaitSemaphores(ctx.device, &wi, UINT64_MAX));
    ctx.completed_value = ctx.current_value;
}

namespace destruction {
    inline auto global_command_context(GlobalCommandContext &ctx) -> void {
        ctx.destroy();
    }
}
