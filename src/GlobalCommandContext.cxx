#include "GlobalCommandContext.hxx"

#include "BindlessHeadless.hxx"

auto create_global_cmd_context(
VkDevice device,
VkQueue queue,
u32 family_index) -> GlobalCommandContext {
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
    set_debug_name(device, VK_OBJECT_TYPE_SEMAPHORE, ctx.timeline, "one-time-submit-timeline");

    VkCommandPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = family_index,
    };
    vk_check(vkCreateCommandPool(device, &pci, nullptr, &ctx.pool));

    return ctx;
}
