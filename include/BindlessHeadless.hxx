#pragma once

#include "Forward.hxx"
#include "GlobalCommandContext.hxx"
#include "Logger.hxx"
#include "Pool.hxx"
#include "Reflection.hxx"
#include "Types.hxx"


#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <expected>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>


#include <vma/vk_mem_alloc.h>

constexpr u32 frames_in_flight = 3; // renderer-side DAG cycle
constexpr u32 max_in_flight = 2; // GPU submit throttle depth


namespace detail {
    auto initialise_debug_name_func(VkInstance) -> void;

    auto set_debug_name_impl(VmaAllocator &, VkObjectType, u64, std::string_view) -> void;
    auto set_debug_name_impl(VkDevice &, VkObjectType, u64, std::string_view) -> void;

    auto submit_and_wait(VkDevice device, VkCommandPool cmd_pool, VkQueue queue, auto &&record) -> void {
        VkCommandBufferAllocateInfo ai{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = cmd_pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1};

        VkCommandBuffer cb{};
        vk_check(vkAllocateCommandBuffers(device, &ai, &cb));

        VkCommandBufferBeginInfo bi{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vk_check(vkBeginCommandBuffer(cb, &bi));

        record(cb);

        vk_check(vkEndCommandBuffer(cb));

        VkSubmitInfo si{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};

        VkFenceCreateInfo fci{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};

        VkFence fence{};
        vk_check(vkCreateFence(device, &fci, nullptr, &fence));

        vk_check(vkQueueSubmit(queue, 1, &si, fence));
        vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmd_pool, 1, &cb);
    }
} // namespace detail

template<typename T>
    requires std::is_pointer_v<T>
auto set_debug_name(VmaAllocator &alloc, VkObjectType t, const T &obj, std::string_view name) -> void {
    detail::set_debug_name_impl(alloc, t, reinterpret_cast<u64>(obj), name);
}

template<typename T>
    requires std::is_pointer_v<T>
auto set_debug_name(VkDevice &dev, VkObjectType t, const T &obj, std::string_view name) -> void {
    detail::set_debug_name_impl(dev, t, reinterpret_cast<u64>(obj), name);
}

enum class Stage : u32 {
    LightCulling = 0,
    GBuffer = 1,
    Predepth = 2,
    Tonemapping = 3,
};

constexpr auto stage_count = static_cast<u32>(Stage::Tonemapping) + 1;

struct FrameState {
    std::array<u64, stage_count> timeline_values{};
    u64 frame_done_value{0}; // This should only be set by the *final* operation in the frame.
};

inline auto stage_index(Stage s) -> std::size_t { return static_cast<std::size_t>(s); }

template<u32 SubmitsPerFrame>
struct Timeline {
    static constexpr u32 submits_per_frame = SubmitsPerFrame;
    static constexpr u32 buffered = submits_per_frame * frames_in_flight;

    VkQueue queue{};
    u32 family_index{};

    VkSemaphore timeline{};
    u64 value{};
    u64 completed{};

    VkCommandPool pool{};
    std::array<VkCommandBuffer, buffered> cmds{};
    std::array<u64, buffered> slot_last_signal{};

    auto destroy(VkDevice device) -> void {
        if (timeline)
            vkDestroySemaphore(device, timeline, nullptr);
        if (pool)
            vkDestroyCommandPool(device, pool, nullptr);
        *this = {};
    }
};

using GraphicsTimeline = Timeline<4>;
using ComputeTimeline = Timeline<1>;

auto create_compute_timeline(VkDevice device, VkQueue queue, u32 family_index) -> ComputeTimeline;
auto create_graphics_timeline(VkDevice device, VkQueue queue, u32 family_index) -> GraphicsTimeline;

auto create_sampler(VmaAllocator &alloc, VkSamplerCreateInfo ci, std::string_view name) -> VkSampler;

auto create_offscreen_target(VmaAllocator alloc, u32 width, u32 height, VkFormat format,
                             std::string_view name = "Empty") -> OffscreenTarget;

auto create_depth_target(VmaAllocator alloc, u32 width, u32 height, VkFormat format, std::string_view name)
        -> OffscreenTarget;

auto create_image_from_span_v2(VmaAllocator alloc, GlobalCommandContext &cmd_ctx, std::uint32_t width,
                               std::uint32_t height, VkFormat format, std::span<const std::uint8_t> data,
                               std::string_view name) -> OffscreenTarget;

struct InstanceWithDebug {
    VkInstance instance{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT messenger{VK_NULL_HANDLE};
};

auto create_instance_with_debug(auto &callback, std::span<const std::string_view> surface_required_extensions,
                                bool is_release) -> InstanceWithDebug {
    VkApplicationInfo app_info{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                               .pNext = nullptr,
                               .pApplicationName = "HeadlessBindless",
                               .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                               .pEngineName = "None",
                               .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                               .apiVersion = VK_API_VERSION_1_3};

    std::array<const char *, 1> enabled_layers = {"VK_LAYER_KHRONOS_validation"};

    std::vector<const char *> enabled_extensions;

    for (const auto &required_extension: surface_required_extensions) {
        enabled_extensions.push_back(required_extension.data());
    }

    bool has_debug_utils = false;
    if (!is_release) {
        u32 ext_count{};
        vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> extensions(ext_count);
        vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, extensions.data());

        for (const auto &ext: extensions) {
            if (std::strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
                has_debug_utils = true;
                break;
            }
        }

        if (has_debug_utils) {
            enabled_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }

    info("Validation layers status: '{}'", has_debug_utils ? "Enabled" : "Disabled");

    VkInstanceCreateInfo create_info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                     .pNext = nullptr,
                                     .flags = 0,
                                     .pApplicationInfo = &app_info,
                                     .enabledLayerCount = is_release ? 0 : static_cast<u32>(enabled_layers.size()),
                                     .ppEnabledLayerNames = enabled_layers.data(),
                                     .enabledExtensionCount = static_cast<u32>(enabled_extensions.size()),
                                     .ppEnabledExtensionNames = enabled_extensions.data()};

    InstanceWithDebug result{};
    vk_check(vkCreateInstance(&create_info, nullptr, &result.instance));

    if (has_debug_utils) {
        VkDebugUtilsMessengerCreateInfoEXT debug_ci{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                                                    .pNext = nullptr,
                                                    .flags = 0,
                                                    .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                                                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                                                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                                                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                                                    .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                                                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                                                   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                                                    .pfnUserCallback = &callback,
                                                    .pUserData = nullptr};

        auto create_debug = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(result.instance, "vkCreateDebugUtilsMessengerEXT"));

        if (create_debug) {
            vk_check(create_debug(result.instance, &debug_ci, nullptr, &result.messenger));
        }
    }

    detail::initialise_debug_name_func(result.instance);

    return result;
}

struct PhysicalDeviceChoice {
    enum class Error { NoDevicesFound, NoQueuesFound };

    Error error;
};

using DeviceChoice = std::tuple<VkPhysicalDevice, u32, u32>;

auto pick_physical_device(VkInstance instance) -> std::expected<DeviceChoice, PhysicalDeviceChoice>;

enum class GpuStamp : u32 { Begin = 0, End = 1, Count = 2 };

inline constexpr u32 query_count = static_cast<u32>(GpuStamp::Count);

auto create_device(VkPhysicalDevice pd, u32 graphics_index, u32 compute_index)
        -> std::tuple<VkDevice, VkQueue, VkQueue>;

auto create_allocator(VkInstance instance, VkPhysicalDevice pd, VkDevice device) -> VmaAllocator;

struct TimelineWait {
    u64 value = 0;
    VkSemaphore semaphore{VK_NULL_HANDLE};
    VkPipelineStageFlags stage{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
};

struct BinaryWait {
    VkSemaphore semaphore{VK_NULL_HANDLE};
    VkPipelineStageFlags stage{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
};

struct SubmitSynchronisation {
    std::span<const TimelineWait> timeline_waits{};
    std::span<const BinaryWait> binary_waits{};
    std::span<const VkSemaphore> binary_signals{};
};
inline constexpr auto no_waits = SubmitSynchronisation{{}, {}, {}};

template<typename TL, typename RecordFn>
auto submit_stage(TL &tl, VkDevice device, RecordFn &&record, SubmitSynchronisation sync = {}) -> u64 {
    const u32 index = static_cast<u32>(tl.value % TL::buffered);
    VkCommandBuffer cmd = tl.cmds[index];

    const u64 last = tl.slot_last_signal[index];
    if (last != 0) {
        VkSemaphoreWaitInfo wi{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                               .semaphoreCount = 1,
                               .pSemaphores = &tl.timeline,
                               .pValues = &last};
        vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));
        tl.completed = std::max(tl.completed, last);
    }

    vk_check(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo bi{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = 0};
    vk_check(vkBeginCommandBuffer(cmd, &bi));

    record(cmd);

    vk_check(vkEndCommandBuffer(cmd));

    const u64 signal_val = tl.value + 1;

    const u32 bwc = static_cast<u32>(sync.binary_waits.size());
    const u32 twc = static_cast<u32>(sync.timeline_waits.size());
    const u32 total_waits = bwc + twc;

    std::vector<VkSemaphore> wait_sems;
    std::vector<VkPipelineStageFlags> wait_stages;
    std::vector<u64> wait_values;

    wait_sems.reserve(total_waits);
    wait_stages.reserve(total_waits);
    wait_values.reserve(total_waits);

    for (auto &&w: sync.binary_waits) {
        wait_sems.push_back(w.semaphore);
        wait_stages.push_back(w.stage);
        wait_values.push_back(0);
    }

    for (auto &&w: sync.timeline_waits) {
        wait_sems.push_back(w.semaphore);
        wait_stages.push_back(w.stage);
        wait_values.push_back(w.value);
    }

    std::vector<VkSemaphore> signal_sems;
    std::vector<u64> signal_values;

    signal_sems.reserve(1 + sync.binary_signals.size());
    signal_values.reserve(1 + sync.binary_signals.size());

    signal_sems.push_back(tl.timeline);
    signal_values.push_back(signal_val);

    for (VkSemaphore s: sync.binary_signals) {
        signal_sems.push_back(s);
        signal_values.push_back(0);
    }

    VkTimelineSemaphoreSubmitInfo timeline_info{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = static_cast<u32>(wait_values.size()),
            .pWaitSemaphoreValues = wait_values.empty() ? nullptr : wait_values.data(),
            .signalSemaphoreValueCount = static_cast<u32>(signal_values.size()),
            .pSignalSemaphoreValues = signal_values.empty() ? nullptr : signal_values.data()};

    VkSubmitInfo si{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .pNext = &timeline_info,
                    .waitSemaphoreCount = static_cast<u32>(wait_sems.size()),
                    .pWaitSemaphores = wait_sems.empty() ? nullptr : wait_sems.data(),
                    .pWaitDstStageMask = wait_stages.empty() ? nullptr : wait_stages.data(),
                    .commandBufferCount = 1,
                    .pCommandBuffers = &cmd,
                    .signalSemaphoreCount = static_cast<u32>(signal_sems.size()),
                    .pSignalSemaphores = signal_sems.data()};

    vk_check(vkQueueSubmit(tl.queue, 1, &si, VK_NULL_HANDLE));

    tl.slot_last_signal[index] = signal_val;
    tl.value = signal_val;
    return signal_val;
}

auto throttle(GraphicsTimeline &, VkDevice device) -> void;
auto throttle(ComputeTimeline &, VkDevice device) -> void;

namespace destruction {
    auto instance(InstanceWithDebug const &inst) -> void;

    inline auto wsi(VkInstance &inst, VkSurfaceKHR &surf, GLFWwindow *win) -> void {
        vkDestroySurfaceKHR(inst, surf, nullptr);
        glfwDestroyWindow(win);
        glfwTerminate();
    }

    auto device(VkDevice &dev) -> void;

    auto bindless_set(VkDevice device, BindlessSet &bs) -> void;

    auto allocator(VmaAllocator &alloc) -> void;

    auto timeline_compute(VkDevice device, GraphicsTimeline &comp) -> void;
    auto timeline_compute(VkDevice device, ComputeTimeline &comp) -> void;

    template<typename T>
    concept PipelineProvider = requires(T t) {
        { t.pipeline } -> std::same_as<VkPipeline &>;
        { t.layout } -> std::same_as<VkPipelineLayout &>;
    } || requires(T t) {
        { std::get<0>(t) } -> std::same_as<VkPipeline &>;
        { std::get<1>(t) } -> std::same_as<VkPipelineLayout &>;
    };

    template<PipelineProvider T>
    auto as_pipeline_refs(T &t) -> std::pair<VkPipeline &, VkPipelineLayout &> {
        if constexpr (requires {
                          t.pipeline;
                          t.layout;
                      }) {
            return {t.pipeline, t.layout};
        } else {
            return {std::get<0>(t), std::get<1>(t)};
        }
    }

    auto pipeline(VkDevice dev, VkPipeline &, VkPipelineLayout &) -> void;

    auto pipeline(VkDevice dev, PipelineProvider auto &val) -> void {
        auto [p, l] = as_pipeline_refs(val);
        destruction::pipeline(dev, p, l);
    }

    template<typename... Ts>
        requires(PipelineProvider<std::remove_reference_t<Ts>> && ...)
    auto pipeline(VkDevice dev, Ts &&...vals) -> void {
        (
                [&] {
                    auto &v = static_cast<std::remove_reference_t<Ts> &>(vals);
                    auto [p, l] = as_pipeline_refs(v);
                    destruction::pipeline(dev, p, l);
                }(),
                ...);
    }
} // namespace destruction
