#pragma once

#include "Forward.hxx"
#include "GlobalCommandContext.hxx"
#include "Logger.hxx"
#include "Pool.hxx"
#include "Reflection.hxx"
#include "Types.hxx"

#include <bitset>
#include <volk.h>

#include <GLFW/glfw3.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>


#include <tl/expected.hpp>
#include <vk_mem_alloc.h>

#include "CreateInfo.hxx"


namespace detail {
    auto initialise_debug_name_func(VkInstance) -> void;

    auto set_debug_name_impl(VmaAllocator &, VkObjectType, u64, std::string_view) -> void;
    auto set_debug_name_impl(VkDevice, VkObjectType, u64, std::string_view) -> void;

    auto submit_and_wait(VkDevice device, VkCommandPool cmd_pool, VkQueue queue, auto &&record) -> void {
        VkCommandBufferAllocateInfo ai{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .pNext = nullptr,
                                       .commandPool = cmd_pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1};

        VkCommandBuffer cb{};
        vk_check(vkAllocateCommandBuffers(device, &ai, &cb));

        VkCommandBufferBeginInfo bi{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .pNext = nullptr,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                    .pInheritanceInfo = nullptr};
        vk_check(vkBeginCommandBuffer(cb, &bi));

        record(cb);

        vk_check(vkEndCommandBuffer(cb));

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;

        auto fci = create_info<VkFenceCreateInfo>();

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
auto set_debug_name(VkDevice dev, VkObjectType t, const T &obj, std::string_view name) -> void {
    detail::set_debug_name_impl(dev, t, reinterpret_cast<u64>(obj), name);
}

enum class Stage : u32 {
    LightCulling = 0,
    GBuffer = 1,
    Predepth = 2,
    Tonemapping = 3,
    CubeRotation = 4,
    DeferredLighting = 5,
    LightClustering = 6,
    UI = 7,
    Count,
};

constexpr auto stage_count = static_cast<u32>(Stage::Count);

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
using ComputeTimeline = Timeline<3>;
using TransferTimeline = Timeline<1>;

auto create_compute_timeline(VkDevice, VkQueue, u32) -> ComputeTimeline;
auto create_graphics_timeline(VkDevice, VkQueue, u32) -> GraphicsTimeline;
auto create_transfer_timeline(VkDevice, VkQueue, u32) -> TransferTimeline;

auto create_sampler(VmaAllocator &alloc, VkSamplerCreateInfo ci, std::string_view name) -> VkSampler;

auto pipeline_cache_path() -> std::optional<std::filesystem::path>;

inline auto pick_msaa_samples(VkPhysicalDevice physical_device) -> VkSampleCountFlagBits {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device, &props);

    const VkSampleCountFlags counts =
            props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;

    if (counts & VK_SAMPLE_COUNT_8_BIT)
        return VK_SAMPLE_COUNT_8_BIT;
    if (counts & VK_SAMPLE_COUNT_4_BIT)
        return VK_SAMPLE_COUNT_4_BIT;
    if (counts & VK_SAMPLE_COUNT_2_BIT)
        return VK_SAMPLE_COUNT_2_BIT;
    return VK_SAMPLE_COUNT_1_BIT;
}

// By default, sets WANT_SAMPLED and WANT_STORAGE and WANT_TRANSFER.
struct TargetSamplerConfiguration {
    std::bitset<3> sampled_storage_transfer{0b111};

    struct Dimensions {
        u32 mip_levels{1};
        u32 array_layers{1};
        VkImageViewType view_type{VK_IMAGE_VIEW_TYPE_2D}; // lets you opt into 2D_ARRAY later
    };

    Dimensions dims{};
};

auto is_block_compressed_format(VkFormat) -> bool;
auto create_texture_image_v2(VmaAllocator, GlobalCommandContext &, u32, u32, VkFormat, std::span<const u8>,
                             std::span<const u32>, std::span<const u32>, std::string_view) -> OffscreenTarget;
auto create_offscreen_target(VmaAllocator &alloc, u32 width, u32 height, VkFormat format, VkSampleCountFlagBits samples,
                             TargetSamplerConfiguration config, std::string_view name) -> OffscreenTarget;
inline auto create_offscreen_target(VmaAllocator &alloc, u32 width, u32 height, VkFormat format,
                                    TargetSamplerConfiguration config, std::string_view name) -> OffscreenTarget {
    return create_offscreen_target(alloc, width, height, format, VK_SAMPLE_COUNT_1_BIT, std::move(config), name);
}

auto create_depth_target(VmaAllocator &alloc, u32 width, u32 height, VkFormat format, VkSampleCountFlagBits samples,
                         bool want_sampled, // usually true only for single-sample depth you intend to sample later
                         std::string_view name) -> OffscreenTarget;
inline auto create_depth_target(VmaAllocator &alloc, u32 width, u32 height, VkFormat format, std::string_view name)
        -> OffscreenTarget {
    return create_depth_target(alloc, width, height, format, VK_SAMPLE_COUNT_1_BIT, true, name);
}

auto create_image_from_mips_v2(VmaAllocator alloc, GlobalCommandContext &cmd_ctx, u32 width, u32 height,
                               VkFormat format, std::span<const std::byte> data, std::span<const u32> mip_offsets,
                               std::span<const u32> mip_sizes, std::string_view name) -> OffscreenTarget;
auto create_image_from_mips_v2(VmaAllocator alloc, GlobalCommandContext &cmd_ctx, u32 width, u32 height,
                               VkFormat format, std::span<const u8> data, std::span<const u32> mip_offsets,
                               std::span<const u32> mip_sizes, std::string_view name) -> OffscreenTarget;
auto create_image_from_span_v2(VmaAllocator alloc, GlobalCommandContext &cmd_ctx, std::uint32_t width,
                               std::uint32_t height, VkFormat format, std::span<const std::uint8_t> data,
                               std::string_view name) -> OffscreenTarget;
auto create_image_from_span_v2(VmaAllocator alloc, GlobalCommandContext &cmd_ctx, std::uint32_t width,
                               std::uint32_t height, VkFormat format, std::span<const std::byte> data,
                               std::string_view name) -> OffscreenTarget;

struct InstanceWithDebug {
    VkInstance instance{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT messenger{VK_NULL_HANDLE};

    explicit(false) operator VkInstance() const { return instance; }
};

inline auto create_instance(std::span<const std::string_view> surface_required_extensions) -> VkInstance {
    vk_check(volkInitialize());

    auto app_info = create_info<VkApplicationInfo>();
    app_info.pApplicationName = "HeadlessBindless";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "None";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_4;

    std::vector<const char *> enabled_extensions;
    for (const auto &required_extension: surface_required_extensions) {
        enabled_extensions.push_back(required_extension.data());
    }

    info("Validation layers status: 'Disabled");

    auto instance_ci = create_info<VkInstanceCreateInfo>();
    instance_ci.pApplicationInfo = &app_info;
    instance_ci.enabledLayerCount = 0;
    instance_ci.ppEnabledLayerNames = nullptr;
    instance_ci.enabledExtensionCount = static_cast<u32>(enabled_extensions.size());
    instance_ci.ppEnabledExtensionNames = enabled_extensions.data();

    VkInstance instance{};
    vk_check(vkCreateInstance(&instance_ci, nullptr, &instance));
    volkLoadInstance(instance);

    detail::initialise_debug_name_func(instance);

    return instance;
}

inline auto create_instance_with_debug(auto &callback, std::span<const std::string_view> surface_required_extensions)
        -> InstanceWithDebug {
    vk_check(volkInitialize());

    auto app_info = create_info<VkApplicationInfo>();
    app_info.pApplicationName = "HeadlessBindless";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "None";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_4;

    std::array<const char *, 1> enabled_layers = {"VK_LAYER_KHRONOS_validation"};

    std::vector<const char *> enabled_extensions;
    for (const auto &required_extension: surface_required_extensions) {
        enabled_extensions.push_back(required_extension.data());
    }

    bool has_debug_utils = false;
    {
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

    auto instance_ci = create_info<VkInstanceCreateInfo>();
    instance_ci.pApplicationInfo = &app_info;
    instance_ci.enabledLayerCount = static_cast<u32>(enabled_layers.size());
    instance_ci.ppEnabledLayerNames = enabled_layers.data();
    instance_ci.enabledExtensionCount = static_cast<u32>(enabled_extensions.size());
    instance_ci.ppEnabledExtensionNames = enabled_extensions.data();

    InstanceWithDebug result{};
    vk_check(vkCreateInstance(&instance_ci, nullptr, &result.instance));
    volkLoadInstance(result.instance);

    if (has_debug_utils) {
        auto debug_ci = create_info<VkDebugUtilsMessengerCreateInfoEXT>();
        debug_ci.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_ci.pfnUserCallback = &callback;

        auto create_debug = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(result.instance, "vkCreateDebugUtilsMessengerEXT"));

        if (create_debug) {
            vk_check(create_debug(result.instance, &debug_ci, nullptr, &result.messenger));
        }
    }

    detail::initialise_debug_name_func(result.instance);

    return result;
}

constexpr auto viewport_scissors(VkExtent2D extent, bool flipped = true) {
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = static_cast<float>(extent.height);
    viewport.width = static_cast<float>(extent.width);
    viewport.height = -static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    if (!flipped) {
        viewport.height = static_cast<float>(extent.height);
        viewport.y = 0;
    }

    VkRect2D scissor{};
    scissor.offset = VkOffset2D{0, 0};
    scissor.extent = extent;
    return std::make_pair(viewport, scissor);
}


struct PhysicalDeviceChoice {
    enum class Error { NoDevicesFound, NoQueuesFound };

    Error error;
};

using DeviceChoice = std::tuple<VkPhysicalDevice, u32, u32, u32>;
auto pick_physical_device(VkInstance instance) -> tl::expected<DeviceChoice, PhysicalDeviceChoice>;

enum class ComputeStamp : u32 { RotateBegin, RotateEnd, CullBegin, CullEnd, ClusteringBegin, ClusteringEnd, Count };
enum class ComputeIndex : u32 { Rotate = 0, Cull = 1, Clustering, Count };

inline constexpr u32 compute_query_count = static_cast<u32>(ComputeStamp::Count);
inline constexpr u32 stats_compute_count = static_cast<u32>(ComputeIndex::Count);

enum class GraphicsStamp : u32 {
    PreDepthBegin,
    PreDepthEnd,
    GbufferBegin,
    GbufferEnd,
    DeferredBegin,
    DeferredEnd,
    TonemapBegin,
    TonemapEnd,
    PresentBegin,
    PresentEnd,
    UIBegin,
    UIEnd,
    Count
};
enum class GraphicsIndex : u32 { PreDepth = 0, GBuffer = 1, Deferred = 2, Tonemap = 3, Present = 4, UI = 5, Count };

inline constexpr u32 graphics_query_count = static_cast<u32>(GraphicsStamp::Count);
inline constexpr u32 stats_graphics_count = static_cast<u32>(GraphicsIndex::Count);

using EnabledFeatureSet = std::unordered_set<std::string, string_hash, string_eq>;
auto create_device(VkPhysicalDevice pd, u32 graphics_index, u32 compute_index, u32 transfer_index)
        -> std::tuple<VkDevice, VkQueue, VkQueue, VkQueue, EnabledFeatureSet>;

auto create_allocator(VkInstance instance, VkPhysicalDevice pd, VkDevice device) -> VmaAllocator;

struct TimelineWait {
    u64 value{0};
    VkSemaphore semaphore{VK_NULL_HANDLE};
    VkPipelineStageFlags2 stage{VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
};

struct BinaryWait {
    VkSemaphore semaphore{VK_NULL_HANDLE};
    VkPipelineStageFlags2 stage{VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
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
        VkSemaphoreWaitInfo wi{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .pNext = nullptr,
                .flags = 0,
                .semaphoreCount = 1,
                .pSemaphores = &tl.timeline,
                .pValues = &last,
        };
        vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));
        tl.completed = std::max(tl.completed, last);
    }

    vk_check(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo bi{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
    };
    vk_check(vkBeginCommandBuffer(cmd, &bi));
    record(cmd);
    vk_check(vkEndCommandBuffer(cmd));

    const u64 signal_val = tl.value + 1;

    const u32 bwc = static_cast<u32>(sync.binary_waits.size());
    const u32 twc = static_cast<u32>(sync.timeline_waits.size());

    // Worst-case sizes (timeline waits may be filtered)
    std::vector<VkSemaphoreSubmitInfo> wait_infos;
    wait_infos.reserve(bwc + twc);

    for (const auto &w: sync.binary_waits) {
        wait_infos.push_back(VkSemaphoreSubmitInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .semaphore = w.semaphore,
                .value = 0,
                .stageMask = w.stage,
                .deviceIndex = 0,
        });
    }

    for (const auto &w: sync.timeline_waits) {
        wait_infos.push_back(VkSemaphoreSubmitInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .semaphore = w.semaphore,
                .value = w.value,
                .stageMask = w.stage,
                .deviceIndex = 0,
        });
    }

    std::vector<VkSemaphoreSubmitInfo> signal_infos;
    signal_infos.reserve(1u + static_cast<u32>(sync.binary_signals.size()));

    // Always signal the timeline
    signal_infos.push_back(VkSemaphoreSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = tl.timeline,
            .value = signal_val,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, // conservative; or tighten if you want
            .deviceIndex = 0,
    });

    // Signal any binary semaphores
    for (VkSemaphore s: sync.binary_signals) {
        signal_infos.push_back(VkSemaphoreSubmitInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .semaphore = s,
                .value = 0,
                .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .deviceIndex = 0,
        });
    }

    VkCommandBufferSubmitInfo cmd_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = nullptr,
            .commandBuffer = cmd,
            .deviceMask = 0,
    };

    VkSubmitInfo2 submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = nullptr,
            .flags = 0,
            .waitSemaphoreInfoCount = static_cast<u32>(wait_infos.size()),
            .pWaitSemaphoreInfos = wait_infos.empty() ? nullptr : wait_infos.data(),
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &cmd_info,
            .signalSemaphoreInfoCount = static_cast<u32>(signal_infos.size()),
            .pSignalSemaphoreInfos = signal_infos.empty() ? nullptr : signal_infos.data(),
    };

    vk_check(vkQueueSubmit2(tl.queue, 1, &submit, VK_NULL_HANDLE));

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
    }

    auto device(VkDevice &dev) -> void;

    auto bindless_set(BindlessSet &bs) -> void;

    auto allocator(VmaAllocator &alloc) -> void;
    auto swapchain(Swapchain &) -> void;

    auto timeline(VkDevice device, GraphicsTimeline &) -> void;
    auto timeline(VkDevice device, TransferTimeline &) -> void;
    auto timeline(VkDevice device, ComputeTimeline &) -> void;

    auto timelines(VkDevice device, auto &&...timelines) -> void { (timeline(device, timelines), ...); }

    template<typename T>
    concept PipelineProvider = requires(T t) {
        { t.pipeline } -> std::same_as<VkPipeline &>;
        { t.layout } -> std::same_as<VkPipelineLayout &>;
    } || requires(T t) {
        { std::get<0>(t) } -> std::same_as<VkPipeline &>;
        { std::get<1>(t) } -> std::same_as<VkPipelineLayout &>;
    } || requires(T t) {
        { t.pipeline } -> std::same_as<const VkPipeline &>;
        { t.layout } -> std::same_as<const VkPipelineLayout &>;
    };

    template<PipelineProvider T>
    auto as_pipeline_refs(T &t) {
        if constexpr (requires {
                          t.pipeline;
                          t.layout;
                      }) {
            return std::make_pair(t.pipeline, t.layout);
        } else {
            return std::make_pair(std::get<0>(t), std::get<1>(t));
        }
    }

    auto pipeline(VkDevice dev, VkPipeline &, VkPipelineLayout &) -> void;

    auto pipeline(VkDevice dev, PipelineProvider auto &val) -> void {
        auto &&[p, l] = as_pipeline_refs(val);
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
