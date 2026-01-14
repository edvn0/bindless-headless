#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "GlobalCommandContext.hxx"
#include "ImageOperations.hxx"
#include "Pool.hxx"
#include "PipelineCache.hxx"
#include "Reflection.hxx"
#include "Compiler.hxx"
#include "Buffer.hxx"

#include "3PP/PerlinNoise.hpp"

#ifdef HAS_RENDERDOC
#include "3PP/renderdoc_app.h"
#endif

#include <chrono>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

auto generate_perlin(auto w, auto h) -> std::vector<std::uint8_t> {
    std::vector<std::uint8_t> data;
    data.resize(w * h);
    const auto seed = static_cast<std::uint32_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const siv::PerlinNoise pn{seed};

    auto z_offset = 0.0;
    for (auto y = 0; y < h; ++y) {
        const auto row_z = z_offset + static_cast<double>(y) * 0.01;
        for (auto x = 0; x < w; ++x) {
            const auto nx = static_cast<double>(x) / static_cast<double>(w);
            auto ny = static_cast<double>(y) / static_cast<double>(h);
            auto value = pn.noise3D(nx * 8.0, ny * 8.0, row_z);
            value = (value + 1.0) / 2.0;
            data[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(x)] =
                    static_cast<std::uint8_t>(value * 255.0);
        }
        z_offset += 0.0001;
    }

    return data;
}

static VkBool32
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
               VkDebugUtilsMessageTypeFlagsEXT,
               const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
               void *) {
    if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "validation layer: " << callback_data->pMessage
                << std::endl;
    }
    return VK_FALSE;
}

auto create_compute_pipeline(VkDevice device,
                             PipelineCache &cache,
                             VkDescriptorSetLayout layout,
                             const std::vector<u32> &code)
    -> std::pair<VkPipeline, VkPipelineLayout> {
    VkShaderModule compute_shader{};
    VkShaderModuleCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = code.size() * sizeof(u32),
        .pCode = code.data()
    };
    vk_check(
        vkCreateShaderModule(device, &create_info, nullptr, &compute_shader));

    VkPushConstantRange push_constant_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 3 * sizeof(u32),
    };

    VkPipelineLayout pi_layout{};
    VkPipelineLayoutCreateInfo plci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
    };
    vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &pi_layout));

    VkComputePipelineCreateInfo cpci{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage =
        VkPipelineShaderStageCreateInfo{
            .sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = compute_shader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        .layout = pi_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1
    };
    VkPipeline pipeline{};
    vk_check(vkCreateComputePipelines(
        device, cache, 1, &cpci, nullptr, &pipeline));

    vkDestroyShaderModule(device, compute_shader, nullptr);
    return std::make_pair(pipeline, pi_layout);
}


auto main(int argc, char **argv) -> int {
#ifdef HAS_RENDERDOC
    RENDERDOC_API_1_6_0* rdoc_api = nullptr;
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll")) {
        auto RENDERDOC_GetAPI =
            reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(mod, "RENDERDOC_GetAPI"));
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void **>(&rdoc_api));
        (void)ret;
    }
#endif
    auto compiler = CompilerSession{};

    constexpr bool is_release = static_cast<bool>(IS_RELEASE);

    auto instance = create_instance_with_debug(debug_callback, is_release);
    auto could_choose = pick_physical_device(instance.instance);
    if (!could_choose) {
        return 1;
    }

    auto &&[physical_device, graphics_index, compute_index] = *could_choose;

    auto &&[device, graphics_queue, compute_queue] =
            create_device(physical_device, graphics_index, compute_index);

    auto cache_path = pipeline_cache_path(argc, argv);
    auto pipeline_cache =
            std::make_unique<PipelineCache>(device, cache_path);

    auto command_context =
            create_global_cmd_context(device, graphics_queue, graphics_index);

    auto simple_random_colour_pipeline = compiler.compile_compute_from_file(
        "shaders/simple_random_colour.slang", "main");

    auto allocator =
            create_allocator(instance.instance, physical_device, device);

    auto tl_compute =
            create_timeline(device, compute_queue, compute_index);
    auto tl_graphics =
            create_timeline(device, graphics_queue, graphics_index);

    BindlessCaps caps = query_bindless_caps(physical_device);
    BindlessSet bindless{};
    bindless.init(device, caps, 8u, 8u, 8u, 0u);

    bindless.grow_if_needed(300u, 40u, 32u, 8u);

    auto &&[compute_pipeline, compute_layout] = create_compute_pipeline(
        device, *pipeline_cache, bindless.layout,
        simple_random_colour_pipeline);

    DestructionContext ctx{
        .allocator = allocator,
        .bindless_set = &bindless,
    };

    ctx.create_texture(create_offscreen_target(
        allocator, 1280u, 720u, VK_FORMAT_R8G8B8A8_UNORM, "white-texture"));
    ctx.create_texture(create_offscreen_target(
        allocator, 1280u, 720u, VK_FORMAT_R8G8B8A8_UNORM, "black-texture"));

    const auto noise = generate_perlin(2048, 2048);
    auto perlin_handle =
            ctx.create_texture(create_image_from_span_v2(
                allocator,
                command_context,
                2048u,
                2048u,
                VK_FORMAT_R8_UNORM,
                std::span{noise},
                "perlin_noise"));

    auto handle = ctx.create_texture(create_offscreen_target(
        allocator, 1280u, 720u, VK_FORMAT_R8G8B8A8_UNORM, "offscreen"));

    ctx.create_sampler(
        VkSamplerCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_FALSE,
            .maxAnisotropy = 1.0f,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        },
        "linear_repeat");

    auto perlin_sampler = ctx.create_sampler(create_sampler(allocator,
                                                            VkSamplerCreateInfo{
                                                                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                                                .magFilter = VK_FILTER_LINEAR,
                                                                .minFilter = VK_FILTER_LINEAR,
                                                                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                                                .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                                .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                                .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                                .mipLodBias = 0.0f,
                                                                .anisotropyEnable = false,
                                                                .compareEnable = false,
                                                                .minLod = 0.0f,
                                                                .maxLod = VK_LOD_CLAMP_NONE,
                                                                .unnormalizedCoordinates = false,
                                                            },
                                                            "noise_sampler"));

    bindless.repopulate_if_needed(ctx.textures, ctx.samplers);

    std::array<FrameState, frames_in_flight> frames{};
    std::uint64_t i = 0;


    struct PointLight {
        std::array<float, 4> position_radius;
        std::array<float, 4> colour_intensity;
    };

    auto all_point_lights = std::vector<PointLight>(500);

    auto point_light_handle =
            ctx.buffers.create(Buffer::from_slice<PointLight>(allocator, VkBufferCreateInfo{
                                                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                              }, VmaAllocationCreateInfo{}, all_point_lights,
                                                              "point_light").value()
            );

    auto stats = FrameStats{};
    for (i = 0; i < 100'000; ++i) {
    auto start_time = std::chrono::high_resolution_clock::now();

#ifdef HAS_RENDERDOC
        if (rdoc_api) rdoc_api->StartFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance.instance), NULL);
#endif
        bindless.repopulate_if_needed(ctx.textures, ctx.samplers);

        const auto frame_index =
                static_cast<std::uint32_t>(i % frames_in_flight);
        auto &fs = frames[frame_index];

        if (fs.frame_done_value > 0) {
            VkSemaphoreWaitInfo wi{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .pNext = nullptr,
                .flags = 0,
                .semaphoreCount = 1,
                .pSemaphores = &tl_graphics.timeline,
                .pValues = &fs.frame_done_value
            };
            vk_check(
                vkWaitSemaphores(device, &wi, UINT64_MAX));
        }

        std::array<VkSemaphore, 0> no_wait_sems{};
        std::array<std::uint64_t, 0> no_wait_vals{};

        auto light_val = submit_stage(
            tl_compute,
            device,
            [&](VkCommandBuffer cmd) {
                auto &target = *ctx.textures.get(handle);

                VkImageMemoryBarrier barrier{
                    .sType =
                    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask =
                    VK_ACCESS_SHADER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED,
                    .image = target.image,
                    .subresourceRange =
                    VkImageSubresourceRange{
                        .aspectMask =
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                    }
                };

                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0,
                    nullptr,
                    0,
                    nullptr,
                    1,
                    &barrier);

                target.initialized = true;

                struct PushConstants {
                    u32 image_index;
                    u32 perlin_index;
                    u32 sampler_index;
                };

                PushConstants pc{
                    .image_index = handle.index(),
                    .perlin_index = perlin_handle.index(), .sampler_index = perlin_sampler.index()
                };

                vkCmdBindDescriptorSets(
                    cmd,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    compute_layout,
                    0,
                    1,
                    &bindless.set,
                    0,
                    nullptr);

                vkCmdBindPipeline(
                    cmd,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    compute_pipeline);

                vkCmdPushConstants(
                    cmd,
                    compute_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    sizeof(PushConstants),
                    &pc);

                vkCmdDispatch(
                    cmd,
                    (1280u + 7u) / 8u,
                    (720u + 7u) / 8u,
                    1u);
            },
            std::span{no_wait_sems},
            std::span{no_wait_vals});

        fs.timeline_values[stage_index(Stage::LightCulling)] =
                light_val;

        std::array<VkSemaphore, 1> gbuffer_wait_sems{
            tl_compute.timeline
        };
        std::array<std::uint64_t, 1> gbuffer_wait_vals{
            fs.timeline_values[stage_index(Stage::LightCulling)]
        };

        auto gbuffer_val = submit_stage(
            tl_graphics,
            device,
            [](VkCommandBuffer) {
            },
            std::span{gbuffer_wait_sems},
            std::span{gbuffer_wait_vals});

        fs.timeline_values[stage_index(Stage::GBuffer)] =
                gbuffer_val;
        fs.frame_done_value = gbuffer_val;

        //throttle(tl_compute, device);
        //throttle(tl_graphics, device);

        const auto completed =
                std::min(tl_compute.completed, tl_graphics.completed);
        ctx.destroy_queue.retire(completed);
#ifdef HAS_RENDERDOC
        if (rdoc_api) rdoc_api->EndFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance.instance), NULL);
#endif
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(frame_end - start_time).count();
        stats.add_sample(ms);
    }

    std::println("frames: {}", stats.samples.size());
    std::println("mean:   {:.3f} ms", stats.mean());
    std::println("median: {:.3f} ms", stats.median());
    std::println("stddev: {:.3f} ms", stats.stddev());

    vkDeviceWaitIdle(device);
    image_operations::write_to_disk(
        ctx.textures, handle, allocator, "output.bmp");
    image_operations::write_to_disk(
        ctx.textures, perlin_handle, allocator, "perlin.bmp");
    vkDeviceWaitIdle(device);

    pipeline_cache.reset();

    vkDestroyPipeline(device, compute_pipeline, nullptr);
    vkDestroyPipelineLayout(device, compute_layout, nullptr);

    ctx.textures.for_each_live(
        [&](auto h, auto &) { destroy(ctx, h); });

    ctx.samplers.for_each_live(
        [&](auto h, auto &) { destroy(ctx, h); });

    ctx.buffers.for_each_live([&](auto h, auto &) { destroy(ctx, h); });

    ctx.destroy_queue.retire(UINT64_MAX);

    destruction::global_command_context(command_context);
    destruction::bindless_set(device, bindless);
    destruction::timeline_compute(device, tl_graphics);
    destruction::timeline_compute(device, tl_compute);
    destruction::allocator(allocator);
    destruction::device(device);
    destruction::instance(instance);

    std::cout << "Bindless headless setup and teardown "
            "completed successfully."
            << std::endl;
    return 0;
}