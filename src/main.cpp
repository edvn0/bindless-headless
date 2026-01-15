#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "GlobalCommandContext.hxx"
#include "ImageOperations.hxx"
#include "Pool.hxx"
#include "PipelineCache.hxx"
#include "Reflection.hxx"
#include "Compiler.hxx"
#include "Buffer.hxx"
#include "Logger.hxx"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "3PP/PerlinNoise.hpp"

#define HAS_RENDERDOC
#ifdef HAS_RENDERDOC
#include "3PP/renderdoc_app.h"
#endif


#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

struct GpuPushConstants {
    const DeviceAddress ubo;
    const DeviceAddress lights;
    const DeviceAddress flags;
    const DeviceAddress prefix;
    const DeviceAddress group_sums;
    const DeviceAddress group_offsets;
    const DeviceAddress compact;
    const u32 image_index;
    const u32 light_count;
    const u32 group_count;
};

struct FrustumPlane {
    glm::vec4 plane; // xyz = normal, w = distance
};

 auto extract_frustum_planes = [](const glm::mat4& inv_proj) -> std::array<FrustumPlane, 6> {
    constexpr std::array<glm::vec4, 8> ndc_corners = {
        glm::vec4{-1, -1, -1, 1}, { 1, -1, -1, 1},
        {-1,  1, -1, 1}, { 1,  1, -1, 1},
        {-1, -1,  1, 1}, { 1, -1,  1, 1},
        {-1,  1,  1, 1}, { 1,  1,  1, 1}
    };

    glm::vec3 view_corners[8];
    for (int i = 0; i < 8; ++i) {
        glm::vec4 v = inv_proj * ndc_corners[i];
        view_corners[i] = glm::vec3(v) / v.w;
    }

    std::array<FrustumPlane, 6> planes;

    // Left plane: from corners 0,2,4
    glm::vec3 v0 = view_corners[2] - view_corners[0];
    glm::vec3 v1 = view_corners[4] - view_corners[0];
    glm::vec3 normal = glm::normalize(glm::cross(v0, v1));  // ← SWAPPED
    planes[0].plane = glm::vec4(normal, -glm::dot(normal, view_corners[0]));

    // Right plane: from corners 1,5,3
    v0 = view_corners[5] - view_corners[1];
    v1 = view_corners[3] - view_corners[1];
    normal = glm::normalize(glm::cross(v0, v1));  // ← SWAPPED
    planes[1].plane = glm::vec4(normal, -glm::dot(normal, view_corners[1]));

    // Bottom plane: from corners 0,4,1
    v0 = view_corners[4] - view_corners[0];
    v1 = view_corners[1] - view_corners[0];
    normal = glm::normalize(glm::cross(v0, v1));  // ← SWAPPED
    planes[2].plane = glm::vec4(normal, -glm::dot(normal, view_corners[0]));

    // Top plane: from corners 2,3,6
    v0 = view_corners[3] - view_corners[2];
    v1 = view_corners[6] - view_corners[2];
    normal = glm::normalize(glm::cross(v0, v1));  // ← SWAPPED
    planes[3].plane = glm::vec4(normal, -glm::dot(normal, view_corners[2]));

    // Near plane: from corners 0,1,2
    v0 = view_corners[1] - view_corners[0];
    v1 = view_corners[2] - view_corners[0];
    normal = glm::normalize(glm::cross(v0, v1));  // ← SWAPPED
    planes[4].plane = glm::vec4(normal, -glm::dot(normal, view_corners[0]));

    // Far plane: from corners 4,6,5
    v0 = view_corners[6] - view_corners[4];
    v1 = view_corners[5] - view_corners[4];
    normal = glm::normalize(glm::cross(v0, v1));  // ← SWAPPED
    planes[5].plane = glm::vec4(normal, -glm::dot(normal, view_corners[4]));

    return planes;
};

struct FrameUBO {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 view_projection;
    glm::mat4 inv_projection;
    glm::vec4 camera_position;
    std::array<FrustumPlane, 6> frustum_planes; // left, right, bottom, top, near, far
    float time;
    float delta_time;
    float _padding[2];
};


auto generate_perlin(auto w, auto h) -> std::vector<std::uint8_t> {
    std::vector<std::uint8_t> data;
    data.resize(w * h);
    const auto seed = static_cast<u32>(
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
debug_callback(const VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
               VkDebugUtilsMessageTypeFlagsEXT type,
               const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
               void *) {
    if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        error("Validation layer: {}", callback_data->pMessage);
    }

    if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT) {
        info("What is this?: {}", callback_data->pMessage);
    }
    return VK_FALSE;
}

auto create_compute_pipeline(VkDevice device,
                             PipelineCache &cache,
                             VkDescriptorSetLayout layout,
                             const std::vector<u32> &code,
                             const std::string_view entry_name)
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
        .size = sizeof(GpuPushConstants),
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
            .pName = entry_name.data(),
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
    RENDERDOC_API_1_6_0 *rdoc_api = nullptr;
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll")) {
        auto RENDERDOC_GetAPI =
                reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(mod, "RENDERDOC_GetAPI"));
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void **>(&rdoc_api));
        (void) ret;
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


    auto flags_code = compiler.compile_compute_from_file(
        "shaders/light_cull_prefix_compact.slang", "LightFlagsCS");
    auto scan_local_code = compiler.compile_compute_from_file(
        "shaders/light_cull_prefix_compact.slang", "LightScanLocalCS");
    auto scan_groups_code = compiler.compile_compute_from_file(
        "shaders/light_cull_prefix_compact.slang", "LightScanGroupsCS");
    ReflectionData light_flags{};
    auto compact_code = compiler.compile_compute_from_file(
        "shaders/light_cull_prefix_compact.slang", "LightCompactCS", &light_flags);
    auto debug_draw_code = compiler.compile_compute_from_file(
        "shaders/light_cull_prefix_compact.slang", "LightDebugDrawCS");


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


    auto &&[flags_pipeline, flags_layout] = create_compute_pipeline(
        device, *pipeline_cache, bindless.layout,
        flags_code, "LightFlagsCS");
    auto &&[scan_local_pipeline, scan_local_layout] = create_compute_pipeline(
        device, *pipeline_cache, bindless.layout,
        scan_local_code, "LightScanLocalCS");
    auto &&[scan_groups_pipeline, scan_groups_layout] = create_compute_pipeline(
        device, *pipeline_cache, bindless.layout,
        scan_groups_code, "LightScanGroupsCS");
    auto &&[compact_pipeline, compact_layout] = create_compute_pipeline(
        device, *pipeline_cache, bindless.layout,
        compact_code, "LightCompactCS");
    auto &&[debug_draw_pipeline, debug_draw_layout] = create_compute_pipeline(
        device, *pipeline_cache, bindless.layout,
        debug_draw_code, "LightDebugDrawCS");


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
    auto all_point_lights_zero = std::vector<PointLight>(500);
    auto light_count = static_cast<u32>(all_point_lights.size());
    constexpr u32 threads_per_group = 64u;
    auto group_count = (light_count + threads_per_group - 1u) / threads_per_group;

    constexpr auto world_size = 50.F;

    auto rng = std::default_random_engine{};
    auto distrib = std::uniform_real_distribution{-world_size, world_size};

    for (u32 idx = 0; idx < light_count; ++idx) {
        auto t = static_cast<float>(idx) / static_cast<float>(light_count);
        auto &[position_radius, colour_intensity] = all_point_lights[idx];

        position_radius = {
            distrib(rng),
            distrib(rng),
            distrib(rng),
            5.0F
        };
        colour_intensity = {t, 1.0f - t, 0.5f, 1.0f};
    }

    auto point_light_handle =
            ctx.buffers.create(Buffer::from_slice<PointLight>(allocator, VkBufferCreateInfo{
                                                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                              }, VmaAllocationCreateInfo{}, all_point_lights,
                                                              "point_light").value()
            );

    std::vector<u32> zeros_lights(light_count, 0u);
    std::vector<u32> zeros_groups(group_count, 0u);
    auto flags_handle =
            ctx.buffers.create(Buffer::from_slice<u32>(allocator, VkBufferCreateInfo{
                                                           .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                       }, VmaAllocationCreateInfo{}, zeros_lights,
                                                       "light_flags").value());
    auto prefix_handle =
            ctx.buffers.create(Buffer::from_slice<u32>(allocator, VkBufferCreateInfo{
                                                           .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                       }, VmaAllocationCreateInfo{}, zeros_lights,
                                                       "light_prefix").value());
    auto group_sums_handle =
            ctx.buffers.create(Buffer::from_slice<u32>(allocator, VkBufferCreateInfo{
                                                           .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                       }, VmaAllocationCreateInfo{}, zeros_groups,
                                                       "group_sums").value());
    auto group_offsets_handle =
            ctx.buffers.create(Buffer::from_slice<u32>(allocator, VkBufferCreateInfo{
                                                           .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                       }, VmaAllocationCreateInfo{}, zeros_groups,
                                                       "group_offsets").value());
    auto compact_lights_handle =
            ctx.buffers.create(Buffer::from_slice<PointLight>(allocator, VkBufferCreateInfo{
                                                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                              }, VmaAllocationCreateInfo{}, all_point_lights_zero,
                                                              "compact_lights").value());
    glm::vec3 camera_pos = glm::vec3(0.0f, 50.0f, -100.0f);
    glm::vec3 camera_target = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

    glm::mat4 view = glm::lookAt(camera_pos, camera_target, camera_up);
    glm::mat4 projection = glm::perspective(
        glm::radians(60.0f), // FOV
        1280.0f / 720.0f, // aspect ratio
        0.1f, // near plane
        1000.0f // far plane
    );


    glm::mat4 inv_proj = glm::inverse(projection);
    auto frustum_planes = extract_frustum_planes(inv_proj);
#ifdef DEBUG_FRUSTA
    glm::vec3 test_point_world = glm::vec3(0.0f, 0.0f, 0.0f); // Origin
    glm::vec4 test_point_view4 = view * glm::vec4(test_point_world, 1.0f);
    glm::vec3 test_point_view = glm::vec3(test_point_view4) / test_point_view4.w;

    info("\n=== Frustum Plane Debug ===");
    info("Test point (world): ({}, {}, {})", test_point_world.x, test_point_world.y, test_point_world.z);
    info("Test point (view):  ({}, {}, {})", test_point_view.x, test_point_view.y, test_point_view.z);

    const char *plane_names[] = {"Left", "Right", "Bottom", "Top", "Near", "Far"};
    for (int x = 0; x < 6; ++x) {
        auto &plane = frustum_planes[x].plane;
        float dist = glm::dot(glm::vec3(plane), test_point_view) + plane.w;
        info("{:6} plane: normal=({:+.3f}, {:+.3f}, {:+.3f}), d={:+.3f}, dist={:+.3f}",
                     plane_names[x], plane.x, plane.y, plane.z, plane.w, dist);
    }
#endif

    FrameUBO ubo_data{
        .view = view,
        .projection = projection,
        .view_projection = projection * view,
        .inv_projection = inv_proj,
        .camera_position = glm::vec4(camera_pos, 1.0f),
        .frustum_planes = {
            frustum_planes[0], frustum_planes[1], frustum_planes[2],
            frustum_planes[3], frustum_planes[4], frustum_planes[5]
        },
        .time = 0.0f,
        .delta_time = 0.0f,
    };

    // Create the UBO buffer
    auto frame_ubo_handle = ctx.buffers.create(
        Buffer::from_slice<FrameUBO>(
            allocator,
            VkBufferCreateInfo{
                .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            },
            VmaAllocationCreateInfo{},
            std::span{&ubo_data, 1},
            "frame_ubo"
        ).value()
    );

    auto frame_ubo = ctx.device_address(frame_ubo_handle);

    auto light_addr = ctx.device_address(point_light_handle);
    auto flags_addr = ctx.device_address(flags_handle);
    auto prefix_addr = ctx.device_address(prefix_handle);
    auto group_sums_addr = ctx.device_address(group_sums_handle);
    auto group_offsets_addr = ctx.device_address(group_offsets_handle);
    auto compact_addr = ctx.device_address(compact_lights_handle);

    auto stats = FrameStats{};
    for (i = 0; i < 5; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

#ifdef HAS_RENDERDOC
        if (rdoc_api) rdoc_api->StartFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance.instance), NULL);
#endif
        bindless.repopulate_if_needed(ctx.textures, ctx.samplers);

        const auto frame_index =
                static_cast<u32>(i % frames_in_flight);
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

                VkImageMemoryBarrier image_barrier{
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
                    &image_barrier);

                target.initialized = true;

                const GpuPushConstants pc{
                    .ubo = frame_ubo,
                    .lights = light_addr,
                    .flags = flags_addr,
                    .prefix = prefix_addr,
                    .group_sums = group_sums_addr,
                    .group_offsets = group_offsets_addr,
                    .compact = compact_addr,
                    .image_index = handle.index(),
                    .light_count = light_count,
                    .group_count = group_count,
                };

                auto bind_and_dispatch = [&](VkPipeline pipeline,
                                             VkPipelineLayout layout,
                                             u32 groups_x) {
                    vkCmdBindDescriptorSets(
                        cmd,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        layout,
                        0,
                        1,
                        &bindless.set,
                        0,
                        nullptr);

                    vkCmdBindPipeline(
                        cmd,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline);

                    vkCmdPushConstants(
                        cmd,
                        layout,
                        VK_SHADER_STAGE_COMPUTE_BIT,
                        0,
                        sizeof(GpuPushConstants),
                        &pc);

                    vkCmdDispatch(
                        cmd,
                        groups_x,
                        1u,
                        1u);
                };

                VkMemoryBarrier mem_barrier{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                };

                bind_and_dispatch(flags_pipeline, flags_layout, group_count);
                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    &mem_barrier,
                    0,
                    nullptr,
                    0,
                    nullptr);

                bind_and_dispatch(scan_local_pipeline, scan_local_layout, group_count);
                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    &mem_barrier,
                    0,
                    nullptr,
                    0,
                    nullptr);

                bind_and_dispatch(scan_groups_pipeline, scan_groups_layout, 1u);
                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    &mem_barrier,
                    0,
                    nullptr,
                    0,
                    nullptr);

                bind_and_dispatch(compact_pipeline, compact_layout, group_count);
                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    &mem_barrier,
                    0,
                    nullptr,
                    0,
                    nullptr);

                bind_and_dispatch(debug_draw_pipeline, debug_draw_layout, group_count);
            },
            std::span{no_wait_sems},
            std::span{no_wait_vals});

        fs.timeline_values[stage_index(Stage::LightCulling)] =
                light_val;

        std::array gbuffer_wait_sems{
            tl_compute.timeline
        };
        std::array gbuffer_wait_vals{
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

        throttle(tl_compute, device);
        throttle(tl_graphics, device);

        const auto completed =
                std::min(tl_compute.completed, tl_graphics.completed);
        ctx.destroy_queue.retire(completed);
#ifdef HAS_RENDERDOC
        if (rdoc_api) rdoc_api->EndFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance.instance), NULL);
#endif
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(frame_end - start_time).
                count();
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

    ctx.textures.for_each_live(
        [&](auto h, auto &) { destroy(ctx, h); });

    ctx.samplers.for_each_live(
        [&](auto h, auto &) { destroy(ctx, h); });

    ctx.buffers.for_each_live([&](auto h, auto &) { destroy(ctx, h); });

    ctx.destroy_queue.retire(UINT64_MAX);

    destruction::pipeline(device, flags_pipeline, flags_layout);
    destruction::pipeline(device, scan_local_pipeline, scan_local_layout);
    destruction::pipeline(device, scan_groups_pipeline, scan_groups_layout);
    destruction::pipeline(device, compact_pipeline, compact_layout);
    destruction::pipeline(device, debug_draw_pipeline, debug_draw_layout);
    destruction::global_command_context(command_context);
    destruction::bindless_set(device, bindless);
    destruction::timeline_compute(device, tl_graphics);
    destruction::timeline_compute(device, tl_compute);
    destruction::allocator(allocator);
    destruction::device(device);
    destruction::instance(instance);

    info("Bindless headless setup and teardown completed successfully.");
    return 0;
}
