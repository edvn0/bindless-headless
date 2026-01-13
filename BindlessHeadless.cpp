#include "BindlessHeadless.hxx"
#include "BindlessSet.hxx"
#include "ImageOperations.hxx"
#include "Pool.hxx"
#include "PipelineCache.hxx"

#include <FastNoise/FastNoise.h>

#include <memory>

static VkBool32
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
               VkDebugUtilsMessageTypeFlagsEXT,
               const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
               void *) {
  if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    std::cerr << "validation layer: " << callback_data->pMessage << std::endl;
  }
  return VK_FALSE;
}

auto create_compute_pipeline(VkDevice device, PipelineCache& cache,
                             VkDescriptorSetLayout layout,
                             const std::vector<u32> &code) {
  VkShaderModule compute_shader{};
  VkShaderModuleCreateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .codeSize = code.size() * sizeof(u32),
      .pCode = code.data()};
  vk_check(
      vkCreateShaderModule(device, &create_info, nullptr, &compute_shader));

  VkPushConstantRange push_constant_range{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = 2 * sizeof(u32), // Padded 4 bytes
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
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = compute_shader,
                .pName = "main",
                .pSpecializationInfo = nullptr},
      .layout = pi_layout,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = -1};
  VkPipeline pipeline{};
  vk_check(
      vkCreateComputePipelines(device, cache, 1, &cpci, nullptr, &pipeline));

  vkDestroyShaderModule(device, compute_shader, nullptr);
  return std::make_pair(pipeline, pi_layout);
}

auto main(int argc, char ** argv) -> int {
  auto compiler = CompilerSession{};

  constexpr bool is_release = bool(IS_RELEASE);

  auto instance = create_instance_with_debug(debug_callback, is_release);
  auto could_choose = pick_physical_device(instance.instance);
  if (!could_choose) {
    return 1;
  }

  auto &&[physical_device, graphics_index, compute_index] = *could_choose;

  auto &&[device, graphics_queue, compute_queue] =
      create_device(physical_device, graphics_index, compute_index);

      auto cache_path = pipeline_cache_path(static_cast<std::int32_t>(argc), argv);
      auto pipeline_cache = std::make_unique<PipelineCache>(device, cache_path);

  auto simple_random_colour_pipeline = compiler.compile_compute_from_file(
      "shaders/simple_random_colour.slang", "main");

  auto allocator = create_allocator(instance.instance, physical_device, device);

  auto tl_compute = create_timeline(device, compute_queue, compute_index);
  auto tl_graphics = create_timeline(device, graphics_queue, graphics_index);

  BindlessCaps caps = query_bindless_caps(physical_device);
  BindlessSet bindless{};
  bindless.init(device, caps,
                128u, // initial textures
                32u,  // initial samplers
                16u,  // initial storage images
                8u);  // initial accel structs

  bindless.grow_if_needed(300u, 40u, 32u, 8u);

  auto &&[compute_pipeline, compute_layout] = create_compute_pipeline(
      device, *pipeline_cache, bindless.layout, simple_random_colour_pipeline);

  DestructionContext ctx{
      .allocator = allocator,
      .bindless_set = &bindless,
  };

  ctx.create_texture(create_offscreen_target(
      allocator, 1280u, 720u, VK_FORMAT_R8G8B8A8_UNORM, "white-texture"));
  ctx.create_texture(create_offscreen_target(
      allocator, 1280u, 720u, VK_FORMAT_R8G8B8A8_UNORM, "black-texture"));

  std::vector<std::uint8_t> noise = generate_perlin(2048, 2048);
  auto tex = create_image_from_span(allocator, cmd_pool, queue, 2048, 2048,
                                    VK_FORMAT_R8_UNORM,
                                    std::span{noise},
                                    "perlin_noise");

  auto handle = ctx.create_texture(create_offscreen_target(
      allocator, 1280u, 720u, VK_FORMAT_R8G8B8A8_UNORM, "offscreen"));

  ctx.create_sampler(
      create_sampler(allocator,
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
                         .anisotropyEnable = false,
                         .maxAnisotropy = 1.0f,
                         .compareEnable = false,
                         .compareOp = VK_COMPARE_OP_ALWAYS,
                         .minLod = 0.0f,
                         .maxLod = VK_LOD_CLAMP_NONE,
                         .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                         .unnormalizedCoordinates = false,
                     },
                     "linear_repeat"));
  bindless.repopulate_if_needed(ctx.textures, ctx.samplers);
  std::cout << "Image handle index: " << handle.index()
            << " (max: " << bindless.max_storage_images << ")" << std::endl;

  std::array<FrameState, frames_in_flight> frames{};
  std::uint64_t i = 0;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (i = 0; i < 10'000; ++i) {

    const std::uint32_t frame_index =
        static_cast<std::uint32_t>(i % frames_in_flight);
    auto &fs = frames[frame_index];

    if (fs.frame_done_value > 0) {
      VkSemaphoreWaitInfo wi{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                             .pNext = nullptr,
                             .flags = 0,
                             .semaphoreCount = 1,
                             .pSemaphores = &tl_graphics.timeline,
                             .pValues = &fs.frame_done_value};
      vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));
    }

    // --- Stage 1: Light culling on compute queue ---
    std::array<VkSemaphore, 0> no_wait_sems{};
    std::array<std::uint64_t, 0> no_wait_vals{};

    auto light_val = submit_stage(
        tl_compute, device,
        [&](VkCommandBuffer cmd) {
          auto &target = *ctx.textures.get(handle);

          VkImageMemoryBarrier barrier{
              .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
              .pNext = nullptr,
              .srcAccessMask = 0,
              .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
              .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .image = target.image,
              .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                   .baseMipLevel = 0,
                                   .levelCount = 1,
                                   .baseArrayLayer = 0,
                                   .layerCount = 1}};

          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                               nullptr, 0, nullptr, 1, &barrier);

          target.initialized = true;

          struct PushConstants {
            u32 image_index;
            u32 padding{0};
          };

          PushConstants pc{.image_index = handle.index()};

          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  compute_layout, 0, 1, &bindless.set, 0,
                                  nullptr);

          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_pipeline);

          vkCmdPushConstants(cmd, compute_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                             0, sizeof(PushConstants), &pc);

          // 1280x720 with 8x8 groups
          vkCmdDispatch(cmd, (1280u + 7u) / 8u, (720u + 7u) / 8u, 1u);
        },
        std::span{no_wait_sems}, std::span{no_wait_vals});

    fs.timeline_values[stage_index(Stage::LightCulling)] = light_val;

    std::array<VkSemaphore, 1> gbuffer_wait_sems{tl_compute.timeline};
    std::array<std::uint64_t, 1> gbuffer_wait_vals{
        fs.timeline_values[stage_index(Stage::LightCulling)]};

    auto gbuffer_val = submit_stage(
        tl_graphics,
        device,
        [](VkCommandBuffer ) {
        },
        std::span{ gbuffer_wait_sems },
        std::span{ gbuffer_wait_vals }
    );

    fs.timeline_values[stage_index(Stage::GBuffer)] = gbuffer_val;
    fs.frame_done_value = gbuffer_val;

    throttle(tl_compute, device);
    throttle(tl_graphics, device);

    const auto completed =
        std::min(tl_compute.completed, tl_graphics.completed);
    ctx.destroy_queue.retire(completed);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_time - start_time)
                      .count();

  std::println("Timeline frametime (ms): {}",
               static_cast<double>(duration) / static_cast<double>(i));

  vkDeviceWaitIdle(device);
  image_operations::write_to_disk(ctx.textures, handle, allocator,
                                  "output.bmp");
  vkDeviceWaitIdle(device);

  pipeline_cache.reset();

  vkDestroyPipeline(device, compute_pipeline, nullptr);
  vkDestroyPipelineLayout(device, compute_layout, nullptr);

  ctx.textures.for_each_live(
      [&](auto handle, auto &) { destroy(ctx, handle); });

  ctx.samplers.for_each_live(
      [&](auto handle, auto &) { destroy(ctx, handle); });


  ctx.destroy_queue.retire(UINT64_MAX);

  destruction::bindless_set(device, bindless);
  destruction::timeline_compute(device, tl_graphics);
  destruction::timeline_compute(device, tl_compute);
  destruction::allocator(allocator);
  destruction::device(device);
  destruction::instance(instance);

  std::cout << "Bindless headless setup and teardown completed successfully."
            << std::endl;
  return 0;
}

auto vk_check(VkResult result) -> void {
  if (result != VK_SUCCESS) {
    std::cerr << "Result: " << string_VkResult(result) << "\n";
    std::abort();
  }
}

namespace destruction {
auto bindless_set(VkDevice device, BindlessSet &bs) -> void {
  if (bs.pool)
    vkDestroyDescriptorPool(device, bs.pool, nullptr);
  if (bs.layout)
    vkDestroyDescriptorSetLayout(device, bs.layout, nullptr);
  bs.pool = VK_NULL_HANDLE;
  bs.layout = VK_NULL_HANDLE;
  bs.set = VK_NULL_HANDLE;
}
} // namespace destruction

namespace detail {

auto set_debug_name_impl(VmaAllocator &alloc, VkObjectType object_type,
                         std::uint64_t object_handle, std::string_view name)
    -> void {
  VmaAllocatorInfo info{};
  vmaGetAllocatorInfo(alloc, &info);

  static PFN_vkSetDebugUtilsObjectNameEXT set_debug_name_func = nullptr;
  if (set_debug_name_func == nullptr) {
    set_debug_name_func = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetInstanceProcAddr(info.instance, "vkSetDebugUtilsObjectNameEXT"));
  }

  if (set_debug_name_func == nullptr) {
    return;
  }

  VkDebugUtilsObjectNameInfoEXT name_info{
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
      .pNext = nullptr,
      .objectType = object_type,
      .objectHandle = object_handle,
      .pObjectName = name.data()};
  vk_check(set_debug_name_func(info.device, &name_info));
}
} // namespace detail
