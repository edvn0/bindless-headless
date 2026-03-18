#include "BindlessSet.hxx"

auto BindlessSet::init(VkDevice dev, BindlessCaps const &caps_init, u32 initial_textures, u32 initial_samplers,
                       u32 initial_comparison_samplers, u32 initial_storage_images, u32 initial_accel_structs) -> void {
    device = dev;
    caps = caps_init;

    max_textures = std::min(initial_textures, caps.max_textures);
    max_samplers = std::min(initial_samplers, caps.max_samplers);
    max_comparison_samplers = std::min(initial_comparison_samplers, caps.max_samplers);
    max_storage_images = std::min(initial_storage_images, caps.max_storage_images);
    max_accel_structs = std::min(initial_accel_structs, caps.max_accel_structs);
    max_cubemaps = std::min(initial_textures, caps.max_textures);
    max_3d_images = std::min(initial_storage_images, caps.max_textures);

    recreate();
}

auto BindlessSet::destroy() -> void {
    if (device == VK_NULL_HANDLE)
        return;

    vkDeviceWaitIdle(device);

    if (pool)
        vkDestroyDescriptorPool(device, pool, nullptr);
    if (pipeline_layout)
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    if (layout)
        vkDestroyDescriptorSetLayout(device, layout, nullptr);
    pool = VK_NULL_HANDLE;
    pipeline_layout = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    set = VK_NULL_HANDLE;
}

auto BindlessSet::grow_if_needed(u32 req_textures, u32 req_samplers, u32 req_storage, u32 req_accel) -> bool {
    bool grow = false;

    NANO_SCOPE("Grow");

    auto grow_and_clamp = [&](u32 &current, u32 requested, u32 cap) {
        if (requested > current) {
            u32 doubled = current * 2u;
            u32 target = std::max(doubled, requested);
            u32 clamped = std::min(target, cap);
            if (clamped > current) {
                current = clamped;
                grow = true;
            }
        }
    };

    grow_and_clamp(max_textures, req_textures, caps.max_textures);
    grow_and_clamp(max_samplers, req_samplers, caps.max_samplers);
    grow_and_clamp(max_storage_images, req_storage, caps.max_storage_images);
    grow_and_clamp(max_comparison_samplers, req_samplers, caps.max_samplers);
    grow_and_clamp(max_accel_structs, req_accel, caps.max_accel_structs);
    grow_and_clamp(max_cubemaps, req_textures, caps.max_textures);
    grow_and_clamp(max_3d_images, req_textures, caps.max_textures);

    if (!grow && layout != VK_NULL_HANDLE) {
        return false;
    }

    info("Bindless set growing: textures={}, samplers={}, comparison_samplers={}, storage={}, cubemaps={}, 3d={}, "
         "accel={}",
         max_textures, max_samplers, max_comparison_samplers, max_storage_images, max_cubemaps, max_3d_images,
         max_accel_structs);

    destroy();
    need_repopulate = true;
    recreate();
    return true;
}

// BindlessSet — new flush path
auto BindlessSet::flush_pending_writes(VkImageView dummy_sampled, VkImageView dummy_storage) -> void {
    if (pending_texture_writes.empty())
        return;

    if (need_repopulate) {
        pending_texture_writes.clear();
        return;
    }

    for (const auto &pw: pending_texture_writes) {
        if (pw.pool_index >= max_textures) {
            need_repopulate = true;
            pending_texture_writes.clear();
            return;
        }
    }

    NANO_SCOPE("Flush pending writes");

    std::vector<VkDescriptorImageInfo> sampled_infos(pending_texture_writes.size());
    std::vector<VkDescriptorImageInfo> storage_infos(pending_texture_writes.size());
    std::vector<VkDescriptorImageInfo> cubemap_infos;
    std::vector<u32> cubemap_indices;
    std::vector<VkDescriptorImageInfo> image_3d_infos;
    std::vector<u32> image_3d_indices;

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(pending_texture_writes.size() * 2);

    for (u32 i = 0; i < static_cast<u32>(pending_texture_writes.size()); ++i) {
        const auto &pw = pending_texture_writes[i];

        const VkImageView sv = (pw.sampled_view != VK_NULL_HANDLE) ? pw.sampled_view : dummy_sampled;
        const VkImageView stv = (pw.storage_view != VK_NULL_HANDLE)   ? pw.storage_view
                                : (pw.sampled_view != VK_NULL_HANDLE) ? pw.sampled_view
                                                                      : dummy_storage;

        sampled_infos[i] = {.sampler = VK_NULL_HANDLE, .imageView = sv, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        storage_infos[i] = {.sampler = VK_NULL_HANDLE, .imageView = stv, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        auto ci = create_info<VkWriteDescriptorSet>();
        ci.dstSet = set;
        ci.dstBinding = 0;
        ci.dstArrayElement = pw.pool_index;
        ci.descriptorCount = 1;
        ci.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        ci.pImageInfo = &sampled_infos[i];
        writes.emplace_back(std::move(ci));

        if (pw.pool_index < max_storage_images) {
            auto ci = create_info<VkWriteDescriptorSet>();
            ci.dstSet = set;
            ci.dstBinding = 2;
            ci.dstArrayElement = pw.pool_index;
            ci.descriptorCount = 1;
            ci.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            ci.pImageInfo = &storage_infos[i];
            writes.emplace_back(std::move(ci));
        }

        if (is_cubemap_view(pw.view_type) && pw.pool_index < max_cubemaps) {
            cubemap_indices.push_back(i);
            cubemap_infos.push_back(sampled_infos[i]);
            auto ci = create_info<VkWriteDescriptorSet>();
            ci.dstSet = set;
            ci.dstBinding = 4;
            ci.dstArrayElement = pw.pool_index;
            ci.descriptorCount = 1;
            ci.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            ci.pImageInfo = &cubemap_infos.back();
            writes.emplace_back(std::move(ci));
        }

        if (is_3d_view(pw.view_type) && pw.pool_index < max_3d_images) {
            image_3d_infos.push_back(sampled_infos[i]);
            auto ci = create_info<VkWriteDescriptorSet>();
            ci.dstSet = set;
            ci.dstBinding = 5;
            ci.dstArrayElement = pw.pool_index;
            ci.descriptorCount = 1;
            ci.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            ci.pImageInfo = &image_3d_infos.back();
            writes.emplace_back(std::move(ci));
        }
    }

    vkUpdateDescriptorSets(device, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
    pending_texture_writes.clear();
}

auto BindlessSet::repopulate_if_needed(TexturePool &textures, SamplerPool &samplers,
                                       ComparisonSamplerPool &comparison_samplers) -> bool {
    // Fast path: incremental only
    if (!need_repopulate) [[likely]] {
        if (!pending_texture_writes.empty()) {
            // Need dummy views — grab from slot 0 as before
            auto &dummy_texture = *textures.get(textures.get_handle(0));
            const VkImageView dummy_sampled = dummy_texture.sampled_view;
            const VkImageView dummy_storage = (dummy_texture.storage_view != VK_NULL_HANDLE)
                                                      ? dummy_texture.storage_view
                                                      : dummy_texture.sampled_view;
            flush_pending_writes(dummy_sampled, dummy_storage);
        }
        return false;
    }

    pending_texture_writes.clear();

    NANO_SCOPE("Resize and grow bindless set.");
    const auto did_resize = grow_if_needed(textures.num_objects(), samplers.num_objects(), textures.num_objects(), 0u);

    auto &dummy_sampler = *samplers.get(samplers.get_handle(0));
    auto &dummy_texture = *textures.get(textures.get_handle(0));

    const VkImageView &dummy_sampled_view = dummy_texture.sampled_view;
    const VkImageView &dummy_storage_view =
            (dummy_texture.storage_view != VK_NULL_HANDLE) ? dummy_texture.storage_view : dummy_texture.sampled_view;
    const VkSampler &dummy_vk_sampler = dummy_sampler;

    std::vector<VkDescriptorImageInfo> sampled_infos(max_textures);
    std::vector<VkDescriptorImageInfo> storage_infos(max_storage_images);
    std::vector<VkDescriptorImageInfo> sampler_infos(max_samplers);
    std::vector<VkDescriptorImageInfo> comparison_sampler_infos(max_comparison_samplers);
    std::vector<VkDescriptorImageInfo> cubemap_infos(max_cubemaps);
    std::vector<VkDescriptorImageInfo> image_3d_infos(max_3d_images);

    for (u32 i = 0; i < max_textures; ++i) {
        sampled_infos[i] = VkDescriptorImageInfo{
                .sampler = VK_NULL_HANDLE,
                .imageView = dummy_sampled_view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
    }

    for (u32 i = 0; i < max_storage_images; ++i) {
        storage_infos[i] = VkDescriptorImageInfo{
                .sampler = VK_NULL_HANDLE,
                .imageView = dummy_storage_view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
    }

    for (u32 i = 0; i < max_samplers; ++i) {
        sampler_infos[i] = VkDescriptorImageInfo{
                .sampler = dummy_vk_sampler,
                .imageView = VK_NULL_HANDLE,
                .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
    }

    for (u32 i = 0; i < max_cubemaps; ++i) {
        cubemap_infos[i] = VkDescriptorImageInfo{
                .sampler = VK_NULL_HANDLE,
                .imageView = dummy_sampled_view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
    }

    for (u32 i = 0; i < max_3d_images; ++i) {
        image_3d_infos[i] = VkDescriptorImageInfo{
                .sampler = VK_NULL_HANDLE,
                .imageView = dummy_sampled_view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
    }

    {
        u32 idx = 0;
        const u32 limit = std::min<u32>(static_cast<u32>(textures.data().size()), max_textures);

        for (const auto &tex_entry: textures.data()) {
            if (idx >= limit) {
                break;
            }

            const auto &texture = tex_entry.object;

            if (texture.sampled_view != VK_NULL_HANDLE) {
                sampled_infos[idx] = VkDescriptorImageInfo{
                        .sampler = VK_NULL_HANDLE,
                        .imageView = texture.sampled_view,
                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                };
            }

            if (idx < max_storage_images && texture.storage_view != VK_NULL_HANDLE) {
                storage_infos[idx] = VkDescriptorImageInfo{
                        .sampler = VK_NULL_HANDLE,
                        .imageView = texture.storage_view,
                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                };
            }

            ++idx;
        }
    }

    {
        u32 idx = 0;
        const u32 limit = std::min<u32>(static_cast<u32>(samplers.data().size()), max_samplers);

        for (const auto &sampler_entry: samplers.data()) {
            if (idx >= limit) {
                break;
            }

            const VkSampler s = sampler_entry.object;
            sampler_infos[idx] = VkDescriptorImageInfo{
                    .sampler = (s != VK_NULL_HANDLE) ? s : dummy_vk_sampler,
                    .imageView = VK_NULL_HANDLE,
                    .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };

            ++idx;
        }
    }

    {
        u32 idx = 0;
        const u32 limit = std::min<u32>(static_cast<u32>(comparison_samplers.data().size()), max_comparison_samplers);

        for (u32 i = 0; i < max_comparison_samplers; ++i) {
            comparison_sampler_infos[i] = VkDescriptorImageInfo{
                    .sampler = dummy_vk_sampler,
                    .imageView = VK_NULL_HANDLE,
                    .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
        }

        for (const auto &sampler_entry: comparison_samplers.data()) {
            if (idx >= limit) {
                break;
            }

            const VkSampler s = sampler_entry.object;
            comparison_sampler_infos[idx] = VkDescriptorImageInfo{
                    .sampler = (s != VK_NULL_HANDLE) ? s : dummy_vk_sampler,
                    .imageView = VK_NULL_HANDLE,
                    .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };

            ++idx;
        }
    }

    {
        u32 idx = 0;
        const u32 limit = std::min<u32>(static_cast<u32>(textures.data().size()), max_cubemaps);

        for (const auto &tex_entry: textures.data()) {
            if (idx >= limit)
                break;

            const auto &texture = tex_entry.object;

            if (texture.sampled_view != VK_NULL_HANDLE && is_cubemap_view(texture.sampled_view_type)) {
                cubemap_infos[idx] = VkDescriptorImageInfo{
                        .sampler = VK_NULL_HANDLE,
                        .imageView = texture.sampled_view,
                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                };
            }
            ++idx;
        }
    }

    {
        u32 idx = 0;
        const u32 limit = std::min<u32>(static_cast<u32>(textures.data().size()), max_3d_images);

        for (const auto &tex_entry: textures.data()) {
            if (idx >= limit)
                break;

            const auto &texture = tex_entry.object;

            if (texture.sampled_view != VK_NULL_HANDLE && is_3d_view(texture.sampled_view_type)) {
                image_3d_infos[idx] = VkDescriptorImageInfo{
                        .sampler = VK_NULL_HANDLE,
                        .imageView = texture.sampled_view,
                        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                };
            }
            ++idx;
        }
    }

    std::array<VkWriteDescriptorSet, 6> writes{};
    u32 num_writes = 0;

    writes[num_writes++] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = max_textures,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .pImageInfo = sampled_infos.data(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
    };

    writes[num_writes++] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = max_samplers,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .pImageInfo = sampler_infos.data(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
    };

    writes[num_writes++] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = max_storage_images,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = storage_infos.data(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
    };

    writes[num_writes++] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = max_comparison_samplers,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .pImageInfo = comparison_sampler_infos.data(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
    };

    writes[num_writes++] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 4,
            .dstArrayElement = 0,
            .descriptorCount = max_cubemaps,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .pImageInfo = cubemap_infos.data(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
    };

    writes[num_writes++] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 5,
            .dstArrayElement = 0,
            .descriptorCount = max_3d_images,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .pImageInfo = image_3d_infos.data(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
    };


    vkUpdateDescriptorSets(device, num_writes, writes.data(), 0, nullptr);
    need_repopulate = false;

    return did_resize;
}

auto BindlessSet::recreate() -> void {
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    bindings.push_back({.binding = 0u,
                        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                        .descriptorCount = max_textures,
                        .stageFlags = VK_SHADER_STAGE_ALL,
                        .pImmutableSamplers = nullptr});

    bindings.push_back({.binding = 1u,
                        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                        .descriptorCount = max_samplers,
                        .stageFlags = VK_SHADER_STAGE_ALL,
                        .pImmutableSamplers = nullptr});

    bindings.push_back({.binding = 2u,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        .descriptorCount = max_storage_images,
                        .stageFlags = VK_SHADER_STAGE_ALL,
                        .pImmutableSamplers = nullptr});

    bindings.push_back({.binding = 3u,
                        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                        .descriptorCount = max_comparison_samplers,
                        .stageFlags = VK_SHADER_STAGE_ALL,
                        .pImmutableSamplers = nullptr});

    bindings.push_back({.binding = 4u,
                        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                        .descriptorCount = max_cubemaps,
                        .stageFlags = VK_SHADER_STAGE_ALL,
                        .pImmutableSamplers = nullptr});

    bindings.push_back({.binding = 5u,
                        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                        .descriptorCount = max_3d_images,
                        .stageFlags = VK_SHADER_STAGE_ALL,
                        .pImmutableSamplers = nullptr});

    bool accel_enabled = (max_accel_structs > 0);
    if (accel_enabled) {
        bindings.push_back({.binding = 6u,
                            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                            .descriptorCount = max_accel_structs,
                            .stageFlags = VK_SHADER_STAGE_ALL,
                            .pImmutableSamplers = nullptr});
    }

    VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                     VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                     VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;

    std::vector<VkDescriptorBindingFlags> binding_flags(bindings.size(), flags);

    VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .pNext = nullptr,
            .bindingCount = static_cast<u32>(binding_flags.size()),
            .pBindingFlags = binding_flags.data()};

    VkDescriptorSetLayoutCreateInfo lci{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                        .pNext = &bfci,
                                        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                        .bindingCount = static_cast<u32>(bindings.size()),
                                        .pBindings = bindings.data()};

    vk_check(vkCreateDescriptorSetLayout(device, &lci, nullptr, &layout));

    std::vector<VkDescriptorPoolSize> pool_sizes;
    pool_sizes.push_back({.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = max_textures});
    pool_sizes.push_back({.type = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = max_samplers});
    pool_sizes.push_back({.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = max_storage_images});
    pool_sizes.push_back({.type = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = max_comparison_samplers});
    pool_sizes.push_back({.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = max_cubemaps});
    pool_sizes.push_back({.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = max_3d_images});
    if (accel_enabled) {
        pool_sizes.push_back(
                {.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, .descriptorCount = max_accel_structs});
    }

    VkPipelineLayoutCreateInfo plci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1u,
            .pSetLayouts = &layout,
            .pushConstantRangeCount = 0u,
            .pPushConstantRanges = nullptr,
    };

    vk_check(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout));
    set_debug_name(device, VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipeline_layout, "bindless_pipeline_layout");


    VkDescriptorPoolCreateInfo pci{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                   .pNext = nullptr,
                                   .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                                   .maxSets = 1u,
                                   .poolSizeCount = static_cast<u32>(pool_sizes.size()),
                                   .pPoolSizes = pool_sizes.data()};

    vk_check(vkCreateDescriptorPool(device, &pci, nullptr, &pool));
    set_debug_name(device, VK_OBJECT_TYPE_DESCRIPTOR_POOL, pool, "bindless_descriptor_pool");

    VkDescriptorSetAllocateInfo dai{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                    .pNext = nullptr,
                                    .descriptorPool = pool,
                                    .descriptorSetCount = 1u,
                                    .pSetLayouts = &layout};

    vk_check(vkAllocateDescriptorSets(device, &dai, &set));
    set_debug_name(device, VK_OBJECT_TYPE_DESCRIPTOR_SET, set, "bindless_descriptor_set");
}
