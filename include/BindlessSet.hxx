#pragma once

#define NO_MIN_MAX

#include "RenderContext.hxx"
#include "Types.hxx"

#include <volk.h>

struct BindlessCaps {
    u32 max_textures;
    u32 max_samplers;
    u32 max_storage_images;
    u32 max_accel_structs;
};

namespace detail {

    inline auto is_depth_format(VkFormat f) -> bool {
        switch (f) {
            case VK_FORMAT_D16_UNORM:
            case VK_FORMAT_D32_SFLOAT:
            case VK_FORMAT_D16_UNORM_S8_UINT:
            case VK_FORMAT_D24_UNORM_S8_UINT:
            case VK_FORMAT_D32_SFLOAT_S8_UINT:
                return true;
            default:
                return false;
        }
    }
} // namespace detail

inline auto query_bindless_caps(VkPhysicalDevice pd) -> BindlessCaps {
    VkPhysicalDeviceVulkan12Properties props12{};
    props12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;

    VkPhysicalDeviceAccelerationStructurePropertiesKHR accel_props{};
    accel_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &props12;

    props12.pNext = &accel_props;

    vkGetPhysicalDeviceProperties2(pd, &props2);

    return BindlessCaps{.max_textures = props12.maxDescriptorSetUpdateAfterBindSampledImages,
                        .max_samplers = props12.maxDescriptorSetUpdateAfterBindSamplers,
                        .max_storage_images = props12.maxDescriptorSetUpdateAfterBindStorageImages,
                        .max_accel_structs = accel_props.maxPerStageDescriptorAccelerationStructures};
}

struct BindlessSet {
    VkDescriptorSetLayout layout{VK_NULL_HANDLE};
    VkDescriptorPool pool{VK_NULL_HANDLE};
    VkDescriptorSet set{VK_NULL_HANDLE};

    u32 max_textures{1};
    u32 max_samplers{1};
    u32 max_storage_images{1};
    u32 max_accel_structs{1};

    bool need_repopulate{false};

    VkDevice device{VK_NULL_HANDLE};
    BindlessCaps caps{};

    auto init(VkDevice dev, BindlessCaps const &caps_init, u32 initial_textures, u32 initial_samplers,
              u32 initial_storage_images, u32 initial_accel_structs) -> void {
        device = dev;
        caps = caps_init;

        max_textures = std::min(initial_textures, caps.max_textures);
        max_samplers = std::min(initial_samplers, caps.max_samplers);
        max_storage_images = std::min(initial_storage_images, caps.max_storage_images);
        max_accel_structs = std::min(initial_accel_structs, caps.max_accel_structs);

        recreate();
    }

    auto destroy() -> void {
        if (device == VK_NULL_HANDLE)
            return;

        vkDeviceWaitIdle(device);

        if (pool)
            vkDestroyDescriptorPool(device, pool, nullptr);
        if (layout)
            vkDestroyDescriptorSetLayout(device, layout, nullptr);
        pool = VK_NULL_HANDLE;
        layout = VK_NULL_HANDLE;
        set = VK_NULL_HANDLE;
    }

    auto grow_if_needed(u32 req_textures, u32 req_samplers, u32 req_storage, u32 req_accel) -> bool {
        bool grow = false;

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
        grow_and_clamp(max_accel_structs, req_accel, caps.max_accel_structs);

        if (!grow && layout != VK_NULL_HANDLE) {
            return false;
        }

        destroy();
        need_repopulate = true;
        recreate();
        return true;
    }


    auto repopulate_if_needed(TexturePool &textures, SamplerPool &samplers) -> void {
        if (!need_repopulate) [[likely]]
            return;

        // Ensure descriptor arrays are big enough for current pools (your existing policy)
        grow_if_needed(textures.num_objects(), samplers.num_objects(), textures.num_objects(), 0u);

        // Guard slot 0 must exist and be valid
        auto &dummy_sampler = *samplers.get(samplers.get_handle(0));
        auto &dummy_texture = *textures.get(textures.get_handle(0));

        const VkImageView &dummy_sampled_view = dummy_texture.sampled_view;
        const VkImageView &dummy_storage_view = (dummy_texture.storage_view != VK_NULL_HANDLE)
                                                        ? dummy_texture.storage_view
                                                        : dummy_texture.sampled_view;
        const VkSampler &dummy_vk_sampler = dummy_sampler;

        // Fill whole descriptor arrays with defaults (no sparse holes)
        std::vector<VkDescriptorImageInfo> sampled_infos(max_textures);
        std::vector<VkDescriptorImageInfo> storage_infos(max_storage_images);
        std::vector<VkDescriptorImageInfo> sampler_infos(max_samplers);

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

        // Now overwrite defaults for slots we actually have, CLAMPED to descriptor sizes.
        // Important: descriptor index == pool slot index.
        {
            u32 idx = 0;
            const u32 limit = std::min<u32>(static_cast<u32>(textures.data().size()), max_textures);

            for (const auto &tex_entry: textures.data()) {
                if (idx >= limit) {
                    break;
                }

                const auto &texture = tex_entry.object;

                // If your pool can contain invalid entries, keep dummy for those.
                // (You can add your own "is_valid" predicate here if you have one.)
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

        VkWriteDescriptorSet writes[3]{};
        u32 num_writes = 0;

        writes[num_writes++] = VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                    .pNext = nullptr,
                                                    .dstSet = set,
                                                    .dstBinding = 0,
                                                    .dstArrayElement = 0,
                                                    .descriptorCount = max_textures,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                                    .pImageInfo = sampled_infos.data(),
                                                    .pBufferInfo = nullptr,
                                                    .pTexelBufferView = nullptr};

        writes[num_writes++] = VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                    .pNext = nullptr,

                                                    .dstSet = set,
                                                    .dstBinding = 1,
                                                    .dstArrayElement = 0,
                                                    .descriptorCount = max_samplers,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                                                    .pImageInfo = sampler_infos.data(),
                                                    .pBufferInfo = nullptr,
                                                    .pTexelBufferView = nullptr};

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

        vkUpdateDescriptorSets(device, num_writes, writes, 0, nullptr);
        need_repopulate = false;
    }


private:
    auto recreate() -> void {
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

        bool accel_enabled = (max_accel_structs > 0);
        if (accel_enabled) {
            bindings.push_back({.binding = 3u,
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
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, max_textures});
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLER, max_samplers});
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, max_storage_images});
        if (accel_enabled) {
            pool_sizes.push_back({VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, max_accel_structs});
        }

        VkDescriptorPoolCreateInfo pci{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .pNext = nullptr,
                                       .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                                       .maxSets = 1u,
                                       .poolSizeCount = static_cast<u32>(pool_sizes.size()),
                                       .pPoolSizes = pool_sizes.data()};

        vk_check(vkCreateDescriptorPool(device, &pci, nullptr, &pool));

        VkDescriptorSetAllocateInfo dai{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .pNext = nullptr,
                                        .descriptorPool = pool,
                                        .descriptorSetCount = 1u,
                                        .pSetLayouts = &layout};

        vk_check(vkAllocateDescriptorSets(device, &dai, &set));
    }
};
