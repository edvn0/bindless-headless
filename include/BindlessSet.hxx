#pragma once

#include "BindlessHeadless.hxx"
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

struct PendingTextureWrite {
    u32 pool_index;
    VkImageView sampled_view; // VK_NULL_HANDLE = use dummy
    VkImageView storage_view; // VK_NULL_HANDLE = use dummy
    VkImageViewType view_type;
};


struct BindlessSet {
    VkDescriptorSetLayout layout{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
    VkDescriptorPool pool{VK_NULL_HANDLE};
    VkDescriptorSet set{VK_NULL_HANDLE};

    u32 max_textures{1};
    u32 max_samplers{1};
    u32 max_comparison_samplers{1};
    u32 max_storage_images{1};
    u32 max_accel_structs{1};
    u32 max_cubemaps{1};
    u32 max_3d_images{1};

    bool need_repopulate{false};

    VkDevice device{VK_NULL_HANDLE};
    BindlessCaps caps{};

    std::vector<PendingTextureWrite> pending_texture_writes;
    auto flush_pending_writes(VkImageView dummy_sampled, VkImageView dummy_storage) -> void;

    auto init(VkDevice dev, BindlessCaps const &caps_init, u32 initial_textures, u32 initial_samplers,
              u32 initial_comparison_samplers, u32 initial_storage_images, u32 initial_accel_structs) -> void;
    auto destroy() -> void;
    auto grow_if_needed(u32 req_textures, u32 req_samplers, u32 req_storage, u32 req_accel) -> bool;
    auto repopulate_if_needed(TexturePool &textures, SamplerPool &samplers, ComparisonSamplerPool &comparison_samplers)
            -> bool;


private:
    auto recreate() -> void;
};
