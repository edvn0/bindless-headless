// AUTO-GENERATED FILE — DO NOT EDIT
#pragma once
#include <vulkan/vulkan.h>

template<typename T>
struct CreateInfoFor;

template<>
struct CreateInfoFor<VkApplicationInfo> {
    static consteval VkApplicationInfo default_() {
        VkApplicationInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceQueueCreateInfo> {
    static consteval VkDeviceQueueCreateInfo default_() {
        VkDeviceQueueCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceCreateInfo> {
    static consteval VkDeviceCreateInfo default_() {
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkInstanceCreateInfo> {
    static consteval VkInstanceCreateInfo default_() {
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMemoryAllocateInfo> {
    static consteval VkMemoryAllocateInfo default_() {
        VkMemoryAllocateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMappedMemoryRange> {
    static consteval VkMappedMemoryRange default_() {
        VkMappedMemoryRange ci{};
        ci.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkWriteDescriptorSet> {
    static consteval VkWriteDescriptorSet default_() {
        VkWriteDescriptorSet ci{};
        ci.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCopyDescriptorSet> {
    static consteval VkCopyDescriptorSet default_() {
        VkCopyDescriptorSet ci{};
        ci.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBufferUsageFlags2CreateInfo> {
    static consteval VkBufferUsageFlags2CreateInfo default_() {
        VkBufferUsageFlags2CreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBufferCreateInfo> {
    static consteval VkBufferCreateInfo default_() {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBufferViewCreateInfo> {
    static consteval VkBufferViewCreateInfo default_() {
        VkBufferViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMemoryBarrier> {
    static consteval VkMemoryBarrier default_() {
        VkMemoryBarrier ci{};
        ci.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBufferMemoryBarrier> {
    static consteval VkBufferMemoryBarrier default_() {
        VkBufferMemoryBarrier ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageMemoryBarrier> {
    static consteval VkImageMemoryBarrier default_() {
        VkImageMemoryBarrier ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageCreateInfo> {
    static consteval VkImageCreateInfo default_() {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageViewCreateInfo> {
    static consteval VkImageViewCreateInfo default_() {
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBindSparseInfo> {
    static consteval VkBindSparseInfo default_() {
        VkBindSparseInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkShaderModuleCreateInfo> {
    static consteval VkShaderModuleCreateInfo default_() {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDescriptorSetLayoutCreateInfo> {
    static consteval VkDescriptorSetLayoutCreateInfo default_() {
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDescriptorPoolCreateInfo> {
    static consteval VkDescriptorPoolCreateInfo default_() {
        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDescriptorSetAllocateInfo> {
    static consteval VkDescriptorSetAllocateInfo default_() {
        VkDescriptorSetAllocateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineShaderStageCreateInfo> {
    static consteval VkPipelineShaderStageCreateInfo default_() {
        VkPipelineShaderStageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkComputePipelineCreateInfo> {
    static consteval VkComputePipelineCreateInfo default_() {
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineCreateFlags2CreateInfo> {
    static consteval VkPipelineCreateFlags2CreateInfo default_() {
        VkPipelineCreateFlags2CreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineVertexInputStateCreateInfo> {
    static consteval VkPipelineVertexInputStateCreateInfo default_() {
        VkPipelineVertexInputStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineInputAssemblyStateCreateInfo> {
    static consteval VkPipelineInputAssemblyStateCreateInfo default_() {
        VkPipelineInputAssemblyStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineTessellationStateCreateInfo> {
    static consteval VkPipelineTessellationStateCreateInfo default_() {
        VkPipelineTessellationStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineViewportStateCreateInfo> {
    static consteval VkPipelineViewportStateCreateInfo default_() {
        VkPipelineViewportStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineRasterizationStateCreateInfo> {
    static consteval VkPipelineRasterizationStateCreateInfo default_() {
        VkPipelineRasterizationStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineMultisampleStateCreateInfo> {
    static consteval VkPipelineMultisampleStateCreateInfo default_() {
        VkPipelineMultisampleStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineColorBlendStateCreateInfo> {
    static consteval VkPipelineColorBlendStateCreateInfo default_() {
        VkPipelineColorBlendStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineDynamicStateCreateInfo> {
    static consteval VkPipelineDynamicStateCreateInfo default_() {
        VkPipelineDynamicStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineDepthStencilStateCreateInfo> {
    static consteval VkPipelineDepthStencilStateCreateInfo default_() {
        VkPipelineDepthStencilStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkGraphicsPipelineCreateInfo> {
    static consteval VkGraphicsPipelineCreateInfo default_() {
        VkGraphicsPipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineCacheCreateInfo> {
    static consteval VkPipelineCacheCreateInfo default_() {
        VkPipelineCacheCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineLayoutCreateInfo> {
    static consteval VkPipelineLayoutCreateInfo default_() {
        VkPipelineLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSamplerCreateInfo> {
    static consteval VkSamplerCreateInfo default_() {
        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCommandPoolCreateInfo> {
    static consteval VkCommandPoolCreateInfo default_() {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCommandBufferAllocateInfo> {
    static consteval VkCommandBufferAllocateInfo default_() {
        VkCommandBufferAllocateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCommandBufferInheritanceInfo> {
    static consteval VkCommandBufferInheritanceInfo default_() {
        VkCommandBufferInheritanceInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCommandBufferBeginInfo> {
    static consteval VkCommandBufferBeginInfo default_() {
        VkCommandBufferBeginInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkRenderPassBeginInfo> {
    static consteval VkRenderPassBeginInfo default_() {
        VkRenderPassBeginInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkRenderPassCreateInfo> {
    static consteval VkRenderPassCreateInfo default_() {
        VkRenderPassCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkEventCreateInfo> {
    static consteval VkEventCreateInfo default_() {
        VkEventCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkFenceCreateInfo> {
    static consteval VkFenceCreateInfo default_() {
        VkFenceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSemaphoreCreateInfo> {
    static consteval VkSemaphoreCreateInfo default_() {
        VkSemaphoreCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkQueryPoolCreateInfo> {
    static consteval VkQueryPoolCreateInfo default_() {
        VkQueryPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkFramebufferCreateInfo> {
    static consteval VkFramebufferCreateInfo default_() {
        VkFramebufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSubmitInfo> {
    static consteval VkSubmitInfo default_() {
        VkSubmitInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDevicePrivateDataCreateInfo> {
    static consteval VkDevicePrivateDataCreateInfo default_() {
        VkDevicePrivateDataCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPrivateDataSlotCreateInfo> {
    static consteval VkPrivateDataSlotCreateInfo default_() {
        VkPrivateDataSlotCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDevicePrivateDataFeatures> {
    static consteval VkPhysicalDevicePrivateDataFeatures default_() {
        VkPhysicalDevicePrivateDataFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceFeatures2> {
    static consteval VkPhysicalDeviceFeatures2 default_() {
        VkPhysicalDeviceFeatures2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceProperties2> {
    static consteval VkPhysicalDeviceProperties2 default_() {
        VkPhysicalDeviceProperties2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkFormatProperties2> {
    static consteval VkFormatProperties2 default_() {
        VkFormatProperties2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageFormatProperties2> {
    static consteval VkImageFormatProperties2 default_() {
        VkImageFormatProperties2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceImageFormatInfo2> {
    static consteval VkPhysicalDeviceImageFormatInfo2 default_() {
        VkPhysicalDeviceImageFormatInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkQueueFamilyProperties2> {
    static consteval VkQueueFamilyProperties2 default_() {
        VkQueueFamilyProperties2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceMemoryProperties2> {
    static consteval VkPhysicalDeviceMemoryProperties2 default_() {
        VkPhysicalDeviceMemoryProperties2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSparseImageFormatProperties2> {
    static consteval VkSparseImageFormatProperties2 default_() {
        VkSparseImageFormatProperties2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceSparseImageFormatInfo2> {
    static consteval VkPhysicalDeviceSparseImageFormatInfo2 default_() {
        VkPhysicalDeviceSparseImageFormatInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDevicePushDescriptorProperties> {
    static consteval VkPhysicalDevicePushDescriptorProperties default_() {
        VkPhysicalDevicePushDescriptorProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceDriverProperties> {
    static consteval VkPhysicalDeviceDriverProperties default_() {
        VkPhysicalDeviceDriverProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVariablePointersFeatures> {
    static consteval VkPhysicalDeviceVariablePointersFeatures default_() {
        VkPhysicalDeviceVariablePointersFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceExternalImageFormatInfo> {
    static consteval VkPhysicalDeviceExternalImageFormatInfo default_() {
        VkPhysicalDeviceExternalImageFormatInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkExternalImageFormatProperties> {
    static consteval VkExternalImageFormatProperties default_() {
        VkExternalImageFormatProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceExternalBufferInfo> {
    static consteval VkPhysicalDeviceExternalBufferInfo default_() {
        VkPhysicalDeviceExternalBufferInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkExternalBufferProperties> {
    static consteval VkExternalBufferProperties default_() {
        VkExternalBufferProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceIDProperties> {
    static consteval VkPhysicalDeviceIDProperties default_() {
        VkPhysicalDeviceIDProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkExternalMemoryImageCreateInfo> {
    static consteval VkExternalMemoryImageCreateInfo default_() {
        VkExternalMemoryImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkExternalMemoryBufferCreateInfo> {
    static consteval VkExternalMemoryBufferCreateInfo default_() {
        VkExternalMemoryBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkExportMemoryAllocateInfo> {
    static consteval VkExportMemoryAllocateInfo default_() {
        VkExportMemoryAllocateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceExternalSemaphoreInfo> {
    static consteval VkPhysicalDeviceExternalSemaphoreInfo default_() {
        VkPhysicalDeviceExternalSemaphoreInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkExternalSemaphoreProperties> {
    static consteval VkExternalSemaphoreProperties default_() {
        VkExternalSemaphoreProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkExportSemaphoreCreateInfo> {
    static consteval VkExportSemaphoreCreateInfo default_() {
        VkExportSemaphoreCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceExternalFenceInfo> {
    static consteval VkPhysicalDeviceExternalFenceInfo default_() {
        VkPhysicalDeviceExternalFenceInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkExternalFenceProperties> {
    static consteval VkExternalFenceProperties default_() {
        VkExternalFenceProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkExportFenceCreateInfo> {
    static consteval VkExportFenceCreateInfo default_() {
        VkExportFenceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceMultiviewFeatures> {
    static consteval VkPhysicalDeviceMultiviewFeatures default_() {
        VkPhysicalDeviceMultiviewFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceMultiviewProperties> {
    static consteval VkPhysicalDeviceMultiviewProperties default_() {
        VkPhysicalDeviceMultiviewProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkRenderPassMultiviewCreateInfo> {
    static consteval VkRenderPassMultiviewCreateInfo default_() {
        VkRenderPassMultiviewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceGroupProperties> {
    static consteval VkPhysicalDeviceGroupProperties default_() {
        VkPhysicalDeviceGroupProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMemoryAllocateFlagsInfo> {
    static consteval VkMemoryAllocateFlagsInfo default_() {
        VkMemoryAllocateFlagsInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBindBufferMemoryInfo> {
    static consteval VkBindBufferMemoryInfo default_() {
        VkBindBufferMemoryInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBindBufferMemoryDeviceGroupInfo> {
    static consteval VkBindBufferMemoryDeviceGroupInfo default_() {
        VkBindBufferMemoryDeviceGroupInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBindImageMemoryInfo> {
    static consteval VkBindImageMemoryInfo default_() {
        VkBindImageMemoryInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBindImageMemoryDeviceGroupInfo> {
    static consteval VkBindImageMemoryDeviceGroupInfo default_() {
        VkBindImageMemoryDeviceGroupInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceGroupRenderPassBeginInfo> {
    static consteval VkDeviceGroupRenderPassBeginInfo default_() {
        VkDeviceGroupRenderPassBeginInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_RENDER_PASS_BEGIN_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceGroupCommandBufferBeginInfo> {
    static consteval VkDeviceGroupCommandBufferBeginInfo default_() {
        VkDeviceGroupCommandBufferBeginInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceGroupSubmitInfo> {
    static consteval VkDeviceGroupSubmitInfo default_() {
        VkDeviceGroupSubmitInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceGroupBindSparseInfo> {
    static consteval VkDeviceGroupBindSparseInfo default_() {
        VkDeviceGroupBindSparseInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_BIND_SPARSE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceGroupDeviceCreateInfo> {
    static consteval VkDeviceGroupDeviceCreateInfo default_() {
        VkDeviceGroupDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDescriptorUpdateTemplateCreateInfo> {
    static consteval VkDescriptorUpdateTemplateCreateInfo default_() {
        VkDescriptorUpdateTemplateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkRenderPassInputAttachmentAspectCreateInfo> {
    static consteval VkRenderPassInputAttachmentAspectCreateInfo default_() {
        VkRenderPassInputAttachmentAspectCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDevice16BitStorageFeatures> {
    static consteval VkPhysicalDevice16BitStorageFeatures default_() {
        VkPhysicalDevice16BitStorageFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceSubgroupProperties> {
    static consteval VkPhysicalDeviceSubgroupProperties default_() {
        VkPhysicalDeviceSubgroupProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures> {
    static consteval VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures default_() {
        VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBufferMemoryRequirementsInfo2> {
    static consteval VkBufferMemoryRequirementsInfo2 default_() {
        VkBufferMemoryRequirementsInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceBufferMemoryRequirements> {
    static consteval VkDeviceBufferMemoryRequirements default_() {
        VkDeviceBufferMemoryRequirements ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageMemoryRequirementsInfo2> {
    static consteval VkImageMemoryRequirementsInfo2 default_() {
        VkImageMemoryRequirementsInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageSparseMemoryRequirementsInfo2> {
    static consteval VkImageSparseMemoryRequirementsInfo2 default_() {
        VkImageSparseMemoryRequirementsInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceImageMemoryRequirements> {
    static consteval VkDeviceImageMemoryRequirements default_() {
        VkDeviceImageMemoryRequirements ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMemoryRequirements2> {
    static consteval VkMemoryRequirements2 default_() {
        VkMemoryRequirements2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSparseImageMemoryRequirements2> {
    static consteval VkSparseImageMemoryRequirements2 default_() {
        VkSparseImageMemoryRequirements2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_SPARSE_IMAGE_MEMORY_REQUIREMENTS_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDevicePointClippingProperties> {
    static consteval VkPhysicalDevicePointClippingProperties default_() {
        VkPhysicalDevicePointClippingProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMemoryDedicatedRequirements> {
    static consteval VkMemoryDedicatedRequirements default_() {
        VkMemoryDedicatedRequirements ci{};
        ci.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMemoryDedicatedAllocateInfo> {
    static consteval VkMemoryDedicatedAllocateInfo default_() {
        VkMemoryDedicatedAllocateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageViewUsageCreateInfo> {
    static consteval VkImageViewUsageCreateInfo default_() {
        VkImageViewUsageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineTessellationDomainOriginStateCreateInfo> {
    static consteval VkPipelineTessellationDomainOriginStateCreateInfo default_() {
        VkPipelineTessellationDomainOriginStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSamplerYcbcrConversionInfo> {
    static consteval VkSamplerYcbcrConversionInfo default_() {
        VkSamplerYcbcrConversionInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSamplerYcbcrConversionCreateInfo> {
    static consteval VkSamplerYcbcrConversionCreateInfo default_() {
        VkSamplerYcbcrConversionCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBindImagePlaneMemoryInfo> {
    static consteval VkBindImagePlaneMemoryInfo default_() {
        VkBindImagePlaneMemoryInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImagePlaneMemoryRequirementsInfo> {
    static consteval VkImagePlaneMemoryRequirementsInfo default_() {
        VkImagePlaneMemoryRequirementsInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceSamplerYcbcrConversionFeatures> {
    static consteval VkPhysicalDeviceSamplerYcbcrConversionFeatures default_() {
        VkPhysicalDeviceSamplerYcbcrConversionFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSamplerYcbcrConversionImageFormatProperties> {
    static consteval VkSamplerYcbcrConversionImageFormatProperties default_() {
        VkSamplerYcbcrConversionImageFormatProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkProtectedSubmitInfo> {
    static consteval VkProtectedSubmitInfo default_() {
        VkProtectedSubmitInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceProtectedMemoryFeatures> {
    static consteval VkPhysicalDeviceProtectedMemoryFeatures default_() {
        VkPhysicalDeviceProtectedMemoryFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceProtectedMemoryProperties> {
    static consteval VkPhysicalDeviceProtectedMemoryProperties default_() {
        VkPhysicalDeviceProtectedMemoryProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceQueueInfo2> {
    static consteval VkDeviceQueueInfo2 default_() {
        VkDeviceQueueInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceSamplerFilterMinmaxProperties> {
    static consteval VkPhysicalDeviceSamplerFilterMinmaxProperties default_() {
        VkPhysicalDeviceSamplerFilterMinmaxProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSamplerReductionModeCreateInfo> {
    static consteval VkSamplerReductionModeCreateInfo default_() {
        VkSamplerReductionModeCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceInlineUniformBlockFeatures> {
    static consteval VkPhysicalDeviceInlineUniformBlockFeatures default_() {
        VkPhysicalDeviceInlineUniformBlockFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceInlineUniformBlockProperties> {
    static consteval VkPhysicalDeviceInlineUniformBlockProperties default_() {
        VkPhysicalDeviceInlineUniformBlockProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkWriteDescriptorSetInlineUniformBlock> {
    static consteval VkWriteDescriptorSetInlineUniformBlock default_() {
        VkWriteDescriptorSetInlineUniformBlock ci{};
        ci.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDescriptorPoolInlineUniformBlockCreateInfo> {
    static consteval VkDescriptorPoolInlineUniformBlockCreateInfo default_() {
        VkDescriptorPoolInlineUniformBlockCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageFormatListCreateInfo> {
    static consteval VkImageFormatListCreateInfo default_() {
        VkImageFormatListCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceMaintenance3Properties> {
    static consteval VkPhysicalDeviceMaintenance3Properties default_() {
        VkPhysicalDeviceMaintenance3Properties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceMaintenance4Features> {
    static consteval VkPhysicalDeviceMaintenance4Features default_() {
        VkPhysicalDeviceMaintenance4Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceMaintenance4Properties> {
    static consteval VkPhysicalDeviceMaintenance4Properties default_() {
        VkPhysicalDeviceMaintenance4Properties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceMaintenance5Features> {
    static consteval VkPhysicalDeviceMaintenance5Features default_() {
        VkPhysicalDeviceMaintenance5Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceMaintenance5Properties> {
    static consteval VkPhysicalDeviceMaintenance5Properties default_() {
        VkPhysicalDeviceMaintenance5Properties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceMaintenance6Features> {
    static consteval VkPhysicalDeviceMaintenance6Features default_() {
        VkPhysicalDeviceMaintenance6Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceMaintenance6Properties> {
    static consteval VkPhysicalDeviceMaintenance6Properties default_() {
        VkPhysicalDeviceMaintenance6Properties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkRenderingAreaInfo> {
    static consteval VkRenderingAreaInfo default_() {
        VkRenderingAreaInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDERING_AREA_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDescriptorSetLayoutSupport> {
    static consteval VkDescriptorSetLayoutSupport default_() {
        VkDescriptorSetLayoutSupport ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceShaderDrawParametersFeatures> {
    static consteval VkPhysicalDeviceShaderDrawParametersFeatures default_() {
        VkPhysicalDeviceShaderDrawParametersFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceShaderFloat16Int8Features> {
    static consteval VkPhysicalDeviceShaderFloat16Int8Features default_() {
        VkPhysicalDeviceShaderFloat16Int8Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceFloatControlsProperties> {
    static consteval VkPhysicalDeviceFloatControlsProperties default_() {
        VkPhysicalDeviceFloatControlsProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceHostQueryResetFeatures> {
    static consteval VkPhysicalDeviceHostQueryResetFeatures default_() {
        VkPhysicalDeviceHostQueryResetFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceQueueGlobalPriorityCreateInfo> {
    static consteval VkDeviceQueueGlobalPriorityCreateInfo default_() {
        VkDeviceQueueGlobalPriorityCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceGlobalPriorityQueryFeatures> {
    static consteval VkPhysicalDeviceGlobalPriorityQueryFeatures default_() {
        VkPhysicalDeviceGlobalPriorityQueryFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkQueueFamilyGlobalPriorityProperties> {
    static consteval VkQueueFamilyGlobalPriorityProperties default_() {
        VkQueueFamilyGlobalPriorityProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceDescriptorIndexingFeatures> {
    static consteval VkPhysicalDeviceDescriptorIndexingFeatures default_() {
        VkPhysicalDeviceDescriptorIndexingFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceDescriptorIndexingProperties> {
    static consteval VkPhysicalDeviceDescriptorIndexingProperties default_() {
        VkPhysicalDeviceDescriptorIndexingProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDescriptorSetLayoutBindingFlagsCreateInfo> {
    static consteval VkDescriptorSetLayoutBindingFlagsCreateInfo default_() {
        VkDescriptorSetLayoutBindingFlagsCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDescriptorSetVariableDescriptorCountAllocateInfo> {
    static consteval VkDescriptorSetVariableDescriptorCountAllocateInfo default_() {
        VkDescriptorSetVariableDescriptorCountAllocateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDescriptorSetVariableDescriptorCountLayoutSupport> {
    static consteval VkDescriptorSetVariableDescriptorCountLayoutSupport default_() {
        VkDescriptorSetVariableDescriptorCountLayoutSupport ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkAttachmentDescription2> {
    static consteval VkAttachmentDescription2 default_() {
        VkAttachmentDescription2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkAttachmentReference2> {
    static consteval VkAttachmentReference2 default_() {
        VkAttachmentReference2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSubpassDescription2> {
    static consteval VkSubpassDescription2 default_() {
        VkSubpassDescription2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSubpassDependency2> {
    static consteval VkSubpassDependency2 default_() {
        VkSubpassDependency2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkRenderPassCreateInfo2> {
    static consteval VkRenderPassCreateInfo2 default_() {
        VkRenderPassCreateInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSubpassBeginInfo> {
    static consteval VkSubpassBeginInfo default_() {
        VkSubpassBeginInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSubpassEndInfo> {
    static consteval VkSubpassEndInfo default_() {
        VkSubpassEndInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SUBPASS_END_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceTimelineSemaphoreFeatures> {
    static consteval VkPhysicalDeviceTimelineSemaphoreFeatures default_() {
        VkPhysicalDeviceTimelineSemaphoreFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceTimelineSemaphoreProperties> {
    static consteval VkPhysicalDeviceTimelineSemaphoreProperties default_() {
        VkPhysicalDeviceTimelineSemaphoreProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSemaphoreTypeCreateInfo> {
    static consteval VkSemaphoreTypeCreateInfo default_() {
        VkSemaphoreTypeCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkTimelineSemaphoreSubmitInfo> {
    static consteval VkTimelineSemaphoreSubmitInfo default_() {
        VkTimelineSemaphoreSubmitInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSemaphoreWaitInfo> {
    static consteval VkSemaphoreWaitInfo default_() {
        VkSemaphoreWaitInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSemaphoreSignalInfo> {
    static consteval VkSemaphoreSignalInfo default_() {
        VkSemaphoreSignalInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineVertexInputDivisorStateCreateInfo> {
    static consteval VkPipelineVertexInputDivisorStateCreateInfo default_() {
        VkPipelineVertexInputDivisorStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVertexAttributeDivisorProperties> {
    static consteval VkPhysicalDeviceVertexAttributeDivisorProperties default_() {
        VkPhysicalDeviceVertexAttributeDivisorProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDevice8BitStorageFeatures> {
    static consteval VkPhysicalDevice8BitStorageFeatures default_() {
        VkPhysicalDevice8BitStorageFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVulkanMemoryModelFeatures> {
    static consteval VkPhysicalDeviceVulkanMemoryModelFeatures default_() {
        VkPhysicalDeviceVulkanMemoryModelFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceShaderAtomicInt64Features> {
    static consteval VkPhysicalDeviceShaderAtomicInt64Features default_() {
        VkPhysicalDeviceShaderAtomicInt64Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVertexAttributeDivisorFeatures> {
    static consteval VkPhysicalDeviceVertexAttributeDivisorFeatures default_() {
        VkPhysicalDeviceVertexAttributeDivisorFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceDepthStencilResolveProperties> {
    static consteval VkPhysicalDeviceDepthStencilResolveProperties default_() {
        VkPhysicalDeviceDepthStencilResolveProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSubpassDescriptionDepthStencilResolve> {
    static consteval VkSubpassDescriptionDepthStencilResolve default_() {
        VkSubpassDescriptionDepthStencilResolve ci{};
        ci.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageStencilUsageCreateInfo> {
    static consteval VkImageStencilUsageCreateInfo default_() {
        VkImageStencilUsageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceScalarBlockLayoutFeatures> {
    static consteval VkPhysicalDeviceScalarBlockLayoutFeatures default_() {
        VkPhysicalDeviceScalarBlockLayoutFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceUniformBufferStandardLayoutFeatures> {
    static consteval VkPhysicalDeviceUniformBufferStandardLayoutFeatures default_() {
        VkPhysicalDeviceUniformBufferStandardLayoutFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceBufferDeviceAddressFeatures> {
    static consteval VkPhysicalDeviceBufferDeviceAddressFeatures default_() {
        VkPhysicalDeviceBufferDeviceAddressFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBufferDeviceAddressInfo> {
    static consteval VkBufferDeviceAddressInfo default_() {
        VkBufferDeviceAddressInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBufferOpaqueCaptureAddressCreateInfo> {
    static consteval VkBufferOpaqueCaptureAddressCreateInfo default_() {
        VkBufferOpaqueCaptureAddressCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceImagelessFramebufferFeatures> {
    static consteval VkPhysicalDeviceImagelessFramebufferFeatures default_() {
        VkPhysicalDeviceImagelessFramebufferFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkFramebufferAttachmentsCreateInfo> {
    static consteval VkFramebufferAttachmentsCreateInfo default_() {
        VkFramebufferAttachmentsCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkFramebufferAttachmentImageInfo> {
    static consteval VkFramebufferAttachmentImageInfo default_() {
        VkFramebufferAttachmentImageInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkRenderPassAttachmentBeginInfo> {
    static consteval VkRenderPassAttachmentBeginInfo default_() {
        VkRenderPassAttachmentBeginInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceTextureCompressionASTCHDRFeatures> {
    static consteval VkPhysicalDeviceTextureCompressionASTCHDRFeatures default_() {
        VkPhysicalDeviceTextureCompressionASTCHDRFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineCreationFeedbackCreateInfo> {
    static consteval VkPipelineCreationFeedbackCreateInfo default_() {
        VkPipelineCreationFeedbackCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceIndexTypeUint8Features> {
    static consteval VkPhysicalDeviceIndexTypeUint8Features default_() {
        VkPhysicalDeviceIndexTypeUint8Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures> {
    static consteval VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures default_() {
        VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkAttachmentReferenceStencilLayout> {
    static consteval VkAttachmentReferenceStencilLayout default_() {
        VkAttachmentReferenceStencilLayout ci{};
        ci.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_STENCIL_LAYOUT;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkAttachmentDescriptionStencilLayout> {
    static consteval VkAttachmentDescriptionStencilLayout default_() {
        VkAttachmentDescriptionStencilLayout ci{};
        ci.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures> {
    static consteval VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures default_() {
        VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceTexelBufferAlignmentProperties> {
    static consteval VkPhysicalDeviceTexelBufferAlignmentProperties default_() {
        VkPhysicalDeviceTexelBufferAlignmentProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceSubgroupSizeControlFeatures> {
    static consteval VkPhysicalDeviceSubgroupSizeControlFeatures default_() {
        VkPhysicalDeviceSubgroupSizeControlFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceSubgroupSizeControlProperties> {
    static consteval VkPhysicalDeviceSubgroupSizeControlProperties default_() {
        VkPhysicalDeviceSubgroupSizeControlProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo> {
    static consteval VkPipelineShaderStageRequiredSubgroupSizeCreateInfo default_() {
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMemoryOpaqueCaptureAddressAllocateInfo> {
    static consteval VkMemoryOpaqueCaptureAddressAllocateInfo default_() {
        VkMemoryOpaqueCaptureAddressAllocateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceMemoryOpaqueCaptureAddressInfo> {
    static consteval VkDeviceMemoryOpaqueCaptureAddressInfo default_() {
        VkDeviceMemoryOpaqueCaptureAddressInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceLineRasterizationFeatures> {
    static consteval VkPhysicalDeviceLineRasterizationFeatures default_() {
        VkPhysicalDeviceLineRasterizationFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceLineRasterizationProperties> {
    static consteval VkPhysicalDeviceLineRasterizationProperties default_() {
        VkPhysicalDeviceLineRasterizationProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineRasterizationLineStateCreateInfo> {
    static consteval VkPipelineRasterizationLineStateCreateInfo default_() {
        VkPipelineRasterizationLineStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDevicePipelineCreationCacheControlFeatures> {
    static consteval VkPhysicalDevicePipelineCreationCacheControlFeatures default_() {
        VkPhysicalDevicePipelineCreationCacheControlFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVulkan11Features> {
    static consteval VkPhysicalDeviceVulkan11Features default_() {
        VkPhysicalDeviceVulkan11Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVulkan11Properties> {
    static consteval VkPhysicalDeviceVulkan11Properties default_() {
        VkPhysicalDeviceVulkan11Properties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVulkan12Features> {
    static consteval VkPhysicalDeviceVulkan12Features default_() {
        VkPhysicalDeviceVulkan12Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVulkan12Properties> {
    static consteval VkPhysicalDeviceVulkan12Properties default_() {
        VkPhysicalDeviceVulkan12Properties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVulkan13Features> {
    static consteval VkPhysicalDeviceVulkan13Features default_() {
        VkPhysicalDeviceVulkan13Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVulkan13Properties> {
    static consteval VkPhysicalDeviceVulkan13Properties default_() {
        VkPhysicalDeviceVulkan13Properties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVulkan14Features> {
    static consteval VkPhysicalDeviceVulkan14Features default_() {
        VkPhysicalDeviceVulkan14Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVulkan14Properties> {
    static consteval VkPhysicalDeviceVulkan14Properties default_() {
        VkPhysicalDeviceVulkan14Properties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

/*
template<>
struct CreateInfoFor<VkFaultData> {
    static consteval VkFaultData default_() {
        VkFaultData ci{};
        ci.sType = VK_STRUCTURE_TYPE_FAULT_DATA;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkFaultCallbackInfo> {
    static consteval VkFaultCallbackInfo default_() {
        VkFaultCallbackInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FAULT_CALLBACK_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};
*/

template<>
struct CreateInfoFor<VkPhysicalDeviceToolProperties> {
    static consteval VkPhysicalDeviceToolProperties default_() {
        VkPhysicalDeviceToolProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

/*
template<>
struct CreateInfoFor<VkPipelineOfflineCreateInfo> {
    static consteval VkPipelineOfflineCreateInfo default_() {
        VkPipelineOfflineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_OFFLINE_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};
*/

template<>
struct CreateInfoFor<VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures> {
    static consteval VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures default_() {
        VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceImageRobustnessFeatures> {
    static consteval VkPhysicalDeviceImageRobustnessFeatures default_() {
        VkPhysicalDeviceImageRobustnessFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBufferCopy2> {
    static consteval VkBufferCopy2 default_() {
        VkBufferCopy2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageCopy2> {
    static consteval VkImageCopy2 default_() {
        VkImageCopy2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageBlit2> {
    static consteval VkImageBlit2 default_() {
        VkImageBlit2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBufferImageCopy2> {
    static consteval VkBufferImageCopy2 default_() {
        VkBufferImageCopy2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageResolve2> {
    static consteval VkImageResolve2 default_() {
        VkImageResolve2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCopyBufferInfo2> {
    static consteval VkCopyBufferInfo2 default_() {
        VkCopyBufferInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCopyImageInfo2> {
    static consteval VkCopyImageInfo2 default_() {
        VkCopyImageInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBlitImageInfo2> {
    static consteval VkBlitImageInfo2 default_() {
        VkBlitImageInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCopyBufferToImageInfo2> {
    static consteval VkCopyBufferToImageInfo2 default_() {
        VkCopyBufferToImageInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCopyImageToBufferInfo2> {
    static consteval VkCopyImageToBufferInfo2 default_() {
        VkCopyImageToBufferInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkResolveImageInfo2> {
    static consteval VkResolveImageInfo2 default_() {
        VkResolveImageInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceShaderTerminateInvocationFeatures> {
    static consteval VkPhysicalDeviceShaderTerminateInvocationFeatures default_() {
        VkPhysicalDeviceShaderTerminateInvocationFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMemoryBarrier2> {
    static consteval VkMemoryBarrier2 default_() {
        VkMemoryBarrier2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageMemoryBarrier2> {
    static consteval VkImageMemoryBarrier2 default_() {
        VkImageMemoryBarrier2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBufferMemoryBarrier2> {
    static consteval VkBufferMemoryBarrier2 default_() {
        VkBufferMemoryBarrier2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDependencyInfo> {
    static consteval VkDependencyInfo default_() {
        VkDependencyInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSemaphoreSubmitInfo> {
    static consteval VkSemaphoreSubmitInfo default_() {
        VkSemaphoreSubmitInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCommandBufferSubmitInfo> {
    static consteval VkCommandBufferSubmitInfo default_() {
        VkCommandBufferSubmitInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSubmitInfo2> {
    static consteval VkSubmitInfo2 default_() {
        VkSubmitInfo2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceSynchronization2Features> {
    static consteval VkPhysicalDeviceSynchronization2Features default_() {
        VkPhysicalDeviceSynchronization2Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceHostImageCopyFeatures> {
    static consteval VkPhysicalDeviceHostImageCopyFeatures default_() {
        VkPhysicalDeviceHostImageCopyFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceHostImageCopyProperties> {
    static consteval VkPhysicalDeviceHostImageCopyProperties default_() {
        VkPhysicalDeviceHostImageCopyProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMemoryToImageCopy> {
    static consteval VkMemoryToImageCopy default_() {
        VkMemoryToImageCopy ci{};
        ci.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageToMemoryCopy> {
    static consteval VkImageToMemoryCopy default_() {
        VkImageToMemoryCopy ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCopyMemoryToImageInfo> {
    static consteval VkCopyMemoryToImageInfo default_() {
        VkCopyMemoryToImageInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCopyImageToMemoryInfo> {
    static consteval VkCopyImageToMemoryInfo default_() {
        VkCopyImageToMemoryInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCopyImageToImageInfo> {
    static consteval VkCopyImageToImageInfo default_() {
        VkCopyImageToImageInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_IMAGE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkHostImageLayoutTransitionInfo> {
    static consteval VkHostImageLayoutTransitionInfo default_() {
        VkHostImageLayoutTransitionInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSubresourceHostMemcpySize> {
    static consteval VkSubresourceHostMemcpySize default_() {
        VkSubresourceHostMemcpySize ci{};
        ci.sType = VK_STRUCTURE_TYPE_SUBRESOURCE_HOST_MEMCPY_SIZE;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkHostImageCopyDevicePerformanceQuery> {
    static consteval VkHostImageCopyDevicePerformanceQuery default_() {
        VkHostImageCopyDevicePerformanceQuery ci{};
        ci.sType = VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY;
        ci.pNext = nullptr;
        return ci;
    }
};

/*template<>
struct CreateInfoFor<VkPhysicalDeviceVulkanSC10Properties> {
    static consteval VkPhysicalDeviceVulkanSC10Properties default_() {
        VkPhysicalDeviceVulkanSC10Properties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_SC_1_0_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelinePoolSize> {
    static consteval VkPipelinePoolSize default_() {
        VkPipelinePoolSize ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_POOL_SIZE;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceObjectReservationCreateInfo> {
    static consteval VkDeviceObjectReservationCreateInfo default_() {
        VkDeviceObjectReservationCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_OBJECT_RESERVATION_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCommandPoolMemoryReservationCreateInfo> {
    static consteval VkCommandPoolMemoryReservationCreateInfo default_() {
        VkCommandPoolMemoryReservationCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_MEMORY_RESERVATION_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCommandPoolMemoryConsumption> {
    static consteval VkCommandPoolMemoryConsumption default_() {
        VkCommandPoolMemoryConsumption ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_MEMORY_CONSUMPTION;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceVulkanSC10Features> {
    static consteval VkPhysicalDeviceVulkanSC10Features default_() {
        VkPhysicalDeviceVulkanSC10Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_SC_1_0_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};*/

template<>
struct CreateInfoFor<VkPhysicalDevicePipelineProtectedAccessFeatures> {
    static consteval VkPhysicalDevicePipelineProtectedAccessFeatures default_() {
        VkPhysicalDevicePipelineProtectedAccessFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_PROTECTED_ACCESS_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceShaderIntegerDotProductFeatures> {
    static consteval VkPhysicalDeviceShaderIntegerDotProductFeatures default_() {
        VkPhysicalDeviceShaderIntegerDotProductFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceShaderIntegerDotProductProperties> {
    static consteval VkPhysicalDeviceShaderIntegerDotProductProperties default_() {
        VkPhysicalDeviceShaderIntegerDotProductProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkFormatProperties3> {
    static consteval VkFormatProperties3 default_() {
        VkFormatProperties3 ci{};
        ci.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineRenderingCreateInfo> {
    static consteval VkPipelineRenderingCreateInfo default_() {
        VkPipelineRenderingCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkRenderingInfo> {
    static consteval VkRenderingInfo default_() {
        VkRenderingInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkRenderingAttachmentInfo> {
    static consteval VkRenderingAttachmentInfo default_() {
        VkRenderingAttachmentInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceDynamicRenderingFeatures> {
    static consteval VkPhysicalDeviceDynamicRenderingFeatures default_() {
        VkPhysicalDeviceDynamicRenderingFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkCommandBufferInheritanceRenderingInfo> {
    static consteval VkCommandBufferInheritanceRenderingInfo default_() {
        VkCommandBufferInheritanceRenderingInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkImageSubresource2> {
    static consteval VkImageSubresource2 default_() {
        VkImageSubresource2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkSubresourceLayout2> {
    static consteval VkSubresourceLayout2 default_() {
        VkSubresourceLayout2 ci{};
        ci.sType = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDevicePipelineRobustnessFeatures> {
    static consteval VkPhysicalDevicePipelineRobustnessFeatures default_() {
        VkPhysicalDevicePipelineRobustnessFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPipelineRobustnessCreateInfo> {
    static consteval VkPipelineRobustnessCreateInfo default_() {
        VkPipelineRobustnessCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDevicePipelineRobustnessProperties> {
    static consteval VkPhysicalDevicePipelineRobustnessProperties default_() {
        VkPhysicalDevicePipelineRobustnessProperties ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_PROPERTIES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkDeviceImageSubresourceInfo> {
    static consteval VkDeviceImageSubresourceInfo default_() {
        VkDeviceImageSubresourceInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_SUBRESOURCE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMemoryMapInfo> {
    static consteval VkMemoryMapInfo default_() {
        VkMemoryMapInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkMemoryUnmapInfo> {
    static consteval VkMemoryUnmapInfo default_() {
        VkMemoryUnmapInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBindMemoryStatus> {
    static consteval VkBindMemoryStatus default_() {
        VkBindMemoryStatus ci{};
        ci.sType = VK_STRUCTURE_TYPE_BIND_MEMORY_STATUS;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkBindDescriptorSetsInfo> {
    static consteval VkBindDescriptorSetsInfo default_() {
        VkBindDescriptorSetsInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPushConstantsInfo> {
    static consteval VkPushConstantsInfo default_() {
        VkPushConstantsInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPushDescriptorSetInfo> {
    static consteval VkPushDescriptorSetInfo default_() {
        VkPushDescriptorSetInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPushDescriptorSetWithTemplateInfo> {
    static consteval VkPushDescriptorSetWithTemplateInfo default_() {
        VkPushDescriptorSetWithTemplateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_WITH_TEMPLATE_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceShaderSubgroupRotateFeatures> {
    static consteval VkPhysicalDeviceShaderSubgroupRotateFeatures default_() {
        VkPhysicalDeviceShaderSubgroupRotateFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceShaderExpectAssumeFeatures> {
    static consteval VkPhysicalDeviceShaderExpectAssumeFeatures default_() {
        VkPhysicalDeviceShaderExpectAssumeFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceShaderFloatControls2Features> {
    static consteval VkPhysicalDeviceShaderFloatControls2Features default_() {
        VkPhysicalDeviceShaderFloatControls2Features ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkPhysicalDeviceDynamicRenderingLocalReadFeatures> {
    static consteval VkPhysicalDeviceDynamicRenderingLocalReadFeatures default_() {
        VkPhysicalDeviceDynamicRenderingLocalReadFeatures ci{};
        ci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkRenderingAttachmentLocationInfo> {
    static consteval VkRenderingAttachmentLocationInfo default_() {
        VkRenderingAttachmentLocationInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};

template<>
struct CreateInfoFor<VkRenderingInputAttachmentIndexInfo> {
    static consteval VkRenderingInputAttachmentIndexInfo default_() {
        VkRenderingInputAttachmentIndexInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO;
        ci.pNext = nullptr;
        return ci;
    }
};
