#pragma once

#include "Pool.hxx"
#include "Forward.hxx"
#include "Types.hxx"

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <print>

#include <slang/slang.h>
#include <slang/slang-com-ptr.h>
#include <slang/slang-com-helper.h>

#include <vma/vk_mem_alloc.h>

constexpr u32 frames_in_flight = 3;   // renderer-side DAG cycle
constexpr u32 max_in_flight = 2;   // GPU submit throttle depth


namespace detail {
	auto set_debug_name_impl(VmaAllocator&, VkObjectType, u64, std::string_view) -> void;
}
template<typename T>
auto set_debug_name(VmaAllocator& alloc, VkObjectType t, const T& obj, std::string_view name) -> void {
	detail::set_debug_name_impl(alloc, t, reinterpret_cast<u64>(obj), name);
}

enum class Stage : u32 {
	LightCulling = 0,
	GBuffer = 1,
};
constexpr auto stage_count = static_cast<u32>(Stage::GBuffer) + 1;

struct FrameState {
	std::array<u64, stage_count> timeline_values{};
	u64 frame_done_value{ 0 }; // This should only be set by the *final* operation in the frame.
};

inline auto stage_index(Stage s) -> std::size_t {
	return static_cast<std::size_t>(s);
}

struct TimelineCompute {
	VkQueue queue{};
	u32 family_index{};

	VkSemaphore timeline{};
	u64 value{};
	u64 completed{};

	static constexpr u32 buffered = 3;

	VkCommandPool pool{};
	std::array<VkCommandBuffer, buffered> cmds{};

	auto destroy(VkDevice device) -> void {
		if (timeline) vkDestroySemaphore(device, timeline, nullptr);
		if (pool) vkDestroyCommandPool(device, pool, nullptr);
		*this = {};
	}
};

inline auto create_timeline(
	VkDevice device,
	VkQueue queue,
	u32 family_index) -> TimelineCompute
{
	TimelineCompute t{};
	t.queue = queue;
	t.family_index = family_index;
	t.value = 0;
	t.completed = 0;

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

	vk_check(vkCreateSemaphore(device, &sci, nullptr, &t.timeline));

	VkCommandPoolCreateInfo pci{
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = family_index,
	};
	vk_check(vkCreateCommandPool(device, &pci, nullptr, &t.pool));

	VkCommandBufferAllocateInfo cai{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = nullptr,
		.commandPool = t.pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = TimelineCompute::buffered
	};
	vk_check(vkAllocateCommandBuffers(device, &cai, t.cmds.data()));

	return t;
}

struct CompilerSession {
	Slang::ComPtr<slang::IGlobalSession> global;
	Slang::ComPtr<slang::ISession> session;

	CompilerSession() {
		createGlobalSession(global.writeRef());

		slang::SessionDesc desc{};
		slang::TargetDesc target{};
		target.format = SLANG_SPIRV;
		target.profile = global->findProfile("spirv_1_6");

		desc.targets = &target;
		desc.targetCount = 1;

		std::array<slang::CompilerOptionEntry, 1> opts = {
			slang::CompilerOptionEntry{
				slang::CompilerOptionName::EmitSpirvDirectly,
				{ slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr }
			}
		};

		desc.compilerOptionEntries = opts.data();
		desc.compilerOptionEntryCount = static_cast<u32>(opts.size());

		global->createSession(desc, session.writeRef());
	}

	auto compile_compute_from_string(
		std::string_view name,
		std::string_view path,
		std::string_view src,
		std::string_view entry = "main") -> std::vector<u32>
	{
		Slang::ComPtr<slang::IBlob> diagnostics;
		Slang::ComPtr<slang::IModule> module;

		module = session->loadModuleFromSourceString(
			name.data(),
			path.data(),
			src.data(),
			diagnostics.writeRef()
		);

		if (diagnostics) {
			std::cerr << static_cast<const char*>(diagnostics->getBufferPointer());
		}

		if (!module) {
			std::abort();
		}

		return compile_compute_module(module, entry);
	}

	auto compile_compute_from_file(
		std::string_view path,
		std::string_view entry) -> std::vector<u32>
	{
		auto extract_module_name = [](const std::filesystem::path& p) {
			return p.filename().string();
			};

		auto load_file_to_string = [](const std::filesystem::path& p) {
			std::ifstream ifs(p);
			if (!ifs) std::abort();
			std::ostringstream oss;
			oss << ifs.rdbuf();
			return oss.str();
			};

		std::filesystem::path p{ path };
		auto name = extract_module_name(p);
		std::string src = load_file_to_string(p);

		return compile_compute_from_string(name, path, src, entry);
	}

private:
	auto compile_compute_module(
		Slang::ComPtr<slang::IModule> const& module,
		std::string_view entry) -> std::vector<u32>
	{
		Slang::ComPtr<slang::IEntryPoint> ep;
		{
			Slang::ComPtr<slang::IBlob> diagnostics;
			module->findEntryPointByName(entry.data(), ep.writeRef());
			if (diagnostics) {
				std::cerr << static_cast<const char*>(diagnostics->getBufferPointer());
			}
			if (!ep) {
				std::abort();
			}
		}

		std::array<slang::IComponentType*, 2> components = {
			module.get(),
			ep.get()
		};

		Slang::ComPtr<slang::IComponentType> composed;
		{
			Slang::ComPtr<slang::IBlob> diagnostics;
			auto result = session->createCompositeComponentType(
				components.data(),
				components.size(),
				composed.writeRef(),
				diagnostics.writeRef()
			);
			if (diagnostics) {
				std::cerr << static_cast<const char*>(diagnostics->getBufferPointer());
			}
			if (SLANG_FAILED(result)) {
				std::abort();
			}
		}

		Slang::ComPtr<slang::IComponentType> linked;
		{
			Slang::ComPtr<slang::IBlob> diagnostics;
			auto result = composed->link(linked.writeRef(), diagnostics.writeRef());
			if (diagnostics) {
				std::cerr << static_cast<const char*>(diagnostics->getBufferPointer());
			}
			if (SLANG_FAILED(result)) {
				std::abort();
			}
		}

		Slang::ComPtr<slang::IBlob> spirv;
		{
			Slang::ComPtr<slang::IBlob> diagnostics;
			auto result = linked->getEntryPointCode(0, 0, spirv.writeRef(), diagnostics.writeRef());
			if (diagnostics) {
				std::cerr << static_cast<const char*>(diagnostics->getBufferPointer());
			}
			if (SLANG_FAILED(result)) {
				std::abort();
			}
		}

		auto bytes = spirv->getBufferSize();
		std::vector<u32> code(bytes / sizeof(u32));
		std::memcpy(code.data(), spirv->getBufferPointer(), bytes);

		return code;
	}
};
inline auto create_sampler(VmaAllocator& alloc, VkSamplerCreateInfo ci, std::string_view name) -> VkSampler {
	VkSampler sampler{};
	VmaAllocatorInfo info{};
	vmaGetAllocatorInfo(alloc, &info);
	vk_check(vkCreateSampler(info.device, &ci, nullptr, &sampler));

	set_debug_name(alloc, VK_OBJECT_TYPE_SAMPLER, sampler, name);

	return sampler;
}

inline auto create_offscreen_target(
	VmaAllocator alloc,
	u32 width,
	u32 height,
	VkFormat format,
	std::string_view name = "Empty") -> OffscreenTarget
{
	OffscreenTarget t{};
	t.width = width;
	t.height = height;
	t.format = format;

	VkImageCreateInfo ici{
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { width, height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage =
			VK_IMAGE_USAGE_SAMPLED_BIT |           // For sampled_view
			VK_IMAGE_USAGE_STORAGE_BIT |           // For storage_view
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
			VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
	};

	VmaAllocationCreateInfo aci{};
	aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	vk_check(vmaCreateImage(alloc, &ici, &aci, &t.image, &t.allocation, nullptr));

	VmaAllocatorInfo info{};
	vmaGetAllocatorInfo(alloc, &info);

	// Create sampled view (for reading in shaders with samplers)
	VkImageViewCreateInfo sampled_vci{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.image = t.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.components = {
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY
		},
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};
	vk_check(vkCreateImageView(info.device, &sampled_vci, nullptr, &t.sampled_view));

	// Create storage view (for reading/writing in compute shaders)
	VkImageViewCreateInfo storage_vci{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.image = t.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.components = {
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY
		},
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};
	vk_check(vkCreateImageView(info.device, &storage_vci, nullptr, &t.storage_view));

	// Set debug names
	auto sampled_view_name = std::format("{}_sampled_view", name);
	auto storage_view_name = std::format("{}_storage_view", name);

	set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE, t.image, name);
	set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.sampled_view, sampled_view_name);
	set_debug_name(alloc, VK_OBJECT_TYPE_IMAGE_VIEW, t.storage_view, storage_view_name);
	vmaSetAllocationName(alloc, t.allocation, name.data());

	return t;
}

struct InstanceWithDebug {
	VkInstance instance{ VK_NULL_HANDLE };
	VkDebugUtilsMessengerEXT messenger{ VK_NULL_HANDLE };
};

inline auto create_instance_with_debug(auto& callback, bool is_release) -> InstanceWithDebug {
	VkApplicationInfo app_info{
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pNext = nullptr,
		.pApplicationName = "HeadlessBindless",
		.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		.pEngineName = "None",
		.engineVersion = VK_MAKE_VERSION(1, 0, 0),
		.apiVersion = VK_API_VERSION_1_3
	};

	std::array<const char*, 1> enabled_layers = {
		"VK_LAYER_KHRONOS_validation"
	};

	u32 ext_count{};
	vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
	std::vector<VkExtensionProperties> extensions(ext_count);
	vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, extensions.data());

	bool has_debug_utils = false;
	for (const auto& ext : extensions) {
		if (std::strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
			has_debug_utils = true;
		}
	}

	has_debug_utils &= !is_release;

	std::vector<const char*> enabled_extensions;

	if (has_debug_utils) {
		enabled_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	VkInstanceCreateInfo create_info{
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.pApplicationInfo = &app_info,
		.enabledLayerCount = is_release ? 0 : static_cast<u32>(enabled_layers.size()),
		.ppEnabledLayerNames = enabled_layers.data(),
		.enabledExtensionCount = is_release ? 0 : static_cast<u32>(enabled_extensions.size()),
		.ppEnabledExtensionNames = enabled_extensions.data()
	};

	InstanceWithDebug result{};
	vk_check(vkCreateInstance(&create_info, nullptr, &result.instance));

	if (has_debug_utils) {
		VkDebugUtilsMessengerCreateInfoEXT debug_ci{
			.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
			.pNext = nullptr,
			.flags = 0,
			.messageSeverity =
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
			.messageType =
				VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
			.pfnUserCallback = &callback,
			.pUserData = nullptr
		};

		auto create_debug = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
			vkGetInstanceProcAddr(result.instance, "vkCreateDebugUtilsMessengerEXT"));

		if (create_debug) {
			vk_check(create_debug(result.instance, &debug_ci, nullptr, &result.messenger));
		}
	}

	return result;
}

struct PhysicalDeviceChoice {
	enum class Error {
		NoDevicesFound,
		NoQueuesFound
	};

	Error error;
};

using DeviceChoice = std::tuple<VkPhysicalDevice, u32, u32>;

inline auto pick_physical_device(VkInstance instance)
-> std::expected<DeviceChoice, PhysicalDeviceChoice>
{
	u32 count{};
	vkEnumeratePhysicalDevices(instance, &count, nullptr);
	if (count == 0u) {
		return std::unexpected(PhysicalDeviceChoice{ PhysicalDeviceChoice::Error::NoDevicesFound });
	}

	std::vector<VkPhysicalDevice> devices(count);
	vkEnumeratePhysicalDevices(instance, &count, devices.data());

	for (VkPhysicalDevice pd : devices) {
		u32 qcount{};
		vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, nullptr);
		if (qcount == 0u) {
			continue;
		}

		std::vector<VkQueueFamilyProperties> qprops(qcount);
		vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops.data());

		std::optional<u32> graphics{};
		std::optional<u32> compute_dedicated{};
		std::optional<u32> compute_shared{};

		for (u32 i = 0u; i < qcount; ++i) {
			VkQueueFlags flags = qprops[i].queueFlags;

			if (flags & VK_QUEUE_GRAPHICS_BIT) {
				if (!graphics) {
					graphics = i;
				}
				if (flags & VK_QUEUE_COMPUTE_BIT) {
					if (!compute_shared) {
						compute_shared = i;
					}
				}
				continue;
			}

			if (flags & VK_QUEUE_COMPUTE_BIT) {
				if (!(flags & VK_QUEUE_GRAPHICS_BIT)) {
					compute_dedicated = i;
				}
			}
		}

		if (graphics && compute_dedicated) {
			return DeviceChoice{ pd, *graphics, *compute_dedicated };
		}

		if (graphics && compute_shared) {
			return DeviceChoice{ pd, *graphics, *compute_shared };
		}
	}

	return std::unexpected(PhysicalDeviceChoice{ PhysicalDeviceChoice::Error::NoQueuesFound });
}

inline auto create_device(
	VkPhysicalDevice pd,
	u32 graphics_index,
	u32 compute_index) -> std::tuple<VkDevice, VkQueue, VkQueue>
{
	u32 ext_count{};
	vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, nullptr);
	std::vector<VkExtensionProperties> dev_exts(ext_count);
	vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, dev_exts.data());

	auto has_ext = [&](char const* name) -> bool {
		for (auto const& e : dev_exts) {
			if (std::strcmp(e.extensionName, name) == 0) {
				return true;
			}
		}
		return false;
		};

	bool accel_supported =
		has_ext(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
		has_ext(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

	std::vector<char const*> enabled_exts;
	if (accel_supported) {
		enabled_exts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
		enabled_exts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
	}

	VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
		.pNext = nullptr
	};

	VkPhysicalDeviceVulkan12Features features12{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &accel_features
	};

	VkPhysicalDeviceVulkan13Features features13{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.pNext = &features12
	};

	VkPhysicalDeviceFeatures2 features2{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &features13,
	};

	vkGetPhysicalDeviceFeatures2(pd, &features2);

	features12.bufferDeviceAddress = VK_TRUE;
	features12.bufferDeviceAddressCaptureReplay = VK_TRUE;
	features12.descriptorIndexing = VK_TRUE;
	features12.runtimeDescriptorArray = VK_TRUE;
	features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
	features12.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
	features12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
	features12.descriptorBindingPartiallyBound = VK_TRUE;
	features12.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
	features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
	features12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
	features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
	features12.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
	features12.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
	features12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
	features12.timelineSemaphore = VK_TRUE;

	features13.dynamicRendering = VK_TRUE;
	features13.synchronization2 = VK_TRUE;

	if (accel_supported) {
		accel_features.accelerationStructure = VK_TRUE;
		accel_features.descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE;
		accel_features.accelerationStructureHostCommands = VK_TRUE;
		accel_features.accelerationStructureCaptureReplay = VK_TRUE;
		accel_features.accelerationStructureIndirectBuild = VK_TRUE;
	}


	float priority_graphics = 0.7f;
	VkDeviceQueueCreateInfo qci_graphics{
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.queueFamilyIndex = graphics_index,
		.queueCount = 1u,
		.pQueuePriorities = &priority_graphics
	};

	float priority_compute = 1.0f;
	VkDeviceQueueCreateInfo qci_compute{
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.queueFamilyIndex = compute_index,
		.queueCount = 1u,
		.pQueuePriorities = &priority_compute
	};

	std::array<VkDeviceQueueCreateInfo, 2> qcis{ qci_graphics, qci_compute };

	u32 qci_count = 0u;
	qcis[qci_count++] = qci_graphics;
	if (compute_index != graphics_index) {
		qcis[qci_count++] = qci_compute;
	}

	VkDeviceCreateInfo dci{
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &features2,
		.flags = 0,
		.queueCreateInfoCount = qci_count,
		.pQueueCreateInfos = qcis.data(),
		.enabledLayerCount = 0,
		.ppEnabledLayerNames = nullptr,
		.enabledExtensionCount = static_cast<u32>(enabled_exts.size()),
		.ppEnabledExtensionNames = enabled_exts.empty() ? nullptr : enabled_exts.data(),
		.pEnabledFeatures = nullptr
	};

	VkDevice device{};
	vk_check(vkCreateDevice(pd, &dci, nullptr, &device));

	VkQueue gq{};
	vkGetDeviceQueue(device, graphics_index, 0u, &gq);

	VkQueue cq{};
	vkGetDeviceQueue(device, compute_index, 0u, &cq);

	return { device, gq, cq };
}

inline auto create_allocator(
	VkInstance instance,
	VkPhysicalDevice pd,
	VkDevice device) -> VmaAllocator
{
	VmaAllocatorCreateInfo info{};
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	info.physicalDevice = pd;
	info.device = device;
	info.instance = instance;

	VmaAllocator alloc{};
	vmaCreateAllocator(&info, &alloc);
	return alloc;
}

template<typename RecordFn>
auto submit_stage(
	TimelineCompute& tl,
	VkDevice,
	RecordFn&& record,
	std::span<const VkSemaphore> wait_semaphores,
	std::span<const u64> wait_values) -> u64
{
	if (wait_semaphores.size() != wait_values.size()) {
		std::abort();
	}

	const u32 index =
		static_cast<u32>(tl.value % TimelineCompute::buffered);
	VkCommandBuffer cmd = tl.cmds[index];

	vk_check(vkResetCommandBuffer(cmd, 0));

	VkCommandBufferBeginInfo bi{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = 0,
		.pInheritanceInfo = nullptr
	};
	vk_check(vkBeginCommandBuffer(cmd, &bi));

	record(cmd);

	vk_check(vkEndCommandBuffer(cmd));

	const u64 signal_val = tl.value + 1;

	VkTimelineSemaphoreSubmitInfo timeline_info{
		.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
		.pNext = nullptr,
		.waitSemaphoreValueCount = static_cast<u32>(wait_values.size()),
		.pWaitSemaphoreValues = wait_values.data(),
		.signalSemaphoreValueCount = 1,
		.pSignalSemaphoreValues = &signal_val
	};

	std::vector<VkPipelineStageFlags> wait_stages(wait_semaphores.size());
	for (std::size_t i = 0; i < wait_semaphores.size(); ++i) {
		wait_stages[i] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	}

	VkSubmitInfo si{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = &timeline_info,
		.waitSemaphoreCount = static_cast<u32>(wait_semaphores.size()),
		.pWaitSemaphores = wait_semaphores.data(),
		.pWaitDstStageMask = wait_semaphores.empty() ? nullptr : wait_stages.data(),
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &tl.timeline
	};

	vk_check(vkQueueSubmit(tl.queue, 1, &si, VK_NULL_HANDLE));
	tl.value = signal_val;
	return signal_val;
}

inline auto throttle(TimelineCompute& tl, VkDevice device) -> void
{
	if (tl.value <= tl.completed + max_in_flight) {
		return;
	}

	const u64 wait_val = tl.value - max_in_flight;

	VkSemaphoreWaitInfo wi{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
		.pNext = nullptr,
		.flags = 0,
		.semaphoreCount = 1,
		.pSemaphores = &tl.timeline,
		.pValues = &wait_val
	};

	vk_check(vkWaitSemaphores(device, &wi, UINT64_MAX));
	tl.completed = wait_val;
}

namespace destruction {
	inline auto instance(InstanceWithDebug const& inst) -> void {
		if (inst.instance == VK_NULL_HANDLE) {
			return;
		}

		if (inst.messenger != VK_NULL_HANDLE) {
			auto destroy_debug = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
				vkGetInstanceProcAddr(inst.instance, "vkDestroyDebugUtilsMessengerEXT"));
			if (destroy_debug) {
				destroy_debug(inst.instance, inst.messenger, nullptr);
			}
		}

		vkDestroyInstance(inst.instance, nullptr);
	}

	inline auto device(VkDevice& dev) -> void {
		if (dev) {
			vkDestroyDevice(dev, nullptr);
		}
		dev = VK_NULL_HANDLE;
	}

	auto bindless_set(VkDevice device, BindlessSet& bs) -> void;

	inline auto allocator(VmaAllocator& alloc) -> void {
		if (alloc) {
			vmaDestroyAllocator(alloc);
		}
		alloc = nullptr;
	}

	inline auto timeline_compute(VkDevice device, TimelineCompute& comp) -> void {
		if (comp.pool)     vkDestroyCommandPool(device, comp.pool, nullptr);
		if (comp.timeline) vkDestroySemaphore(device, comp.timeline, nullptr);
		comp = {};
	}
}
