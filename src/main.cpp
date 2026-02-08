#include <GLFW/glfw3.h>
#include <cassert>
#include <chrono>
#include <deque>
#include <efsw/efsw.hpp>
#include <execution>
#include <future>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <imgui.h>
#include <iostream>
#include <random>
#include <ranges>
#include <thread>

#include "BindlessHeadless.hxx"

#include "app/app.hxx"

namespace {

    auto debug_callback(const VkDebugUtilsMessageSeverityFlagBitsEXT message_severity, VkDebugUtilsMessageTypeFlagsEXT,
                        const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *) -> VkBool32 {
        if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            error("Validation layer: {}", callback_data->pMessage);
        }

        if (message_severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
            trace("Validation info: {}", callback_data->pMessage);
        }

        return VK_FALSE;
    }

} // namespace


auto main(int argc, char **argv) -> int {
    BindlessApp app;

    if (auto init = glfwInit(); init != GLFW_TRUE) {
        error("Could not initialize GLFW");
        return 1;
    }

    auto opts = parse_cli(argc, argv);

    uint32_t count{};
    const char **extensions_raw = glfwGetRequiredInstanceExtensions(&count);
    std::vector<std::string_view> extensions(extensions_raw, extensions_raw + count);

    bool enable_validation = opts.validation_layers.value_or(!static_cast<bool>(IS_RELEASE));

    InstanceWithDebug instance;
    if (enable_validation) {
        // With validation layers
        instance = create_instance_with_debug(debug_callback, extensions);
    } else {
        // No validation
        auto raw_instance = create_instance(extensions);
        instance.instance = raw_instance;
        instance.messenger = VK_NULL_HANDLE;
    }

    auto result = app.run(opts, instance);
    if (!result) {
        error("Application error: {}", result.error());
        return 1;
    }

    info("Bindless headless setup and teardown completed successfully.");
    return 0;
}
