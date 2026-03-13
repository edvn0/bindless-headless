#include <GLFW/glfw3.h>
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

#include "Logger.hxx"
#include "SceneLoader.hxx"

#include "BindlessHeadless.hxx"

#include "app/app.hxx"

namespace {

    auto debug_callback(const VkDebugUtilsMessageSeverityFlagBitsEXT message_severity, VkDebugUtilsMessageTypeFlagsEXT,
                        const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *) -> VkBool32 {
        if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            auto object_info = std::string{};
            if (callback_data->objectCount > 0) {
                object_info += "Objects involved:\n";
                for (u32 i = 0; i < callback_data->objectCount; ++i) {
                    const auto &obj = callback_data->pObjects[i];
                    if (obj.pObjectName) {
                        object_info += std::format("    Name: {}\n", obj.pObjectName);
                    }
                    object_info += std::format("  - Object {}: Type={}, Handle=0x{:X}\n", i,
                                               static_cast<i32>(obj.objectType), obj.objectHandle);
                }
            }

            error("Validation layer: {}. {}", callback_data->pMessage, object_info);
        }

        return VK_FALSE;
    }

} // namespace


auto main(int argc, char **argv) -> int {
    BindlessApp app;

    if (glfwPlatformSupported(GLFW_PLATFORM_WAYLAND)) {
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
    } else {
        error("Could not initialise");
        return 1;
    }

    if (auto init = glfwInit(); init != GLFW_TRUE) {
        error("Could not initialize GLFW");
        return 1;
    }

    auto maybe_opts = parse_cli(argc, argv);
    if (!maybe_opts) {
        return 1;
    }
    auto opts = std::move(maybe_opts.value());

    {
        Tooling::SceneLoader loader{"assets/meshes"};

        auto f0 = std::async(std::launch::async, [&loader] {
            return loader.convert_gltf("/home/edwin/Assets/Meshes/SponzaGLTF/Sponza.gltf",
                                       "SponzaGLTF/sponza_converted");
        });

        auto f1 = std::async(std::launch::async, [&loader] {
            return loader.convert_gltf("/home/edwin/Assets/Meshes/DamagedHelmet/glTF/DamagedHelmet.gltf",
                                       "DamagedHelmetGLTF/damaged_helmet_converted");
        });

        std::ignore = f0.get();
        std::ignore = f1.get();
    }
    u32 extension_count{};
    const char **extensions_raw = glfwGetRequiredInstanceExtensions(&extension_count);
    std::vector<std::string_view> extensions(extensions_raw, extensions_raw + extension_count);

    bool enable_validation =
            opts.validation_layers.value_or(!static_cast<bool>(IS_RELEASE)); // NOLINT(modernize-use-bool-literals)

    InstanceWithDebug instance;
    if (enable_validation) {
        instance = create_instance_with_debug(debug_callback, extensions);
    } else {
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
