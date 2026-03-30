#include <GLFW/glfw3.h>
#include <chrono>
#include <cstdlib>
#include <cxxabi.h>
#include <deque>
#include <efsw/efsw.hpp>
#include <exception>
#include <execution>
#include <future>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <imgui.h>
#include <iostream>
#include <lyra/cli.hpp>
#include <lyra/opt.hpp>
#include <random>
#include <ranges>
#include <thread>

#include "Logger.hxx"
#include "RenderDoc.hxx"
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
    auto renderdoc = renderdoc_init();

    if (glfwPlatformSupported(GLFW_PLATFORM_WAYLAND)) {
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
    }

    if (renderdoc.is_active()) {
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
        info("RenderDoc prefers X11.");
    }

    if (auto init = glfwInit(); init != GLFW_TRUE) {
        error("Could not initialize GLFW");
        return 1;
    }

    EngineOptions opts{};
    auto cli = build_base_cli(opts) | lyra::opt(opts.iteration_count, "n")["--iterations"]("Render iterations") |
               lyra::opt(opts.light_count, "n")["--lights"]("Number of lights") |
               lyra::opt(opts.msaa, "n")["--msaa"]("MSAA sample count") |
               lyra::opt(opts.disable_output_images)["--no-output"]("Skip writing output images");

    if (auto r = cli.parse({argc, argv}); !r) {
        error("Argument error: {}", r.message());
        return 1;
    }


    {
        std::counting_semaphore<> transcode_sem{std::thread::hardware_concurrency()};
        Tooling::SceneLoader loader{"assets/meshes"};

        std::vector<std::future<bool>> futures;

#define LOAD_MESH(src, dst)                                                                                            \
    futures.emplace_back(std::async(std::launch::async, [&sem = transcode_sem, &loader] {                              \
        sem.acquire();                                                                                                 \
        auto res = loader.convert_gltf(src, dst).has_value();                                                          \
        sem.release();                                                                                                 \
        return res;                                                                                                    \
    }));

        // LOAD_MESH("/home/edwin/Assets/Meshes/SponzaGLTF/Sponza.gltf", "SponzaGLTF/sponza_converted");
        // LOAD_MESH("/home/edwin/Assets/Meshes/DamagedHelmet/glTF/DamagedHelmet.gltf",
        //           "DamagedHelmetGLTF/damaged_helmet_converted");
        LOAD_MESH("/home/edwin/Assets/Meshes/main_sponza/NewSponza_Main_glTF_003.gltf", "NewSponza/new_sponza");
        LOAD_MESH("/home/edwin/Assets/Meshes/pkg_a_curtains/NewSponza_Curtains_glTF.gltf", "NewSponza/curtains");

#undef LOAD_MESH

        for (auto &f: futures) {
            std::ignore = f.get();
        }
    }
    u32 extension_count{};
    const char **extensions_raw = glfwGetRequiredInstanceExtensions(&extension_count);
    std::vector<std::string_view> extensions(extensions_raw, extensions_raw + extension_count);

    const bool enable_validation = opts.validation_layers && !static_cast<bool>(IS_RELEASE);

    InstanceWithDebug instance;
    if (enable_validation) {
        instance = create_instance_with_debug(debug_callback, extensions);
    } else {
        auto raw_instance = create_instance(extensions);
        instance.instance = raw_instance;
        instance.messenger = VK_NULL_HANDLE;
    }

    auto result = app.run(opts, instance, &renderdoc);
    if (!result) {
        error("Application error: {}", result.error());
        return 1;
    }

    info("Bindless headless setup and teardown completed successfully.");
    return 0;
}
