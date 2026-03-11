// BaseApplication.hxx
// Minimal base for ImGui-only tools.
//
// Usage — your entire tool is one translation unit:
//
//   #include "BaseApplication.hxx"
//
//   class ImageViewer final : public BaseApplication {
//   protected:
//       auto on_init()          -> tl::expected<void, Error> override { ... }
//       auto on_frame(float dt) -> void                     override { ... }
//       auto on_shutdown()      -> void                     override {}
//   };
//
//   REGISTER_APPLICATION(ImageViewer)
//
#pragma once

#include <memory>
#include <tl/expected.hpp>

#include "Error.hxx"
#include "Numeric.hxx"
#include "RenderContext.hxx"

struct CLIOptions;
struct InstanceWithDebug;
class ImGuiRenderer;

using SemVer = std::array<u16, 3>;

struct AppInfo {
    SemVer version{0, 0, 1};
    std::string name{"App"};
};

struct Timestep {
    float ts{0.0F};
};

class BaseApplication {
public:
    BaseApplication(AppInfo);
    virtual ~BaseApplication();

protected:
    // Override points ---------------------------------------------------
    virtual auto on_init() -> tl::expected<void, Error>;
    virtual auto on_frame(Timestep) -> void = 0;
    virtual auto on_shutdown() -> void {}

    auto set_should_exit_app(bool val) { should_exit = val; }

    // State exposed to subclasses ---------------------------------------
    RenderContext *ctx{nullptr};
    VmaAllocator allocator{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    GLFWwindow *window{nullptr};
    ImGuiRenderer *gui{nullptr};
    u64 frame_index{0};
    std::atomic_bool should_exit{false};

private:
    friend auto run_application(BaseApplication &, int, char **) -> int;

    auto init_vulkan(CLIOptions &, InstanceWithDebug &) -> tl::expected<void, Error>;
    auto render_frame(u32 fi, float dt) -> tl::expected<void, Error>;
    auto wait_frame(u32 fi) -> void;

    struct Impl;

    struct ImplDeleter {
        auto operator()(Impl *ptr) const noexcept -> void;
    };

    AppInfo app_info{};

    std::unique_ptr<Impl, ImplDeleter> m_impl;
};

auto run_application(BaseApplication &app, int argc, char **argv) -> int;

#define REGISTER_APPLICATION(AppClass)                                                                                 \
    int main(int argc, char **argv) {                                                                                  \
        AppClass app;                                                                                                  \
        return run_application(app, argc, argv);                                                                       \
    }
