#pragma once

#include "Forward.hxx"

#include <efsw/efsw.hpp>
#include <thread>

class ShaderSourceCodeChangeListener final : public efsw::FileWatchListener {
    ResizeGraph *resize_graph{nullptr};

    std::jthread worker_thread;
    std::mutex work_mutex;
    std::condition_variable work_cv;
    std::atomic<bool> should_exit{false};

    std::unordered_map<std::string, std::chrono::steady_clock::time_point> pending_files;
    const std::chrono::milliseconds debounce_delay{100};

public:
    explicit ShaderSourceCodeChangeListener(ResizeGraph *r) : resize_graph(r) {
        worker_thread = std::jthread(&ShaderSourceCodeChangeListener::worker_loop, this);
    }

    ~ShaderSourceCodeChangeListener() {
        should_exit = true;
        work_cv.notify_all();
    }

    void handleFileAction(efsw::WatchID, const std::string &dir, const std::string &filename, efsw::Action action,
                          std::string) override;

private:
    auto worker_loop() -> void;
    auto compile_shader(const std::string &path) -> void;
};
