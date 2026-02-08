#include "app/listeners.hxx"

#include "Logger.hxx"
#include "ResizeableGraph.hxx"

auto ShaderSourceCodeChangeListener::compile_shader(const std::string &path) -> void {
        info("Shader changed: {}", path);
        resize_graph->trigger_resize(ResizeTrigger::Shaders);
}

void ShaderSourceCodeChangeListener::worker_loop() {
    while (!should_exit) {
        std::string file_to_compile;

        {
            std::unique_lock<std::mutex> lock(work_mutex);

            // Wait until there is work OR we need to shut down
            work_cv.wait(lock, [this] { return !pending_files.empty() || should_exit; });

            if (should_exit)
                return;

            auto now = std::chrono::steady_clock::now();
            auto it = pending_files.begin();

            // Check if the oldest pending file has aged enough
            if (now - it->second >= debounce_delay) {
                file_to_compile = it->first;
                pending_files.erase(it);
            } else {
                // Not ready yet, sleep until it is ready
                work_cv.wait_for(lock, debounce_delay);
                continue;
            }
        }

        if (!file_to_compile.empty()) {
            compile_shader(file_to_compile);
        }
    }
}
void ShaderSourceCodeChangeListener::handleFileAction(efsw::WatchID, const std::string &dir,
                                                      const std::string &filename, efsw::Action action, std::string) {
    if (action == efsw::Actions::Modified || action == efsw::Actions::Add) {
        std::lock_guard<std::mutex> lock(work_mutex);
        // Update the "last seen" time for this file
        pending_files[dir + filename] = std::chrono::steady_clock::now();
        work_cv.notify_one();
    }
}
