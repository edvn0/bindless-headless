#pragma once

#include "Error.hxx"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <tl/expected.hpp>
#include <vector>
#include <volk.h>

class DeviceThreadPool {
public:
    struct Config {
        VkDevice device{};
        u32 queue_family{};
        u32 thread_count{std::max(1u, std::thread::hardware_concurrency() - 1u)};
    };

    using RecordFunction = std::function<tl::expected<void, Error>(VkCommandBuffer)>;
    using CompleteFunction = std::function<void()>;

    [[nodiscard]] static auto create(const Config &cfg) -> tl::expected<DeviceThreadPool, Error>;

    DeviceThreadPool() = default;
    DeviceThreadPool(DeviceThreadPool &&) = default;
    DeviceThreadPool &operator=(DeviceThreadPool &&) = default;
    DeviceThreadPool(const DeviceThreadPool &) = delete;
    DeviceThreadPool &operator=(const DeviceThreadPool &) = delete;
    ~DeviceThreadPool() = default;

    void enqueue(RecordFunction record_fn, CompleteFunction complete_fn = {});

    [[nodiscard]] auto wait_recordings() -> tl::expected<void, Error>;
    [[nodiscard]] auto submit_batch(VkQueue queue) -> tl::expected<std::optional<VkFence>, Error>;
    void on_batch_complete();

    [[nodiscard]] auto poll(VkQueue queue) -> tl::expected<bool, Error>;

private:
    struct PendingTask {
        RecordFunction record_fn;
        CompleteFunction complete_fn;
    };

    struct RecordedCmd {
        VkCommandBuffer cmd{VK_NULL_HANDLE};
        usize worker_idx{};
        CompleteFunction complete_fn;
    };

    struct Worker {
        std::thread thread;
        VkCommandPool pool{VK_NULL_HANDLE};
    };

    struct Impl {
        VkDevice device{VK_NULL_HANDLE};

        std::vector<Worker> workers;
        VkCommandPool primary_pool{VK_NULL_HANDLE};

        std::queue<PendingTask> task_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        bool stopping{false};

        std::vector<RecordedCmd> recorded;
        std::mutex recorded_mutex;

        std::vector<Error> worker_errors;
        std::mutex errors_mutex;

        std::atomic<i32> in_flight{0};
        std::mutex idle_mutex;
        std::condition_variable idle_cv;

        VkFence batch_fence{VK_NULL_HANDLE};

        enum class BatchState { idle, recording, submitted };
        BatchState batch_state{BatchState::idle};

        void worker_fn(usize worker_idx);
        void shutdown();
        ~Impl();
    };

    std::unique_ptr<Impl> impl_;

    friend class StreamingUploader;
    friend class AssetStreamer;
};

class StreamingUploader {
public:
    struct Config {
        VkDevice device{};
        u32 queue_family{};
        u32 chunk_size{8}; // icons per submission
        u32 thread_count{std::max(1u, std::thread::hardware_concurrency() - 1u)};
    };

    explicit StreamingUploader(Config cfg) : cfg_{cfg} {}

    auto enqueue(DeviceThreadPool::RecordFunction record_fn, DeviceThreadPool::CompleteFunction complete_fn = {})
            -> tl::expected<void, Error>;

    [[nodiscard]] auto poll(VkQueue queue) -> tl::expected<bool, Error>;
    void abort();

private:
    auto ensure_active_chunk() -> tl::expected<void, Error>;

    Config cfg_;

    struct Chunk {
        DeviceThreadPool pool;
        u32 enqueued{0};
    };

    std::vector<Chunk> chunks_;
    u32 current_chunk_enqueued_{0};
};

// General-purpose streaming system for any GPU upload work.
//
// Usage:
//   1. Call submit() freely from any setup code — work is queued, not started.
//   2. Call poll() once per frame from the main loop.
//      - Each poll feeds up to submissions_per_frame items to the GPU uploader.
//      - complete_fns fire on the main thread as individual GPU fences signal.
//   3. is_idle() returns true once all submitted work has completed.
//
class AssetStreamer {
public:
    struct Config {
        VkDevice device{};
        u32 queue_family{};
        u32 submissions_per_frame{1};
        u32 chunk_size{1};
        u32 thread_count{std::max(1u, std::thread::hardware_concurrency() - 1u)};
    };

    using RecordFunction = DeviceThreadPool::RecordFunction;
    using CompleteFunction = DeviceThreadPool::CompleteFunction;

    explicit AssetStreamer(Config cfg);

    void submit(RecordFunction record_fn, CompleteFunction complete_fn = {});

    [[nodiscard]] auto poll(VkQueue queue) -> tl::expected<bool, Error>;

    [[nodiscard]] auto pending_count() const -> usize { return pending_.size(); }
    [[nodiscard]] auto is_idle() const -> bool { return pending_.empty() && !uploader_; }

    void reset();
    void abort();

    void emergency_shutdown();

private:
    struct PendingItem {
        RecordFunction record_fn;
        CompleteFunction complete_fn;
    };

    auto ensure_uploader() -> tl::expected<void, Error>;
    void feed_pending();

    Config cfg_;
    std::vector<PendingItem> pending_;
    std::unique_ptr<StreamingUploader> uploader_;
};
