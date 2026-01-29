#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    template<class F>
    auto enqueue(F &&f) -> void;

    auto wait_all() -> void;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable wait_condition;
    std::atomic<bool> stop{false};
    std::atomic<std::size_t> active_tasks{0};
    std::atomic<std::size_t> queued_tasks{0};

    auto worker_thread() -> void;
};

template<class F>
auto ThreadPool::enqueue(F &&f) -> void {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace(std::forward<F>(f));
        queued_tasks.fetch_add(1, std::memory_order_relaxed);
    }
    condition.notify_one();
}
