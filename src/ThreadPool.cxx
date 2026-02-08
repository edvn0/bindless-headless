#include "ThreadPool.hxx"


ThreadPool::ThreadPool(std::size_t num_threads) {
    workers.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop.store(true, std::memory_order_relaxed);
    }
    condition.notify_all();

    for (auto &worker: workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    wait_condition.wait(lock, [this] {
        return queued_tasks.load(std::memory_order_relaxed) == 0 && active_tasks.load(std::memory_order_relaxed) == 0;
    });
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this] { return stop.load(std::memory_order_relaxed) || !tasks.empty(); });

            if (stop.load(std::memory_order_relaxed) && tasks.empty()) {
                return;
            }

            if (!tasks.empty()) {
                task = std::move(tasks.front());
                tasks.pop();
                queued_tasks.fetch_sub(1, std::memory_order_relaxed);
                active_tasks.fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (task) {
            task();
            active_tasks.fetch_sub(1, std::memory_order_relaxed);
            wait_condition.notify_all();
        }
    }
}
