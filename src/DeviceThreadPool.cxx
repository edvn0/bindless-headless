#include "DeviceThreadPool.hxx"
#include "CreateInfo.hxx"
#include "Logger.hxx"

#include "Types.hxx"

DeviceThreadPool::Impl::~Impl() {
    if (device == VK_NULL_HANDLE)
        return;

    {
        std::lock_guard lock{queue_mutex};
        stopping = true;
    }
    queue_cv.notify_all();

    for (auto &w: workers) {
        if (w.thread.joinable())
            w.thread.join();
        if (w.pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(device, w.pool, nullptr);
    }

    if (primary_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(device, primary_pool, nullptr);

    if (batch_fence != VK_NULL_HANDLE)
        vkDestroyFence(device, batch_fence, nullptr);
}

auto DeviceThreadPool::create(const Config &cfg) -> tl::expected<DeviceThreadPool, Error> {
    auto impl = std::make_unique<Impl>();
    impl->device = cfg.device;
    impl->workers.resize(cfg.thread_count);

    for (usize i = 0; i < cfg.thread_count; ++i) {
        auto pool_info = create_info<VkCommandPoolCreateInfo>();
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = cfg.queue_family;

        if (const auto result = vkCreateCommandPool(cfg.device, &pool_info, nullptr, &impl->workers[i].pool);
            result != VK_SUCCESS) {
            for (usize j = 0; j < i; ++j)
                vkDestroyCommandPool(cfg.device, impl->workers[j].pool, nullptr);
            impl->device = VK_NULL_HANDLE;
            return tl::unexpected{Error::make_error(Error::Type::RenderError,
                                                    "DeviceThreadPool: vkCreateCommandPool failed: {}",
                                                    static_cast<i32>(result))};
        }

        impl->workers[i].thread = std::thread{[&impl_ref = *impl, i] { impl_ref.worker_fn(i); }};
    }

    auto primary_pool_info = create_info<VkCommandPoolCreateInfo>();
    primary_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    primary_pool_info.queueFamilyIndex = cfg.queue_family;

    if (const auto result = vkCreateCommandPool(cfg.device, &primary_pool_info, nullptr, &impl->primary_pool);
        result != VK_SUCCESS) {
        for (auto &w: impl->workers)
            vkDestroyCommandPool(cfg.device, w.pool, nullptr);
        impl->device = VK_NULL_HANDLE;
        return tl::unexpected{Error::make_error(Error::Type::RenderError,
                                                "DeviceThreadPool: vkCreateCommandPool (primary) failed: {}",
                                                static_cast<i32>(result))};
    }

    DeviceThreadPool pool{};
    pool.impl_ = std::move(impl);
    return pool;
}

void DeviceThreadPool::enqueue(RecordFunction record_fn, CompleteFunction complete_fn) {
    impl_->in_flight.fetch_add(1, std::memory_order_relaxed);
    impl_->batch_state = Impl::BatchState::recording;
    {
        std::lock_guard lock{impl_->queue_mutex};
        impl_->task_queue.push({std::move(record_fn), std::move(complete_fn)});
    }
    impl_->queue_cv.notify_one();
}

void DeviceThreadPool::Impl::worker_fn(const usize worker_idx) {
    auto &worker = workers[worker_idx];

    auto push_error = [&](Error err) {
        {
            std::lock_guard lock{errors_mutex};
            worker_errors.push_back(std::move(err));
        }
        if (in_flight.fetch_sub(1, std::memory_order_acq_rel) == 1)
            idle_cv.notify_all();
    };

    while (true) {
        PendingTask task;
        {
            std::unique_lock lock{queue_mutex};
            queue_cv.wait(lock, [this] { return !task_queue.empty() || stopping; });
            if (stopping && task_queue.empty())
                return;
            task = std::move(task_queue.front());
            task_queue.pop();
        }

        VkCommandBuffer cmd{VK_NULL_HANDLE};
        auto alloc_info = create_info<VkCommandBufferAllocateInfo>();
        alloc_info.commandPool = worker.pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        alloc_info.commandBufferCount = 1;

        if (const auto result = vkAllocateCommandBuffers(device, &alloc_info, &cmd); result != VK_SUCCESS) {
            push_error(Error::make_error(Error::Type::RenderError, "vkAllocateCommandBuffers failed: {}",
                                         static_cast<i32>(result)));
            continue;
        }

        auto inheritance = create_info<VkCommandBufferInheritanceInfo>();

        auto begin_info = create_info<VkCommandBufferBeginInfo>();
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        begin_info.pInheritanceInfo = &inheritance;

        if (const auto result = vkBeginCommandBuffer(cmd, &begin_info); result != VK_SUCCESS) {
            push_error(Error::make_error(Error::Type::RenderError, "vkBeginCommandBuffer failed: {}",
                                         static_cast<i32>(result)));
            continue;
        }

        auto record_result = task.record_fn(cmd);
        if (!record_result) {
            push_error(std::move(record_result.error()));
            continue;
        }

        if (const auto result = vkEndCommandBuffer(cmd); result != VK_SUCCESS) {
            push_error(Error::make_error(Error::Type::RenderError, "vkEndCommandBuffer failed: {}",
                                         static_cast<i32>(result)));
            continue;
        }

        {
            std::lock_guard lock{recorded_mutex};
            recorded.push_back({cmd, worker_idx, std::move(task.complete_fn)});
        }

        if (in_flight.fetch_sub(1, std::memory_order_acq_rel) == 1)
            idle_cv.notify_all();
    }
}

namespace {
    auto do_submit(auto &impl, VkQueue queue) -> tl::expected<VkFence, Error> {

        std::vector<VkCommandBuffer> secondaries;
        secondaries.reserve(impl.recorded.size());
        for (const auto &r: impl.recorded)
            secondaries.push_back(r.cmd);

        VkCommandBuffer primary{VK_NULL_HANDLE};
        auto alloc_info = create_info<VkCommandBufferAllocateInfo>();
        alloc_info.commandPool = impl.primary_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        if (const auto result = vkAllocateCommandBuffers(impl.device, &alloc_info, &primary); result != VK_SUCCESS)
            return tl::unexpected{Error::make_error(Error::Type::RenderError,
                                                    "do_submit: vkAllocateCommandBuffers failed: {}",
                                                    static_cast<i32>(result))};

        auto begin_info = create_info<VkCommandBufferBeginInfo>();
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (const auto result = vkBeginCommandBuffer(primary, &begin_info); result != VK_SUCCESS)
            return tl::unexpected{Error::make_error(
                    Error::Type::RenderError, "do_submit: vkBeginCommandBuffer failed: {}", static_cast<i32>(result))};

        vkCmdExecuteCommands(primary, static_cast<u32>(secondaries.size()), secondaries.data());

        if (const auto result = vkEndCommandBuffer(primary); result != VK_SUCCESS)
            return tl::unexpected{Error::make_error(
                    Error::Type::RenderError, "do_submit: vkEndCommandBuffer failed: {}", static_cast<i32>(result))};

        auto fence_info = create_info<VkFenceCreateInfo>();
        if (const auto result = vkCreateFence(impl.device, &fence_info, nullptr, &impl.batch_fence);
            result != VK_SUCCESS)
            return tl::unexpected{Error::make_error(Error::Type::RenderError, "do_submit: vkCreateFence failed: {}",
                                                    static_cast<i32>(result))};

        auto submit_info = create_info<VkSubmitInfo>();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &primary;

        if (const auto result = vkQueueSubmit(queue, 1, &submit_info, impl.batch_fence); result != VK_SUCCESS) {
            vkDestroyFence(impl.device, impl.batch_fence, nullptr);
            impl.batch_fence = VK_NULL_HANDLE;
            return tl::unexpected{Error::make_error(Error::Type::RenderError, "do_submit: vkQueueSubmit failed: {}",
                                                    static_cast<i32>(result))};
        }

        return impl.batch_fence;
    }
} // namespace

auto DeviceThreadPool::wait_recordings() -> tl::expected<void, Error> {
    {
        std::unique_lock lock{impl_->idle_mutex};
        impl_->idle_cv.wait(lock, [this] { return impl_->in_flight.load(std::memory_order_acquire) == 0; });
    }

    std::lock_guard lock{impl_->errors_mutex};
    if (!impl_->worker_errors.empty()) {
        auto first = std::move(impl_->worker_errors.front());
        impl_->worker_errors.clear();
        return tl::unexpected{std::move(first)};
    }
    return {};
}

auto DeviceThreadPool::submit_batch(VkQueue queue) -> tl::expected<std::optional<VkFence>, Error> {
    if (impl_->recorded.empty())
        return std::nullopt;

    TRY_PROPAGATE(fence, do_submit(*impl_, queue), "submit_batch: failed to submit");
    return fence;
}

void DeviceThreadPool::on_batch_complete() {
    for (auto &r: impl_->recorded)
        if (r.complete_fn)
            r.complete_fn();
    impl_->recorded.clear();

    for (auto &w: impl_->workers)
        vkResetCommandPool(impl_->device, w.pool, 0);

    vkResetCommandPool(impl_->device, impl_->primary_pool, 0);

    if (impl_->batch_fence != VK_NULL_HANDLE) {
        vkDestroyFence(impl_->device, impl_->batch_fence, nullptr);
        impl_->batch_fence = VK_NULL_HANDLE;
    }

    impl_->batch_state = Impl::BatchState::idle;
}

auto DeviceThreadPool::poll(VkQueue queue) -> tl::expected<bool, Error> {
    if (!impl_ || impl_->batch_state == Impl::BatchState::idle)
        return true;

    if (impl_->batch_state == Impl::BatchState::recording) {
        if (impl_->in_flight.load(std::memory_order_acquire) > 0)
            return false;

        {
            std::lock_guard lock{impl_->errors_mutex};
            if (!impl_->worker_errors.empty()) {
                auto first = std::move(impl_->worker_errors.front());
                impl_->worker_errors.clear();
                return tl::unexpected{std::move(first)};
            }
        }

        if (impl_->recorded.empty()) {
            impl_->batch_state = Impl::BatchState::idle;
            return true;
        }

        if (auto result = do_submit(*impl_, queue); !result)
            return tl::unexpected{std::move(result.error())};

        impl_->batch_state = Impl::BatchState::submitted;
        return false;
    }

    const auto status = vkGetFenceStatus(impl_->device, impl_->batch_fence);
    if (status == VK_NOT_READY)
        return false;

    if (status != VK_SUCCESS)
        return tl::unexpected{Error::make_error(Error::Type::RenderError, "poll: vkGetFenceStatus failed: {}",
                                                static_cast<i32>(status))};

    on_batch_complete();
    return true;
}

auto StreamingUploader::ensure_active_chunk() -> tl::expected<void, Error> {
    const bool needs_new = chunks_.empty() || current_chunk_enqueued_ >= cfg_.chunk_size;

    if (!needs_new)
        return {};

    TRY_PROPAGATE(pool,
                  DeviceThreadPool::create({
                          .device = cfg_.device,
                          .queue_family = cfg_.queue_family,
                          .thread_count = cfg_.thread_count,
                  }),
                  "StreamingUploader: failed to create chunk pool");

    chunks_.push_back(Chunk{.pool = std::move(pool)});
    current_chunk_enqueued_ = 0;
    return {};
}

auto StreamingUploader::enqueue(DeviceThreadPool::RecordFunction record_fn,
                                DeviceThreadPool::CompleteFunction complete_fn) -> tl::expected<void, Error> {

    if (auto r = ensure_active_chunk(); !r)
        return tl::make_unexpected(std::move(r.error()));

    chunks_.back().pool.enqueue(std::move(record_fn), std::move(complete_fn));
    ++current_chunk_enqueued_;
    return {};
}

auto StreamingUploader::poll(VkQueue queue) -> tl::expected<bool, Error> {
    bool all_done = true;
    for (auto &chunk: chunks_) {
        TRY_PROPAGATE(done, chunk.pool.poll(queue), "StreamingUploader: chunk poll failed");
        if (!done)
            all_done = false;
    }

    if (all_done)
        chunks_.clear();

    return all_done;
}

AssetStreamer::AssetStreamer(Config cfg) : cfg_{cfg} {}

void AssetStreamer::submit(RecordFunction record_fn, CompleteFunction complete_fn) {
    pending_.push_back({std::move(record_fn), std::move(complete_fn)});
}

auto AssetStreamer::ensure_uploader() -> tl::expected<void, Error> {
    if (uploader_)
        return {};

    uploader_ = std::make_unique<StreamingUploader>(StreamingUploader::Config{
            .device = cfg_.device,
            .queue_family = cfg_.queue_family,
            .chunk_size = cfg_.chunk_size,
            .thread_count = cfg_.thread_count,
    });
    return {};
}

void AssetStreamer::feed_pending() {
    for (u32 i = 0; i < cfg_.submissions_per_frame && !pending_.empty(); ++i) {
        auto item = std::move(pending_.back());
        pending_.pop_back();

        auto result = uploader_->enqueue(std::move(item.record_fn), std::move(item.complete_fn));

        if (!result)
            error("AssetStreamer: failed to enqueue item: {}", result.error().message);
    }
}

auto AssetStreamer::poll(VkQueue queue) -> tl::expected<bool, Error> {
    // Nothing ever submitted
    if (pending_.empty() && !uploader_)
        return true;

    if (auto r = ensure_uploader(); !r)
        return tl::unexpected{std::move(r.error())};

    feed_pending();

    auto poll_result = uploader_->poll(queue);
    if (!poll_result)
        return tl::unexpected{std::move(poll_result.error())};

    const bool uploader_done = *poll_result;
    const bool all_done = uploader_done && pending_.empty();

    if (all_done)
        uploader_.reset(); // free all GPU resources once idle

    return all_done;
}

void AssetStreamer::reset() {
    pending_.clear();
    uploader_.reset();
}
