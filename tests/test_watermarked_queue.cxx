#include <doctest/doctest.h>
#include "RenderSubmission.hxx"

TEST_CASE("WatermarkedQueue Basic Operations") {
    WatermarkedQueue<MeshSubmission> queue;

    SUBCASE("Initialization state") {
        CHECK(queue.objects.empty());
        CHECK(queue.high_watermark == 0);
    }

    SUBCASE("Submission and Flush") {
        MeshSubmission sub{};
        sub.mesh_index = 42;
        queue.submit(std::move(sub));

        auto results = queue.flush();
        CHECK(results.size() == 1);
        CHECK(results[0].mesh_index == 42);
    }
}

TEST_CASE("WatermarkedQueue Memory Management Logic") {
    WatermarkedQueue<MeshSubmission> queue;

    // 1. Establish a peak
    for (u32 i = 0; i < 100; ++i)
        queue.submit({});
    queue.reset();
    CHECK(queue.high_watermark == 100);

    SUBCASE("Does not shrink immediately") {
        // Submit low amount (below 75% of 100)
        for (u32 i = 0; i < 10; ++i)
            queue.submit({});
        queue.reset();

        CHECK(queue.frames_below_watermark == 1);
        // Capacity shouldn't have dropped yet
        CHECK(queue.high_watermark == 100);
    }

    SUBCASE("Shrinks after threshold (120 frames)") {
        // Simulate 119 frames of low usage
        for (int i = 0; i < 119; ++i) {
            queue.reset();
        }
        CHECK(queue.frames_below_watermark == 119);

        // Frame 120 triggers the shrink
        queue.reset();

        // high_watermark should now be (current_usage * 1.25) + 8
        // Current usage was 0, so (0 * 1.25) + 8 = 8
        CHECK(queue.high_watermark == 8);
        CHECK(queue.frames_below_watermark == 0);
    }

    SUBCASE("Spike resets the cooldown") {
        // 50 frames of low usage
        for (int i = 0; i < 50; ++i)
            queue.reset();
        CHECK(queue.frames_below_watermark == 50);

        // Spike usage back to 90 (above 75% of 100)
        for (u32 i = 0; i < 90; ++i)
            queue.submit({});
        queue.reset();

        // Cooldown should reset to 0 because we aren't "below watermark" anymore
        CHECK(queue.frames_below_watermark == 0);
        CHECK(queue.high_watermark == 100);
    }
}
