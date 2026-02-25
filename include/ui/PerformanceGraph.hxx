#pragma once
#include <array>
#include <implot.h>
#include <string_view>

#include "Profiler.hxx"

#include <implot_internal.h>
#include "Types.hxx"


template<std::size_t MaxLines = 8, std::size_t MaxSamples = 120>
class PerformanceGraph {
public:
    struct Line {
        std::string_view name;
        std::array<float, MaxSamples> samples;
        std::size_t write_index = 0;
        std::size_t sample_count = 0;
    };

    PerformanceGraph() = default;

    auto add_line(std::string_view name) -> std::size_t {
        ZoneScopedNC("PerformanceGraph::add_line", 0x8B5CF6);

        if (line_count >= MaxLines)
            return MaxLines;

        graph_lines[line_count].name = name;
        return line_count++;
    }

    auto push_sample(std::size_t line_index, float value) {
        ZoneScopedNC("PerformanceGraph::push_sample", 0x10B981);

        if (line_index >= line_count)
            return;

        auto &line = graph_lines[line_index];

        line.samples[line.write_index] = value;
        line.write_index = (line.write_index + 1) % MaxSamples;

        if (line.sample_count < MaxSamples)
            ++line.sample_count;
    }

    auto push_sample(std::size_t line_index, std::floating_point auto value) {
        return push_sample(line_index, static_cast<f32>(value));
    }

    void render_split(const char *title_prefix, ImVec2 single_plot_size = ImVec2(-1, 80), bool shared_scale = false) {
        ZoneScopedNC("PerformanceGraph::render_split", 0xF59E0B);

        if (line_count == 0)
            return;

        float global_max = 0.0f;
        if (shared_scale) {
            ZoneScopedNC("calculate_global_max", 0xEF4444);

            for (size_t i = 0; i < line_count; ++i) {
                const auto &line = graph_lines[i];
                if (line.sample_count > 0) {
                    auto it = std::max_element(line.samples.begin(), line.samples.begin() + line.sample_count);
                    global_max = std::max(global_max, *it);
                }
            }
            global_max = std::max(global_max * 1.1f, 0.1f);
        }

        for (size_t i = 0; i < line_count; ++i) {
            ZoneScopedNC("render_line", 0x06B6D4);

            const auto &line = graph_lines[i];

            if (line.sample_count == 0)
                continue;

            auto plot_title = std::format("{} - {}", title_prefix, line.name);

            // Use global max or calculate local max
            float y_max;
            if (shared_scale) {
                y_max = global_max;
            } else {
                ZoneScopedNC("calculate_local_max", 0xEF4444);

                float local_max = 0.0f;
                for (size_t j = 0; j < line.sample_count; ++j) {
                    local_max = std::max(local_max, line.samples[j]);
                }
                y_max = std::max(local_max * 1.1f, 0.1f);
            }

            ImVec2 plot_size = single_plot_size;
            if (plot_size.x <= 0.0f) {
                plot_size.x = ImGui::GetContentRegionAvail().x;
            }

            {
                ZoneScopedNC("ImPlot::render", 0x8B5CF6);

                if (ImPlot::BeginPlot(plot_title.c_str(), plot_size, ImPlotFlags_NoTitle | ImPlotFlags_NoMouseText)) {
                    ImPlot::SetupAxes(nullptr, "ms", ImPlotAxisFlags_NoTickLabels, 0);
                    ImPlot::SetupAxisLimits(ImAxis_X1, 0, MaxSamples, ImGuiCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, y_max, ImGuiCond_Always);

                    ImPlot::PlotLine(line.name.data(), line.samples.data(), static_cast<int>(line.sample_count), 1.0,
                                     0.0);

                    ImPlot::EndPlot();
                }
            }

            // Show inline stats
            if (line.sample_count > 0) {
                ZoneScopedNC("calculate_stats", 0xA78BFA);

                float sum = 0.0f;
                float min_val = line.samples[0];
                float max_val = line.samples[0];

                for (size_t j = 0; j < line.sample_count; ++j) {
                    float val = line.samples[j];
                    sum += val;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }

                float avg = sum / line.sample_count;

                auto stats_text = std::format("avg: {:.3f}  min: {:.3f}  max: {:.3f}", avg, min_val, max_val);
                ImGui::SameLine();
                ImGui::TextDisabled("%s", stats_text.c_str());
            }
        }
    }

    auto render(const std::string_view title, ImVec2 size = ImVec2(-1, 150)) -> void {
        ZoneScopedNC("PerformanceGraph::render", 0xF59E0B);

        if (line_count == 0)
            return;

        if (size.x <= 0.0f) {
            size.x = ImGui::GetContentRegionAvail().x;
        }

        if (ImPlot::BeginPlot(title.data(), size, ImPlotFlags_NoTitle)) {
            ImPlot::SetupLegend(ImPlotLocation_South, ImPlotLegendFlags_Outside | ImPlotLegendFlags_Horizontal);
            ZoneScopedNC("ImPlot::render_combined", 0x8B5CF6);

            ImPlot::SetupAxes("Frame", "Time (ms)", ImPlotAxisFlags_NoTickLabels, 0);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, MaxSamples, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, auto_scale_max, ImGuiCond_Always);

            float new_max = 0.0f;

            for (std::size_t i = 0; i < line_count; ++i) {
                const auto &line = graph_lines[i];

                if (line.sample_count == 0)
                    continue;

                if (ImPlot::BeginItem(line.name.data())) {
                    ImPlot::PlotLine(line.name.data(), line.samples.data(), static_cast<int>(line.sample_count), 1.0,
                                     0.0);

                    for (std::size_t j = 0; j < line.sample_count; ++j) {
                        new_max = std::max(new_max, line.samples[j]);
                    }

                    ImPlot::EndItem();
                }
            }

            auto_scale_max = std::max(new_max * 1.1f, 1.0f);
            ImPlot::EndPlot();
        }
    }

    auto clear() -> void {
        ZoneScopedNC("PerformanceGraph::clear", 0xEF4444);

        for (std::size_t i = 0; i < line_count; ++i) {
            auto &line = graph_lines[i];
            line.write_index = 0;
            line.sample_count = 0;
        }
        auto_scale_max = 1.0f;
    }

private:
    std::array<Line, MaxLines> graph_lines{};
    std::size_t line_count = 0;
    float auto_scale_max = 1.0f;
};
