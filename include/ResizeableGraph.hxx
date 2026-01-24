#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "Forward.hxx"
#include "Types.hxx"


struct ResizeContext {
    RenderContext &ctx;
    u64 retire_value{0};

    auto get_device() const -> VkDevice;
    auto get_allocator() const -> VmaAllocator;
    auto get_instance() const -> VkInstance;
};

struct ResizeGraph {
    using ResizeCallback = std::function<void(VkExtent2D, const ResizeContext &)>;

    using NodeId = std::uint32_t;

    struct Node {
        NodeId id{0};
        std::string name{};
        std::vector<NodeId> deps{}; // edges: dep -> this
        ResizeCallback rebuild{}; // called in topo order
        std::uint32_t insertion_index{0};
    };

    auto add_node(std::string_view name, ResizeCallback &&rebuild) -> NodeId {
        const NodeId id = next_id++;
        nodes.push_back(Node{
                .id = id,
                .name = std::string{name},
                .deps = {},
                .rebuild = std::move(rebuild),
                .insertion_index = static_cast<std::uint32_t>(nodes.size()),
        });
        id_to_index[id] = nodes.size() - 1;
        return id;
    }

    auto add_dependency(NodeId node, NodeId depends_on) -> void {
        auto *n = find_node(node);
        if (!n)
            std::abort();
        n->deps.push_back(depends_on);
    }

    auto rebuild(VkExtent2D extent, const ResizeContext &rc) -> void {
        if (extent.width == 0 || extent.height == 0) return;

        ensure_topo_cache();
        for (NodeId id: topo_order_cache) {
            nodes.at(id_to_index.at(id)).rebuild(extent, rc);
        }
    }

    [[nodiscard]] auto to_graphviz_dot(bool include_topo_rank = true) const -> std::string;

private:
    auto ensure_topo_cache() -> void {
        if (topo_cache_valid)
            return;
        topo_order_cache = topo_sort_stable(); // your Kahn stable topo
        topo_cache_valid = true;
        topo_cache_revision = graph_revision;
    }

    auto find_node(const NodeId id) -> Node * {
        const auto it = id_to_index.find(id);
        if (it == id_to_index.end())
            return nullptr;
        return &nodes[it->second];
    }

    auto topo_sort_stable() const -> std::vector<NodeId> {
        // Build adjacency and indegrees.
        // deps: dep -> node edge
        std::unordered_map<NodeId, std::vector<NodeId>> outgoing{};
        outgoing.reserve(nodes.size());

        std::unordered_map<NodeId, std::uint32_t> indegree{};
        indegree.reserve(nodes.size());

        for (auto &n: nodes) {
            indegree[n.id] = 0;
        }

        for (auto &n: nodes) {
            for (NodeId dep: n.deps) {
                outgoing[dep].push_back(n.id);
                indegree[n.id] += 1;
            }
        }

        // Ready set: nodes with indegree 0.
        // Stable tie-breaker: insertion_index.
        std::vector<NodeId> ready{};
        ready.reserve(nodes.size());

        for (auto &n: nodes) {
            if (indegree[n.id] == 0) {
                ready.push_back(n.id);
            }
        }

        auto insertion_index_of = [&](NodeId id) -> std::uint32_t { return nodes[id_to_index.at(id)].insertion_index; };

        auto stable_ready_sort = [&] {
            std::ranges::sort(ready, [&](NodeId a, NodeId b) { return insertion_index_of(a) < insertion_index_of(b); });
        };

        stable_ready_sort();

        std::vector<NodeId> out{};
        out.reserve(nodes.size());

        while (!ready.empty()) {
            const NodeId n = ready.front();
            ready.erase(ready.begin()); // small N, fine; can optimize later
            out.push_back(n);

            auto it = outgoing.find(n);
            if (it == outgoing.end())
                continue;

            for (NodeId m: it->second) {
                auto &deg = indegree[m];
                deg -= 1;
                if (deg == 0) {
                    ready.push_back(m);
                }
            }

            // Preserve insertion order among newly-ready nodes too.
            stable_ready_sort();
        }

        if (out.size() != nodes.size()) {
            // Cycle detected. Optional: print node names here.
            std::abort();
        }

        return out;
    }

    NodeId next_id{1};
    std::vector<Node> nodes{};
    std::unordered_map<NodeId, std::size_t> id_to_index{};

    std::uint64_t graph_revision{0};
    std::uint64_t topo_cache_revision{0};
    bool topo_cache_valid{false};
    std::vector<NodeId> topo_order_cache{};
};
