#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vk_mem_alloc.h>

#include "Forward.hxx"
#include "Types.hxx"


struct ResizeContext {
    RenderContext &ctx;
    u64 retire_value{0};

    [[nodiscard]] auto get_device() const -> VkDevice;
    [[nodiscard]] auto get_allocator() const -> VmaAllocator;
    [[nodiscard]] auto get_instance() const -> VkInstance;
};

namespace detail {
    constexpr auto bit(std::unsigned_integral auto const x) -> u32 { return 1 << x; }
} // namespace detail


enum class ResizeTrigger : u32 {
    Empty = 0,
    Extent = detail::bit(0u), // Window/Resolution change
    Shaders = detail::bit(1u), // Hot-reloading pipelines
    Bindings = detail::bit(2u), // Descriptor set changes
    ShadowMap = detail::bit(3u), // Shadow map resolution change
    Clustering = detail::bit(4u), // Cluster config change
    All = 0xFFFFFFFF
};
inline auto to_string(ResizeTrigger flags) -> std::string;

constexpr auto operator|(ResizeTrigger a, ResizeTrigger b) -> ResizeTrigger {
    return static_cast<ResizeTrigger>(static_cast<u32>(a) | static_cast<u32>(b));
}

constexpr auto operator&(ResizeTrigger a, ResizeTrigger b) -> ResizeTrigger {
    return static_cast<ResizeTrigger>(static_cast<u32>(a) & static_cast<u32>(b));
}

constexpr auto operator!=(ResizeTrigger a, ResizeTrigger b) -> bool {
    return static_cast<u32>(a) != static_cast<u32>(b);
}

constexpr auto operator!=(std::integral auto a, ResizeTrigger b) -> bool {
    return static_cast<u32>(a) != static_cast<u32>(b);
}

constexpr auto operator!=(ResizeTrigger a, std::integral auto b) -> bool {
    return static_cast<u32>(a) != static_cast<u32>(b);
}

struct ResizeGraph {
    using ResizeCallback = std::function<void(VkExtent2D, const ResizeContext &)>;

    using NodeId = u32;

    struct Node {
        NodeId id{0};
        std::string name{};
        std::vector<NodeId> deps{}; // edges: dep -> this
        ResizeCallback rebuild{}; // called in topo order
        u32 insertion_index{0};
        ResizeTrigger category{ResizeTrigger::Extent}; // Default to Extent
    };

    auto add_node(std::string_view name, ResizeCallback &&rebuild, ResizeTrigger category = ResizeTrigger::All)
            -> NodeId {
        const NodeId id = next_id++;
        nodes.push_back(Node{
                .id = id,
                .name = std::string{name},
                .deps = {},
                .rebuild = std::move(rebuild),
                .insertion_index = static_cast<u32>(nodes.size()),
                .category = category,
        });
        id_to_index[id] = nodes.size() - 1;
        topo_cache_valid = false;
        return id;
    }

    auto add_dependency(NodeId node, NodeId depends_on) -> void {
        auto *n = find_node(node);
        if (!n)
            std::abort();
        n->deps.push_back(depends_on);
        topo_cache_valid = false;
    }

    auto rebuild(const VkExtent2D extent, const ResizeContext &rc, ResizeTrigger trigger = ResizeTrigger::All) -> void {
        if (extent.width == 0 || extent.height == 0)
            return;

        ensure_topo_cache();

        std::unordered_set<NodeId> dirty_nodes;

        for (NodeId id: topo_order_cache) {
            auto &node = nodes.at(id_to_index.at(id));

            const bool category_match = (node.category & trigger) != 0;
            const bool dependency_dirty =
                    std::ranges::any_of(node.deps, [&](const NodeId &dep_id) { return dirty_nodes.contains(dep_id); });

            if (category_match || dependency_dirty) {
                node.rebuild(extent, rc);
                dirty_nodes.insert(id);
            }
        }
    }

    auto trigger_resize(ResizeTrigger trigger) { pending_triggers.fetch_or(static_cast<u32>(trigger)); }

    auto get_and_clear_triggers() -> ResizeTrigger { return static_cast<ResizeTrigger>(pending_triggers.exchange(0)); }

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
        std::unordered_map<NodeId, std::vector<NodeId>> outgoing{};
        outgoing.reserve(nodes.size());

        std::unordered_map<NodeId, u32> indegree{};
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

        std::vector<NodeId> ready{};
        ready.reserve(nodes.size());

        for (auto &n: nodes) {
            if (indegree[n.id] == 0) {
                ready.push_back(n.id);
            }
        }

        auto insertion_index_of = [&](NodeId id) -> u32 { return nodes[id_to_index.at(id)].insertion_index; };

        auto stable_ready_sort = [&] {
            std::ranges::sort(ready, [&](NodeId a, NodeId b) { return insertion_index_of(a) < insertion_index_of(b); });
        };

        stable_ready_sort();

        std::vector<NodeId> out{};
        out.reserve(nodes.size());

        while (!ready.empty()) {
            const NodeId n = ready.front();
            ready.erase(ready.begin());
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

            stable_ready_sort();
        }

        if (out.size() != nodes.size()) {
            std::abort();
        }

        return out;
    }

    NodeId next_id{1};
    std::vector<Node> nodes{};
    std::unordered_map<NodeId, std::size_t> id_to_index{};

    std::atomic<u32> pending_triggers{0};

    std::uint64_t graph_revision{0};
    std::uint64_t topo_cache_revision{0};
    bool topo_cache_valid{false};
    std::vector<NodeId> topo_order_cache{};
};

inline auto to_string(ResizeTrigger flags) -> std::string {
    if (flags == ResizeTrigger::Empty)
        return "None";

    if (flags == ResizeTrigger::All)
        return "All";

    struct FlagName {
        ResizeTrigger flag;
        std::string_view name;
    };

    constexpr std::array<FlagName, 4> names = {
            FlagName{.flag = ResizeTrigger::Extent, .name = "Extent"},
            FlagName{.flag = ResizeTrigger::Shaders, .name = "Shaders"},
            FlagName{.flag = ResizeTrigger::Bindings, .name = "Bindings"},
            FlagName{.flag = ResizeTrigger::ShadowMap, .name = "ShadowMap"},
    };

    std::string result;
    for (auto &[flag, name]: names) {
        if ((flags & flag) == flag) {
            if (!result.empty())
                result += " | ";
            result += name;
        }
    }

    return result;
}
