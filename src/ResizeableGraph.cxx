#include "ResizeableGraph.hxx"

auto ResizeGraph::to_graphviz_dot(bool include_topo_rank) const -> std::string {
    std::ostringstream out;
    out << "digraph ResizeGraph {\n";
    out << "  rankdir=LR;\n";
    out << "  node [shape=box, fontname=\"Consolas\", fontsize=10];\n";
    out << "  edge [fontname=\"Consolas\", fontsize=9];\n\n";

    // Optional: compute topo order to display rank / detect cycles.
    std::unordered_map<NodeId, std::uint32_t> topo_rank{};
    if (include_topo_rank) {
        // topo_sort_stable() aborts on cycle in your code; if you want DOT even on cycles,
        // replace this with a non-aborting variant and mark unknown ranks.
        auto topo = topo_sort_stable();
        topo_rank.reserve(topo.size());
        for (std::uint32_t i = 0; i < topo.size(); ++i) {
            topo_rank[topo[i]] = i;
        }
    }

    auto escape = [](std::string_view s) -> std::string {
        std::string r;
        r.reserve(s.size());
        for (char c: s) {
            switch (c) {
                case '\\': r += "\\\\";
                    break;
                case '\"': r += "\\\"";
                    break;
                case '\n': r += "\\n";
                    break;
                case '\r': break;
                case '\t': r += "\\t";
                    break;
                default: r += c;
                    break;
            }
        }
        return r;
    };

    // Emit nodes.
    for (auto const &n: nodes) {
        out << "  n" << n.id << " [label=\""
                << escape(n.name);

        out << "\\n"
                << "id=" << n.id
                << "  ins=" << n.insertion_index;

        if (include_topo_rank) {
            if (auto it = topo_rank.find(n.id); it != topo_rank.end()) {
                out << "  topo=" << it->second;
            } else {
                out << "  topo=?";
            }
        }

        out << "\"];\n";
    }
    out << "\n";

    // Emit edges dep -> node.
    // n.deps stores "depends_on", so outgoing edge is dep -> n.
    for (auto const &n: nodes) {
        for (NodeId dep: n.deps) {
            out << "  n" << dep << " -> n" << n.id << ";\n";
        }
    }

    // Optional: align nodes by topo rank (nice for stable topo visual).
    if (include_topo_rank) {
        // Group by rank into same-rank subgraphs (DOT uses subgraph rank=same)
        // Here each rank is unique, so it’s mostly to force ordering; still helpful.
        auto topo = topo_sort_stable();
        out << "\n";
        out << "  { rank=same; ";
        for (auto id: topo) {
            out << "n" << id << "; ";
        }
        out << "}\n";
    }

    out << "}\n";
    return out.str();
}
