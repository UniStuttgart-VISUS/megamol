#ifndef INCLUDE_IMAGESERIES_GRAPHDATA2D_H_
#define INCLUDE_IMAGESERIES_GRAPHDATA2D_H_

#include "../util/AsyncData.h"

#include "glm/vec2.hpp"

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace megamol::ImageSeries::graph {

class GraphData2D {
public:
    using NodeID = std::uint32_t;

    static constexpr NodeID NodeIDNone = std::numeric_limits<NodeID>::max();

    struct Node {
        std::uint32_t frameIndex = 0;
        glm::vec2 centerOfMass;
        glm::vec2 velocity;
        float area = 0.f;
        float interfaceLength = 0.f;
        std::uint8_t edgeCountIn = 0;
        std::uint8_t edgeCountOut = 0;
    };

    struct Edge {
        NodeID from = NodeIDNone;
        NodeID to = NodeIDNone;
    };

    GraphData2D() = default;

    GraphData2D(const GraphData2D&) = default;
    GraphData2D(GraphData2D&&) = default;
    GraphData2D& operator=(const GraphData2D&) = default;
    GraphData2D& operator=(GraphData2D&&) = default;

    NodeID addNode(Node node);
    void addEdge(Edge edge);

    bool hasNode(NodeID id) const;
    Node& getNode(NodeID id);
    const Node& getNode(NodeID id) const;

    const std::vector<Node>& getNodes() const;
    const std::vector<Edge>& getEdges() const;

private:
    std::vector<Node> nodes;
    std::vector<Edge> edges;

    Node placeholderNode;
};

using AsyncGraphData2D = util::AsyncData<const GraphData2D>;
using AsyncGraphPtr = std::shared_ptr<const AsyncGraphData2D>;

} // namespace megamol::ImageSeries::graph

#endif
