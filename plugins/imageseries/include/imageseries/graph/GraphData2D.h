#ifndef INCLUDE_IMAGESERIES_GRAPHDATA2D_H_
#define INCLUDE_IMAGESERIES_GRAPHDATA2D_H_

#include "../util/AsyncData.h"

#include "glm/vec2.hpp"

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace megamol::ImageSeries::graph {

class GraphData2D {
public:
    using NodeID = std::uint32_t;

    static constexpr NodeID NodeIDNone = std::numeric_limits<NodeID>::max();

    struct Rect {
        int x1 = 0;
        int y1 = 0;
        int x2 = -1;
        int y2 = -1;

        bool valid() const {
            return x1 <= x2 && y1 <= y2;
        }
    };

    struct Node {
        std::size_t flowFrontIndex = 0;

        std::uint32_t frameIndex = 0;
        glm::vec2 centerOfMass = {};
        glm::vec2 velocity = {};
        float velocityMagnitude = 0.f;
        float area = 0.f;
        float interfaceFluid = 0.f;
        float interfaceSolid = 0.f;
        float averageChordLength = 0.f;
        Rect boundingBox;
        std::uint8_t modified = 0;
        bool valid = true;

        std::uint8_t edgeCountIn = 0;
        std::uint8_t edgeCountOut = 0;
        std::unordered_set<NodeID> parentNodes, childNodes;
    };

    struct Edge {
        inline bool operator==(const GraphData2D::Edge& other) {
            return this->from == other.from && this->to == other.to;
        }
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
    NodeID getNodeCount() const;
    Node& getNode(NodeID id);
    const Node& getNode(NodeID id) const;

    const std::vector<Node>& getNodes() const;
    const std::vector<Edge>& getEdges() const;

    std::vector<std::vector<NodeID>> getOutboundEdges() const;
    std::vector<std::vector<NodeID>> getInboundEdges() const;

private:
    std::vector<Node> nodes;
    std::vector<Edge> edges;

    Node placeholderNode;
};

using AsyncGraphData2D = megamol::ImageSeries::util::AsyncData<const GraphData2D>;
using AsyncGraphPtr = std::shared_ptr<const AsyncGraphData2D>;

} // namespace megamol::ImageSeries::graph

#endif
