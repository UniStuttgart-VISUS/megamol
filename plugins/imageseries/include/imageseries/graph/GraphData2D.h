#ifndef INCLUDE_IMAGESERIES_GRAPHDATA2D_H_
#define INCLUDE_IMAGESERIES_GRAPHDATA2D_H_

#include "../util/AsyncData.h"
#include "../util/ImageUtils.h"

#include "glm/vec2.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <vector>

namespace {
using Label_ = std::uint16_t;
using Timestamp_ = std::uint16_t;
} // namespace

namespace std {
using TimeLabelPair = std::pair<Timestamp_, Label_>;

template<>
struct hash<TimeLabelPair> {
    inline std::size_t operator()(const TimeLabelPair& val) const {
        return megamol::ImageSeries::util::combineHash(val.first, val.second);
    }
};
} // namespace std

namespace megamol::ImageSeries::graph {

class GraphData2D {
public:
    using NodeID = std::uint32_t;
    using Pixel = std::uint32_t;
    using Label = Label_;
    using Timestamp = Timestamp_;

    static constexpr NodeID NodeIDNone = std::numeric_limits<NodeID>::max();

    struct Rect {
        int x1 = 0;
        int y1 = 0;
        int x2 = -1;
        int y2 = -1;

        bool valid() const {
            return x1 <= x2 && y1 <= y2;
        }

        Rect& Union(const Rect& other) {
            x1 = std::min(x1, other.x1);
            y1 = std::min(y1, other.y1);
            x2 = std::max(x2, other.x2);
            y2 = std::max(y2, other.y2);

            return *this;
        }
    };

    struct Node {
        Label label = 0;
        Timestamp frameIndex = 0;
        std::vector<Pixel> pixels;

        float area = 0.f;
        float interfaceFluid = 0.f;
        float interfaceSolid = 0.f;

        glm::vec2 centerOfMass = {};
        glm::vec2 velocity = {};
        float velocityMagnitude = 0.f;

        using rect_t = Rect;
        Rect boundingBox;
        std::map<Label, std::unordered_set<Pixel>> interfaces;

        float averageChordLength = 0.f;
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

        float weight = 0.0;
    };

    GraphData2D() = default;

    GraphData2D(const GraphData2D&) = default;
    GraphData2D(GraphData2D&&) = default;
    GraphData2D& operator=(const GraphData2D&) = default;
    GraphData2D& operator=(GraphData2D&&) = default;

    NodeID addNode(Node node);
    void addEdge(Edge edge);

    Node removeNode(NodeID id, bool lazy = false);
    void removeEdge(NodeID from, NodeID to);
    void finalizeLazyRemoval();

    bool hasNode(NodeID id) const;
    NodeID getNodeCount() const;
    Node& getNode(NodeID id);
    const Node& getNode(NodeID id) const;

    Edge& getEdge(NodeID from, NodeID to);
    const Edge& getEdge(NodeID from, NodeID to) const;

    std::pair<NodeID, std::reference_wrapper<Node>> findNode(Timestamp time, Label label);
    std::pair<NodeID, std::reference_wrapper<const Node>> findNode(Timestamp time, Label label) const;

    std::vector<Node>& getNodes();
    const std::vector<Node>& getNodes() const;
    std::vector<Edge>& getEdges();
    const std::vector<Edge>& getEdges() const;

    std::vector<std::vector<NodeID>> getOutboundEdges() const;
    std::vector<std::vector<NodeID>> getInboundEdges() const;

private:
    std::vector<Node> nodes;
    std::vector<Edge> edges;

    std::unordered_map<std::pair<Timestamp, Label>, NodeID> nodeMap;

    Node placeholderNode;
};

using AsyncGraphData2D = megamol::ImageSeries::util::AsyncData<const GraphData2D>;
using AsyncGraphPtr = std::shared_ptr<const AsyncGraphData2D>;

} // namespace megamol::ImageSeries::graph

#endif
