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
        return megamol::ImageSeries::util::computeHash(val.first, val.second);
    }
};
} // namespace std

namespace megamol::ImageSeries::graph {

class GraphData2D {
public:
    using NodeID = std::size_t;
    using Pixel = std::uint32_t;
    using Label = Label_;
    using Timestamp = Timestamp_;

    struct Rect {
        int x1 = 0;
        int y1 = 0;
        int x2 = -1;
        int y2 = -1;

        bool valid() const {
            return x1 <= x2 && y1 <= y2;
        }

        Rect& Union(const Rect& other) {
            if (!valid()) {
                *this = other;
                return *this;
            } else if (!other.valid()) {
                return *this;
            }

            x1 = std::min(x1, other.x1);
            y1 = std::min(y1, other.y1);
            x2 = std::max(x2, other.x2);
            y2 = std::max(y2, other.y2);

            return *this;
        }
    };

    struct Node {
        Node(const Timestamp frameIndex, const Label label)
                : id(GraphData2D::getNodeID(frameIndex, label))
                , frameIndex(frameIndex)
                , label(label) {}
        Node(const Node& src)
                : id(GraphData2D::getNodeID(src.frameIndex, src.label))
                , frameIndex(src.frameIndex)
                , label(src.label)
                , parentNodes(src.parentNodes)
                , childNodes(src.childNodes) {

            pixels = src.pixels;
            area = src.area;

            centerOfMass = src.centerOfMass;
            velocity = src.velocity;
            velocityMagnitude = src.velocityMagnitude;

            boundingBox = src.boundingBox;

            interfaces = src.interfaces;
            interfaceFluid = src.interfaceFluid;
            interfaceSolid = src.interfaceSolid;

            valid = src.valid;
        }
        Node(Node&& src) noexcept
                : id(GraphData2D::getNodeID(src.frameIndex, src.label))
                , frameIndex(src.frameIndex)
                , label(src.label)
                , parentNodes(std::move(src.parentNodes))
                , childNodes(std::move(src.childNodes)) {

            std::swap(pixels, src.pixels);
            std::swap(area, src.area);

            std::swap(centerOfMass, src.centerOfMass);
            std::swap(velocity, src.velocity);
            std::swap(velocityMagnitude, src.velocityMagnitude);

            std::swap(boundingBox, src.boundingBox);

            std::swap(interfaces, src.interfaces);
            std::swap(interfaceFluid, src.interfaceFluid);
            std::swap(interfaceSolid, src.interfaceSolid);

            std::swap(valid, src.valid);
        }

        NodeID getID() const {
            return id;
        }
        Label getLabel() const {
            return label;
        }
        Timestamp getFrameIndex() const {
            return frameIndex;
        }

        const std::unordered_set<NodeID>& getParentNodes() const {
            return parentNodes;
        }
        const std::unordered_set<NodeID>& getChildNodes() const {
            return childNodes;
        }

        std::size_t getEdgeCountIn() const {
            return parentNodes.size();
        }
        std::size_t getEdgeCountOut() const {
            return childNodes.size();
        }

        bool isRemoved() const {
            return removed;
        }

        std::vector<Pixel> pixels;
        float area = 0.f;

        glm::vec2 centerOfMass = {};
        glm::vec2 velocity = {};
        float velocityMagnitude = 0.f;

        using rect_t = Rect;
        Rect boundingBox;

        std::map<Timestamp, std::unordered_set<Pixel>> interfaces;
        float interfaceFluid = 0.f;
        float interfaceSolid = 0.f;

        bool valid = true;

    private:
        const NodeID id;

        const Timestamp frameIndex;
        const Label label;

        std::unordered_set<NodeID> parentNodes, childNodes;

        bool removed = false;
        friend class GraphData2D;
    };

    struct Edge {
        Edge(NodeID from, NodeID to, float weight = 0.0f) : from(from), to(to), weight(weight) {}
        Edge(const Edge& src) = default;
        Edge(Edge&& src) noexcept = default;
        Edge& operator=(const Edge& rhs) = default;
        Edge& operator=(Edge&& rhs) noexcept = default;

        inline bool operator==(const GraphData2D::Edge& other) {
            return this->from == other.from && this->to == other.to;
        }

        NodeID from, to;
        float weight;
    };

    GraphData2D() = default;

    GraphData2D(const GraphData2D&) = default;
    GraphData2D(GraphData2D&&) = default;
    GraphData2D& operator=(const GraphData2D&) = default;
    GraphData2D& operator=(GraphData2D&&) = default;

    // Nodes
    NodeID addNode(Node node);

    std::size_t getNodeCount() const;

    bool hasNode(NodeID id) const;
    bool hasNode(Timestamp time, Label label) const;

    Node& getNode(NodeID id);
    Node& getNode(Timestamp time, Label label);
    const Node& getNode(NodeID id) const;
    const Node& getNode(Timestamp time, Label label) const;

    Node removeNode(NodeID id, bool lazy = false);
    Node removeNode(Timestamp time, Label label, bool lazy = false);

    // Edges
    void addEdge(Edge edge);

    std::size_t getEdgeCount() const;

    bool hasEdge(NodeID from, NodeID to) const;

    Edge& getEdge(NodeID from, NodeID to);
    const Edge& getEdge(NodeID from, NodeID to) const;

    void removeEdge(NodeID from, NodeID to);
    void removeEdge(const Edge& edge);

    // Lazy removal
    void finalizeLazyRemoval();

    // Access functions
    std::unordered_map<NodeID, Node>& getNodes();
    const std::unordered_map<NodeID, Node>& getNodes() const;
    std::vector<Edge>& getEdges();
    const std::vector<Edge>& getEdges() const;

    std::vector<std::vector<NodeID>> getOutboundEdges() const;
    std::vector<std::vector<NodeID>> getInboundEdges() const;

private:
    std::unordered_map<NodeID, Node> nodes;
    std::vector<Edge> edges;

    static NodeID getNodeID(Timestamp time, Label label);
    friend class Node;
};

using AsyncGraphData2D = megamol::ImageSeries::util::AsyncData<const GraphData2D>;
using AsyncGraphPtr = std::shared_ptr<const AsyncGraphData2D>;

} // namespace megamol::ImageSeries::graph

#endif
