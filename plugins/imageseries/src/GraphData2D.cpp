#include "imageseries/graph/GraphData2D.h"

#include <type_traits>

using namespace megamol::ImageSeries::graph;

GraphData2D::NodeID GraphData2D::addNode(Node node) {
    nodes.push_back(std::move(node));
    nodeMap[std::make_pair(node.frameIndex, node.label)] = nodes.size() - 1;
    return nodes.size() - 1;
}

void GraphData2D::addEdge(Edge edge) {
    if (std::find(edges.begin(), edges.end(), edge) == edges.end()) {
        edges.push_back(std::move(edge));

        getNode(edge.from).edgeCountOut++;
        getNode(edge.to).edgeCountIn++;

        getNode(edge.from).childNodes.insert(edge.to);
        getNode(edge.to).parentNodes.insert(edge.from);
    }
}

bool GraphData2D::hasNode(NodeID id) const {
    return id < nodes.size();
}

GraphData2D::NodeID GraphData2D::getNodeCount() const {
    return nodes.size();
}

GraphData2D::Node& GraphData2D::getNode(NodeID id) {
    return id < nodes.size() ? nodes[id] : placeholderNode;
}

const GraphData2D::Node& GraphData2D::getNode(NodeID id) const {
    return id < nodes.size() ? nodes[id] : placeholderNode;
}

std::pair<std::size_t, std::reference_wrapper<GraphData2D::Node>> GraphData2D::findNode(Timestamp time, Label label) {
    auto it = nodeMap.find(std::make_pair(time, label));

    if (it == nodeMap.end()) {
        throw std::runtime_error("Node not found for given time and label.");
    }

    return std::make_pair(it->second, std::reference_wrapper<Node>(getNode(it->second)));
}

std::pair<std::size_t, std::reference_wrapper<const GraphData2D::Node>> GraphData2D::findNode(
    Timestamp time, Label label) const {
    auto it = nodeMap.find(std::make_pair(time, label));

    if (it == nodeMap.end()) {
        throw std::runtime_error("Node not found for given time and label.");
    }

    return std::make_pair(it->second, std::reference_wrapper<const Node>(getNode(it->second)));
}

std::vector<GraphData2D::Node>& GraphData2D::getNodes() {
    return nodes;
}

const std::vector<GraphData2D::Node>& GraphData2D::getNodes() const {
    return nodes;
}

std::vector<GraphData2D::Edge>& GraphData2D::getEdges() {
    return edges;
}

const std::vector<GraphData2D::Edge>& GraphData2D::getEdges() const {
    return edges;
}

std::vector<std::vector<GraphData2D::NodeID>> GraphData2D::getOutboundEdges() const {
    std::vector<std::vector<GraphData2D::NodeID>> result(nodes.size());
    for (auto& edge : edges) {
        if (edge.from < result.size()) {
            result[edge.from].push_back(edge.to);
        }
    }
    return result;
}

std::vector<std::vector<GraphData2D::NodeID>> GraphData2D::getInboundEdges() const {
    std::vector<std::vector<GraphData2D::NodeID>> result(nodes.size());
    for (auto& edge : edges) {
        if (edge.to < result.size()) {
            result[edge.to].push_back(edge.from);
        }
    }
    return result;
}
