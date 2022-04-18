#include "imageseries/graph/GraphData2D.h"

using namespace megamol::ImageSeries::graph;

GraphData2D::NodeID GraphData2D::addNode(Node node) {
    nodes.push_back(std::move(node));
    return nodes.size() - 1;
}

void GraphData2D::addEdge(Edge edge) {
    edges.push_back(std::move(edge));
}

bool GraphData2D::hasNode(NodeID id) const {
    return id < nodes.size();
}

GraphData2D::Node& GraphData2D::getNode(NodeID id) {
    return id < nodes.size() ? nodes[id] : placeholderNode;
}

const GraphData2D::Node& GraphData2D::getNode(NodeID id) const {
    return id < nodes.size() ? nodes[id] : placeholderNode;
}

const std::vector<GraphData2D::Node>& GraphData2D::getNodes() const {
    return nodes;
}

const std::vector<GraphData2D::Edge>& GraphData2D::getEdges() const {
    return edges;
}
