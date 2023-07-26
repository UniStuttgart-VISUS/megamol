/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "imageseries/graph/GraphData2D.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace megamol::ImageSeries::graph;

GraphData2D::NodeID GraphData2D::addNode(Node node) {
    const auto newNodeID = getNodeID(node.frameIndex, node.label);

    nodes.insert(std::make_pair(newNodeID, std::move(node)));

    return newNodeID;
}

std::size_t GraphData2D::getNodeCount() const {
    return nodes.size();
}

bool GraphData2D::hasNode(const NodeID id) const {
    return nodes.find(id) != nodes.end();
}

bool GraphData2D::hasNode(const Timestamp time, const Label label) const {
    return hasNode(getNodeID(time, label));
}

GraphData2D::Node& GraphData2D::getNode(const NodeID id) {
    return nodes.at(id);
}

GraphData2D::Node& GraphData2D::getNode(const Timestamp time, const Label label) {
    return getNode(getNodeID(time, label));
}

const GraphData2D::Node& GraphData2D::getNode(const NodeID id) const {
    return nodes.at(id);
}

const GraphData2D::Node& GraphData2D::getNode(const Timestamp time, const Label label) const {
    return getNode(getNodeID(time, label));
}

GraphData2D::Node GraphData2D::removeNode(NodeID id, bool lazy) {
    GraphData2D::Node toBeRemoved(nodes.at(id));

    if (lazy) {
        nodes.at(id).removed = true;
    } else {
        for (auto& parent : nodes.at(id).parentNodes) {
            removeEdge(parent, id);
        }
        for (auto& child : nodes.at(id).childNodes) {
            removeEdge(id, child);
        }

        nodes.erase(id);
    }

    return toBeRemoved;
}

GraphData2D::Node GraphData2D::removeNode(const Timestamp time, const Label label, bool lazy) {
    return removeNode(getNodeID(time, label), lazy);
}

void GraphData2D::addEdge(Edge edge) {
    if (std::find(edges.begin(), edges.end(), edge) == edges.end()) {
        getNode(edge.from).childNodes.insert(edge.to);
        getNode(edge.to).parentNodes.insert(edge.from);

        edges.push_back(std::move(edge));
    }
}

std::size_t GraphData2D::getEdgeCount() const {
    return edges.size();
}

bool GraphData2D::hasEdge(NodeID from, NodeID to) const {
    for (auto& edge : edges) {
        if (edge.from == from && edge.to == to) {
            return true;
        }
    }

    return false;
}

GraphData2D::Edge& GraphData2D::getEdge(NodeID from, NodeID to) {
    for (auto& edge : edges) {
        if (edge.from == from && edge.to == to) {
            return edge;
        }
    }

    throw std::runtime_error("Edge not found for given nodes.");
}

const GraphData2D::Edge& GraphData2D::getEdge(NodeID from, NodeID to) const {
    for (const auto& edge : edges) {
        if (edge.from == from && edge.to == to) {
            return edge;
        }
    }

    throw std::runtime_error("Edge not found for given nodes.");
}

void GraphData2D::removeEdge(NodeID from, NodeID to) {
    getNode(from).childNodes.erase(to);
    getNode(to).parentNodes.erase(from);

    for (auto it = edges.begin(); it != edges.end(); ++it) {
        if (it->from == from && it->to == to) {
            edges.erase(it);
            break;
        }
    }
}

void GraphData2D::removeEdge(const Edge& edge) {
    removeEdge(edge.from, edge.to);
}

void GraphData2D::finalizeLazyRemoval() {
    // Remove invalidated edges and remove nodes
    for (auto it = nodes.begin(); it != nodes.end();) {
        if (it->second.removed) {
            for (auto& parent : it->second.parentNodes) {
                removeEdge(parent, it->first);
            }
            for (auto& child : it->second.childNodes) {
                removeEdge(it->first, child);
            }

            it = nodes.erase(it);
        } else {
            ++it;
        }
    }
}

std::unordered_map<GraphData2D::NodeID, GraphData2D::Node>& GraphData2D::getNodes() {
    return nodes;
}

const std::unordered_map<GraphData2D::NodeID, GraphData2D::Node>& GraphData2D::getNodes() const {
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

GraphData2D::NodeID GraphData2D::getNodeID(Timestamp time, Label label) {
    static std::hash<std::pair<Timestamp, Label>> hasher;
    return hasher(std::make_pair(time, label));
}
