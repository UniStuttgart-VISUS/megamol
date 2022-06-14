#include "GraphSimplifier.h"

#include <vector>

namespace megamol::ImageSeries::graph::util {

GraphData2D simplifyGraph(GraphData2D graph) {
    auto outbound = graph.getOutboundEdges();
    auto inbound = graph.getInboundEdges();

    double areaRatioThreshold = 10.0;
    double areaAbsoluteMin = 3.0;
    double areaAbsoluteMax = 30.0;

    auto idNone = GraphData2D::NodeIDNone;
    auto nodeCount = graph.getNodeCount();

    for (GraphData2D::NodeID nodeID = 0; nodeID < nodeCount; ++nodeID) {
        auto parentID = inbound[nodeID].empty() ? idNone : inbound[nodeID][0];
        if (parentID < nodeCount) {
            auto& node = graph.getNode(nodeID);
            auto& parent = graph.getNode(parentID);

            // Merge small nodes into parents
            if (node.area <= areaAbsoluteMin ||
                (node.area <= areaAbsoluteMax && node.area * areaRatioThreshold <= parent.area)) {
                parent.area += node.area;
                node.area = 0;

                // Update child count
                parent.edgeCountOut = parent.edgeCountOut + node.edgeCountOut - 1;

                // Update parents of any child nodes
                for (auto childID : outbound[nodeID]) {
                    if (childID < nodeCount) {
                        if (inbound[childID].size() == 1) {
                            // Single-parent node: replace directly
                            inbound[childID][0] = parentID;
                        } else {
                            // Multi-parent node: need to search
                            for (auto& childParentID : inbound[childID]) {
                                if (childParentID == nodeID) {
                                    childParentID = parentID;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<GraphData2D::NodeID> nodeIDMap(nodeCount, idNone);
    auto translateNodeID = [&](GraphData2D::NodeID nodeID) {
        return nodeID < nodeIDMap.size() ? nodeIDMap[nodeID] : idNone;
    };

    GraphData2D result;

    for (GraphData2D::NodeID nodeID = 0; nodeID < nodeCount; ++nodeID) {
        auto& node = graph.getNode(nodeID);
        if (node.area > 0) {
            nodeIDMap[nodeID] = result.addNode(node);
        }
    }

    for (GraphData2D::NodeID nodeID = 0; nodeID < nodeCount; ++nodeID) {
        GraphData2D::Edge edge;
        edge.to = translateNodeID(nodeID);
        if (edge.to != idNone) {
            for (auto parentID : inbound[nodeID]) {
                edge.from = translateNodeID(parentID);
                if (edge.from != idNone) {
                    result.addEdge(edge);
                }
            }
        }
    }

    return result;
}

} // namespace megamol::ImageSeries::graph::util
