#include "GraphGDFExporter.h"

#include <filesystem>
#include <fstream>

namespace megamol::ImageSeries::graph::util {

bool exportToGDF(const GraphData2D& graph, const std::string& outfileName, GDFExportMeta meta) {

    static std::atomic_int counter;

    std::string tempname = outfileName + std::to_string(counter++);

    std::ofstream file(tempname);
    if (!file) {
        return false;
    }

    std::map<std::size_t, std::size_t> idToNum;

    file << "nodedef>name VARCHAR,time DOUBLE,label VARCHAR,x DOUBLE,y DOUBLE,xPos DOUBLE,"
            "yPos DOUBLE,velocity DOUBLE,area DOUBLE,edgeIn DOUBLE,edgeOut DOUBLE\n";
    std::size_t i = 0;
    for (const auto& node_info : graph.getNodes()) {
        auto& node = node_info.second;
        idToNum[node.getID()] = i;

        if (!(meta.stopAtBreakthrough && node.getFrameIndex() > meta.breakthroughTime)) {
            file << "s" << i++ << ","
                 << node.getFrameIndex() << ","
                 << "'Time: " << node.getFrameIndex() << "',"
                 << node.centerOfMass.x << "," << -node.centerOfMass.y << "," // original position loaded, e.g., in Gephi
                 << node.centerOfMass.x << "," << -node.centerOfMass.y << "," // same position as "backup"
                 << node.velocityMagnitude << ","
                 << node.area << ","
                 << node.getEdgeCountIn() << ","
                 << node.getEdgeCountOut() << "\n";
        }
    }

    file << "edgedef>node1 VARCHAR,node2 VARCHAR,directed BOOLEAN\n";
    for (auto& edge : graph.getEdges()) {
        if (!(meta.stopAtBreakthrough && (graph.getNode(edge.from).getFrameIndex() > meta.breakthroughTime ||
                graph.getNode(edge.to).getFrameIndex() > meta.breakthroughTime))) {

            file << "s" << idToNum.at(edge.from) << ","
                 << "s" << idToNum.at(edge.to) << ","
                 << "true\n";
        }
    }

    file.close();

    std::filesystem::rename(tempname, outfileName);

    return true;
}

} // namespace megamol::ImageSeries::graph::util
