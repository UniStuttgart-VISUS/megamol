#include "GraphLuaExporter.h"

#include <filesystem>
#include <fstream>

namespace megamol::ImageSeries::graph::util {

bool exportToLua(const GraphData2D& graph, const std::string& outfileName, LuaExportMeta meta) {

    static std::atomic_int counter;

    std::string tempname = outfileName + std::to_string(counter++);

    std::ofstream file(tempname);
    if (!file) {
        return false;
    }

    file << "local graphData = {}\n";
    file << "graphData.imgDir = [[" << meta.path << "]]\n";
    file << "graphData.minRange = " << meta.minRange << "\n";
    file << "graphData.maxRange = " << meta.maxRange << "\n";
    file << "graphData.imgW = " << meta.imgW << "\n";
    file << "graphData.imgH = " << meta.imgH << "\n";
    file << "graphData.start = " << meta.startTime << "\n";
    file << "graphData.breakthrough = " << meta.breakthroughTime << "\n";
    file << "graphData.end = " << meta.endTime << "\n";

    file << "graphData.Nodes = {\n";
    std::size_t i = 0;
    for (const auto& node_info : graph.getNodes()) {
        auto& node = node_info.second;
        file << "{" << node.getFrameIndex() << ", " << i++ << ", " << node.centerOfMass.x << ", " << node.centerOfMass.y
             << ", " << node.velocityMagnitude << ", " << 0 << ", " << int(node.area) << ", "
             << int(node.getEdgeCountIn()) << ", " << int(node.getEdgeCountOut()) << "},\n";
    }
    file << "}\n\n";

    file << "graphData.Rects = {\n";
    for (const auto& node_info : graph.getNodes()) {
        auto& node = node_info.second;
        file << "{" << node.boundingBox.x1 << ", " << node.boundingBox.y1 << ", " << node.boundingBox.x2 << ", "
             << node.boundingBox.y2 << "},\n";
    }
    file << "}\n\n";

    file << "graphData.Interfaces = {\n";
    for (const auto& node_info : graph.getNodes()) {
        auto& node = node_info.second;
        file << node.interfaceFluid << ", " << node.interfaceSolid << ",\n";
    }
    file << "}\n\n";

    file << "graphData.Edges = {\n";
    for (auto& edge : graph.getEdges()) {
        file << edge.from << ", " << edge.to << ",\n";
    }
    file << "}\n\n";

    file << "return graphData" << std::endl;
    file.close();

    std::filesystem::rename(tempname, outfileName);


    return true;
}

} // namespace megamol::ImageSeries::graph::util
