#include "GraphLuaExporter.h"

#include <filesystem>
#include <fstream>

namespace megamol::ImageSeries::graph::util {

bool exportToLua(const GraphData2D& graph, const std::string& outfileName, const std::string& metaPath) {

    static std::atomic_int counter;

    std::string tempname = outfileName + std::to_string(counter++);

    std::ofstream file(tempname);
    if (!file) {
        return false;
    }

    // TODO metadata
    // clang-format off
	file << "local graphData = {}\n";
	file << "graphData.imgDir = [[" << metaPath << "]]\n";
	file << "graphData.minRange = " << 0.0 << "\n";
	file << "graphData.maxRange = " << 1.0 << "\n";
    // clang-format on

    file << "graphData.Nodes = {\n";
    for (std::size_t i = 0; i < graph.getNodes().size(); ++i) {
        auto& node = graph.getNodes()[i];
        file << "{" << node.frameIndex << ", " << i << ", " << node.centerOfMass.x << ", " << node.centerOfMass.y
             << ", " << node.velocityMagnitude << ", " << 0 << ", " << int(node.area) << ", " << int(node.edgeCountIn)
             << ", " << int(node.edgeCountOut) << "},\n";
    }
    file << "}\n\n";

    file << "graphData.Rects = {\n";
    for (std::size_t i = 0; i < graph.getNodes().size(); ++i) {
        auto& node = graph.getNodes()[i];
        file << "{" << node.boundingBox.x1 << ", " << node.boundingBox.y1 << ", " << node.boundingBox.x2 << ", "
             << node.boundingBox.y2 << "},\n";
    }
    file << "}\n\n";

    file << "graphData.Interfaces = {\n";
    for (std::size_t i = 0; i < graph.getNodes().size(); ++i) {
        auto& node = graph.getNodes()[i];
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
