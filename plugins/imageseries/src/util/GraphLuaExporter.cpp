/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "GraphLuaExporter.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace megamol::ImageSeries::graph::util {

bool exportToLua(const GraphData2D& graph, const std::string& outfileName, LuaExportMeta meta,
    const std::vector<std::size_t>& velocityDistribution) {

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
    file << "graphData.startTime = " << meta.startTime << "\n";
    file << "graphData.breakthroughTime = " << meta.breakthroughTime << "\n";
    file << "graphData.endTime = " << meta.endTime << "\n";

    std::map<std::size_t, std::size_t> idToNum;

    file << "graphData.Nodes = {\n";
    std::size_t i = 0;
    for (const auto& node_info : graph.getNodes()) {
        auto& node = node_info.second;
        idToNum[node.getID()] = i;
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
    for (const auto& edge : graph.getEdges()) {
        file << idToNum.at(edge.from) << ", " << idToNum.at(edge.to) << ",\n";
    }
    file << "}\n\n";

    file << "graphData.Velocities = {\n";
    for (const auto velocity : velocityDistribution) {
        file << velocity << ",\n";
    }
    file << "}\n\n";

    file << "return graphData" << std::endl;
    file.close();

    std::filesystem::rename(tempname, outfileName);


    return true;
}

} // namespace megamol::ImageSeries::graph::util
