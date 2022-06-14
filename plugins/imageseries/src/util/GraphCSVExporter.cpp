#include "GraphCSVExporter.h"

#include <filesystem>
#include <fstream>

namespace megamol::ImageSeries::graph::util {

bool exportToCSV(const GraphData2D& graph, const std::string& outfileName) {

    static std::atomic_int counter;

    std::string tempname = outfileName + std::to_string(counter++);

    std::ofstream file(tempname);
    if (!file) {
        return false;
    }

    file << "ID,FrameIndex,CenterOfMassX,CenterOfMassY,Velocity,Area,InterfaceFluid,InterfaceSolid,ChordLengthAverage,"
         << "EdgeCountOut,EdgeCountIn,ParentID,BoundingBoxX,BoundingBoxY,BoundingBoxW,BoundingBoxH\n";

    auto inbound = graph.getInboundEdges();

    auto getOr = [](const std::vector<GraphData2D::NodeID>& vec, std::size_t index) {
        return index < vec.size() ? static_cast<int>(vec[index]) : -1;
    };

    for (std::size_t i = 0; i < graph.getNodes().size(); ++i) {
        auto& node = graph.getNodes()[i];
        file << i                                                //
             << "," << node.frameIndex                           //
             << "," << node.centerOfMass.x                       //
             << "," << node.centerOfMass.y                       //
             << "," << node.velocityMagnitude                    //
             << "," << node.area                                 //
             << "," << node.interfaceFluid                       //
             << "," << node.interfaceSolid                       //
             << "," << node.averageChordLength                   //
             << "," << int(node.edgeCountOut)                    //
             << "," << int(node.edgeCountIn)                     //
             << "," << getOr(inbound[i], 0)                      //
             << "," << node.boundingBox.x1                       //
             << "," << node.boundingBox.y1                       //
             << "," << node.boundingBox.x2 - node.boundingBox.x1 //
             << "," << node.boundingBox.y2 - node.boundingBox.y1 //
             << "\n";
    }

    file.close();

    std::filesystem::rename(tempname, outfileName);

    return true;
}

} // namespace megamol::ImageSeries::graph::util
