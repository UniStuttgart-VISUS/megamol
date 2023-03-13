#pragma once

#include "imageseries/graph/GraphData2D.h"

#include <string>

namespace megamol::ImageSeries::graph::util {

struct LuaExportMeta {
    std::string path;

    float minRange = 0.0;
    float maxRange = 1.0;

    int imgW = 0;
    int imgH = 0;

    int startTime = 0;
    int breakthroughTime = 0;
    int endTime = 0;
};

bool exportToLua(const GraphData2D& graph, const std::string& outfileName, LuaExportMeta meta,
    const std::vector<std::size_t>& velocityDistribution);

} // namespace megamol::ImageSeries::graph::util
