#ifndef IMAGESERIES_SRC_UTIL_GRAPHLUAEXPORTER_HPP_
#define IMAGESERIES_SRC_UTIL_GRAPHLUAEXPORTER_HPP_

#include "imageseries/graph/GraphData2D.h"

#include <string>

namespace megamol::ImageSeries::graph::util {

struct LuaExportMeta {
    std::string path;
    float minRange = 0.0;
    float maxRange = 1.0;
    int imgW = 0;
    int imgH = 0;
};

bool exportToLua(const GraphData2D& graph, const std::string& outfileName, LuaExportMeta meta);

} // namespace megamol::ImageSeries::graph::util

#endif
