#ifndef IMAGESERIES_SRC_UTIL_GRAPHLUAEXPORTER_HPP_
#define IMAGESERIES_SRC_UTIL_GRAPHLUAEXPORTER_HPP_

#include "imageseries/graph/GraphData2D.h"

#include <string>

namespace megamol::ImageSeries::graph::util {

bool exportToLua(const GraphData2D& graph, const std::string& outfileName);

}

#endif
