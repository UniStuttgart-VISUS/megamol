#ifndef IMAGESERIES_SRC_UTIL_GRAPHCSVEXPORTER_HPP_
#define IMAGESERIES_SRC_UTIL_GRAPHCSVEXPORTER_HPP_

#include "imageseries/graph/GraphData2D.h"

#include <string>

namespace megamol::ImageSeries::graph::util {

bool exportToCSV(const GraphData2D& graph, const std::string& outfileName);

}

#endif
