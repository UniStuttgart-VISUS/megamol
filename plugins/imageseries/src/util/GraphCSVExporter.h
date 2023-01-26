#pragma once

#include "imageseries/graph/GraphData2D.h"

#include <string>

namespace megamol::ImageSeries::graph::util {

bool exportToCSV(const GraphData2D& graph, const std::string& outfileName);

}
