/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "imageseries/graph/GraphData2D.h"

#include <string>

namespace megamol::ImageSeries::graph::util {

struct GDFExportMeta {
    int startTime = 0;
    int breakthroughTime = 0;
    int endTime = 0;

    bool stopAtBreakthrough = false;
};

bool exportToGDF(const GraphData2D& graph, const std::string& outfileName, GDFExportMeta meta);

} // namespace megamol::ImageSeries::graph::util
