/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "ImageWrapper.h"

namespace megamol::frontend_resources {

struct ImagePresentationSink {
    std::string name = "";
    std::function<void(std::vector<ImageWrapper> const&)> present_images;
};

} // namespace megamol::frontend_resources
