/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "imageseries/AsyncImageData2D.h"

#include <memory>
#include <vector>

namespace megamol::ImageSeries::filter {

class TimeOffsetFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D<>>;
    using ImagePtr = std::shared_ptr<const typename AsyncImageData2D<>::BitmapImage>;

    struct Input {
        struct Frame {
            AsyncImagePtr image;
            float weight = 1.f;
            float certainty = 1.f;
            bool primary = false;
        };
        std::vector<Frame> frames;
        AsyncImagePtr reference;
    };

    TimeOffsetFilter(Input input);
    ImagePtr operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter
