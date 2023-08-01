/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace megamol::ImageSeries::filter {

class ImageSamplingFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D<>>;
    using ImagePtr = std::shared_ptr<const typename AsyncImageData2D<>::BitmapImage>;

    struct Input {
        AsyncImagePtr indexMap;
        std::size_t iterations;
        std::size_t neighborThreshold;
    };

    ImageSamplingFilter(Input input);
    ImagePtr operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter
