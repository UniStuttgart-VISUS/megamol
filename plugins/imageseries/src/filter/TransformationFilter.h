/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "imageseries/AsyncImageData2D.h"

#include <glm/mat3x2.hpp>

#include <memory>

namespace megamol::ImageSeries::filter {

class TransformationFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D<>>;
    using ImagePtr = std::shared_ptr<const typename AsyncImageData2D<>::BitmapImage>;

    struct Input {
        AsyncImagePtr image;
        glm::mat3x2 transform;
    };

    TransformationFilter(Input input);
    TransformationFilter(AsyncImagePtr image, glm::mat3x2 transform);
    ImagePtr operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter
