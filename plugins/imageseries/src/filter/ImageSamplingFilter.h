#pragma once

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace megamol::ImageSeries::filter {

class ImageSamplingFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

    struct Input {
        AsyncImagePtr indexMap;
    };

    ImageSamplingFilter(Input input);
    ImageSamplingFilter(AsyncImagePtr indexMap);
    ImagePtr operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter
