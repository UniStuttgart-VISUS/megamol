#pragma once

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace megamol::ImageSeries::filter {

class GenericFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

    enum class Operation {
        Difference,
    };

    struct Input {
        AsyncImagePtr image1;
        AsyncImagePtr image2;
        Operation operation = Operation::Difference;
    };

    GenericFilter(Input input);
    GenericFilter(AsyncImagePtr image1, AsyncImagePtr image2, Operation op);
    ImagePtr operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter
