#ifndef SRC_IMAGESERIES_FILTER_DERIVATIVEFILTER_HPP_
#define SRC_IMAGESERIES_FILTER_DERIVATIVEFILTER_HPP_

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace megamol::ImageSeries::filter {

class DerivativeFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

    struct Input {
        AsyncImagePtr image;
    };

    DerivativeFilter(Input input);
    DerivativeFilter(AsyncImagePtr image);
    ImagePtr operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter


#endif
