#ifndef SRC_IMAGESERIES_FILTER_SEGMENTATIONFILTER_HPP_
#define SRC_IMAGESERIES_FILTER_SEGMENTATIONFILTER_HPP_

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace megamol::ImageSeries::filter {

class SegmentationFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

    struct Input {
        AsyncImagePtr image;
        double threshold = 0.5;
        bool negateOutput = false;
    };

    SegmentationFilter(Input input);
    SegmentationFilter(AsyncImagePtr image, double threshold = 0.5, bool negateOutput = false);
    ImagePtr operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter


#endif
