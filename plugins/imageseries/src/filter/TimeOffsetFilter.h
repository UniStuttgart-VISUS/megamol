#ifndef SRC_IMAGESERIES_FILTER_TIMEOFFSETFILTER_HPP_
#define SRC_IMAGESERIES_FILTER_TIMEOFFSETFILTER_HPP_

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace megamol::ImageSeries::filter {

class TimeOffsetFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

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


#endif
