#ifndef SRC_IMAGESERIES_FILTER_DEINTERLACEFILTER_HPP_
#define SRC_IMAGESERIES_FILTER_DEINTERLACEFILTER_HPP_

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace megamol::ImageSeries::filter {

class DeinterlaceFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

    struct Input {
        AsyncImagePtr image;
        int offset = 0;
    };

    DeinterlaceFilter(Input input);
    DeinterlaceFilter(AsyncImagePtr image, int offset = 0);
    ImagePtr operator()();

    std::size_t getByteSize() const;
    AsyncImageData2D::Hash getHash() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter


#endif
