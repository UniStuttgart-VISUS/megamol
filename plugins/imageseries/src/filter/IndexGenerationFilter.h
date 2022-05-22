#ifndef SRC_IMAGESERIES_FILTER_INDEXGENERATIONFILTER_HPP_
#define SRC_IMAGESERIES_FILTER_INDEXGENERATIONFILTER_HPP_

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace megamol::ImageSeries::filter {

class IndexGenerationFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

    struct Input {
        AsyncImagePtr image;
        AsyncImagePtr indexMap;
        std::uint16_t frameIndex = 0;
    };

    IndexGenerationFilter(Input input);
    IndexGenerationFilter(AsyncImagePtr image, AsyncImagePtr indexMap, std::size_t frameIndex);
    ImagePtr operator()();

    std::size_t getByteSize() const;
    AsyncImageData2D::Hash getHash() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter


#endif
