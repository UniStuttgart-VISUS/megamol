#ifndef SRC_FILTER_IMAGELOADFILTER_HPP_
#define SRC_FILTER_IMAGELOADFILTER_HPP_

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace vislib::graphics {
class BitmapCodecCollection;
}

namespace megamol::ImageSeries::filter {

class ImageLoadFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

    struct Input {
        std::shared_ptr<vislib::graphics::BitmapCodecCollection> codecs;
        std::string filename;
        std::size_t sizeEstimate = 0;
    };

    ImageLoadFilter(Input input);
    ImageLoadFilter(std::shared_ptr<vislib::graphics::BitmapCodecCollection> codecs, std::string filename,
        std::size_t sizeEstimate = 0);
    ImagePtr operator()();

    std::size_t getByteSize() const;
    AsyncImageData2D::Hash getHash() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter

#endif
