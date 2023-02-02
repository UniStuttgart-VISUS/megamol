#pragma once

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace vislib::graphics {
class BitmapCodecCollection;
}

namespace megamol::ImageSeries::filter {

class ImageLoadFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D<>>;
    using ImagePtr = std::shared_ptr<const typename AsyncImageData2D<>::BitmapImage>;

    struct Input {
        std::shared_ptr<vislib::graphics::BitmapCodecCollection> codecs;
        std::string filename;
        ImageMetadata metadata;
    };

    ImageLoadFilter(Input input);
    ImageLoadFilter(
        std::shared_ptr<vislib::graphics::BitmapCodecCollection> codecs, std::string filename, ImageMetadata metadata);
    ImagePtr operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter
