#include "DeinterlaceFilter.h"

#include "vislib/graphics/BitmapImage.h"

namespace megamol::ImageSeries::filter {

DeinterlaceFilter::DeinterlaceFilter(Input input) : input(std::move(input)) {}

DeinterlaceFilter::DeinterlaceFilter(AsyncImagePtr image, int offset) {
    input.image = image;
    input.offset = offset;
}

DeinterlaceFilter::ImagePtr DeinterlaceFilter::operator()() {
    using Image = AsyncImageData2D::BitmapImage;

    // Wait for image data to be ready
    auto image = input.image ? input.image->getImageData() : nullptr;

    // Empty -> return nothing
    if (!image) {
        return nullptr;
    }

    // TODO: add compatibility with non-byte multichannel images
    if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE) {
        return nullptr;
    }

    // Create output image
    auto result = std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_BYTE);

    const auto* dataIn = image->PeekDataAs<std::uint8_t>();
    auto* dataOut = result->PeekDataAs<std::uint8_t>();
    std::size_t size = result->Width() * result->Height();
    int width = result->Width();
    int odd = input.offset / 2;
    int even = odd - input.offset;

    // Apply offset to each pixel
    for (std::size_t i = 0; i < size; i++) {
        int x = i % width;
        int offset = (i / width) % 2 == 0 ? even : odd;
        int shiftedX = std::max<int>(0, std::min<int>(x + offset, width - 1));
        std::size_t shiftedIndex = i + (shiftedX - x);
        if (shiftedIndex < size) {
            dataOut[i] = dataIn[shiftedIndex];
        }
    }

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata DeinterlaceFilter::getMetadata() const {
    if (input.image) {
        ImageMetadata metadata = input.image->getMetadata();
        metadata.hash = util::computeHash(input.image, input.offset);
        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
