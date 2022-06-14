#include "MaskFilter.h"

#include "vislib/graphics/BitmapImage.h"

namespace megamol::ImageSeries::filter {

MaskFilter::MaskFilter(Input input) : input(std::move(input)) {}

MaskFilter::MaskFilter(AsyncImagePtr image, AsyncImagePtr mask) {
    input.image = image;
    input.mask = mask;
}

MaskFilter::ImagePtr MaskFilter::operator()() {
    using Image = AsyncImageData2D::BitmapImage;

    // Wait for image data to be ready
    auto image = input.image ? input.image->getImageData() : nullptr;
    auto mask = input.mask ? input.mask->getImageData() : nullptr;

    // Empty -> return nothing
    if (!image || !mask) {
        return nullptr;
    }

    // Size must match
    if (image->Width() != mask->Width() || image->Height() != mask->Height()) {
        return nullptr;
    }

    // TODO: add compatibility with non-byte multichannel images
    if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE ||
        mask->GetChannelCount() != 1 || mask->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE) {
        return nullptr;
    }

    // Create output image
    auto result = std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_BYTE);

    const auto* dataIn = image->PeekDataAs<std::uint8_t>();
    const auto* maskIn = mask->PeekDataAs<std::uint8_t>();
    auto* dataOut = result->PeekDataAs<std::uint8_t>();
    std::size_t size = result->Width() * result->Height();

    // Apply mask to each pixel
    for (std::size_t i = 0; i < size; i++) {
        dataOut[i] = std::max(dataIn[i], maskIn[i]);
    }

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata MaskFilter::getMetadata() const {
    if (input.image) {
        ImageMetadata metadata = input.image->getMetadata();
        metadata.hash = util::computeHash(input.image, input.mask);
        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
