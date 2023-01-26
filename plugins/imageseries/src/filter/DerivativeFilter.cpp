#include "DerivativeFilter.h"

#include "vislib/graphics/BitmapImage.h"

#include <cstring>

namespace megamol::ImageSeries::filter {

DerivativeFilter::DerivativeFilter(Input input) : input(std::move(input)) {}

DerivativeFilter::DerivativeFilter(AsyncImagePtr image) {
    input.image = std::move(image);
}

DerivativeFilter::ImagePtr DerivativeFilter::operator()() {
    using Image = AsyncImageData2D::BitmapImage;

    // Wait for image data to be ready
    auto image = input.image ? input.image->getImageData() : nullptr;

    // Empty or too small -> return nothing
    if (!image || image->Width() < 3 || image->Height() < 3) {
        return nullptr;
    }

    // TODO: add compatibility with non-byte images
    if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE) {
        return nullptr;
    }

    // Create output image
    auto result = std::make_shared<Image>(image->Width(), image->Height(), 2, Image::ChannelType::CHANNELTYPE_BYTE);

    const auto* dataIn = image->PeekDataAs<std::uint8_t>();
    auto* dataOut = result->PeekDataAs<std::uint8_t>();
    std::size_t width = result->Width();
    std::size_t height = result->Height();

    auto clamp = [](int low, int value, int up) { return std::min(std::max(value, low), up); };
    auto getIndex = [=](std::size_t x, std::size_t y) { return x + y * width; };

    // Compute derivative for each pixel
    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            auto inIndex = getIndex(clamp(1, x, width - 2), clamp(1, y, height - 2));

            // Red channel: horizontal derivative
            *(dataOut++) = clamp(0, (dataIn[inIndex + 1] - dataIn[inIndex - 1] + 255) / 2, 255);

            // Green channel: vertical derivative
            *(dataOut++) = clamp(0, (dataIn[inIndex + width] - dataIn[inIndex - width] + 255) / 2, 255);
        }
    }

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata DerivativeFilter::getMetadata() const {
    if (input.image) {
        ImageMetadata metadata = input.image->getMetadata();
        metadata.channels = 2;
        metadata.hash = util::computeHash(input.image);
        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
