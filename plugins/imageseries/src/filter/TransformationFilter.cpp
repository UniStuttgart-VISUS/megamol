#include "TransformationFilter.h"

#include "vislib/graphics/BitmapImage.h"

#include "glm/glm.hpp"

#include <cmath>

namespace megamol::ImageSeries::filter {

TransformationFilter::TransformationFilter(Input input) : input(std::move(input)) {}

TransformationFilter::TransformationFilter(AsyncImagePtr image, glm::mat3x2 transform) {
    input.image = image;
    input.transform = transform;
}

TransformationFilter::ImagePtr TransformationFilter::operator()() {
    using Image = typename AsyncImageData2D<>::BitmapImage;

    // Wait for image data to be ready
    auto image = input.image ? input.image->getImageData() : nullptr;

    // Empty -> return nothing
    if (!image) {
        return nullptr;
    }

    // TODO: add compatibility with non-byte / multichannel images
    if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE) {
        return nullptr;
    }

    // Create output image
    auto result = std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_BYTE);

    const auto* dataIn = image->PeekDataAs<std::uint8_t>();
    auto* dataOut = result->PeekDataAs<std::uint8_t>();

    std::size_t width = image->Width();
    std::size_t height = image->Height();

    auto clamp = [](int low, int value, int up) { return std::min(std::max(value, low), up); };
    auto getIndex = [=](int x, int y) { return x + y * width; };
    auto getPixel = [=](int x, int y) {
        return dataIn[getIndex(clamp(0, x, width - 1), clamp(0, y, height - 1))] / 255.f;
    };
    auto setPixel = [=](int x, int y, float value) { return dataOut[getIndex(x, y)] = clamp(0, value * 255, 255); };
    auto bilerp = [=](float x, float y) {
        float xi, yi;
        float xf = std::modf(x, &xi), yf = std::modf(y, &yi);
        return getPixel(xi, yi) * (1 - xf) * (1 - yf) + getPixel(xi + 1, yi) * xf * (1 - yf) +
               getPixel(xi, yi + 1) * (1 - xf) * yf + getPixel(xi + 1, yi + 1) * xf * yf;
    };

    // Transform each pixel according to the supplied matrix
    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            auto transformed = input.transform * glm::vec3(x, y, 1);
            setPixel(x, y, bilerp(transformed.x, transformed.y));
        }
    }

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata TransformationFilter::getMetadata() const {
    if (input.image) {
        ImageMetadata metadata = input.image->getMetadata();
        metadata.hash = util::computeHash(input.image, util::hashBytes(&input.transform, sizeof(input.transform)));
        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
