#include "SegmentationFilter.h"

#include "vislib/graphics/BitmapImage.h"

namespace megamol::ImageSeries::filter {

SegmentationFilter::SegmentationFilter(Input input) : input(std::move(input)) {}

SegmentationFilter::SegmentationFilter(AsyncImagePtr image, double threshold, bool negateOutput) {
    input.image = image;
    input.threshold = threshold;
    input.negateOutput = negateOutput;
}

SegmentationFilter::ImagePtr SegmentationFilter::operator()() {
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
    std::int32_t threshold = input.threshold * 255;

    // Apply threshold to each pixel
    if (input.negateOutput) {
        for (std::size_t i = 0; i < size; i++) {
            dataOut[i] = dataIn[i] < threshold ? 255 : 0;
        }
    } else {
        for (std::size_t i = 0; i < size; i++) {
            dataOut[i] = dataIn[i] >= threshold ? 255 : 0;
        }
    }

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata SegmentationFilter::getMetadata() const {
    if (input.image) {
        ImageMetadata metadata = input.image->getMetadata();
        metadata.hash = util::computeHash(input.image, input.threshold, input.negateOutput);

        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
