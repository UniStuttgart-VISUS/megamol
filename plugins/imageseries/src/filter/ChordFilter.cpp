#include "ChordFilter.h"

#include "vislib/graphics/BitmapImage.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace megamol::ImageSeries::filter {

ChordFilter::ChordFilter(Input input) : input(std::move(input)) {}

ChordFilter::ChordFilter(AsyncImagePtr image, double threshold, bool clearEdges) {
    input.image = image;
    input.threshold = threshold;
    input.clearEdges = clearEdges;
}

ChordFilter::ImagePtr ChordFilter::operator()() {
    using Image = AsyncImageData2D<>::BitmapImage;

    // Wait for image data to be ready
    auto image = input.image ? input.image->getImageData() : nullptr;

    // Empty or too small -> return nothing
    if (!image || image->Width() < 1 || image->Height() < 1) {
        return nullptr;
    }

    if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE) {
        return nullptr;
    }

    // Create output image with 2 channels (R = horizontal chords, G = vertical chords)
    auto result = std::make_shared<Image>(image->Width(), image->Height(), 2, Image::ChannelType::CHANNELTYPE_WORD);

    const auto* dataIn = image->PeekDataAs<std::uint8_t>();
    auto* dataOut = result->PeekDataAs<std::uint16_t>();
    std::size_t width = result->Width();
    std::size_t height = result->Height();

    std::int32_t threshold = input.threshold * 255;

    // Compute chords along X-axis
    for (std::size_t y = 0; y < height; y++) {
        // Ignore chords touching borders
        std::size_t x = 0;
        for (; x < width; x++) {
            if (dataIn[x + y * width] < threshold) {
                break;
            }
        }

        std::size_t length = 0;
        for (; x < width; x++) {
            if (dataIn[x + y * width] >= threshold) {
                length++;
            } else if (length != 0) {
                for (std::size_t x2 = x - length; x2 < x; x2++) {
                    dataOut[(x2 + y * width) * 2] = length;
                }
                length = 0;
            }
        }
    }

    // Compute chords along Y-axis
    for (std::size_t x = 0; x < width; x++) {
        // Ignore chords touching borders
        std::size_t y = 0;
        for (; y < height; y++) {
            if (dataIn[y + x * width] < threshold) {
                break;
            }
        }

        std::size_t length = 0;
        for (; y < height; y++) {
            if (dataIn[y + x * width] >= threshold) {
                length++;
            } else if (length != 0) {
                for (std::size_t y2 = y - length; y2 < y; y2++) {
                    dataOut[(y2 + x * width) * 2 + 1] = length;
                }
                length = 0;
            }
        }
    }

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata ChordFilter::getMetadata() const {
    if (input.image) {
        ImageMetadata metadata = input.image->getMetadata();
        metadata.bytesPerChannel = 2;
        metadata.hash = util::computeHash(input.image, input.threshold, input.clearEdges);
        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
