#include "BlobLabelFilter.h"

#include "vislib/graphics/BitmapImage.h"

#include <array>
#include <deque>
#include <vector>

namespace megamol::ImageSeries::filter {

BlobLabelFilter::BlobLabelFilter(Input input) : input(std::move(input)) {}

BlobLabelFilter::ImagePtr BlobLabelFilter::operator()() {
    using Image = AsyncImageData2D::BitmapImage;

    // Wait for image data to be ready
    auto image = input.image ? input.image->getImageData() : nullptr;
    auto mask = input.mask ? input.mask->getImageData() : nullptr;
    auto prev = input.prevImage ? input.prevImage->getImageData() : nullptr;

    // Empty -> return nothing
    if (!image) {
        return nullptr;
    }

    // TODO: add compatibility with non-byte multichannel images
    if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE) {
        return nullptr;
    }

    // Mask must have matching size and channels, otherwise, it is ignored
    if (mask && (image->Width() != mask->Width() || image->Height() != mask->Height() || mask->GetChannelCount() != 1 ||
                    mask->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE)) {
        mask = nullptr;
    }

    // Prev image must have matching size and channels, otherwise, it is ignored
    if (prev && (image->Width() != prev->Width() || image->Height() != prev->Height() || prev->GetChannelCount() != 1 ||
                    prev->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE)) {
        prev = nullptr;
    }

    // Create output image
    auto result = std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_BYTE);

    using Index = std::uint32_t;

    const auto* dataIn = image->PeekDataAs<std::uint8_t>();
    auto* dataOut = result->PeekDataAs<std::uint8_t>();
    const auto* prevIn = prev ? prev->PeekDataAs<std::uint8_t>() : nullptr;
    Index width = result->Width();
    Index height = result->Height();
    Index size = width * height;
    std::int32_t threshold = input.threshold * 255;

    // Apply mask, if available
    if (mask) {
        // TODO separate mask threshold/negation option?
        const auto* maskIn = mask->PeekDataAs<std::uint8_t>();
        if (input.maskPriority) {
            // Mask has priority: override pixels unconditionally
            for (Index i = 0; i < size; ++i) {
                dataOut[i] = (maskIn[i] < threshold) != input.negateMask ? LabelMask : LabelBackground;
            }
        } else {
            // Mask does not have priority: only override pixels that don't meet the threshold
            for (Index i = 0; i < size; ++i) {
                dataOut[i] =
                    (maskIn[i] < threshold) != input.negateMask && (dataIn[i] < threshold) == input.negateThreshold
                        ? LabelMask
                        : LabelBackground;
            }
        }
    }

    Label nextLabel = LabelFirst;
    Label labelLimit = LabelFirst + input.blobCountLimit - 1;

    // TODO: Track number of pixels per blob
    //std::array<Index, LabelCount> blobSizes;
    //blobSizes.fill(0);

    std::vector<Index> pendingPixels;
    std::vector<Index> pendingFlow;
    std::deque<Index> queue;

    // Reserve enough space to perform first pass
    pendingPixels.reserve(input.minBlobSize + 4);

    auto testPixel = [&](Index index) {
        return dataOut[index] == LabelBackground && (dataIn[index] < threshold) != input.negateThreshold;
    };

    auto testPrev = [&](Index index) { return prevIn && (prevIn[index] < threshold) != input.negateThreshold; };

    auto markMinimal = [&](Index index) {
        if (testPixel(index)) {
            if (testPrev(index)) {
                pendingFlow.push_back(index);
            } else {
                dataOut[index] = LabelMinimal;
                pendingPixels.push_back(index);
            }
        }
    };

    auto fillMinimal = [&](Index startIndex) {
        pendingPixels.clear();
        pendingFlow.clear();
        markMinimal(startIndex);
        Index queueIndex = 0;

        // Try to collect at least MinBlobSize connected pixels
        while (queueIndex < pendingPixels.size() && pendingPixels.size() < input.minBlobSize) {
            Index index = pendingPixels[queueIndex++];

            // Pixel is not on the right boundary
            if (index % width < width - 1) {
                markMinimal(index + 1);
            }
            // Pixel is not on the left boundary
            if (index % width > 0) {
                markMinimal(index - 1);
            }
            // Pixel is not on the bottom boundary
            if (index < size - width) {
                markMinimal(index + width);
            }
            // Pixel is not on the top boundary
            if (index >= width) {
                markMinimal(index - width);
            }
        }

        // Minimal stage complete: return true if minimum blob size condition was met
        return pendingPixels.size() >= input.minBlobSize;
    };

    auto mark = [&](Index index) {
        if (testPixel(index)) {
            if (testPrev(index)) {
                dataOut[index] = LabelFlow;
            } else {
                dataOut[index] = nextLabel;
                queue.push_back(index);
            }
        }
    };

    auto fill = [&](Index startIndex) {
        // Perform minimal first pass, checking if the blob is large enough
        if (!fillMinimal(startIndex)) {
            // Small blob: avoid assigning unique label ID
            return;
        }

        queue.clear();

        // Large blob: fill out all pixels with proper label ID
        for (auto& index : pendingPixels) {
            dataOut[index] = nextLabel;
            queue.push_back(index);
        }

        for (auto& index : pendingFlow) {
            dataOut[index] = LabelFlow;
        }

        // Perform proper flood fill
        while (!queue.empty()) {
            auto index = queue.front();
            queue.pop_front();

            // Pixel is not on the right boundary
            if (index % width < width - 1) {
                mark(index + 1);
            }
            // Pixel is not on the left boundary
            if (index % width > 0) {
                mark(index - 1);
            }
            // Pixel is not on the bottom boundary
            if (index < size - width) {
                mark(index + width);
            }
            // Pixel is not on the top boundary
            if (index >= width) {
                mark(index - width);
            }
        }

        // Check if label limit has been reached
        if (nextLabel == labelLimit) {
            // TODO clear smallest blob
            return;
        }

        // Increment label index
        nextLabel++;
    };

    // Perform flood fill
    for (Index i = 0; i < size; ++i) {
        // Found a possible seed point: start flood filling!
        if (testPixel(i) && !testPrev(i)) {
            fill(i);
        }
    }

    return std::const_pointer_cast<const Image>(result);
}

std::size_t BlobLabelFilter::getByteSize() const {
    return input.image ? input.image->getByteSize() : 0;
}

} // namespace megamol::ImageSeries::filter
