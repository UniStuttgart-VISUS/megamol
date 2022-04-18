#include "BlobAnalyzer.h"

#include "../filter/AsyncFilterRunner.h"
#include "../util/ImageUtils.h"

#include <bitset>

namespace megamol::ImageSeries::blob {

BlobAnalyzer::Output BlobAnalyzer::apply(Input input) {
    using Image = vislib::graphics::BitmapImage;

    Output out;

    // Wait for image data to be ready
    auto image = input.image ? input.image->getImageData() : nullptr;
    auto label = input.labels ? input.labels->getImageData() : nullptr;

    // Empty -> return nothing
    if (!image || !label) {
        return out;
    }

    // Sizes must match
    if (image->Width() != label->Width() || image->Height() != label->Height()) {
        return out;
    }

    // TODO: add compatibility with non-byte multichannel images
    if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE ||
        label->GetChannelCount() != 1 || label->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE) {
        return out;
    }

    std::array<Blob, filter::BlobLabelFilter::LabelCount> blobArray;

    const auto* imgIn = image->PeekDataAs<std::uint8_t>();
    const auto* labIn = label->PeekDataAs<std::uint8_t>();
    std::size_t width = image->Width();
    std::size_t height = image->Height();
    std::size_t size = width * height;

    for (std::size_t i = 0; i < size; ++i) {
        int x = i % width;
        int y = i / width;
        Label label = labIn[i];
        int value = imgIn[i];

        Blob& blob = blobArray[label];

        if (blob.pixelCount == 0) {
            blob.boundingBox.x1 = x;
            blob.boundingBox.y1 = y;
            blob.boundingBox.x2 = x;
            blob.boundingBox.y2 = y;
        } else {
            blob.boundingBox.x1 = std::min<int>(blob.boundingBox.x1, x);
            blob.boundingBox.y1 = std::min<int>(blob.boundingBox.y1, y);
            blob.boundingBox.x2 = std::max<int>(blob.boundingBox.x2, x);
            blob.boundingBox.y2 = std::max<int>(blob.boundingBox.y2, y);
        }

        blob.pixelCount++;
        blob.valueSum += value;
        blob.centerOfMass.x += x;
        blob.centerOfMass.y += y;
        blob.weightedCenterOfMass.x += x * value;
        blob.weightedCenterOfMass.y += y * value;
    }

    for (std::size_t i = 0; i < filter::BlobLabelFilter::LabelCount; ++i) {
        auto& blob = blobArray[i];
        if (blob.pixelCount > 0) {
            blob.label = i;
            blob.valueMean = static_cast<double>(blob.valueSum) / (blob.pixelCount);
            blob.centerOfMass /= static_cast<float>(blob.pixelCount);
            if (blob.valueSum > 0) {
                blob.weightedCenterOfMass /= static_cast<float>(blob.valueSum);
            } else {
                blob.weightedCenterOfMass = blob.centerOfMass;
            }
            out.blobs.push_back(blob);
        }
    }

    return out;
}

} // namespace megamol::ImageSeries::blob
