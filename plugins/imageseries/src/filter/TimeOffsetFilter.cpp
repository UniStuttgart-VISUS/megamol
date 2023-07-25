/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "TimeOffsetFilter.h"

#include "vislib/graphics/BitmapImage.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

namespace megamol::ImageSeries::filter {

TimeOffsetFilter::TimeOffsetFilter(Input input) : input(std::move(input)) {}

TimeOffsetFilter::ImagePtr TimeOffsetFilter::operator()() {
    using Image = typename AsyncImageData2D<>::BitmapImage;

    // No frames -> no result.
    if (input.frames.empty()) {
        return nullptr;
    }

    // Normalize frame weights
    float totalWeight = 0;
    float totalCertainty = 0;
    for (const auto& frame : input.frames) {
        totalWeight += std::abs(frame.weight);
        totalCertainty += frame.certainty;
    }

    // Wait for reference data to be ready
    auto reference = input.reference ? input.reference->getImageData() : nullptr;
    if (!reference || reference->Width() == 0 || reference->Height() == 0) {
        return nullptr;
    }

    // TODO: add compatibility with non-byte multichannel images
    if (reference->GetChannelCount() != 1 || reference->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE) {
        return nullptr;
    }

    const auto* dataRefIn = reference->PeekDataAs<std::uint8_t>();

    // Create output image
    auto result =
        std::make_shared<Image>(reference->Width(), reference->Height(), 4, Image::ChannelType::CHANNELTYPE_BYTE);

    std::size_t size = result->Width() * result->Height();
    auto* dataOut = result->PeekDataAs<std::uint8_t>();
    for (std::size_t i = 0; i < size; i++) {
        // Initialize weights to 127 and certainty to 0
        dataOut[i * 4] = 127;
        dataOut[i * 4 + 1] = 0;

        // Add original reference image
        dataOut[i * 4 + 2] = dataRefIn[i];
    }

    for (const auto& frame : input.frames) {
        int weight = 127 * (frame.weight / totalWeight);
        int certainty = 255 * std::max<float>(0.f, frame.certainty / totalCertainty);

        // Skip weightless images
        if (weight == 0 && certainty == 0) {
            continue;
        }

        // Wait for image data to be ready
        auto image = frame.image ? frame.image->getImageData() : nullptr;

        // Skip empty or size-mismatched images
        if (!image || image->Width() != result->Width() || image->Height() != result->Height()) {
            continue;
        }

        // TODO: add compatibility with non-byte multichannel images
        if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE) {
            continue;
        }

        const auto* dataFrameIn = image->PeekDataAs<std::uint8_t>();

        for (std::size_t i = 0; i < size; i++) {
            // Mismatch: add weight. Match: increase certainty.
            bool match = dataRefIn[i] == dataFrameIn[i];
            dataOut[i * 4] += match ? 0 : weight;
            dataOut[i * 4 + 1] += match ? certainty : 0;
        }

        // Copy primary frame to last channel
        if (frame.primary) {
            for (std::size_t i = 0; i < size; i++) {
                dataOut[i * 4 + 3] = dataFrameIn[i];
            }
        }
    }

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata TimeOffsetFilter::getMetadata() const {
    if (input.reference) {
        ImageMetadata metadata = input.reference->getMetadata();
        metadata.channels = 4;
        metadata.hash = util::computeHash(input.reference);
        for (auto& frame : input.frames) {
            metadata.hash = util::computeHash(metadata.hash, frame.image, frame.weight, frame.certainty, frame.primary);
        }
        return metadata;
    } else {
        return {};
    }
}


} // namespace megamol::ImageSeries::filter
