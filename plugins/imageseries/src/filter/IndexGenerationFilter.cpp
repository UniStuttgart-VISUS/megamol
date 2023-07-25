/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "IndexGenerationFilter.h"

#include "vislib/graphics/BitmapImage.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

namespace megamol::ImageSeries::filter {

IndexGenerationFilter::IndexGenerationFilter(Input input) : input(std::move(input)) {}

IndexGenerationFilter::IndexGenerationFilter(AsyncImagePtr image, AsyncImagePtr indexMap, std::size_t frameIndex) {
    input.image = image;
    input.indexMap = indexMap;
    input.frameIndex = frameIndex;
}

IndexGenerationFilter::ImagePtr IndexGenerationFilter::operator()() {
    using Image = typename AsyncImageData2D<>::BitmapImage;

    // Wait for image data to be ready.
    auto map = input.indexMap ? input.indexMap->getImageData() : nullptr;
    auto image = input.image ? input.image->getImageData() : nullptr;

    // Empty -> return nothing
    if (!image) {
        return nullptr;
    }

    // Size must match
    if (map && (image->Width() != map->Width() || image->Height() != map->Height())) {
        return nullptr;
    }

    // TODO: add compatibility with non-byte multichannel images
    if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE ||
        (map && (map->GetChannelCount() != 1 || map->GetChannelType() != Image::ChannelType::CHANNELTYPE_WORD))) {
        return nullptr;
    }

    util::PerfTimer timer("IndexGenerationFilter", input.image->getMetadata().filename);

    // Create output image
    auto result = std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);

    const auto* imageIn = image->PeekDataAs<std::uint8_t>();
    auto* mapOut = result->PeekDataAs<std::uint16_t>();
    std::size_t size = result->Width() * result->Height();

    if (map) {
        // Map image given: use min value
        const auto* mapIn = map->PeekDataAs<std::uint16_t>();
        for (std::size_t i = 0; i < size; i++) {
            mapOut[i] = !imageIn[i] ? std::min<std::uint16_t>(input.frameIndex, mapIn[i]) : mapIn[i];
        }
    } else {
        // No input map: assume int16_max for existing map
        std::uint16_t value = std::numeric_limits<std::uint16_t>::max();
        for (std::size_t i = 0; i < size; i++) {
            mapOut[i] = !imageIn[i] ? input.frameIndex : value;
        }
    }

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata IndexGenerationFilter::getMetadata() const {
    if (input.image) {
        ImageMetadata metadata = input.image->getMetadata();
        metadata.bytesPerChannel = 2;
        metadata.hash = util::computeHash(input.image, input.indexMap, input.frameIndex);
        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
