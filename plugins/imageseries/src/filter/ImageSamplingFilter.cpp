#include "ImageSamplingFilter.h"

#include "vislib/graphics/BitmapImage.h"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace megamol::ImageSeries::filter {

ImageSamplingFilter::ImageSamplingFilter(Input input) : input(std::move(input)) {}

ImageSamplingFilter::ImagePtr ImageSamplingFilter::operator()() {
    using Image = typename AsyncImageData2D<>::BitmapImage;

    // Wait for image data to be ready.
    auto map = input.indexMap ? input.indexMap->getImageData() : nullptr;

    // Empty -> return nothing
    if (!map) {
        return nullptr;
    }

    // TODO: add compatibility with non-byte multichannel images
    if (map && (map->GetChannelCount() != 1 || map->GetChannelType() != Image::ChannelType::CHANNELTYPE_WORD)) {
        return nullptr;
    }

    util::PerfTimer timer("ImageSamplingFilter", "-");

    // Create output image
    auto result = std::make_shared<Image>(map->Width(), map->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);
    auto temp = std::make_shared<Image>(map->Width(), map->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);

    auto size_result = std::make_shared<Image>(map->Width(), map->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);

    const auto width = result->Width();
    const auto height = result->Height();
    const auto size = static_cast<std::size_t>(width) * height;

    const auto get_index = [&width](const unsigned int x, const unsigned int y) { return y * width + x; };

    const auto* mapIn = map->PeekDataAs<std::uint16_t>();
    auto* mapOut = result->PeekDataAs<std::uint16_t>();
    auto* mapTemp = temp->PeekDataAs<std::uint16_t>();
    auto* sizeOut = size_result->PeekDataAs<std::uint16_t>();

    std::memcpy(mapTemp, mapIn, size * sizeof(std::uint16_t));

    constexpr std::uint16_t LabelSolid = 0;
    constexpr std::uint16_t LabelUnassigned = std::numeric_limits<std::uint16_t>::max() - 1;
    constexpr std::uint16_t LabelEmpty = std::numeric_limits<std::uint16_t>::max();

    uint16_t max = 0;

    for (std::size_t index = 0; index < size; ++index) {
        if (mapIn[index] == LabelSolid || mapIn[index] == LabelEmpty) {
            mapOut[index] = mapIn[index];
        } else {
            max = std::max(max, mapIn[index]);
            mapOut[index] = LabelUnassigned;
        }

        sizeOut[index] = 1;
    }

    struct area_t {
        std::vector<std::size_t> pixels;
        std::vector<std::size_t> neighbors;
    };

    std::unordered_map<std::size_t, area_t> areas;

    auto floodFill = [&](const std::size_t index) {
        areas[index].pixels.push_back(index);
        mapOut[index] = mapTemp[index];

        for (std::size_t queueIndex = 0; queueIndex < areas.at(index).pixels.size(); ++queueIndex) {
            const auto currentIndex = areas.at(index).pixels[queueIndex];

            const auto x = currentIndex % width;
            const auto y = currentIndex / width;

            // TODO: face-connected vs. kernel (now: kernel)
            for (auto j = ((y > 0) ? -1 : 0); j <= ((y < height - 1) ? 1 : 0); ++j) {
                for (auto i = ((x > 0) ? -1 : 0); i <= ((x < width - 1) ? 1 : 0); ++i) {
                    const auto neighborIndex = (y + j) * width + (x + i);

                    if (mapOut[neighborIndex] == LabelUnassigned && mapTemp[index] == mapTemp[neighborIndex] &&
                        mapTemp[neighborIndex] != LabelSolid && mapTemp[neighborIndex] != LabelEmpty) {

                        areas.at(index).pixels.push_back(neighborIndex);
                        mapOut[neighborIndex] = mapTemp[index];
                    } else if (mapTemp[index] != mapTemp[neighborIndex] && mapTemp[neighborIndex] != LabelSolid &&
                               mapTemp[neighborIndex] != LabelEmpty) {

                        areas.at(index).neighbors.push_back(neighborIndex);
                    }
                }
            }
        }

        for (auto currentIndex : areas.at(index).pixels) {
            sizeOut[currentIndex] = areas.at(index).pixels.size();
        }
    };

    for (std::size_t index = 0; index < size; ++index) {
        if (mapOut[index] == LabelUnassigned) {
            floodFill(index);
        }
    }

    std::size_t iteration = 1;

    while (!areas.empty() && iteration <= input.iterations) {
        // Only keep small areas
        for (auto it = areas.cbegin(); it != areas.cend();) {
            if (it->second.pixels.size() < input.neighborThreshold) {
                ++it;
            } else {
                it = areas.erase(it);
            }
        }

        // Adjust time information of small areas
        const std::size_t exponent = std::pow(2, iteration - 1);

        for (const auto& area : areas) {
            const auto representative = area.first;

            const auto range = exponent * 2;
            const auto mod = mapOut[representative] % range;

            std::uint16_t neighbor_min = std::numeric_limits<std::uint16_t>::max();
            std::uint16_t neighbor_max = std::numeric_limits<std::uint16_t>::lowest();

            for (const auto& neighbor : area.second.neighbors) {
                if (sizeOut[neighbor] >= input.neighborThreshold && mapTemp[neighbor] != LabelSolid &&
                    mapTemp[neighbor] != LabelEmpty) {
                    neighbor_min = std::min(neighbor_min, mapTemp[neighbor]);
                    neighbor_max = std::max(neighbor_max, mapTemp[neighbor]);
                }
            }

            if (neighbor_min == std::numeric_limits<std::uint16_t>::max()) {
                neighbor_min = std::numeric_limits<std::uint16_t>::lowest();
            }
            if (neighbor_max == std::numeric_limits<std::uint16_t>::lowest()) {
                neighbor_max = std::numeric_limits<std::uint16_t>::max();
            }

            std::uint16_t newValue = mapOut[representative] + (-mod)/*((mod >= exponent) ? (range - mod) : (-mod))*/;
            newValue = std::max(neighbor_min, newValue);
            newValue = std::min(neighbor_max, newValue);

            if (newValue == 0) {
                newValue = LabelEmpty;
            } else if (newValue > max) {
                newValue = max;
            }

            for (const auto pixel : area.second.pixels) {
                mapOut[pixel] = newValue;
            }
        }

        // Perform flood fill on adjusted areas only
        std::unordered_map<std::size_t, area_t> adjustedAreas;
        std::swap(areas, adjustedAreas);

        std::swap(mapTemp, mapOut);
        std::for_each(mapOut, mapOut + size, [&LabelUnassigned](auto& value) { value = LabelUnassigned; });

        for (const auto& pixel : adjustedAreas) {
            const auto index = pixel.first;

            if (mapOut[index] == LabelUnassigned) {
                floodFill(index);
            }
        }

        for (std::size_t index = 0; index < size; ++index) {
            if (mapOut[index] == LabelUnassigned) {
                mapOut[index] = mapTemp[index];
            }
        }

        ++iteration;
    }

    std::memcpy(result->PeekDataAs<std::uint16_t>(), mapTemp, size * sizeof(std::uint16_t));

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata ImageSamplingFilter::getMetadata() const {
    if (input.indexMap) {
        ImageMetadata metadata = input.indexMap->getMetadata();
        metadata.bytesPerChannel = 2;
        metadata.hash = util::computeHash(input.indexMap, input.iterations, input.neighborThreshold);
        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
