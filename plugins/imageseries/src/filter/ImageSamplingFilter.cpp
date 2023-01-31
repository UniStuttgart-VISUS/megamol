#include "ImageSamplingFilter.h"

#include "vislib/graphics/BitmapImage.h"

#include <limits>
#include <unordered_map>

namespace megamol::ImageSeries::filter {

ImageSamplingFilter::ImageSamplingFilter(Input input) : input(std::move(input)) {}

ImageSamplingFilter::ImageSamplingFilter(AsyncImagePtr indexMap) : input(Input{indexMap}) {}

ImageSamplingFilter::ImagePtr ImageSamplingFilter::operator()() {
    using Image = AsyncImageData2D::BitmapImage;

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

    const auto width = result->Width();
    const auto height = result->Height();
    const auto size = static_cast<std::size_t>(width) * height;

    const auto get_index = [&width](const unsigned int x, const unsigned int y) { return y * width + x; };

    const auto* mapIn = map->PeekDataAs<std::uint16_t>();
    auto* mapOut = result->PeekDataAs<std::uint16_t>();
    auto* mapTemp = temp->PeekDataAs<std::uint16_t>();

    std::memcpy(mapTemp, mapIn, size * sizeof(std::uint16_t));

    auto num_iterations = 0;

    do {
        for (std::size_t index = 0; index < size; ++index) {
            auto value = mapTemp[index];

            const auto x = index % width;
            const auto y = index / width;

            bool border = x == 0 || y == 0 || static_cast<int>(x) == width - 1 || static_cast<int>(y) == height - 1;

            if (!border) {
                std::unordered_map<unsigned int, unsigned int> num_neighbors;
                auto num_same_neighbors = -1;

                for (int j = -1; j <= 1; ++j) {
                    for (int i = -1; i <= 1; ++i) {
                        const auto neighbor_value = mapTemp[get_index(x + i, y + j)];

                        if (neighbor_value != value) {
                            if (num_neighbors.find(neighbor_value) == num_neighbors.end()) {
                                num_neighbors[neighbor_value] = 1;
                            } else {
                                ++num_neighbors[neighbor_value];
                            }
                        } else {
                            ++num_same_neighbors;
                        }
                    }
                }

                if (num_same_neighbors < 3) { // TODO: reason?
                    auto best_neighbor = value;
                    auto best_neighbor_num = 0u;

                    for (auto it = num_neighbors.cbegin(); it != num_neighbors.cend(); ++it) {
                        if (it->second > best_neighbor_num) {
                            best_neighbor_num = it->second;
                            best_neighbor = it->first;
                        }
                    }

                    value = best_neighbor;
                }
            }

            mapOut[index] = value;
        }

        std::swap(mapOut, mapTemp);
    } while (++num_iterations < 2); // just get rid of the fine-grained noise

    if (num_iterations % 2 == 0) {
        std::memcpy(mapOut, mapTemp, size * sizeof(std::uint16_t));
    }

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata ImageSamplingFilter::getMetadata() const {
    if (input.indexMap) {
        ImageMetadata metadata = input.indexMap->getMetadata();
        metadata.bytesPerChannel = 2;
        metadata.hash = util::computeHash(input.indexMap);
        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
