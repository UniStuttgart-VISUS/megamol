#include "BlobRegistrator.h"

#include "../filter/AsyncFilterRunner.h"
#include "imageseries/util/ImageUtils.h"

#include <bitset>

namespace megamol::ImageSeries::blob {

BlobRegistrator::Output BlobRegistrator::apply(Input input) {
    using Image = vislib::graphics::BitmapImage;

    Output out;

    // Wait for image data to be ready
    auto image = input.image ? input.image->getImageData() : nullptr;
    auto prev = input.predecessor ? input.predecessor->getImageData() : nullptr;

    // Empty -> return nothing
    if (!image || !prev) {
        return out;
    }

    // Sizes must match
    if (image->Width() != prev->Width() || image->Height() != prev->Height()) {
        return out;
    }

    // TODO: add compatibility with non-byte multichannel images
    if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE ||
        prev->GetChannelCount() != 1 || prev->GetChannelType() != Image::ChannelType::CHANNELTYPE_BYTE) {
        return out;
    }

    // Tracks which unique label links exist between the previous and current images
    std::bitset<65536> linkMask;

    auto addLink = [&](Label preLabel, Label curLabel) {
        if (preLabel < filter::BlobLabelFilter::LabelFirst || curLabel < filter::BlobLabelFilter::LabelFirst) {
            // Ignore links with the background/mask/minimal pixels
            return;
        }

        std::size_t index = preLabel | (curLabel << 8);
        if (!linkMask.test(index)) {
            linkMask.set(index);
            out.links.push_back({preLabel, curLabel});
        }
    };

    const auto* preIn = prev->PeekDataAs<std::uint8_t>();
    const auto* curIn = image->PeekDataAs<std::uint8_t>();
    std::size_t size = image->Width() * image->Height();

    if (input.flowFrontMode) {
        std::size_t width = image->Width();

        // Flow front mode: look for adjacent overlap around flow labels
        for (std::size_t i = 0; i < size; ++i) {
            if (curIn[i] == filter::BlobLabelFilter::LabelFlow) {
                auto preLabel = preIn[i];
                if (i % width < width - 1) {
                    addLink(preLabel, curIn[i + 1]);
                }
                if (i % width > 0) {
                    addLink(preLabel, curIn[i - 1]);
                }
                if (i < size - width) {
                    addLink(preLabel, curIn[i + width]);
                }
                if (i >= width) {
                    addLink(preLabel, curIn[i - width]);
                }
            }
        }
    } else {
        // Basic label mode: look for direct overlap
        for (std::size_t i = 0; i < size; ++i) {
            addLink(preIn[i], curIn[i]);
        }
    }

    return out;
}

} // namespace megamol::ImageSeries::blob
