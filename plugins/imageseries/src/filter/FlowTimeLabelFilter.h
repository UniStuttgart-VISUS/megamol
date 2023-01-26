#pragma once

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace megamol::ImageSeries::filter {

class FlowTimeLabelFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;

    using Label = std::uint8_t;

    // Special labels
    enum LabelType : Label {
        // Background pixel (no blob at this location)
        LabelBackground = 0,

        // Foreground mask pixel (wall/obstacle at this location)
        LabelMask = 1,

        // Indicates that this pixel is part of a small blob that fell below the size threshold
        LabelMinimal = 2,

        // Interface pixel between temporally adjacent frames
        LabelFlow = 3,

        // Minimum auto-assigned ID for pixels belonging to a cohesive blob
        LabelFirst = 4,

        // Maximum auto-assigned ID for pixels belonging to a cohesive blob
        LabelLast = 255,
    };

    static constexpr int LabelCount = 256;

    struct Input {
        // Timestamp map within which to search for connected blobs.
        AsyncImagePtr timeMap;

        // Maximum number of blobs to track (NYI)
        std::size_t blobCountLimit = 250;

        // Minimum number of pixels required for a blob to be tracked
        std::size_t minBlobSize = 50;

        // Maximum number of frames by which a pixel may differ from its neighbor.
        std::size_t timeThreshold = 20;

        // Maximum number of frames by which a pixel may differ from its neighbor.
        std::size_t minimumTimestamp = 10;

        // Maximum number of frames by which a pixel may differ from its neighbor.
        std::size_t maximumTimestamp = 65535;
    };

    FlowTimeLabelFilter(Input input);
    ImagePtr operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter
