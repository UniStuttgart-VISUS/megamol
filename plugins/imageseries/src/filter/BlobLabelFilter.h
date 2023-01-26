#pragma once

#include "imageseries/AsyncImageData2D.h"

#include <memory>

namespace megamol::ImageSeries::filter {

class BlobLabelFilter {
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
        // Input image within which to look for connected blobs.
        AsyncImagePtr image;

        // Predecessor frame to the input image.
        AsyncImagePtr prevImage;

        // Optional difference frame by which to decide whether labels should be included or not.
        AsyncImagePtr diffImage;

        // Optional mask image from which to determine the locations of static obstacles/boundaries.
        AsyncImagePtr mask;

        // Maximum number of blobs to track - if exceeded, the smallest blob is evicted and replaced with LabelMinimal.
        // Currently does not support values >252.
        std::size_t blobCountLimit = 250;

        // Minimum number of pixels required for a blob to be tracked. Smaller blobs are replaced with LabelMinimal.
        // Setting this value too high will decrease performance.
        std::size_t minBlobSize = 10;

        // If false, tracks connected blobs within dark pixels. If true, tracks connected blobs within bright pixels.
        bool negateThreshold = false;

        // Specifies the normalized brightness threshold under which a pixel is considered part of a blob.
        // If negateThreshold is true, this condition is inverted.
        float threshold = 0.5f;

        // Negates the mask's threshold.
        bool negateMask = false;

        // If false, applies the mask after labeling (labels take priority).
        // If true, applies the mask before labeling (mask takes priority).
        bool maskPriority = false;

        // If true, marks active pixels from the predecessor + current frames with the "small blob" label.
        bool markPrevious = true;
    };

    BlobLabelFilter(Input input);
    ImagePtr operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter
