#ifndef SRC_IMAGESERIES_BLOB_BLOBANALYZER_HPP_
#define SRC_IMAGESERIES_BLOB_BLOBANALYZER_HPP_

#include "imageseries/AsyncImageData2D.h"

#include "../filter/BlobLabelFilter.h"

#include "glm/vec2.hpp"

#include <memory>

namespace megamol::ImageSeries::blob {

class BlobAnalyzer {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;
    using ImagePtr = std::shared_ptr<const AsyncImageData2D::BitmapImage>;
    using Label = filter::BlobLabelFilter::Label;

    struct Rect {
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
    };

    struct Blob {
        Label label = filter::BlobLabelFilter::LabelBackground;
        int pixelCount = 0;
        int valueSum = 0;
        float valueMean = 0.f;
        Rect boundingBox;
        glm::vec2 centerOfMass = {};
        glm::vec2 weightedCenterOfMass = {};
    };

    struct Input {
        AsyncImagePtr image;
        AsyncImagePtr labels;
    };

    struct Output {
        std::vector<Blob> blobs;
    };

    Output apply(Input input);
};

} // namespace megamol::ImageSeries::blob


#endif
