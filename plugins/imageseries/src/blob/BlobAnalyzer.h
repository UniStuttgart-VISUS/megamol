#pragma once

#include "imageseries/AsyncImageData2D.h"
#include "imageseries/graph/GraphData2D.h"

#include "../filter/BlobLabelFilter.h"

#include "glm/vec2.hpp"

#include <memory>

namespace megamol::ImageSeries::blob {

class BlobAnalyzer {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D<>>;
    using ImagePtr = std::shared_ptr<const typename AsyncImageData2D<>::BitmapImage>;
    using Label = filter::BlobLabelFilter::Label;

    struct Blob {
        Label label = filter::BlobLabelFilter::LabelBackground;
        int pixelCount = 0;
        int valueSum = 0;
        float valueMean = 0.f;
        megamol::ImageSeries::graph::GraphData2D::Rect boundingBox;
        glm::vec2 centerOfMass = {};
        glm::vec2 weightedCenterOfMass = {};
    };

    struct Input {
        AsyncImagePtr image;
        AsyncImagePtr labels;
        AsyncImagePtr prev;
    };

    struct Output {
        std::vector<Blob> blobs;
    };

    Output apply(Input input);
};

} // namespace megamol::ImageSeries::blob
