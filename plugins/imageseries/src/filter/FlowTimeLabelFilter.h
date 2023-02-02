#pragma once

#include "imageseries/AsyncImageData2D.h"
#include "imageseries/ImageSeries2DCall.h"
#include "imageseries/graph/GraphData2D.h"
#include "imageseries/graph/GraphData2DCall.h"
#include "imageseries/util/ImageUtils.h"

#include <memory>

namespace megamol::ImageSeries::filter {

class FlowTimeLabelFilter {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D<>>;
    using ImagePtr = std::shared_ptr<const typename AsyncImageData2D<>::BitmapImage>;

    struct Output {
        std::shared_ptr<const vislib::graphics::BitmapImage> image;
        std::shared_ptr<const graph::GraphData2D> graph;
    };

    using Label = std::uint16_t;

    // Special labels
    enum LabelType : Label {
        // Pixel belonging to the solid phase / unoccupied space
        LabelSolid = 0,

        // Minimal label for flow
        LabelMinimum = 1,

        // Maximal label available
        LabelMaximum = 65533,

        // Label for flow fronts that are "invalid" (e.g., too small)
        LabelInvalid = 65534,

        // Initially unassigned pixel
        LabelUnassigned = 65535,
    };

    static constexpr int LabelCount = 65536;

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
    std::shared_ptr<Output> operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;
};

} // namespace megamol::ImageSeries::filter

namespace std {
template<>
struct hash<std::shared_ptr<const megamol::ImageSeries::graph::GraphData2D>> {
    std::size_t operator()(std::shared_ptr<const megamol::ImageSeries::graph::GraphData2D> graphData) const {
        return graphData
                   ? megamol::ImageSeries::util::computeHash(graphData->getNodes().size(), graphData->getEdges().size())
                         : 0;
    }
};

template<>
struct hash<std::shared_ptr<const megamol::ImageSeries::filter::FlowTimeLabelFilter::Output>> {
    std::size_t operator()(
        std::shared_ptr<const megamol::ImageSeries::filter::FlowTimeLabelFilter::Output> data) const {
        return data ? megamol::ImageSeries::util::combineHash(std::hash<decltype(data->image)>()(data->image),
                          std::hash<decltype(data->graph)>()(data->graph))
                    : 0;
    }
};
} // namespace std
