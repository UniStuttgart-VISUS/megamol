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

    using Index = graph::GraphData2D::Pixel;
    using Timestamp = graph::GraphData2D::Timestamp;
    using Label = graph::GraphData2D::Label;

    // Special labels
    enum LabelType : Label {
        // Pixel belonging to the solid phase / unoccupied space
        LabelSolid = 0,

        // Minimal label for flow
        LabelMinimum = 1,

        // Maximal label available
        LabelMaximum = 65532,

        // Empty space
        LabelEmpty = 65533,

        // Label for flow fronts that are "invalid" (e.g., too small)
        LabelInvalid = 65534,

        // Initially unassigned pixel
        LabelUnassigned = 65535,
    };

    static constexpr int LabelCount = 65536;

    struct Input {
        // Timestamp map within which to search for connected blobs.
        AsyncImagePtr timeMap;

        // Select output image
        enum class image_t : int { simplified, invalid, original } outputImage;

        // Inflow area
        enum class inflow_t : int { left, bottom, right, top } inflowArea;
        int inflowMargin;

        // Minimum area, used as parameter for combining small areas
        int minArea;

        // Use Hausdorff distance or centers of mass for velocity calculation
        bool hausdorff = false;

        // Applied graph fixes/simplifications
        enum class fixes_t : int {
            nope = 0,
            isolated = 1,
            false_sources = 2,
            false_sinks = 4,
            resolve_diamonds = 16,
            combine_trivial = 32,
            combine_tiny = 64,
            remove_trivial = 128
        } fixes;
    };

    FlowTimeLabelFilter(Input input);
    std::shared_ptr<Output> operator()();

    ImageMetadata getMetadata() const;

private:
    Input input;

    graph::GraphData2D::Node combineNodes(
        const std::vector<graph::GraphData2D::Node>& nodesToCombine, Label& nextLabel) const;
    std::vector<graph::GraphData2D::Edge> combineEdges(const graph::GraphData2D& nodeGraph,
        const std::vector<graph::GraphData2D::NodeID>& nodesToCombine, graph::GraphData2D::NodeID newNodeID) const;

    bool combineSmallNodes(graph::GraphData2D& nodeGraph, Label& nextLabel, float tiny_area_threshold) const;
    void combineTrivialNodes(graph::GraphData2D& nodeGraph, Label& nextLabel) const;
    bool resolveDiamonds(graph::GraphData2D& nodeGraph, Label& nextLabel) const;
    void removeTrivialNodes(graph::GraphData2D& nodeGraph, Label& nextLabel) const;
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

static auto operator&(megamol::ImageSeries::filter::FlowTimeLabelFilter::Input::fixes_t lhs,
    megamol::ImageSeries::filter::FlowTimeLabelFilter::Input::fixes_t rhs) {
    return static_cast<int>(lhs) & static_cast<int>(rhs);
}

static auto operator|(megamol::ImageSeries::filter::FlowTimeLabelFilter::Input::fixes_t lhs,
    megamol::ImageSeries::filter::FlowTimeLabelFilter::Input::fixes_t rhs) {
    return static_cast<megamol::ImageSeries::filter::FlowTimeLabelFilter::Input::fixes_t>(
        static_cast<int>(lhs) | static_cast<int>(rhs));
}
