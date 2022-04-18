#ifndef SRC_IMAGESERIES_BLOB_BLOBGRAPHBUILDER_HPP_
#define SRC_IMAGESERIES_BLOB_BLOBGRAPHBUILDER_HPP_

#include "imageseries/AsyncImageData2D.h"
#include "imageseries/graph/GraphData2D.h"

#include "BlobAnalyzer.h"
#include "BlobRegistrator.h"

#include <atomic>
#include <memory>

namespace megamol::ImageSeries::blob {

class BlobGraphBuilder {
public:
    using AsyncImagePtr = std::shared_ptr<const AsyncImageData2D>;

    BlobGraphBuilder();

    // TODO add timestamp
    void addFrame(AsyncImagePtr labels, AsyncImagePtr values);
    std::shared_ptr<const graph::AsyncGraphData2D> finalize();

    std::size_t getTotalFrameCount() const;
    std::size_t getPendingFrameCount() const;

private:
    struct PendingFrame {
        std::size_t index = 0;
        std::shared_ptr<util::AsyncData<BlobAnalyzer::Output>> analyzerResult;
        std::shared_ptr<util::AsyncData<BlobRegistrator::Output>> registratorResult;
    };

    struct PartialResult {
        graph::GraphData2D graph;
        std::array<graph::GraphData2D::NodeID, 256> labelMapping;
    };

    std::size_t index = 0;
    bool finalized = false;
    std::atomic_int pendingFrameCount = ATOMIC_VAR_INIT(0);

    AsyncImagePtr previousLabels;
    std::shared_ptr<util::AsyncData<PartialResult>> partialResult;
};

} // namespace megamol::ImageSeries::blob


#endif
