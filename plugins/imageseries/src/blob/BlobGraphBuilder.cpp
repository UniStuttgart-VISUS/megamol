#include "BlobGraphBuilder.h"

namespace megamol::ImageSeries::blob {

BlobGraphBuilder::BlobGraphBuilder() {}

void BlobGraphBuilder::addFrame(AsyncImagePtr labels, AsyncImagePtr values) {
    auto pending = std::make_shared<PendingFrame>();
    pending->index = index++;

    pending->analyzerResult = std::make_shared<util::AsyncData<BlobAnalyzer::Output>>(
        [labels, values]() {
            BlobAnalyzer::Input input;
            input.labels = labels;
            input.image = values;
            return std::make_shared<BlobAnalyzer::Output>(BlobAnalyzer().apply(input));
        },
        0);

    if (previousLabels) {
        pending->registratorResult = std::make_shared<util::AsyncData<BlobRegistrator::Output>>(
            [labels, prev = previousLabels]() {
                BlobRegistrator::Input input;
                input.image = labels;
                input.predecessor = prev;
                return std::make_shared<BlobRegistrator::Output>(BlobRegistrator().apply(input));
            },
            0);
    }

    pendingFrameCount++;

    auto partial = partialResult;
    partialResult = std::make_shared<util::AsyncData<PartialResult>>(
        [this, partial, pending]() {
            auto result = partial ? partial->getData() : std::make_shared<PartialResult>();
            auto analyzerResult = pending->analyzerResult ? pending->analyzerResult->getData() : nullptr;
            auto registratorResult = pending->registratorResult ? pending->registratorResult->getData() : nullptr;

            auto prevMapping = result->labelMapping;
            result->labelMapping.fill(graph::GraphData2D::NodeIDNone);

            if (analyzerResult) {
                for (const auto& blob : analyzerResult->blobs) {
                    if (blob.label >= filter::BlobLabelFilter::LabelFirst) {
                        graph::GraphData2D::Node node;
                        node.frameIndex = pending->index;
                        node.centerOfMass = blob.centerOfMass;
                        node.area = blob.pixelCount;
                        // TODO interface length
                        result->labelMapping[blob.label] = result->graph.addNode(node);
                    }
                }
            }

            if (registratorResult) {
                for (const auto& link : registratorResult->links) {
                    graph::GraphData2D::Edge edge;
                    edge.from = prevMapping[link.source];
                    edge.to = result->labelMapping[link.dest];
                    if (edge.from != graph::GraphData2D::NodeIDNone && edge.to != graph::GraphData2D::NodeIDNone) {
                        auto& nodeFrom = result->graph.getNode(edge.from);
                        auto& nodeTo = result->graph.getNode(edge.to);
                        nodeTo.velocity += nodeTo.centerOfMass - nodeFrom.centerOfMass;
                        nodeFrom.edgeCountOut++;
                        nodeTo.edgeCountIn++;
                        result->graph.addEdge(std::move(edge));
                    }
                }
            }

            pendingFrameCount--;

            return result;
        },
        0);

    previousLabels = labels;
}

std::shared_ptr<const graph::AsyncGraphData2D> BlobGraphBuilder::finalize() {
    previousLabels = nullptr;
    if (auto result = partialResult) {
        partialResult = nullptr;
        return std::make_shared<const graph::AsyncGraphData2D>(
            [result]() { return std::make_shared<const graph::GraphData2D>(std::move(result->getData()->graph)); }, 0);
    } else {
        return nullptr;
    }
}

std::size_t BlobGraphBuilder::getTotalFrameCount() const {
    return index;
}

std::size_t BlobGraphBuilder::getPendingFrameCount() const {
    return pendingFrameCount;
}


} // namespace megamol::ImageSeries::blob
