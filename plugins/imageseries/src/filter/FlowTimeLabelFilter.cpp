#include "FlowTimeLabelFilter.h"

#include "imageseries/graph/GraphData2D.h"

#include "vislib/graphics/BitmapImage.h"

#include "../util/GraphCSVExporter.h"
#include "../util/GraphLuaExporter.h"
#include "../util/GraphSimplifier.h"

#include <array>
#include <deque>
#include <vector>

namespace megamol::ImageSeries::filter {

FlowTimeLabelFilter::FlowTimeLabelFilter(Input input) : input(std::move(input)) {}

FlowTimeLabelFilter::ImagePtr FlowTimeLabelFilter::operator()() {
    using Image = AsyncImageData2D::BitmapImage;

    // Wait for image data to be ready
    auto image = input.timeMap ? input.timeMap->getImageData() : nullptr;

    // Empty -> return nothing
    if (!image) {
        return nullptr;
    }

    if (image->GetChannelCount() != 1 || image->GetChannelType() != Image::ChannelType::CHANNELTYPE_WORD) {
        return nullptr;
    }

    // Create output image
    auto result = std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_BYTE);

    using Index = std::uint32_t;
    using Timestamp = std::uint16_t;

    const auto* dataIn = image->PeekDataAs<Timestamp>();
    auto* dataOut = result->PeekDataAs<std::uint8_t>();
    Index width = result->Width();
    Index height = result->Height();
    Index size = width * height;

    std::vector<std::vector<graph::GraphData2D::NodeID>> nodeIDs;
    graph::GraphData2D nodeGraph;

    auto getOrCreateNodeID = [&](Timestamp ts, Label label) {
        if (nodeIDs.size() <= ts) {
            nodeIDs.resize(ts + 1);
        }
        while (nodeIDs[ts].size() <= label) {
            nodeIDs[ts].push_back(graph::GraphData2D::NodeIDNone);
        }
        auto nodeID = nodeIDs[ts][label];
        if (nodeID == graph::GraphData2D::NodeIDNone) {
            graph::GraphData2D::Node node;
            node.frameIndex = ts;
            nodeID = nodeGraph.addNode(std::move(node));
            nodeIDs[ts][label] = nodeID;
        }
        return nodeID;
    };

    auto getOrCreateNode = [&](Timestamp ts, Label label) -> graph::GraphData2D::Node& {
        return nodeGraph.getNode(getOrCreateNodeID(ts, label));
    };

    auto addEdge = [&](Timestamp srcTS, Label srcLabel, Timestamp destTS, Label destLabel) {
        graph::GraphData2D::Edge edge;
        edge.from = getOrCreateNodeID(srcTS, srcLabel);
        edge.to = getOrCreateNodeID(destTS, destLabel);

        auto& nodeFrom = nodeGraph.getNode(edge.from);
        auto& nodeTo = nodeGraph.getNode(edge.to);

        nodeFrom.edgeCountOut++;
        nodeTo.edgeCountIn++;

        nodeGraph.addEdge(std::move(edge));
    };

    Timestamp timeThreshold = input.timeThreshold;

    Timestamp currentTimestamp = 0;
    Timestamp minimumTimestamp = input.minimumTimestamp;
    Timestamp maximumTimestamp = 0;

    for (Index i = 0; i < size; ++i) {
        auto timestamp = dataIn[i];
        // Init background/mask
        dataOut[i] = timestamp < minimumTimestamp ? LabelMask : LabelBackground;
        if (maximumTimestamp < timestamp && timestamp < input.maximumTimestamp) {
            maximumTimestamp = timestamp;
        }
    }

    struct FrontQueueEntry {
        FrontQueueEntry(Index index = 0, Label label = LabelBackground) : index(index), label(label) {}
        Index index; // Image pixel index of this queue entry
        Label label; // Timestamp of this queue entry
    };

    std::vector<Index> floodQueue;
    std::vector<FrontQueueEntry> nextFront;

    auto mark = [&](Index index, Label label) {
        // Skip already-visited pixels
        if (dataOut[index] != LabelBackground) {
            return;
        }

        auto sample = dataIn[index];
        if (sample == currentTimestamp) {
            // Mark as filled
            dataOut[index] = label;
            floodQueue.push_back(index);
        } else if (sample > currentTimestamp && sample < static_cast<std::uint32_t>(currentTimestamp) + timeThreshold) {
            // Mark as pending
            dataOut[index] = LabelFlow;
            nextFront.emplace_back(index, label);
        }
    };

    auto rectExpand = [](graph::GraphData2D::Rect r, int x, int y) -> graph::GraphData2D::Rect {
        if (r.valid()) {
            r.x1 = std::min(r.x1, x);
            r.x2 = std::max(r.x2, x);
            r.y1 = std::min(r.y1, y);
            r.y2 = std::max(r.y2, y);
        } else {
            r.x1 = x;
            r.x2 = x;
            r.y1 = y;
            r.y2 = y;
        }
        return r;
    };

    auto rectUnion = [](graph::GraphData2D::Rect r1, graph::GraphData2D::Rect r2) -> graph::GraphData2D::Rect {
        if (!r1.valid()) {
            return r2;
        } else if (!r2.valid()) {
            return r1;
        } else {
            graph::GraphData2D::Rect r;
            r.x1 = std::min(r1.x1, r2.x1);
            r.x2 = std::max(r1.x2, r2.x2);
            r.y1 = std::min(r1.y1, r2.y1);
            r.y2 = std::max(r1.y2, r2.y2);
            return r;
        }
    };

    int lastFloodFillCenterX = 0;
    int lastFloodFillCenterY = 0;

    graph::GraphData2D::Rect floodFillRect;

    auto floodFill = [&](Index index, Label label) {
        floodQueue.clear();
        floodQueue.push_back(index);

        lastFloodFillCenterX = 0;
        lastFloodFillCenterY = 0;
        floodFillRect.x1 = index % width;
        floodFillRect.x2 = index % width;
        floodFillRect.y1 = index / width;
        floodFillRect.y2 = index / width;

        for (std::size_t queueIndex = 0; queueIndex < floodQueue.size(); ++queueIndex) {
            index = floodQueue[queueIndex];

            int x = index % width;
            int y = index / width;

            lastFloodFillCenterX += x;
            lastFloodFillCenterY += y;

            floodFillRect = rectExpand(floodFillRect, x, y);

            // Pixel is not on the right boundary
            if (index % width < width - 1) {
                mark(index + 1, label);
            }
            // Pixel is not on the left boundary
            if (index % width > 0) {
                mark(index - 1, label);
            }
            // Pixel is not on the bottom boundary
            if (index < size - width) {
                mark(index + width, label);
            }
            // Pixel is not on the top boundary
            if (index >= width) {
                mark(index - width, label);
            }
        }
        return floodQueue.size();
    };

    // Phase 1: find initial frame
    for (currentTimestamp = minimumTimestamp; currentTimestamp <= maximumTimestamp; ++currentTimestamp) {
        Label frontLabel = LabelFirst;
        for (Index i = 0; i < size; ++i) {
            if (dataIn[i] == currentTimestamp) {
                dataOut[i] = frontLabel;

                // Remember old size of nextFront
                std::size_t nextFrontPrevSize = nextFront.size();

                // Perform speculative flood fill. If the region is big enough, keep the changes. Otherwise, revert them
                std::size_t fillCount = floodFill(i, frontLabel);
                if (fillCount >= input.minBlobSize) {
                    // Region large enough: advance label counter and keep filled pixels
                    auto& node = getOrCreateNode(currentTimestamp, frontLabel);
                    node.area += fillCount;
                    node.centerOfMass.x += lastFloodFillCenterX;
                    node.centerOfMass.y += lastFloodFillCenterY;
                    node.boundingBox = rectUnion(node.boundingBox, floodFillRect);
                    if (frontLabel != LabelLast) {
                        frontLabel++;
                    }
                } else {
                    // Region too small: mask out any involved pixels and revert front queue
                    dataOut[i] = LabelMask;
                    // Mask flooded pixels
                    for (auto index : floodQueue) {
                        dataOut[index] = LabelMask;
                    }
                    // Revert queued pixels
                    for (std::size_t frontIndex = nextFrontPrevSize; frontIndex < nextFront.size(); ++frontIndex) {
                        dataOut[nextFront[frontIndex].index] = LabelBackground;
                    }
                    nextFront.resize(nextFrontPrevSize);
                }
            }
        }

        // Enter phase 2 if any suitably sized region was found
        if (frontLabel != LabelFirst) {
            break;
        }
    }

    // Tracks the list of pixels associated with the current front of each label
    std::vector<std::vector<Index>> currentInterfaces;

    // Phase 2: follow fluid flow
    for (; currentTimestamp <= maximumTimestamp; ++currentTimestamp) {
        std::vector<std::vector<Label>> interfaceEdges;
        std::vector<std::vector<Index>> previousInterfaces = std::move(currentInterfaces);
        currentInterfaces.clear();

        // Split pending pixels into sections with unique labels
        std::vector<Index> frontQueue;
        Label frontLabel = LabelFirst;
        auto splitFront = [&](FrontQueueEntry entry) {
            if (dataOut[entry.index] != LabelFlow || frontLabel == LabelLast) {
                return;
            }
            frontQueue.clear();
            frontQueue.push_back(entry.index);
            dataOut[entry.index] = frontLabel;

            // Insert edge into graph
            // TODO - this does not handle merges yet
            addEdge(currentTimestamp - 1, entry.label, currentTimestamp, frontLabel);

            // Track interface edges for velocity computation
            interfaceEdges.resize(
                std::max<std::size_t>(interfaceEdges.size(), static_cast<std::size_t>(entry.label) + 1));
            interfaceEdges[entry.label].push_back(frontLabel);

            graph::GraphData2D::Rect rect;
            for (std::size_t queueIndex = 0; queueIndex < frontQueue.size(); ++queueIndex) {
                auto index = frontQueue[queueIndex];
                int cx = index % width;
                int cy = index / width;
                rect = rectExpand(rect, cx, cy);
                for (int y = std::max<int>(0, cy - 2); y < std::min<int>(cy + 3, height); ++y) {
                    for (int x = std::max<int>(0, cx - 2); x < std::min<int>(cx + 3, width); ++x) {
                        Index nextIndex = x + y * width;
                        if (dataOut[nextIndex] == LabelFlow) {
                            frontQueue.push_back(nextIndex);
                            dataOut[nextIndex] = frontLabel;
                        }
                    }
                }
            }

            auto& node = getOrCreateNode(currentTimestamp, frontLabel);
            node.boundingBox = rectUnion(node.boundingBox, rect);
            node.interfaceFluid += frontQueue.size();

            // Store all pixels in this interface for velocity computation
            currentInterfaces.push_back(std::move(frontQueue));

            frontLabel++;
        };

        // Split different non-connected fronts
        for (const auto& entry : nextFront) {
            splitFront(entry);
        }

        std::vector<FrontQueueEntry> currentFront = std::move(nextFront);
        nextFront.clear();
        for (const auto& entry : currentFront) {
            if (dataIn[entry.index] == currentTimestamp) {
                // Same timestamp: perform flood fill from this pixel
                auto fillLabel = dataOut[entry.index];
                auto fillCount = floodFill(entry.index, fillLabel);
                auto& node = getOrCreateNode(currentTimestamp, fillLabel);
                node.area += fillCount;
                node.centerOfMass.x += lastFloodFillCenterX;
                node.centerOfMass.y += lastFloodFillCenterY;
                node.boundingBox = rectUnion(node.boundingBox, floodFillRect);
            } else {
                // Preserve front pixels with a steeper timestamp gradient, requeueing them for the next iteration
                nextFront.emplace_back(entry.index, dataOut[entry.index]);
                dataOut[entry.index] = LabelFlow;
            }
        }

        auto computeHausdorffDistance = [&](const std::vector<Index>& sourceInterface,
                                            const std::vector<Index>& targetInterface) -> double {
            if (sourceInterface.empty() || targetInterface.empty()) {
                return 0.0;
            }

            // Use Hausdorff distance as a proxy for velocity
            int64_t maxSquareDistance = 0;
            for (Index sourceIndex : sourceInterface) {
                int sourceX = sourceIndex % width;
                int sourceY = sourceIndex / width;
                int64_t minSquareDistance = std::numeric_limits<int>::max();
                for (Index targetIndex : targetInterface) {
                    int64_t differenceX = sourceX - static_cast<int>(targetIndex % width);
                    int64_t differenceY = sourceY - static_cast<int>(targetIndex / width);
                    int64_t squareDistance = differenceX * differenceX + differenceY * differenceY;
                    minSquareDistance = std::min(minSquareDistance, squareDistance);
                }
                maxSquareDistance = std::max(minSquareDistance, maxSquareDistance);
            }

            return std::sqrt(static_cast<double>(maxSquareDistance));
        };

        // Compute interface velocities
        for (std::size_t i = 0; i < interfaceEdges.size(); ++i) {
            auto& edge = interfaceEdges[i];
            if (edge.empty() || static_cast<std::size_t>(i - LabelFirst) >= previousInterfaces.size()) {
                continue;
            }

            auto& sourceInterface = previousInterfaces[i - LabelFirst];

            if (edge.size() == 1) {
                // Fast path: 1-1 edge
                if (static_cast<std::size_t>(edge[0] - LabelFirst) < currentInterfaces.size()) {
                    getOrCreateNode(currentTimestamp, edge[0]).velocityMagnitude +=
                        computeHausdorffDistance(sourceInterface, currentInterfaces[edge[0] - LabelFirst]);
                }
            } else {
                // For 1-n connections, we need to merge the target nodes together to avoid treating interface splits
                // as having very high velocities
                std::vector<Index> mergedInterface;
                for (auto targetLabel : edge) {
                    if (static_cast<std::size_t>(targetLabel - LabelFirst) < currentInterfaces.size()) {
                        auto& targetInterface = currentInterfaces[targetLabel - LabelFirst];
                        mergedInterface.insert(mergedInterface.end(), targetInterface.begin(), targetInterface.end());
                    }
                }

                // Compute merged velocity and apply it to all target nodes
                double velocity = computeHausdorffDistance(sourceInterface, mergedInterface);
                for (auto targetLabel : edge) {
                    getOrCreateNode(currentTimestamp, targetLabel).velocityMagnitude += velocity;
                }
            }
        }
    }

    // TODO remove small mask blobs

    auto checkMaskNeighbor = [&](Index index, bool& mask) {
        if (dataOut[index] == LabelMask) {
            mask = true;
            return true;
        } else {
            return dataOut[index] == LabelBackground;
        }
    };

    auto checkMaskVicinity = [&](int cx, int cy) {
        for (int y = std::max<int>(0, cy - 2); y < std::min<int>(cy + 3, height); ++y) {
            for (int x = std::max<int>(0, cx - 2); x < std::min<int>(cx + 3, width); ++x) {
                if (dataOut[x + y * width] == LabelMask) {
                    return true;
                }
            }
        }
        return false;
    };

    // Phase 3: compute solid interface length
    for (Index y = 1; y < height - 1; ++y) {
        for (Index x = 1; x < width - 1; ++x) {
            Index index = x + y * width;
            auto label = dataOut[index];
            if (label > LabelFirst) {
                bool mask = false;
                bool hasBackgroundOrMask = checkMaskNeighbor(index - 1, mask) || checkMaskNeighbor(index + 1, mask) ||
                                           checkMaskNeighbor(index - width, mask) ||
                                           checkMaskNeighbor(index + width, mask);
                if (hasBackgroundOrMask && (mask || checkMaskVicinity(x, y))) {
                    auto timestamp = dataIn[index];
                    getOrCreateNode(timestamp, label).interfaceSolid++;
                }
            }
        }
    }

    // Postprocess nodes
    for (std::size_t i = 0; i < nodeGraph.getNodes().size(); ++i) {
        auto& node = nodeGraph.getNode(i);
        if (node.area > 0.000001f) {
            node.centerOfMass /= node.area;
        }
    }

    nodeGraph = graph::util::simplifyGraph(nodeGraph);

    return std::const_pointer_cast<const Image>(result);
}

ImageMetadata FlowTimeLabelFilter::getMetadata() const {
    if (input.timeMap) {
        ImageMetadata metadata = input.timeMap->getMetadata();
        metadata.bytesPerChannel = 1;
        metadata.hash = util::computeHash(input.timeMap, input.blobCountLimit, input.minBlobSize, input.timeThreshold,
            input.minimumTimestamp, input.maximumTimestamp);
        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
