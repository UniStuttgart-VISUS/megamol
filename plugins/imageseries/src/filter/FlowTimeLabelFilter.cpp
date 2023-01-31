#include "FlowTimeLabelFilter.h"

#include "mmcore/utility/log/Log.h"

#include "imageseries/graph/GraphData2D.h"

#include "vislib/graphics/PngBitmapCodec.h"
#include "vislib/graphics/BitmapImage.h"

#include "../util/GraphCSVExporter.h"
#include "../util/GraphLuaExporter.h"
#include "../util/GraphSimplifier.h"

#include <array>
#include <deque>
#include <iostream>
#include <cmath>
#include <regex>
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

    util::PerfTimer timer("FlowTimeLabelFilter", input.timeMap->getMetadata().filename);

    // Create output image
    auto result = std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);

    using Index = std::uint32_t;
    using Timestamp = std::uint16_t;

    const auto* dataIn = image->PeekDataAs<Timestamp>();
    auto* dataOut = result->PeekDataAs<Label>();
    Index width = result->Width();
    Index height = result->Height();
    Index size = width * height;





    // Assign unique labels to connected areas of same time
    struct FlowFront {
        Timestamp time = 0;
        std::vector<Index> pixels;

        float area = 0.0f;
        float fluid_fluid_interface = 0.0f;
        float fluid_solid_interface = 0.0f;
        struct Vec2 {
            float x;
            float y;
        } center_of_mass{}, velocity{};
        struct Rect2 {
            float x_min;
            float y_min;
            float x_max;
            float y_max;
        } bounding_rectangle{};
        // ...

        bool valid = true;
    };

    std::vector<FlowFront> flow_fronts(1);

    for (Index index = 0; index < size; ++index) {
        if (dataIn[index] == 0 || dataIn[index] == std::numeric_limits<Timestamp>::max()) {
            dataOut[index] = LabelSolid;

            flow_fronts[LabelSolid].pixels.push_back(index);
        } else {
            dataOut[index] = LabelUnassigned;
        }
    }

    std::vector<Index> floodQueue;

    auto floodFill = [&flow_fronts, &floodQueue, &width, &height, &dataIn, &dataOut](
                         const Index index, const Label label) {
        floodQueue.clear();
        floodQueue.push_back(index);

        auto& current_region = flow_fronts.emplace_back();
        current_region.time = dataIn[index];
        current_region.pixels.push_back(index);

        dataOut[index] = label;

        auto add = [&current_region, &floodQueue, &label, &index, &dataIn, &dataOut](const Index newIndex) {
            if (dataOut[newIndex] == LabelUnassigned && dataIn[index] == dataIn[newIndex]) {
                floodQueue.push_back(newIndex);
                dataOut[newIndex] = label;

                current_region.pixels.push_back(newIndex);
            }
        };

        for (std::size_t queueIndex = 0; queueIndex < floodQueue.size(); ++queueIndex) {
            const auto currentIndex = floodQueue[queueIndex];

            const auto x = currentIndex % width;
            const auto y = currentIndex / width;

            // TODO: face-connected vs. kernel
            for (auto j = ((y > 0) ? -1 : 0); j <= ((y < height - 1) ? 1 : 0); ++j) {
                for (auto i = ((x > 0) ? -1 : 0); i <= ((x < width - 1) ? 1 : 0); ++i) {
                    const auto neighborIndex = (y + j) * width + (x + i);

                    add(neighborIndex);
                }
            }
        }
    };

    Label next_label = LabelMinimum;

    for (Index index = 0; index < size; ++index) {
        if (dataOut[index] == LabelUnassigned && next_label <= LabelMaximum) {
            floodFill(index, next_label++);
        }
    }

    if (next_label > LabelMaximum) {
        throw std::runtime_error("Too many labels.");
    }

    // Calculate quantities for each flow front
    for (std::size_t i = 1; i < flow_fronts.size(); ++i) {
        const auto& pixels = flow_fronts[i].pixels;

        flow_fronts[i].area = static_cast<float>(pixels.size());

        if (!pixels.empty()) {
            const auto first_x = static_cast<float>(pixels[0] % width);
            const auto first_y = static_cast<float>(pixels[0] / width);

            flow_fronts[i].center_of_mass = FlowFront::Vec2{0, 0};
            flow_fronts[i].bounding_rectangle = FlowFront::Rect2{first_x, first_y, first_x, first_y};

            for (const auto& pixel : pixels) {
                const auto x = static_cast<float>(pixel % width);
                const auto y = static_cast<float>(pixel / width);

                flow_fronts[i].center_of_mass.x += static_cast<float>(x);
                flow_fronts[i].center_of_mass.y += static_cast<float>(y);

                flow_fronts[i].bounding_rectangle.x_min =
                    std::min(flow_fronts[i].bounding_rectangle.x_min, static_cast<float>(x));
                flow_fronts[i].bounding_rectangle.x_max =
                    std::max(flow_fronts[i].bounding_rectangle.x_max, static_cast<float>(x));
                flow_fronts[i].bounding_rectangle.y_min =
                    std::min(flow_fronts[i].bounding_rectangle.y_min, static_cast<float>(y));
                flow_fronts[i].bounding_rectangle.y_max =
                    std::max(flow_fronts[i].bounding_rectangle.y_max, static_cast<float>(y));
            }

            flow_fronts[i].center_of_mass.x /= pixels.size();
            flow_fronts[i].center_of_mass.y /= pixels.size();

            // TODO: compute more stuff
            flow_fronts[i].velocity = FlowFront::Vec2{0, 0};
            flow_fronts[i].fluid_fluid_interface = 0.0f;
            flow_fronts[i].fluid_solid_interface = 0.0f;
        }
    }

    // Filter or combine flow fronts
    const bool remove = true; // TODO: user input
    const auto min_size = 5; // TODO: user input

    for (std::size_t i = 1; i < flow_fronts.size(); ++i) {
        if (flow_fronts[i].area < min_size) {
            flow_fronts[i].valid = !remove;

            if (remove) {
                for (const auto& pixel : flow_fronts[i].pixels) {
                    dataOut[pixel] = LabelInvalid;
                }
            } else {
                // TODO
            }
        }
    }

    // Create graph
    std::vector<std::vector<graph::GraphData2D::NodeID>> nodeIDs;
    graph::GraphData2D nodeGraph;

    auto getOrCreateNodeID = [&](Timestamp ts, Label label) {
        if (nodeIDs.size() <= ts) {
            nodeIDs.resize(ts + 1uLL);
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












    graph::util::LuaExportMeta luaExportMeta;
    luaExportMeta.path = input.timeMap->getMetadata().filename;
    luaExportMeta.minRange = 0.0f;
    luaExportMeta.maxRange = 1.0f / input.timeMap->getMetadata().imageCount;
    luaExportMeta.imgW = input.timeMap->getMetadata().width;
    luaExportMeta.imgH = input.timeMap->getMetadata().height;

    // Export to Lua file
    //graph::util::exportToLua(nodeGraph, "temp/CurrentGraph.lua", luaExportMeta);

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
