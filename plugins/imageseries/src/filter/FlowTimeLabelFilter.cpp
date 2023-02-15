#include "FlowTimeLabelFilter.h"

#include "mmcore/utility/log/Log.h"

#include "imageseries/graph/GraphData2D.h"

#include "vislib/graphics/BitmapImage.h"
#include "vislib/graphics/PngBitmapCodec.h"

#include "../util/GraphCSVExporter.h"
#include "../util/GraphLuaExporter.h"
#include "../util/GraphSimplifier.h"

#include <array>
#include <deque>
#include <iostream>
#include <cmath>
#include <regex>
#include <map>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace megamol::ImageSeries::filter {

FlowTimeLabelFilter::FlowTimeLabelFilter(Input input) : input(std::move(input)) {}

std::shared_ptr<FlowTimeLabelFilter::Output> FlowTimeLabelFilter::operator()() {
    using Image = vislib::graphics::BitmapImage;

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

    // Create interface images
    const enum class interface_t : Timestamp {
        none = -1,
        full
    } interface_output = interface_t::none; //static_cast<interface_t>(75); // TODO: parameter?!

    std::shared_ptr<Image> interfaceFluidImage, interfaceSolidImage, interfaceImage;
    uint16_t* interfaceFluidOut, * interfaceSolidOut, * interfaceOut = nullptr;
    if (interface_output != interface_t::none) {
        interfaceFluidImage =
            std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);
        interfaceFluidImage->SetChannelLabel(0, vislib::graphics::BitmapImage::ChannelLabel::CHANNEL_GRAY);
        interfaceFluidOut = interfaceFluidImage->PeekDataAs<uint16_t>();

        interfaceSolidImage =
            std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);
        interfaceSolidImage->SetChannelLabel(0, vislib::graphics::BitmapImage::ChannelLabel::CHANNEL_GRAY);
        interfaceSolidOut = interfaceSolidImage->PeekDataAs<uint16_t>();

        interfaceImage =
            std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);
        interfaceImage->SetChannelLabel(0, vislib::graphics::BitmapImage::ChannelLabel::CHANNEL_GRAY);
        interfaceOut = interfaceImage->PeekDataAs<uint16_t>();

        for (Index index = 0; index < size; ++index) {
            interfaceFluidOut[index] = interfaceSolidOut[index] = interfaceOut[index] = -1;
        }
    }

    // Setup graph
    std::vector<std::vector<graph::GraphData2D::NodeID>> nodeIDs;
    auto nodeGraph = std::make_shared<graph::GraphData2D>();

    auto getOrCreateNodeID = [&](const Timestamp ts, const Label label) {
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
            nodeID = nodeGraph->addNode(std::move(node));
            nodeIDs[ts][label] = nodeID;
        }
        return nodeID;
    };

    auto getOrCreateNode = [&](const Timestamp ts, const Label label) -> graph::GraphData2D::Node& {
        return nodeGraph->getNode(getOrCreateNodeID(ts, label));
    };

    auto addEdgeByID = [&](const graph::GraphData2D::NodeID srcNode, const graph::GraphData2D::NodeID destNode) {
        graph::GraphData2D::Edge edge;
        edge.from = srcNode;
        edge.to = destNode;

        nodeGraph->getNode(edge.from).edgeCountOut++;
        nodeGraph->getNode(edge.to).edgeCountIn++;

        nodeGraph->addEdge(std::move(edge));
    };

    auto addEdge = [&](const Timestamp srcTS, const Label srcLabel, const Timestamp destTS, const Label destLabel) {
        return addEdgeByID(getOrCreateNodeID(srcTS, srcLabel), getOrCreateNodeID(destTS, destLabel));
    };

    // Assign unique labels to connected areas of same time
    struct FlowFront {
        graph::GraphData2D::NodeID node_id = -1;

        Label label = 0;
        Timestamp time = 0;
        std::vector<Index> pixels;

        float area = 0.0f;
        float fluid_fluid_interface_length = 0.0f;
        float fluid_solid_interface_length = 0.0f;
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

        std::map<Timestamp, std::unordered_set<Index>> interfaces;

        bool valid = true;
    };

    std::vector<FlowFront> flow_fronts(1);
    std::map<Timestamp, std::vector<std::reference_wrapper<FlowFront>>> flow_fronts_by_time; // without solid

    for (Index index = 0; index < size; ++index) {
        if (dataIn[index] == 0) {
            dataOut[index] = LabelSolid;

            flow_fronts[LabelSolid].pixels.push_back(index);
        } else if (dataIn[index] == std::numeric_limits<Timestamp>::max()) {
            dataOut[index] = LabelEmpty;
        } else {
            dataOut[index] = LabelUnassigned;
        }
    }

    std::vector<Index> floodQueue;

    auto floodFill = [&flow_fronts, &flow_fronts_by_time, &floodQueue, &width, &height, &dataIn, &dataOut,
                         &interfaceFluidOut, &interfaceSolidOut, &interfaceOut, &interface_output,
                         &getOrCreateNodeID](
                         const Index index, const Label label) {
        floodQueue.clear();
        floodQueue.push_back(index);

        auto& current_region = flow_fronts.emplace_back();
        current_region.label = label;
        current_region.node_id = getOrCreateNodeID(dataIn[index], label);
        current_region.time = dataIn[index];
        current_region.pixels.push_back(index);

        dataOut[index] = label;

        for (std::size_t queueIndex = 0; queueIndex < floodQueue.size(); ++queueIndex) {
            const auto currentIndex = floodQueue[queueIndex];

            const auto x = currentIndex % width;
            const auto y = currentIndex / width;

            // TODO: face-connected vs. kernel
            for (auto j = ((y > 0) ? -1 : 0); j <= ((y < height - 1) ? 1 : 0); ++j) {
                for (auto i = ((x > 0) ? -1 : 0); i <= ((x < width - 1) ? 1 : 0); ++i) {
                    const auto neighborIndex = (y + j) * width + (x + i);

                    if (dataOut[neighborIndex] == LabelUnassigned && dataIn[index] == dataIn[neighborIndex]) {
                        floodQueue.push_back(neighborIndex);
                        dataOut[neighborIndex] = label;

                        current_region.pixels.push_back(neighborIndex);
                    } else if (dataOut[neighborIndex] == LabelSolid) {
                        // Add current fluid index to interfaces, indicating a fluid-solid interface
                        current_region.interfaces[LabelSolid].insert(currentIndex);

                        if (interface_output == interface_t::full) {
                            interfaceSolidOut[currentIndex] = interfaceOut[currentIndex] = 0;
                        }
                    } else if (dataIn[index] > dataIn[neighborIndex]) {
                        // Add neighboring fluid index to interfaces, indicating a past fluid-fluid interface
                        current_region.interfaces[dataIn[neighborIndex]].insert(neighborIndex);
                    } else if (dataIn[index] < dataIn[neighborIndex]) {
                        // Add current fluid index to interfaces, indicating a current fluid-fluid interface
                        current_region.interfaces[dataIn[neighborIndex]].insert(currentIndex);

                        if (interface_output == interface_t::full) {
                            interfaceFluidOut[neighborIndex] = interfaceOut[currentIndex] = current_region.time;
                        }
                    }

                    if (interface_output != interface_t::none) {
                        const auto targetInterface = static_cast<Timestamp>(interface_output);
                        if (dataIn[index] <= targetInterface && dataIn[neighborIndex] > targetInterface) {
                            interfaceFluidOut[neighborIndex] = interfaceOut[currentIndex] = 0;
                        }
                        if (dataIn[index] <= targetInterface && dataIn[neighborIndex] == LabelSolid) {
                            interfaceSolidOut[neighborIndex] = interfaceOut[currentIndex] = 0;
                        }
                    }
                }
            }
        }

        current_region.fluid_solid_interface_length = current_region.interfaces[LabelSolid].size();
        current_region.fluid_fluid_interface_length = 0.0f;

        for (const auto& fluid_interface : current_region.interfaces) {
            if (fluid_interface.first != LabelSolid) {
                current_region.fluid_fluid_interface_length += fluid_interface.second.size();
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
        }

        flow_fronts_by_time[flow_fronts[i].time].push_back(flow_fronts[i]);
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

    // Create graph by iterating over flow fronts with time monotonically increasing
    for (auto& flow_fronts : flow_fronts_by_time) {
        for (auto& flow_front_ref : flow_fronts.second) {
            FlowFront& flow_front = flow_front_ref;

            // Find neighboring flow fronts and add edge if the neighboring front is (1) from the past and (2) the local maximum
            std::unordered_set<Label> past_neighboring_flow_fronts;
            Timestamp past_time{};

            for (auto it = flow_front.interfaces.crbegin(); it != flow_front.interfaces.crend(); ++it) {
                if (it->first != LabelSolid && it->first < flow_front.time) {
                    past_time = it->first;

                    for (const auto index : it->second) {
                        if (dataOut[index] != LabelInvalid) {
                            past_neighboring_flow_fronts.insert(dataOut[index]);
                        }
                    }

                    break;
                }
            }

            for (const auto& past_flow_front : past_neighboring_flow_fronts) {
                addEdge(past_time, past_flow_front, flow_front.time, flow_front.label);
            }

            // TODO: Use flow fronts to calculate velocity
            flow_front.velocity = FlowFront::Vec2{0, 0};
        }
    }

    // Copy flow front information into graph nodes
    for (std::size_t i = 1; i < flow_fronts.size(); ++i) {
        auto& node = nodeGraph->getNode(flow_fronts[i].node_id);
        node.frameIndex = flow_fronts[i].time;
        node.centerOfMass = glm::vec2{flow_fronts[i].center_of_mass.x, flow_fronts[i].center_of_mass.y};
        node.velocity = glm::vec2{flow_fronts[i].velocity.x, flow_fronts[i].velocity.y};
        node.velocityMagnitude = std::sqrt(flow_fronts[i].velocity.x * flow_fronts[i].velocity.x +
                                           flow_fronts[i].velocity.y * flow_fronts[i].velocity.y);
        node.area = flow_fronts[i].area;
        node.interfaceFluid = flow_fronts[i].fluid_fluid_interface_length;
        node.interfaceSolid = flow_fronts[i].fluid_solid_interface_length;
        node.boundingBox = graph::GraphData2D::Rect{
            static_cast<int>(flow_fronts[i].bounding_rectangle.x_min),
            static_cast<int>(flow_fronts[i].bounding_rectangle.y_min),
            static_cast<int>(flow_fronts[i].bounding_rectangle.x_max),
            static_cast<int>(flow_fronts[i].bounding_rectangle.y_max)};
    }

    // TODO: Simplify graph







    graph::util::LuaExportMeta luaExportMeta;
    luaExportMeta.path = input.timeMap->getMetadata().filename;
    luaExportMeta.minRange = 0.0f;
    luaExportMeta.maxRange = 1.0f;
    luaExportMeta.imgW = input.timeMap->getMetadata().width;
    luaExportMeta.imgH = input.timeMap->getMetadata().height;

    // Export to Lua file
    graph::util::exportToLua(*nodeGraph, "temp/CurrentGraph.lua", luaExportMeta);

    // Output interface image to hard disk
    if (interface_output != interface_t::none) {
        std::stringstream value;
        value << "_" << static_cast<Timestamp>(interface_output);

        const std::string path = "T:\\temp\\adrian\\interface";

        std::stringstream filenameFluid;
        filenameFluid << path << "_fluid" << (interface_output == interface_t::full ? "" : value.str().c_str())
                      << ".png";

        std::stringstream filenameSolid;
        filenameSolid << path << "_solid" << (interface_output == interface_t::full ? "" : value.str().c_str())
                      << ".png";

        std::stringstream filename;
        filename << path << (interface_output == interface_t::full ? "" : value.str().c_str()) << ".png";

        sg::graphics::PngBitmapCodec png_codec;
        png_codec.Image() = interfaceFluidImage.get();
        png_codec.Save(filenameFluid.str().c_str());
        png_codec.Image() = interfaceSolidImage.get();
        png_codec.Save(filenameSolid.str().c_str());
        png_codec.Image() = interfaceImage.get();
        png_codec.Save(filename.str().c_str());
    }

    // Set output
    auto output = std::make_shared<Output>();
    output->image = result;
    output->graph = nodeGraph;
    return output;
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
