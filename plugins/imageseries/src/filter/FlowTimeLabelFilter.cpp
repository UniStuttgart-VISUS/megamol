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
    std::uint16_t *interfaceFluidOut, *interfaceSolidOut, *interfaceOut = nullptr;
    if (interface_output != interface_t::none) {
        interfaceFluidImage =
            std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);
        interfaceFluidImage->SetChannelLabel(0, vislib::graphics::BitmapImage::ChannelLabel::CHANNEL_GRAY);
        interfaceFluidOut = interfaceFluidImage->PeekDataAs<std::uint16_t>();

        interfaceSolidImage =
            std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);
        interfaceSolidImage->SetChannelLabel(0, vislib::graphics::BitmapImage::ChannelLabel::CHANNEL_GRAY);
        interfaceSolidOut = interfaceSolidImage->PeekDataAs<std::uint16_t>();

        interfaceImage =
            std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);
        interfaceImage->SetChannelLabel(0, vislib::graphics::BitmapImage::ChannelLabel::CHANNEL_GRAY);
        interfaceOut = interfaceImage->PeekDataAs<std::uint16_t>();

        for (Index index = 0; index < size; ++index) {
            interfaceFluidOut[index] = interfaceSolidOut[index] = interfaceOut[index] = -1;
        }
    }

    // Assign unique labels to connected areas of same time
    auto nodeGraph = std::make_shared<graph::GraphData2D>();
    std::map<Timestamp, std::vector<std::reference_wrapper<graph::GraphData2D::Node>>> nodesByTime;

    for (Index index = 0; index < size; ++index) {
        if (dataIn[index] == 0) {
            dataOut[index] = LabelSolid;
        } else if (dataIn[index] == std::numeric_limits<Timestamp>::max()) {
            dataOut[index] = LabelEmpty;
        } else {
            dataOut[index] = LabelUnassigned;
        }
    }

    std::vector<Index> floodQueue;

    auto floodFill = [&nodeGraph, &floodQueue, &width, &height, &dataIn, &dataOut,
                         &interfaceFluidOut, &interfaceSolidOut, &interfaceOut, &interface_output](
                         const Index index, const Label label) {
        floodQueue.clear();
        floodQueue.push_back(index);

        graph::GraphData2D::Node current_region;
        current_region.label = label;
        current_region.frameIndex = dataIn[index];
        current_region.pixels.push_back(index);

        dataOut[index] = label;

        for (std::size_t queueIndex = 0; queueIndex < floodQueue.size(); ++queueIndex) {
            const auto currentIndex = floodQueue[queueIndex];

            const auto x = currentIndex % width;
            const auto y = currentIndex / width;

            // TODO: face-connected vs. kernel (now: kernel)
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
                            interfaceFluidOut[neighborIndex] = interfaceOut[currentIndex] = current_region.frameIndex;
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

        current_region.interfaceSolid = current_region.interfaces[LabelSolid].size();
        current_region.interfaceFluid = 0.0f;

        for (const auto& fluid_interface : current_region.interfaces) {
            if (fluid_interface.first != LabelSolid) {
                current_region.interfaceFluid += fluid_interface.second.size();
            }
        }

        nodeGraph->addNode(current_region);
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
    for (auto& node : nodeGraph->getNodes()) {
        const auto& pixels = node.pixels;

        node.area = static_cast<float>(pixels.size());

        if (!pixels.empty()) {
            const int first_x = pixels[0] % width;
            const int first_y = pixels[0] / width;

            node.centerOfMass = glm::vec2{0, 0};
            node.boundingBox = graph::GraphData2D::Rect{first_x, first_y, first_x, first_y};

            for (const auto& pixel : pixels) {
                const int x = pixel % width;
                const int y = pixel / width;

                node.centerOfMass.x += static_cast<float>(x);
                node.centerOfMass.y += static_cast<float>(y);

                node.boundingBox.x1 = std::min(node.boundingBox.x1, x);
                node.boundingBox.x2 = std::max(node.boundingBox.x2, x);
                node.boundingBox.y1 = std::min(node.boundingBox.y1, y);
                node.boundingBox.y2 = std::max(node.boundingBox.y2, y);
            }

            node.centerOfMass.x /= pixels.size();
            node.centerOfMass.y /= pixels.size();
        }

        nodesByTime[node.frameIndex].push_back(node);
    }

    // Create edges by iterating over flow fronts with time monotonically increasing
    for (auto& flow_fronts : nodesByTime) {
        for (auto& flow_front_ref : flow_fronts.second) {
            graph::GraphData2D::Node& flow_front = flow_front_ref;

            // Find neighboring flow fronts and add edge if the neighboring front is (1) from the past and (2) the local maximum
            std::unordered_set<Label> past_neighboring_flow_fronts;
            Timestamp past_time{};

            for (auto it = flow_front.interfaces.crbegin(); it != flow_front.interfaces.crend(); ++it) {
                if (it->first != LabelSolid && it->first < flow_front.frameIndex) {
                    past_time = it->first;

                    for (const auto index : it->second) {
                        past_neighboring_flow_fronts.insert(dataOut[index]);
                    }

                    break;
                }
            }

            const auto destNode = nodeGraph->findNode(flow_front.frameIndex, flow_front.label);

            for (const auto& past_flow_front : past_neighboring_flow_fronts) {
                const auto srcNode = nodeGraph->findNode(past_time, past_flow_front);

                graph::GraphData2D::Edge edge;
                edge.from = srcNode.first;
                edge.to = destNode.first;
                edge.weight = glm::distance(destNode.second.get().centerOfMass, srcNode.second.get().centerOfMass);

                nodeGraph->addEdge(std::move(edge));
            }

            // TODO: Use flow fronts to calculate velocity
            flow_front.velocity = glm::vec2{0, 0};
        }
    }

    // Set nodes to be invalid if they meet specific criteria
    for (auto& node : nodeGraph->getNodes()) {
        bool invalid = !node.valid;

        // (+) Remove isolated nodes, i.e., nodes that do not have incoming and outgoing connections
        if (input.fixes & Input::fixes_t::isolated) {
            invalid |= node.edgeCountIn == 0 && node.edgeCountOut == 0;
        }

        // (+) Remove source nodes that are not in an area where sources are expected (per user input)
        if (input.fixes & Input::fixes_t::false_sources) {
            int inflowRef = 0;
            switch (input.inflowArea) {
            case Input::inflow_t::left:
                inflowRef = node.boundingBox.x1;
                break;
            case Input::inflow_t::bottom:
                inflowRef = node.boundingBox.y1;
                break;
            case Input::inflow_t::right:
                inflowRef = node.boundingBox.x2 - input.timeMap->getMetadata().width;
                break;
            case Input::inflow_t::top:
                inflowRef = node.boundingBox.y2 - input.timeMap->getMetadata().height;
                break;
            }

            invalid |= node.edgeCountIn == 0 && std::abs(inflowRef) > input.inflowMargin;

            // (+) Remove nodes that have incoming edges only from invalid nodes, as these were originally connected
            //     to now removed sources, as per above directive
            node.valid = !invalid;

            std::list<graph::GraphData2D::NodeID> checkNodes(node.childNodes.begin(), node.childNodes.end());
            for (auto it = checkNodes.begin(); it != checkNodes.end(); checkNodes.erase(it++)) {
                auto& node = nodeGraph->getNode(*it);

                bool is_valid = false;
                for (const auto& parent : node.parentNodes) {
                    is_valid |= nodeGraph->getNode(parent).valid;
                }

                if (!is_valid || !node.valid) {
                    node.valid = false;
                    checkNodes.insert(checkNodes.end(), node.childNodes.begin(), node.childNodes.end());
                }
            }
        }

        // (+) Remove sinks that have neighbors with larger time step value, as these cannot be real sinks
        if (input.fixes & Input::fixes_t::false_sinks && node.edgeCountOut == 0) {
            for (auto it = node.interfaces.crbegin(); it != node.interfaces.crend(); ++it) {
                if (it->first != LabelSolid && it->first <= LabelMaximum && it->first > node.frameIndex) {
                    invalid |= true;

                    break;
                }
            }
        }

        // (?) Remove sinks whose single parent is not a node with one incoming and one outgoing edge,
        //     as this probably indicates an irrelevant sink
        if (input.fixes & Input::fixes_t::unimportant_sinks) {
            invalid |= node.edgeCountIn == 1 && node.edgeCountOut == 0 &&
                       !(nodeGraph->getNode(*node.parentNodes.begin()).edgeCountIn == 1 &&
                           nodeGraph->getNode(*node.parentNodes.begin()).edgeCountOut == 1);
        }

        // TODO: further filters/heuristics?

        node.valid = !invalid;
    }

    // Mark pixels of invalid nodes as invalid
    if (input.outputImage == Input::image_t::invalid) {
        for (auto& node : nodeGraph->getNodes()) {
            if (!node.valid) {
                for (const auto& pixel : node.pixels) {
                    dataOut[pixel] = LabelInvalid;
                }
            }
        }
    }

    // Re-build graph with valid nodes only
    auto oldNodeGraph = std::make_shared<graph::GraphData2D>();
    std::swap(nodeGraph, oldNodeGraph);

    std::unordered_map<graph::GraphData2D::NodeID, graph::GraphData2D::NodeID> nodeMap;

    for (std::size_t i = 0; i < oldNodeGraph->getNodeCount(); ++i) {
        const auto& oldNode = oldNodeGraph->getNode(i);

        if (oldNode.valid) {
            auto newNode(oldNode);
            newNode.edgeCountIn = 0;
            newNode.edgeCountOut = 0;
            newNode.childNodes.clear();
            newNode.parentNodes.clear();

            const auto newNodeID = nodeGraph->addNode(newNode);

            nodeMap[i] = newNodeID;
        }
    }

    for (const auto& oldEdge : oldNodeGraph->getEdges()) {
        const auto& oldNodeFrom = oldNodeGraph->getNode(oldEdge.from);
        const auto& oldNodeTo = oldNodeGraph->getNode(oldEdge.to);

        if (oldNodeFrom.valid && oldNodeTo.valid) {
            nodeGraph->addEdge(graph::GraphData2D::Edge{nodeMap.at(oldEdge.from), nodeMap.at(oldEdge.to), oldEdge.weight});
        }
    }

    // Simplify graph by combining tiny areas that result most likely from very small local velocities
    const auto tiny_area_threshold = 10u; // TODO: parameter

    if (input.fixes & Input::fixes_t::combine_tiny) {

    }

    // Simplify graph by combining subsequent nodes of 1-to-1 connections
    if (input.fixes & Input::fixes_t::combine_trivial) {

    }

    // Simplify graph by resolving diamond patterns into 1-to-1 connected nodes:
    // Resolve diamond patterns if and only if the nodes involved are
    // below the user-defined threshold for minimum obstacle size
    const float diamond_threshold = input.minObstacleSize;

    if (input.fixes & Input::fixes_t::resolve_diamonds) {

    }

    // Update pixels to match the resulting simplified graph
    if (input.outputImage == Input::image_t::simplified) {

    }








    // Export to Lua file
    graph::util::LuaExportMeta luaExportMeta;
    luaExportMeta.path = input.timeMap->getMetadata().filename;
    luaExportMeta.minRange = 0.0f;
    luaExportMeta.maxRange = 1.0f;
    luaExportMeta.imgW = input.timeMap->getMetadata().width;
    luaExportMeta.imgH = input.timeMap->getMetadata().height;

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
        metadata.hash = util::computeHash(
            input.timeMap, input.outputImage, input.inflowArea, input.inflowMargin, input.minObstacleSize, input.fixes);
        return metadata;
    } else {
        return {};
    }
}

} // namespace megamol::ImageSeries::filter
