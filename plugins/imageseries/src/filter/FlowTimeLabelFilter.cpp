#include "FlowTimeLabelFilter.h"

#include "mmcore/utility/log/Log.h"

#include "imageseries/graph/GraphData2D.h"

#include "vislib/graphics/BitmapImage.h"
#include "vislib/graphics/PngBitmapCodec.h"

#include "../util/GraphLuaExporter.h"
#include "../util/GraphGDFExporter.h"

#include <array>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <list>
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

    std::vector<Label> imageOrig, imageFixed, imageSimplified;

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

    // Calculate global velocity distribution as a measure of pixels added per frame
    std::map<Timestamp, double> pixelsPerFrame;

    for (Index index = 0; index < size; ++index) {
        if (dataIn[index] != 0 && dataIn[index] != std::numeric_limits<Timestamp>::max()) {
            if (pixelsPerFrame.find(dataIn[index]) == pixelsPerFrame.end()) {
                pixelsPerFrame[dataIn[index]] = 0.0;
            }

            pixelsPerFrame.at(dataIn[index]) += 1.0;
        }
    }

    // Assign unique labels to connected areas of same time
    auto nodeGraph = std::make_shared<graph::GraphData2D>();
    std::map<Timestamp, std::vector<graph::GraphData2D::NodeID>> nodesByTime;

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

    auto floodFill = [&](const Index index, const Label label) {
        floodQueue.clear();
        floodQueue.push_back(index);

        graph::GraphData2D::Node current_region(dataIn[index], label);
        current_region.pixels.push_back(index);

        dataOut[index] = label;

        for (std::size_t queueIndex = 0; queueIndex < floodQueue.size(); ++queueIndex) {
            const auto currentIndex = floodQueue[queueIndex];

            const auto x = currentIndex % width;
            const auto y = currentIndex / width;

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
                            interfaceFluidOut[neighborIndex] = interfaceOut[currentIndex] = current_region.getFrameIndex();
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

        nodesByTime[current_region.getFrameIndex()].push_back(nodeGraph->addNode(current_region));
    };

    Label next_label = LabelMinimum;

    for (Index index = 0; index < size; ++index) {
        if (dataOut[index] == LabelUnassigned && next_label <= LabelMaximum) {
            floodFill(index, next_label++);
        }
    }

    if (next_label > LabelMaximum) {
        core::utility::log::Log::DefaultLog.WriteError("[FlowTimeLabelFilter]: Too many labels! Consider denoising the input images.");

        // return black image and empty graph
        auto output = std::make_shared<Output>();
        output->image =
            std::make_shared<Image>(image->Width(), image->Height(), 1, Image::ChannelType::CHANNELTYPE_WORD);
        output->graph = std::make_shared<graph::GraphData2D>();
        return output;
    }

    auto printNode = [&nodeGraph](graph::GraphData2D::NodeID id) {
        const auto& node = nodeGraph->getNode(id);

        core::utility::log::Log::DefaultLog.WriteInfo(
            "ID: %d\nLabel: %d\nFrame: %d\nNum pixels: %d\nArea: %.0f\nInterface fluid: "
            "%.0f\nInterface solid: %.0f\nCenter: (%.0f, %.0f)\nBbox: (%d, %d, %d, "
            "%d)\nInterfaces: %d\nIncoming: %d\nOutgoing: %d",
            id, node.getLabel(), node.getFrameIndex(), node.pixels.size(), node.area, node.interfaceFluid,
            node.interfaceSolid, node.centerOfMass.x, node.centerOfMass.y, node.boundingBox.x1, node.boundingBox.y1,
            node.boundingBox.x2, node.boundingBox.y2, node.interfaces.size(), node.getEdgeCountIn(),
            node.getEdgeCountOut());
    };

    // Save image content for file dump
    imageOrig.resize(size);
    std::memcpy(imageOrig.data(), dataOut, size * sizeof(Label));

    // Calculate quantities for each flow front
    for (auto& node_info : nodeGraph->getNodes()) {
        auto& node = node_info.second;
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
    }

    // Create edges by iterating over flow fronts with time monotonically increasing
    for (const auto& flow_fronts : nodesByTime) {
        for (const auto flow_front_ID : flow_fronts.second) {
            const auto& flow_front = nodeGraph->getNode(flow_front_ID);

            // Find neighboring flow fronts and add edge if the neighboring front is (1) from the past and (2) the local maximum
            std::unordered_set<Label> past_neighboring_flow_fronts;
            Timestamp past_time{};

            for (auto it = flow_front.interfaces.crbegin(); it != flow_front.interfaces.crend(); ++it) {
                if (it->first != LabelSolid && it->first < flow_front.getFrameIndex()) {
                    past_time = it->first;

                    for (const auto index : it->second) {
                        past_neighboring_flow_fronts.insert(dataOut[index]);
                    }

                    break;
                }
            }

            for (const auto& past_flow_front : past_neighboring_flow_fronts) {
                const auto& srcNode = nodeGraph->getNode(past_time, past_flow_front);

                graph::GraphData2D::Edge edge(
                    srcNode.getID(), flow_front_ID, glm::distance(flow_front.centerOfMass, srcNode.centerOfMass));

                nodeGraph->addEdge(std::move(edge));
            }
        }
    }

    auto originalGraph = graph::GraphData2D(*nodeGraph);

    // Set nodes to be invalid if they meet specific criteria
    for (auto& node_info : nodeGraph->getNodes()) {
        auto& node = node_info.second;
        bool invalid = !node.valid;

        // (+) Remove isolated nodes, i.e., nodes that do not have incoming and outgoing connections
        if (input.fixes & Input::fixes_t::isolated) {
            invalid |= node.getEdgeCountIn() == 0 && node.getEdgeCountOut() == 0;
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

            invalid |= node.getEdgeCountIn() == 0 && std::abs(inflowRef) > input.inflowMargin;

            // (+) Remove nodes that have incoming edges only from invalid nodes, as these were originally connected
            //     to now removed sources, as per above directive
            node.valid = !invalid;

            std::list<graph::GraphData2D::NodeID> checkNodes(node.getChildNodes().begin(), node.getChildNodes().end());
            for (auto it = checkNodes.begin(); it != checkNodes.end(); checkNodes.erase(it++)) {
                auto& node = nodeGraph->getNode(*it);

                bool is_valid = false;
                for (const auto& parent : node.getParentNodes()) {
                    is_valid |= nodeGraph->getNode(parent).valid;
                }

                if (!is_valid || !node.valid) {
                    node.valid = false;
                    checkNodes.insert(checkNodes.end(), node.getChildNodes().begin(), node.getChildNodes().end());
                }
            }
        }

        // (+) Remove sinks that have neighbors with larger time step value, as these cannot be real sinks
        if (input.fixes & Input::fixes_t::false_sinks && node.getEdgeCountOut() == 0) {
            for (auto it = node.interfaces.crbegin(); it != node.interfaces.crend(); ++it) {
                if (it->first != LabelSolid && it->first <= LabelMaximum && it->first > node.getFrameIndex()) {
                    invalid |= true;

                    break;
                }
            }
        }

        node.valid = !invalid;
    }

    // Mark pixels of invalid nodes as invalid
    if (input.outputImage == Input::image_t::invalid || input.outputImage == Input::image_t::simplified) {
        for (const auto& node_info : nodeGraph->getNodes()) {
            const auto& node = node_info.second;

            if (!node.valid) {
                for (const auto& pixel : node.pixels) {
                    dataOut[pixel] = LabelInvalid;
                }
            }
        }
    }

    // Save image content for file dump
    imageFixed.resize(size);
    std::memcpy(imageFixed.data(), dataOut, size * sizeof(Label));

    // Remove non-valid nodes
    for (const auto& node_info : nodeGraph->getNodes()) {
        const auto& node = node_info.second;

        if (!node.valid) {
            nodeGraph->removeNode(node_info.first, true);
        }
    }

    nodeGraph->finalizeLazyRemoval();

    auto fixedGraph = graph::GraphData2D(*nodeGraph);

    // Compute velocities
    computeVelocities(originalGraph);
    computeVelocities(fixedGraph);
    computeVelocities(*nodeGraph);

    // Combine tiny areas that result most likely from very small local velocities
    if (input.fixes & Input::fixes_t::combine_tiny) {
        while (combineSmallNodes(*nodeGraph, next_label, input.minArea)) {
            // Every iteration, combine nodes that have edges in both directions, which
            // is a side effect of greedily combining neighboring nodes
            for (const auto& node_info : nodeGraph->getNodes()) {
                const auto nodeID = node_info.first;
                const auto& node = node_info.second;

                if (!node.isRemoved()) {
                    std::vector<graph::GraphData2D::NodeID> nodeIDsToCombine;
                    nodeIDsToCombine.push_back(nodeID);

                    for (const auto parentID : node.getParentNodes()) {
                        if (nodeGraph->hasEdge(nodeID, parentID)) {
                            nodeIDsToCombine.push_back(parentID);
                        }
                    }
                    for (const auto childID : node.getChildNodes()) {
                        if (nodeGraph->hasEdge(childID, nodeID)) {
                            nodeIDsToCombine.push_back(childID);
                        }
                    }

                    if (nodeIDsToCombine.size() > 1) {
                        std::vector<graph::GraphData2D::Node> nodesToCombine;
                        nodesToCombine.reserve(nodeIDsToCombine.size());

                        for (const auto& nodeID : nodeIDsToCombine) {
                            nodesToCombine.push_back(nodeGraph->removeNode(nodeID, true));
                        }

                        // Modify graph
                        const auto newNodeID = nodeGraph->addNode(combineNodes(nodesToCombine, next_label));
                        const auto newEdges = combineEdges(*nodeGraph, nodeIDsToCombine, newNodeID);

                        for (const auto& newEdge : newEdges) {
                            nodeGraph->addEdge(newEdge);
                        }
                    }
                }
            }

            nodeGraph->finalizeLazyRemoval();
        }
    }

    // Get landmark times
    auto startTime = std::numeric_limits<int>::max();
    auto breakthroughTime = std::numeric_limits<int>::max();
    auto endTime = 0;

    for (const auto& node_info : nodeGraph->getNodes()) {
        const auto& node = node_info.second;

        startTime = std::min(startTime, static_cast<int>(node.getFrameIndex()));
        endTime = std::max(endTime, static_cast<int>(node.getFrameIndex()));

        int outflowRef = 0;
        switch (input.inflowArea) {
        case Input::inflow_t::left:
            outflowRef = node.boundingBox.x2 - input.timeMap->getMetadata().width;
            break;
        case Input::inflow_t::bottom:
            outflowRef = node.boundingBox.y2 - input.timeMap->getMetadata().height;
            break;
        case Input::inflow_t::right:
            outflowRef = node.boundingBox.x1;
            break;
        case Input::inflow_t::top:
            outflowRef = node.boundingBox.y1;
            break;
        }

        if (std::abs(outflowRef) <= input.inflowMargin) {
            breakthroughTime = std::min(breakthroughTime, static_cast<int>(node.getFrameIndex()));
        }
    }

    breakthroughTime = std::clamp(breakthroughTime, startTime, endTime);

    const auto keepBreakthroughTime = (input.fixes & Input::fixes_t::keep_breakthrough_nodes) ? breakthroughTime : -1;

    // Iteratively improve graph
    if (input.fixes & (Input::fixes_t::combine_trivial | Input::fixes_t::remove_trivial)) {
        combineTrivialNodes(*nodeGraph, next_label, keepBreakthroughTime);
    }

    if (input.fixes & Input::fixes_t::resolve_diamonds) {
        if (input.fixes & (Input::fixes_t::combine_trivial | Input::fixes_t::remove_trivial)) {
            combineTrivialNodes(*nodeGraph, next_label, keepBreakthroughTime);
        }

        while (resolveDiamonds(*nodeGraph, next_label, keepBreakthroughTime)) {
            if (input.fixes & (Input::fixes_t::combine_trivial | Input::fixes_t::remove_trivial)) {
                combineTrivialNodes(*nodeGraph, next_label, keepBreakthroughTime);
            }
        }
    }

    // Update pixels to match the resulting simplified graph
    if (input.outputImage == Input::image_t::simplified) {
        for (const auto& node_info : nodeGraph->getNodes()) {
            const auto& node = node_info.second;

            for (const auto& pixel : node.pixels) {
                dataOut[pixel] = node.getLabel();
            }
        }
    }

    // Improve graph (but do this after updating the label image, as nodes are being removed
    // without substitution, thus leading to the previous version of the image being reused)
    if (input.fixes & Input::fixes_t::remove_trivial) {
        removeTrivialNodes(*nodeGraph, next_label, keepBreakthroughTime);
    }

    auto simplifiedGraph = graph::GraphData2D(*nodeGraph);

    // Save image content for file dump
    imageSimplified.resize(size);
    std::memcpy(imageSimplified.data(), dataOut, size * sizeof(Label));

    // Compute velocity distributions
    auto getVelocities = [](const graph::GraphData2D& graph) {
        std::map<Timestamp, double> weightedVelocity, weight;

        for (const auto& node : graph.getNodes()) {
            const auto time = node.second.getFrameIndex();

            if (weightedVelocity.find(time) == weightedVelocity.end()) {
                weightedVelocity[time] = 0.0;
                weight[time] = 0.0;
            }

            weightedVelocity.at(time) += static_cast<double>(node.second.velocityMagnitude) * node.second.area;
            weight.at(time) += node.second.area;
        }

        for (auto& velocity : weightedVelocity) {
            velocity.second /= weight.at(velocity.first);
        }

        return weightedVelocity;
    };

    const auto numTimesteps = static_cast<std::size_t>(endTime) - startTime + 1;

    const auto origVelocities = getVelocities(originalGraph);
    const auto fixedVelocities = getVelocities(fixedGraph);
    const auto simplifiedVelocities = getVelocities(simplifiedGraph);

    std::vector<std::size_t> origGraphDistribution(numTimesteps, 0uLL);
    std::vector<std::size_t> fixedGraphDistribution(numTimesteps, 0uLL);
    std::vector<std::size_t> simplifiedGraphDistribution(numTimesteps, 0uLL);

    for (std::size_t t = startTime; t <= endTime; ++t) {
        if (origVelocities.find(t) != origVelocities.end()) {
            origGraphDistribution[t - startTime] = static_cast<std::size_t>(std::floor(origVelocities.at(t)));
        }
        if (fixedVelocities.find(t) != fixedVelocities.end()) {
            fixedGraphDistribution[t - startTime] = static_cast<std::size_t>(std::floor(fixedVelocities.at(t)));
        }
        if (simplifiedVelocities.find(t) != simplifiedVelocities.end()) {
            simplifiedGraphDistribution[t - startTime] =
                static_cast<std::size_t>(std::floor(simplifiedVelocities.at(t)));
        }
    }

    // Create output directory for file dumps
    if (input.outputPath.empty()) {
        input.outputPath = ".";
    }

    if (!input.outputPath.empty() && !std::filesystem::is_directory(input.outputPath)) {
        std::filesystem::create_directories(input.outputPath);
    }

    if (std::filesystem::is_directory(input.outputPath)) {
        sg::graphics::PngBitmapCodec png_codec;

        // Export to Lua and GDF files
        if (input.outputGraphs) {
            graph::util::LuaExportMeta luaExportMeta;
            luaExportMeta.path = input.timeMap->getMetadata().filename;
            luaExportMeta.minRange = 0.0f;
            luaExportMeta.maxRange = 1.0f;
            luaExportMeta.imgW = input.timeMap->getMetadata().width;
            luaExportMeta.imgH = input.timeMap->getMetadata().height;
            luaExportMeta.startTime = startTime;
            luaExportMeta.breakthroughTime = breakthroughTime;
            luaExportMeta.endTime = endTime;

            graph::util::exportToLua(originalGraph, (input.outputPath / "graph_0_original.lua").string(), luaExportMeta,
                origGraphDistribution);
            graph::util::exportToLua(
                fixedGraph, (input.outputPath / "graph_1_fixed.lua").string(), luaExportMeta, fixedGraphDistribution);
            graph::util::exportToLua(simplifiedGraph, (input.outputPath / "graph_2_simplified.lua").string(),
                luaExportMeta, simplifiedGraphDistribution);

            graph::util::GDFExportMeta gdfExportMeta;
            gdfExportMeta.startTime = startTime;
            gdfExportMeta.breakthroughTime = breakthroughTime;
            gdfExportMeta.endTime = endTime;
            gdfExportMeta.stopAtBreakthrough = false;

            graph::util::exportToGDF(
                originalGraph, (input.outputPath / "graph_0_original.gdf").string(), gdfExportMeta);
            graph::util::exportToGDF(fixedGraph, (input.outputPath / "graph_1_fixed.gdf").string(), gdfExportMeta);
            graph::util::exportToGDF(
                simplifiedGraph, (input.outputPath / "graph_2_simplified.gdf").string(), gdfExportMeta);

            gdfExportMeta.stopAtBreakthrough = true;

            graph::util::exportToGDF(
                originalGraph, (input.outputPath / "graph_0_original_bt.gdf").string(), gdfExportMeta);
            graph::util::exportToGDF(fixedGraph, (input.outputPath / "graph_1_fixed_bt.gdf").string(), gdfExportMeta);
            graph::util::exportToGDF(
                simplifiedGraph, (input.outputPath / "graph_2_simplified_bt.gdf").string(), gdfExportMeta);
        }

        // Output label images to hard disk
        if (input.outputLabelImages) {
            if (!imageOrig.empty()) {
                auto image =
                    std::make_shared<Image>(width, height, 1, Image::ChannelType::CHANNELTYPE_WORD, imageOrig.data());
                image->SetChannelLabel(0, vislib::graphics::BitmapImage::ChannelLabel::CHANNEL_GRAY);

                png_codec.Image() = image.get();
                png_codec.Save((input.outputPath / "label_0_original.png").string().c_str());
            }

            if (!imageFixed.empty()) {
                auto image =
                    std::make_shared<Image>(width, height, 1, Image::ChannelType::CHANNELTYPE_WORD, imageFixed.data());
                image->SetChannelLabel(0, vislib::graphics::BitmapImage::ChannelLabel::CHANNEL_GRAY);

                png_codec.Image() = image.get();
                png_codec.Save((input.outputPath / "label_1_fixed.png").string().c_str());
            }

            if (!imageSimplified.empty()) {
                auto image = std::make_shared<Image>(
                    width, height, 1, Image::ChannelType::CHANNELTYPE_WORD, imageSimplified.data());
                image->SetChannelLabel(0, vislib::graphics::BitmapImage::ChannelLabel::CHANNEL_GRAY);

                png_codec.Image() = image.get();
                png_codec.Save((input.outputPath / "label_2_simplified.png").string().c_str());
            }
        }

        // Output time images to hard disk
        if (input.outputTimeImages) {
            const auto basePath = input.outputPath / "time";

            auto filename(basePath);
            filename += ".png";

            auto timeImage = std::make_shared<Image>(*image);
            timeImage->SetChannelLabel(0, vislib::graphics::BitmapImage::ChannelLabel::CHANNEL_GRAY);

            auto* timeOut = timeImage->PeekDataAs<Timestamp>();

            for (std::size_t i = 0; i < size; ++i) {
                timeOut[i] = (static_cast<float>(timeOut[i] - startTime) / (endTime - startTime)) *
                             std::numeric_limits<Timestamp>::max();
            }

            png_codec.Image() = timeImage.get();
            png_codec.Save(filename.c_str());
        }

        // Output interface image to hard disk
        if (interface_output != interface_t::none) {
            std::string value;
            if (interface_output != interface_t::full) {
                std::stringstream valueStream;
                valueStream << "_" << static_cast<Timestamp>(interface_output);

                value += valueStream.str();
            }

            const auto basePath = input.outputPath / "interface";

            auto filenameFluid(basePath);
            filenameFluid += std::string("_fluid") + value + ".png";

            auto filenameSolid(basePath);
            filenameSolid += std::string("_solid") + value + ".png";

            auto filename(basePath);
            filename += value + ".png";

            png_codec.Image() = interfaceFluidImage.get();
            png_codec.Save(filenameFluid.c_str());
            png_codec.Image() = interfaceSolidImage.get();
            png_codec.Save(filenameSolid.c_str());
            png_codec.Image() = interfaceImage.get();
            png_codec.Save(filename.c_str());
        }
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
        metadata.hash = util::computeHash(input.timeMap, input.outputImage, input.inflowArea, input.inflowMargin,
            input.minArea, input.hausdorff, input.fixes);
        return metadata;
    } else {
        return {};
    }
}

graph::GraphData2D::Node FlowTimeLabelFilter::combineNodes(
    const std::vector<graph::GraphData2D::Node>& nodesToCombine, Label& nextLabel) const {

    Timestamp time = 0;
    for (const auto& node : nodesToCombine) {
        time += node.getFrameIndex();
    }
    time /= nodesToCombine.size();

    graph::GraphData2D::Node combinedNode(time, nextLabel++);

    for (const auto& node : nodesToCombine) {
        combinedNode.area += node.area;
        combinedNode.boundingBox.Union(node.boundingBox);
        combinedNode.centerOfMass += node.area * node.centerOfMass;
        combinedNode.velocity += node.area * node.velocity;
        combinedNode.pixels.insert(combinedNode.pixels.end(), node.pixels.begin(), node.pixels.end());
        combinedNode.interfaceSolid += node.interfaceSolid;

        for (const auto& fluid_interface : node.interfaces) {
            combinedNode.interfaces[fluid_interface.first].insert(
                fluid_interface.second.begin(), fluid_interface.second.end());
        }
    }

    combinedNode.centerOfMass /= combinedNode.area;
    combinedNode.velocity /= combinedNode.area;
    combinedNode.velocityMagnitude = glm::length(combinedNode.velocity);

    for (const auto& node : nodesToCombine) {
        combinedNode.interfaces.erase(node.getLabel());
    }

    for (const auto& fluid_interface : combinedNode.interfaces) {
        if (fluid_interface.first != LabelSolid) {
            combinedNode.interfaceFluid += fluid_interface.second.size();
        }
    }

    return combinedNode;
}

std::vector<graph::GraphData2D::Edge> FlowTimeLabelFilter::combineEdges(const graph::GraphData2D& nodeGraph,
    const std::vector<graph::GraphData2D::NodeID>& nodesToCombine, const graph::GraphData2D::NodeID newNodeID) const {

    std::vector<graph::GraphData2D::Edge> newEdges;

    const auto& newNode = nodeGraph.getNode(newNodeID);

    for (const auto nodeID : nodesToCombine) {
        const auto& node = nodeGraph.getNode(nodeID);

        for (const auto parentID : node.getParentNodes()) {
            const auto& neighbor = nodeGraph.getNode(parentID);

            if (!neighbor.isRemoved()) {
                graph::GraphData2D::Edge inEdge(parentID, newNodeID);
                inEdge.weight = glm::distance(newNode.centerOfMass, neighbor.centerOfMass);

                newEdges.push_back(inEdge);
            }
        }

        for (const auto childID : node.getChildNodes()) {
            const auto& neighbor = nodeGraph.getNode(childID);

            if (!neighbor.isRemoved()) {
                graph::GraphData2D::Edge outEdge(newNodeID, childID);
                outEdge.weight = glm::distance(newNode.centerOfMass, neighbor.centerOfMass);

                newEdges.push_back(outEdge);
            }
        }
    }

    return newEdges;
}

bool FlowTimeLabelFilter::combineSmallNodes(
    graph::GraphData2D& nodeGraph, Label& nextLabel, float tiny_area_threshold) const {

    bool has_changes = false;

    // Gather small nodes
    std::map<float, std::list<graph::GraphData2D::NodeID>> smallNodes;
    for (const auto& node_info : nodeGraph.getNodes()) {
        const auto nodeID = node_info.first;
        const auto& node = node_info.second;

        if (node.area < tiny_area_threshold) {
            smallNodes[node.area].push_back(nodeID);
        }
    }

    // Combine nodes, beginning with the smallest
    for (const auto& nodeIDs : smallNodes) {
        for (const auto nodeID : nodeIDs.second) {
            const auto& node = nodeGraph.getNode(nodeID);

            if (!node.isRemoved()) {
                float smallestNeighborArea = std::numeric_limits<float>::max();
                graph::GraphData2D::NodeID smallestNeighborID{};

                for (const auto parentID : node.getParentNodes()) {
                    const auto& neighbor = nodeGraph.getNode(parentID);

                    if (!neighbor.isRemoved() && neighbor.area < smallestNeighborArea) {
                        smallestNeighborArea = neighbor.area;
                        smallestNeighborID = parentID;
                    }
                }
                for (const auto childID : node.getChildNodes()) {
                    const auto& neighbor = nodeGraph.getNode(childID);

                    if (!neighbor.isRemoved() && neighbor.area < smallestNeighborArea) {
                        smallestNeighborArea = neighbor.area;
                        smallestNeighborID = childID;
                    }
                }

                if (smallestNeighborArea < std::numeric_limits<float>::max()) {
                    has_changes = true;

                    std::vector<graph::GraphData2D::Node> nodesToCombine;
                    nodesToCombine.push_back(nodeGraph.removeNode(nodeID, true));
                    nodesToCombine.push_back(nodeGraph.removeNode(smallestNeighborID, true));

                    // Modify graph
                    const auto newNodeID = nodeGraph.addNode(combineNodes(nodesToCombine, nextLabel));
                    const auto newEdges = combineEdges(nodeGraph, {nodeID, smallestNeighborID}, newNodeID);

                    for (const auto& newEdge : newEdges) {
                        nodeGraph.addEdge(newEdge);
                    }
                }
            }
        }
    }

    nodeGraph.finalizeLazyRemoval();

    return has_changes;
}

void FlowTimeLabelFilter::combineTrivialNodes(
    graph::GraphData2D& nodeGraph, Label& nextLabel, const int breakthroughTime) const {
    // Simplify graph by combining subsequent nodes of 1-to-1 connections
    for (const auto& node_info : nodeGraph.getNodes()) {
        const auto nodeID = node_info.first;
        const auto& node = node_info.second;

        if (!node.isRemoved() && node.getEdgeCountIn() == 1 && node.getEdgeCountOut() == 1 &&
            node.getFrameIndex() != breakthroughTime) {

            std::list<graph::GraphData2D::NodeID> nodeIDsToCombine;
            nodeIDsToCombine.insert(nodeIDsToCombine.begin(), nodeID);

            // Create list of subsequent 1-to-1 connected nodes
            auto parentID = *node.getParentNodes().begin();
            do {
                const auto& parent = nodeGraph.getNode(parentID);
                if (!parent.isRemoved() && parent.getEdgeCountIn() == 1 && parent.getEdgeCountOut() == 1 &&
                    parent.getFrameIndex() != breakthroughTime) {

                    nodeIDsToCombine.insert(nodeIDsToCombine.begin(), parentID);
                    parentID = *parent.getParentNodes().begin();
                } else {
                    break;
                }
            } while (true);

            auto childID = *node.getChildNodes().begin();
            do {
                const auto& child = nodeGraph.getNode(childID);
                if (!child.isRemoved() && child.getEdgeCountIn() == 1 && child.getEdgeCountOut() == 1 &&
                    child.getFrameIndex() != breakthroughTime) {

                    nodeIDsToCombine.insert(nodeIDsToCombine.end(), childID);
                    childID = *child.getChildNodes().begin();
                } else {
                    break;
                }
            } while (true);

            // Combine nodes
            if (nodeIDsToCombine.size() > 1) {
                std::vector<graph::GraphData2D::Node> nodesToCombine;
                nodesToCombine.reserve(nodeIDsToCombine.size());

                for (const auto& nodeID : nodeIDsToCombine) {
                    nodesToCombine.push_back(nodeGraph.removeNode(nodeID, true));
                }

                // Calculate sum of distances between nodes
                float distance = 0.0f;
                for (auto it = nodeIDsToCombine.begin(); it != std::prev(nodeIDsToCombine.end()); ++it) {
                    distance += glm::distance(
                        nodeGraph.getNode(*it).centerOfMass, nodeGraph.getNode(*std::next(it)).centerOfMass);
                }

                // Modify graph
                const auto newNodeID = nodeGraph.addNode(combineNodes(nodesToCombine, nextLabel));

                graph::GraphData2D::Edge inEdge(*nodesToCombine.front().getParentNodes().begin(), newNodeID);
                inEdge.weight = nodeGraph.getEdge(inEdge.from, nodeIDsToCombine.front()).weight + distance / 2.0f;

                graph::GraphData2D::Edge outEdge(newNodeID, *nodesToCombine.back().getChildNodes().begin());
                outEdge.weight = nodeGraph.getEdge(nodeIDsToCombine.back(), outEdge.to).weight + distance / 2.0f;

                nodeGraph.addEdge(inEdge);
                nodeGraph.addEdge(outEdge);
            }
        }
    }

    nodeGraph.finalizeLazyRemoval();
}

bool FlowTimeLabelFilter::resolveDiamonds(
    graph::GraphData2D& nodeGraph, Label& nextLabel, const int breakthroughTime) const {

    bool has_changes = false;

    // Simplify graph by resolving diamond patterns by combining parallel 1-to-1 connected nodes.
    // Resolve diamond patterns if and only if the edges between nodes involved are
    // below the user-defined threshold for minimum obstacle size
    for (const auto& node_info : nodeGraph.getNodes()) {
        const auto nodeID = node_info.first;
        const auto& node = node_info.second;

        if (!node.isRemoved() && node.getEdgeCountIn() >= 1 && node.getEdgeCountOut() >= 1) {
            const auto& nodesIn = node.getParentNodes();
            const auto& nodesOut = node.getChildNodes();

            for (auto nodeIn : nodesIn) {
                for (auto nodeOut : nodesOut) {
                    if (nodeGraph.hasEdge(nodeIn, nodeID) && nodeGraph.hasEdge(nodeID, nodeOut)) {
                        const auto& edgeIn = nodeGraph.getEdge(nodeIn, nodeID);
                        const auto& edgeOut = nodeGraph.getEdge(nodeID, nodeOut);

                        if (nodeGraph.hasEdge(nodeIn, nodeOut)) {
                            const auto& directEdge = nodeGraph.getEdge(nodeIn, nodeOut);

                            if (edgeOut.weight > directEdge.weight && node.getEdgeCountOut() > 1) {
                                nodeGraph.removeEdge(edgeOut);
                            } else {
                                nodeGraph.removeEdge(directEdge);
                            }

                            has_changes = true;
                        }
                    }
                }
            }
        }

        if (!node.isRemoved() && node.getEdgeCountIn() == 1 && node.getEdgeCountOut() == 1) {
            const auto& edgeIn = nodeGraph.getEdge(*node.getParentNodes().begin(), nodeID);
            const auto& edgeOut = nodeGraph.getEdge(nodeID, *node.getChildNodes().begin());

            const auto& origin = nodeGraph.getNode(edgeIn.from);
            const auto& target = nodeGraph.getNode(edgeOut.to);

            std::vector<graph::GraphData2D::NodeID> nodeIDsToCombine;

            for (const auto& originChild : origin.getChildNodes()) {
                if (target.getParentNodes().find(originChild) != target.getParentNodes().end()) {
                    const auto& otherNode = nodeGraph.getNode(originChild);

                    if (!otherNode.isRemoved() && otherNode.getEdgeCountIn() == 1 && otherNode.getEdgeCountOut() == 1 &&
                        otherNode.getFrameIndex() != breakthroughTime) {

                        nodeIDsToCombine.push_back(originChild);
                    }
                }
            }

            // Combine nodes
            if (nodeIDsToCombine.size() > 1) {
                std::vector<graph::GraphData2D::Node> nodesToCombine;
                nodesToCombine.reserve(nodeIDsToCombine.size());

                float distance_in = 0.0f;
                float distance_out = 0.0f;

                for (const auto& nodeID : nodeIDsToCombine) {
                    nodesToCombine.push_back(nodeGraph.removeNode(nodeID, true));

                    distance_in += nodeGraph.getEdge(edgeIn.from, nodeID).weight;
                    distance_out += nodeGraph.getEdge(nodeID, edgeOut.to).weight;
                }

                distance_in /= nodeIDsToCombine.size();
                distance_out /= nodeIDsToCombine.size();

                // Modify graph
                const auto newNodeID = nodeGraph.addNode(combineNodes(nodesToCombine, nextLabel));

                nodeGraph.addEdge(graph::GraphData2D::Edge(edgeIn.from, newNodeID, distance_in));
                nodeGraph.addEdge(graph::GraphData2D::Edge(newNodeID, edgeOut.to, distance_out));

                has_changes = true;
            }
        }
    }

    nodeGraph.finalizeLazyRemoval();

    return has_changes;
}

void FlowTimeLabelFilter::removeTrivialNodes(
    graph::GraphData2D& nodeGraph, Label& nextLabel, const int breakthroughTime) const {
    // Simplify graph by combining subsequent nodes of 1-to-1 connections
    for (const auto& node_info : nodeGraph.getNodes()) {
        const auto nodeID = node_info.first;
        const auto& node = node_info.second;

        if (!node.isRemoved() && node.getEdgeCountIn() == 1 && node.getEdgeCountOut() == 1 &&
            node.getFrameIndex() != breakthroughTime) {

            const auto oldNode = nodeGraph.removeNode(nodeID, true);

            const auto parentID = *oldNode.getParentNodes().begin();
            const auto childID = *oldNode.getChildNodes().begin();

            graph::GraphData2D::Edge edge(parentID, childID);
            edge.weight = nodeGraph.getEdge(parentID, nodeID).weight + nodeGraph.getEdge(nodeID, childID).weight;

            nodeGraph.addEdge(edge);
        }
    }

    nodeGraph.finalizeLazyRemoval();
}

void FlowTimeLabelFilter::computeVelocities(graph::GraphData2D& nodeGraph) const {
    for (auto& node_info : nodeGraph.getNodes()) {
        auto& node = node_info.second;

        node.velocity = glm::vec2{0, 0};
        node.velocityMagnitude = 0.0f;

        if (input.hausdorff) {
            // TODO
        } else {
            for (const auto parentID : node.getParentNodes()) {
                const auto& parent = nodeGraph.getNode(parentID);

                node.velocity += (node.centerOfMass - parent.centerOfMass) /
                                 static_cast<float>(node.getFrameIndex() - parent.getFrameIndex());

                node.velocityMagnitude += glm::length(node.centerOfMass - parent.centerOfMass) /
                                          std::abs(static_cast<float>(node.getFrameIndex() - parent.getFrameIndex()));
            }
            for (const auto childID : node.getChildNodes()) {
                const auto& child = nodeGraph.getNode(childID);

                node.velocity += (node.centerOfMass - child.centerOfMass) /
                                 static_cast<float>(node.getFrameIndex() - child.getFrameIndex());

                node.velocityMagnitude += glm::length(node.centerOfMass - child.centerOfMass) /
                                          std::abs(static_cast<float>(node.getFrameIndex() - child.getFrameIndex()));
            }
        }

        node.velocityMagnitude = glm::length(node.velocity);
    }
}

} // namespace megamol::ImageSeries::filter
