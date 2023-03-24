#include "ImageSeriesFlowLabeler.h"
#include "imageseries/ImageSeries2DCall.h"
#include "imageseries/graph/GraphData2DCall.h"
#include "imageseries/util/AsyncData.h"

#include "vislib/graphics/PngBitmapCodec.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"

#include "../filter/FlowTimeLabelFilter.h"
#include "imageseries/util/ImageUtils.h"

using Log = megamol::core::utility::log::Log;
using Presentation = megamol::core::param::AbstractParamPresentation::Presentation;

namespace megamol::ImageSeries {

ImageSeriesFlowLabeler::ImageSeriesFlowLabeler()
        : getDataCallee("getData", "Returns data from the image series.")
        , getGraphCallee("getGraph", "Returns the constructed graph from the image series.")
        , getTimeMapCaller("requestTimeMapImageSeries", "Requests timestamp data from an image series.")

        , inflowAreaParam("Inflow area", "Set border where inflow is expected.")
        , inflowMarginParam("Inflow margin", "Margin from the sides of the image for inflow detection.")
        , velocityMethodParam("Velocity calculation", "Method used for calculating flow velocity.")
        , outputImageParam("Image output", "Select output image.")

        , isolatedParam("fixes::Remove isolated nodes", "Remove isolated nodes, which result from noise.")
        , falseSourcesParam("fixes::Remove unexpected sources",
              "Remove false sources, which result from noise, as well as connected nodes.")
        , falseSinksParam("fixes::Remove false sinks", "Remove false sinks, where neighbors have a higher time value.")

        , combineTrivialParam("simplification::Combine trivial nodes",
              "Combine 1-to-1 connected nodes, as these do not provide any valuable information.")
        , removeTrivialParam("simplification::Remove trivial nodes",
              "Remove 1-to-1 connected nodes, as these do not provide any valuable information.")
        , resolveDiamondsParam("simplification::Resolve diamond patterns",
              "Resolve diamond patterns, as these usually result from small local velocities.")
        , minAreaParam("simplification::Minimum area", "Minimum area, used for combining small areas.")
        , keepBreakthroughNodesParam("simplification::Keep breakthrough nodes",
              "Keep breakthrough nodes, although they would be removed using other fixes.")

        , outputGraphsParam("output::Write graphs to file", "Option to write the (intermediate) graphs to file.")
        , outputLabelImagesParam(
              "output::Write label images to file", "Option to write the (intermediate) label images to file.")
        , outputTimeImagesParam("output::Write time image to file", "Option to write the input time image to file.")
        , outputPathParam("output::Path", "Path where files for above options are written.")

        , imageCache([](const AsyncImageData2D<filter::FlowTimeLabelFilter::Output>& imageData) {
            return imageData.getByteSize();
        }) {

    getTimeMapCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getTimeMapCaller);

    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetData), &ImageSeriesFlowLabeler::getDataCallback);
    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetMetaData),
        &ImageSeriesFlowLabeler::getMetaDataCallback);
    MakeSlotAvailable(&getDataCallee);

    getGraphCallee.SetCallback(GraphData2DCall::ClassName(),
        GraphData2DCall::FunctionName(GraphData2DCall::CallGetData), &ImageSeriesFlowLabeler::getDataCallback);
    MakeSlotAvailable(&getGraphCallee);

    // General parameters
    inflowAreaParam << new core::param::EnumParam(0);
    inflowAreaParam.Param<core::param::EnumParam>()->SetTypePair(0, "left");
    inflowAreaParam.Param<core::param::EnumParam>()->SetTypePair(1, "bottom");
    inflowAreaParam.Param<core::param::EnumParam>()->SetTypePair(2, "right");
    inflowAreaParam.Param<core::param::EnumParam>()->SetTypePair(3, "top");
    inflowAreaParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&inflowAreaParam);

    inflowMarginParam << new core::param::IntParam(5, 1, 100);
    inflowMarginParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    inflowMarginParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&inflowMarginParam);

    velocityMethodParam << new core::param::EnumParam(0);
    velocityMethodParam.Param<core::param::EnumParam>()->SetTypePair(0, "Centers of mass");
    velocityMethodParam.Param<core::param::EnumParam>()->SetTypePair(1, "Hausdorff distance");
    velocityMethodParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&velocityMethodParam);

    outputImageParam << new core::param::EnumParam(0);
    outputImageParam.Param<core::param::EnumParam>()->SetTypePair(0, "After graph simplification");
    outputImageParam.Param<core::param::EnumParam>()->SetTypePair(1, "Original with invalid labels");
    outputImageParam.Param<core::param::EnumParam>()->SetTypePair(2, "Original labels");
    outputImageParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&outputImageParam);

    // Parameters for fixes
    isolatedParam << new core::param::BoolParam(true);
    isolatedParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&isolatedParam);

    falseSourcesParam << new core::param::BoolParam(true);
    falseSourcesParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&falseSourcesParam);

    falseSinksParam << new core::param::BoolParam(true);
    falseSinksParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&falseSinksParam);

    // Parameters for simplifications
    combineTrivialParam << new core::param::BoolParam(false);
    combineTrivialParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&combineTrivialParam);

    removeTrivialParam << new core::param::BoolParam(false);
    removeTrivialParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&removeTrivialParam);

    resolveDiamondsParam << new core::param::BoolParam(false);
    resolveDiamondsParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&resolveDiamondsParam);

    minAreaParam << new core::param::IntParam(10, 1, 1000);
    minAreaParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    minAreaParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&minAreaParam);

    keepBreakthroughNodesParam << new core::param::BoolParam(true);
    keepBreakthroughNodesParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&keepBreakthroughNodesParam);

    // Parameters for output
    outputGraphsParam << new core::param::BoolParam(false);
    outputGraphsParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&outputGraphsParam);

    outputLabelImagesParam << new core::param::BoolParam(false);
    outputLabelImagesParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&outputLabelImagesParam);

    outputTimeImagesParam << new core::param::BoolParam(false);
    outputTimeImagesParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&outputTimeImagesParam);

    outputPathParam << new core::param::FilePathParam("", core::param::FilePathParam::Flag_Directory_ToBeCreated);
    outputPathParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&outputPathParam);

    // Set default image cache size to 512 MB
    imageCache.setMaximumSize(512 * 1024 * 1024);
}

ImageSeriesFlowLabeler::~ImageSeriesFlowLabeler() {
    Release();
}

bool ImageSeriesFlowLabeler::create() {
    filterRunner = std::make_unique<filter::AsyncFilterRunner<AsyncImageData2D<filter::FlowTimeLabelFilter::Output>>>();
    return true;
}

void ImageSeriesFlowLabeler::release() {
    filterRunner = nullptr;
    imageCache.clear();
}

bool ImageSeriesFlowLabeler::getDataCallback(core::Call& caller) {
    filter::FlowTimeLabelFilter::Input filterInput;
    filterInput.outputImage = static_cast<filter::FlowTimeLabelFilter::Input::image_t>(
        outputImageParam.Param<core::param::EnumParam>()->Value());
    filterInput.inflowArea = static_cast<filter::FlowTimeLabelFilter::Input::inflow_t>(
        inflowAreaParam.Param<core::param::EnumParam>()->Value());
    filterInput.inflowMargin = inflowMarginParam.Param<core::param::IntParam>()->Value();
    filterInput.minArea = minAreaParam.Param<core::param::IntParam>()->Value();
    filterInput.hausdorff = velocityMethodParam.Param<core::param::EnumParam>()->Value() == 1;

    using bool_pt = core::param::BoolParam;
    using fixes_t = filter::FlowTimeLabelFilter::Input ::fixes_t;

    filterInput.fixes =
        (isolatedParam.Param<bool_pt>()->Value() ? fixes_t::isolated : fixes_t::nope) |
        (falseSourcesParam.Param<bool_pt>()->Value() ? fixes_t::false_sources : fixes_t::nope) |
        (falseSinksParam.Param<bool_pt>()->Value() ? fixes_t::false_sinks : fixes_t::nope) |
        (resolveDiamondsParam.Param<bool_pt>()->Value() ? fixes_t::resolve_diamonds : fixes_t::nope) |
        (combineTrivialParam.Param<bool_pt>()->Value() ? fixes_t::combine_trivial : fixes_t::nope) |
        (removeTrivialParam.Param<bool_pt>()->Value() ? fixes_t::remove_trivial : fixes_t::nope) |
        (keepBreakthroughNodesParam.Param<bool_pt>()->Value() ? fixes_t::keep_breakthrough_nodes : fixes_t::nope);

    filterInput.outputGraphs = outputGraphsParam.Param<bool_pt>()->Value();
    filterInput.outputLabelImages = outputLabelImagesParam.Param<bool_pt>()->Value();
    filterInput.outputTimeImages = outputTimeImagesParam.Param<bool_pt>()->Value();
    filterInput.outputPath = outputPathParam.Param<core::param::FilePathParam>()->Value();

    // Try to get time map
    if (auto* getTimeMap = getTimeMapCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
        getTimeMap->SetInput(ImageSeries2DCall::Input{0});
        if ((*getTimeMap)(ImageSeries::ImageSeries2DCall::CallGetData)) {
            ImageSeries2DCall::Output timeMap = getTimeMap->GetOutput();

            filterInput.timeMap = timeMap.imageData;

            auto output = imageCache.findOrCreate(
                timeMap.getHash(), [=](typename AsyncImageData2D<filter::FlowTimeLabelFilter::Output>::Hash) {
                    return filterRunner->run<filter::FlowTimeLabelFilter>(filterInput);
                });
        }
    }

    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        const auto hash = getTimeMapCaller.CallAs<ImageSeries::ImageSeries2DCall>()->GetOutput().getHash();

        auto output = imageCache.find(hash);

        if (output != nullptr) {
            auto intermediate =
                std::make_shared<AsyncImageData2D<>>([output]() { return output->getImageData()->image; },
                    filter::FlowTimeLabelFilter(filterInput).getMetadata());

            ImageSeries2DCall::Output timeMap;
            timeMap.imageData = intermediate;

            call->SetOutput(timeMap);

            return true;
        }
    } else if (auto* call = dynamic_cast<GraphData2DCall*>(&caller)) {
        const auto hash = getTimeMapCaller.CallAs<ImageSeries::ImageSeries2DCall>()->GetOutput().getHash();

        auto output = imageCache.find(hash);

        if (output != nullptr) {
            auto intermediate = std::make_shared<graph::AsyncGraphData2D>(
                [output]() { return output->getImageData()->graph; }, 0); // TODO: size of graph?

            call->SetOutput(GraphData2DCall::Output{intermediate});
            call->SetDataHash(output->getHash());

            return true;
        }
    }

    return false;
}

bool ImageSeriesFlowLabeler::getMetaDataCallback(core::Call& caller) {
    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        if (auto* getInput = getTimeMapCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
            if ((*getInput)(ImageSeries::ImageSeries2DCall::CallGetMetaData)) {
                // Pass through metadata
                call->SetOutput(getInput->GetOutput());
                return true;
            }
        }
    }
    return false;
}

bool ImageSeriesFlowLabeler::filterParametersChangedCallback(core::param::ParamSlot& param) {
    imageCache.clear();
    return true;
}


} // namespace megamol::ImageSeries
