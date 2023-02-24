#include "ImageSeriesFlowLabeler.h"
#include "imageseries/ImageSeries2DCall.h"
#include "imageseries/graph/GraphData2DCall.h"
#include "imageseries/util/AsyncData.h"

#include "vislib/graphics/PngBitmapCodec.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"

#include "../filter/FlowTimeLabelFilter.h"
#include "imageseries/util/ImageUtils.h"

using Log = megamol::core::utility::log::Log;
using Presentation = megamol::core::param::AbstractParamPresentation::Presentation;

namespace megamol::ImageSeries {

ImageSeriesFlowLabeler::ImageSeriesFlowLabeler()
        : getDataCallee("getData", "Returns data from the image series.")
        , getGraphCallee("getGraph", "Returns the constructed graph from the image series.")
        , getTimeMapCaller("requestTimeMapImageSeries", "Requests timestamp data from an image series.")
        , outputImageParam("Image output", "Select output image.")
        , inflowAreaParam("Inflow area", "Set border where inflow is expected.")
        , inflowMarginParam("Inflow margin", "Margin from the sides of the image for inflow detection.")
        , minAreaParam("Minimum area", "Minimum area, used for combining small areas.")
        , velocityMethodParam("Velocity calculation", "Method used for calculating flow velocity.")
        , isolatedParam("Remove isolated nodes", "Remove isolated nodes, which result from noise.")
        , falseSourcesParam("Remove unexpected sources", "Remove false sources, which result from noise, as well as connected nodes.")
        , falseSinksParam("Remove false sinks", "Remove false sinks, where neighbors have a higher time value.")
        , resolveDiamondsParam("Resolve diamond patterns", "Resolve diamond patterns, as these usually result from small local velocities.")
        , combineTrivialParam("Combine trivial nodes", "Combine 1-to-1 connected nodes, as these do not provide any valuable information.")
        , combineTinyParam("Combine small areas", "Combine small areas, as these usually result from small local velocities.")
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

    outputImageParam << new core::param::EnumParam(0);
    outputImageParam.Param<core::param::EnumParam>()->SetTypePair(0, "After graph simplification");
    outputImageParam.Param<core::param::EnumParam>()->SetTypePair(1, "Original with invalid labels");
    outputImageParam.Param<core::param::EnumParam>()->SetTypePair(2, "Original labels");
    outputImageParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&outputImageParam);

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

    minAreaParam << new core::param::IntParam(10, 1, 1000);
    minAreaParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    minAreaParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&minAreaParam);

    velocityMethodParam << new core::param::EnumParam(0);
    velocityMethodParam.Param<core::param::EnumParam>()->SetTypePair(0, "Centers of mass");
    velocityMethodParam.Param<core::param::EnumParam>()->SetTypePair(1, "Hausdorff distance");
    velocityMethodParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&velocityMethodParam);

    isolatedParam << new core::param::BoolParam(true);
    isolatedParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&isolatedParam);

    falseSourcesParam << new core::param::BoolParam(true);
    falseSourcesParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&falseSourcesParam);

    falseSinksParam << new core::param::BoolParam(true);
    falseSinksParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&falseSinksParam);

    combineTinyParam << new core::param::BoolParam(false);
    combineTinyParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&combineTinyParam);

    combineTrivialParam << new core::param::BoolParam(false);
    combineTrivialParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&combineTrivialParam);

    resolveDiamondsParam << new core::param::BoolParam(false);
    resolveDiamondsParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&resolveDiamondsParam);

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
    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        // Try to get time map
        if (auto* getTimeMap = getTimeMapCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
            getTimeMap->SetInput(call->GetInput());
            if ((*getTimeMap)(ImageSeries::ImageSeries2DCall::CallGetData)) {
                ImageSeries2DCall::Output timeMap = getTimeMap->GetOutput();

                filter::FlowTimeLabelFilter::Input filterInput;
                filterInput.timeMap = timeMap.imageData;
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
                    (combineTinyParam.Param<bool_pt>()->Value() ? fixes_t::combine_tiny : fixes_t::nope);

                auto output = imageCache.findOrCreate(
                    timeMap.getHash(), [=](typename AsyncImageData2D<filter::FlowTimeLabelFilter::Output>::Hash) {
                    return filterRunner->run<filter::FlowTimeLabelFilter>(filterInput);
                });

                auto intermediate =
                    std::make_shared<AsyncImageData2D<>>([output]() { return output->getImageData()->image; },
                        filter::FlowTimeLabelFilter(filterInput).getMetadata());

                timeMap.imageData = intermediate;

                call->SetOutput(timeMap);
                return true;
            }
        }
    } else if (auto* call = dynamic_cast<GraphData2DCall*>(&caller)) {
        const auto hash = getTimeMapCaller.CallAs<ImageSeries::ImageSeries2DCall>()->GetOutput().getHash();

        auto output = imageCache.find(hash);

        if (output != nullptr) {
            auto intermediate = std::make_shared<graph::AsyncGraphData2D>(
                [output]() { return output->getImageData()->graph; }, 0); // TODO: size of graph?

            call->SetOutput(GraphData2DCall::Output{intermediate});
            call->SetDataHash(output->getHash());
        }

        return true;
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
