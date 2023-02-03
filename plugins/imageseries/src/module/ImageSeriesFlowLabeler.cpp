#include "ImageSeriesFlowLabeler.h"
#include "imageseries/ImageSeries2DCall.h"
#include "imageseries/graph/GraphData2DCall.h"
#include "imageseries/util/AsyncData.h"

#include "vislib/graphics/PngBitmapCodec.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"

#include "../filter/FlowTimeLabelFilter.h"
#include "imageseries/util/ImageUtils.h"

using Log = megamol::core::utility::log::Log;
using Presentation = megamol::core::param::AbstractParamPresentation::Presentation;

namespace megamol::ImageSeries {

ImageSeriesFlowLabeler::ImageSeriesFlowLabeler()
        : getDataCallee("getData", "Returns data from the image series.")
        , getGraphCallee("getGraph", "Returns the constructed graph from the image series.")
        , getTimeMapCaller("requestTimeMapImageSeries", "Requests timestamp data from an image series.")
        , timeThresholdParam("Time threshold", "Maximum allowed time offset between adjacent pixels, in frames.")
        , minTimestampParam("Min frame index", "Minimum frame index to start flood filling from.")
        , maxTimestampParam("Max frame index", "Maximum frame index to stop flood filling at.")
        , initThresholdParam("Initial area threshold", "Minimum required number of connected pixels to start filling.")
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

    timeThresholdParam << new core::param::IntParam(20, 1, 50);
    timeThresholdParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    timeThresholdParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&timeThresholdParam);

    minTimestampParam << new core::param::IntParam(10);
    minTimestampParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    minTimestampParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&minTimestampParam);

    maxTimestampParam << new core::param::IntParam(1000);
    maxTimestampParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    maxTimestampParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&maxTimestampParam);

    initThresholdParam << new core::param::IntParam(30);
    initThresholdParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    initThresholdParam.SetUpdateCallback(&ImageSeriesFlowLabeler::filterParametersChangedCallback);
    MakeSlotAvailable(&initThresholdParam);

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
                filterInput.timeThreshold = timeThresholdParam.Param<core::param::IntParam>()->Value();
                filterInput.minimumTimestamp = minTimestampParam.Param<core::param::IntParam>()->Value();
                filterInput.maximumTimestamp = maxTimestampParam.Param<core::param::IntParam>()->Value();
                filterInput.minBlobSize = initThresholdParam.Param<core::param::IntParam>()->Value();

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
            call->SetDataHash(hash);
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
