#include "ImageSeriesFlowPreprocessor.h"
#include "imageseries/ImageSeries2DCall.h"

#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/graphics/BitmapCodecCollection.h"

#include "../filter/SegmentationFilter.h"

using Log = megamol::core::utility::log::Log;
using Presentation = megamol::core::param::AbstractParamPresentation::Presentation;

namespace megamol::ImageSeries {

ImageSeriesFlowPreprocessor::ImageSeriesFlowPreprocessor()
        : getDataCallee("getData", "Returns data from the image series for the requested timestamp.")
        , getInputCaller("requestInputImageSeries", "Requests image data from a series.")
        , segmentationThresholdParam("Segmentation threshold", "Per-pixel threshold for image segmentation.")
        , imageCache([](const AsyncImageData2D& imageData) { return imageData.getByteSize(); }) {

    getInputCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getInputCaller);

    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetData), &ImageSeriesFlowPreprocessor::getDataCallback);
    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetMetaData),
        &ImageSeriesFlowPreprocessor::getMetaDataCallback);
    MakeSlotAvailable(&getDataCallee);

    segmentationThresholdParam << new core::param::FloatParam(0.5f, 0.f, 1.f);
    segmentationThresholdParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    segmentationThresholdParam.SetUpdateCallback(&ImageSeriesFlowPreprocessor::filterParametersChangedCallback);
    MakeSlotAvailable(&segmentationThresholdParam);

    // Set default image cache size to 512 MB
    imageCache.setMaximumSize(512 * 1024 * 1024);
}

ImageSeriesFlowPreprocessor::~ImageSeriesFlowPreprocessor() {
    Release();
}

bool ImageSeriesFlowPreprocessor::create() {
    filterRunner = std::make_unique<filter::AsyncFilterRunner>();
    return true;
}

void ImageSeriesFlowPreprocessor::release() {
    filterRunner = nullptr;
    imageCache.clear();
}

bool ImageSeriesFlowPreprocessor::getDataCallback(core::Call& caller) {
    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        if (auto* getInput = getInputCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
            // Pass through input parameters
            getInput->SetInput(call->GetInput());
            if ((*getInput)(ImageSeries::ImageSeries2DCall::CallGetData)) {
                auto output = getInput->GetOutput();

                // Retrieve cached image or run filter on input data
                output.imageData = imageCache.findOrCreate(output.getHash(), [&](AsyncImageData2D::Hash) {
                    return filterRunner->run<filter::SegmentationFilter>(
                        output.imageData, segmentationThresholdParam.Param<core::param::FloatParam>()->Value());
                });

                call->SetOutput(output);
                return true;
            }
        }
    }
    return false;
}

bool ImageSeriesFlowPreprocessor::getMetaDataCallback(core::Call& caller) {
    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        if (auto* getInput = getInputCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
            if ((*getInput)(ImageSeries::ImageSeries2DCall::CallGetMetaData)) {
                // Pass through metadata
                call->SetOutput(getInput->GetOutput());
                return true;
            }
        }
    }
    return false;
}

bool ImageSeriesFlowPreprocessor::filterParametersChangedCallback(core::param::ParamSlot& param) {
    imageCache.clear();
    return true;
}


} // namespace megamol::ImageSeries
