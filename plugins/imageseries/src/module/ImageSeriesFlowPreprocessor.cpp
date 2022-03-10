#include "ImageSeriesFlowPreprocessor.h"
#include "imageseries/ImageSeries2DCall.h"

#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/graphics/BitmapCodecCollection.h"

#include "../filter/MaskFilter.h"
#include "../filter/SegmentationFilter.h"
#include "../util/ImageUtils.h"

using Log = megamol::core::utility::log::Log;
using Presentation = megamol::core::param::AbstractParamPresentation::Presentation;

namespace megamol::ImageSeries {

ImageSeriesFlowPreprocessor::ImageSeriesFlowPreprocessor()
        : getDataCallee("getData", "Returns data from the image series for the requested timestamp.")
        , getInputCaller("requestInputImageSeries", "Requests image data from a series.")
        , getMaskCaller("requestMaskImageSeries", "Requests mask data from an image series.")
        , segmentationThresholdParam("Segmentation threshold", "Per-pixel threshold for image segmentation.")
        , maskFrameParam("Mask frame", "Image timestamp to use for applying a mask.")
        , imageCache([](const AsyncImageData2D& imageData) { return imageData.getByteSize(); }) {

    getInputCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getInputCaller);

    getMaskCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getMaskCaller);

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

    maskFrameParam << new core::param::FloatParam(0.f, 0.f, 1.f);
    maskFrameParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    maskFrameParam.SetUpdateCallback(&ImageSeriesFlowPreprocessor::filterParametersChangedCallback);
    MakeSlotAvailable(&maskFrameParam);

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
        // Try to get mask
        ImageSeries2DCall::Output mask;
        if (auto* getMask = getMaskCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
            ImageSeries2DCall::Input maskInput;
            maskInput.time = maskFrameParam.Param<core::param::FloatParam>()->Value();
            getMask->SetInput(maskInput);
            if ((*getMask)(ImageSeries::ImageSeries2DCall::CallGetData)) {
                mask = getMask->GetOutput();
            }
        }

        if (auto* getInput = getInputCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
            // Pass through input parameters
            getInput->SetInput(call->GetInput());
            if ((*getInput)(ImageSeries::ImageSeries2DCall::CallGetData)) {
                auto output = getInput->GetOutput();

                // Retrieve cached image or run filter on input data
                output.imageData = imageCache.findOrCreate(
                    util::combineHash(output.getHash(), mask.getHash()), [=](AsyncImageData2D::Hash) {
                        auto image = output.imageData;
                        if (mask.imageData) {
                            image = filterRunner->run<filter::MaskFilter>(image, mask.imageData);
                        }
                        return filterRunner->run<filter::SegmentationFilter>(
                            image, segmentationThresholdParam.Param<core::param::FloatParam>()->Value());
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
