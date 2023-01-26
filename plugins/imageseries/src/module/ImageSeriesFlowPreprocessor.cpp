#include "ImageSeriesFlowPreprocessor.h"
#include "imageseries/ImageSeries2DCall.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "vislib/graphics/BitmapCodecCollection.h"
#include "vislib/graphics/PngBitmapCodec.h"

#include "../filter/DeinterlaceFilter.h"
#include "../filter/MaskFilter.h"
#include "../filter/SegmentationFilter.h"
#include "imageseries/util/ImageUtils.h"

using Log = megamol::core::utility::log::Log;
using Presentation = megamol::core::param::AbstractParamPresentation::Presentation;

namespace megamol::ImageSeries {

ImageSeriesFlowPreprocessor::ImageSeriesFlowPreprocessor()
        : getDataCallee("getData", "Returns data from the image series for the requested timestamp.")
        , getInputCaller("requestInputImageSeries", "Requests image data from a series.")
        , getMaskCaller("requestMaskImageSeries", "Requests mask data from an image series.")
        , maskFrameParam("Mask frame", "Image timestamp to use for applying a mask.")
        , deinterlaceParam("Deinterlace", "Number of pixels of horizontal interlacing correction to apply.")
        , segmentationEnabledParam("Enable segmentation", "Toggles the image segmentation step on/off.")
        , segmentationThresholdParam("Segmentation threshold", "Per-pixel threshold for image segmentation.")
        , segmentationNegationParam("Segmentation negation", "Negates the output of the segmentation filter.")
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

    maskFrameParam << new core::param::FloatParam(0.f, 0.f, 1.f);
    maskFrameParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    maskFrameParam.SetUpdateCallback(&ImageSeriesFlowPreprocessor::filterParametersChangedCallback);
    MakeSlotAvailable(&maskFrameParam);

    deinterlaceParam << new core::param::IntParam(0, -10, 10);
    deinterlaceParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    deinterlaceParam.SetUpdateCallback(&ImageSeriesFlowPreprocessor::filterParametersChangedCallback);
    MakeSlotAvailable(&deinterlaceParam);

    segmentationEnabledParam << new core::param::BoolParam(1);
    segmentationEnabledParam.Parameter()->SetGUIPresentation(Presentation::Checkbox);
    segmentationEnabledParam.SetUpdateCallback(&ImageSeriesFlowPreprocessor::filterParametersChangedCallback);
    MakeSlotAvailable(&segmentationEnabledParam);

    segmentationThresholdParam << new core::param::FloatParam(0.5f, 0.f, 1.f);
    segmentationThresholdParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    segmentationThresholdParam.SetUpdateCallback(&ImageSeriesFlowPreprocessor::filterParametersChangedCallback);
    MakeSlotAvailable(&segmentationThresholdParam);

    segmentationNegationParam << new core::param::BoolParam(0);
    segmentationNegationParam.Parameter()->SetGUIPresentation(Presentation::Checkbox);
    segmentationNegationParam.SetUpdateCallback(&ImageSeriesFlowPreprocessor::filterParametersChangedCallback);
    MakeSlotAvailable(&segmentationNegationParam);

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

                // Get deinterlacing offset
                int deinterlace = deinterlaceParam.Param<core::param::IntParam>()->Value();

                // Retrieve cached image or run filter on input data
                auto hash = util::combineHash(output.getHash(), mask.getHash());
                output.imageData = imageCache.findOrCreate(hash, [=](AsyncImageData2D::Hash) {
                    auto image = output.imageData;
                    if (mask.imageData) {
                        image = filterRunner->run<filter::MaskFilter>(image, mask.imageData);
                    }
                    if (deinterlace != 0) {
                        image = filterRunner->run<filter::DeinterlaceFilter>(image, deinterlace);
                    }
                    if (segmentationEnabledParam.Param<core::param::BoolParam>()->Value()) {
                        image = filterRunner->run<filter::SegmentationFilter>(image,
                            segmentationThresholdParam.Param<core::param::FloatParam>()->Value(),
                            segmentationNegationParam.Param<core::param::BoolParam>()->Value());
                    }
                    return image;
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
