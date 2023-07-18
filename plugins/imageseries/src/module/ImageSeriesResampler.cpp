#include "ImageSeriesResampler.h"
#include "imageseries/ImageSeries2DCall.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "vislib/graphics/BitmapCodecCollection.h"
#include "vislib/graphics/PngBitmapCodec.h"

#include "../filter/TransformationFilter.h"

using Log = megamol::core::utility::log::Log;
using Presentation = megamol::core::param::AbstractParamPresentation::Presentation;

namespace megamol::ImageSeries {

ImageSeriesResampler::ImageSeriesResampler()
        : getDataCallee("getData", "Returns data from the image series for the requested timestamp.")
        , getTransformCallee("getTransform", "Returns the affine matrix by which the image is transformed.")
        , getInputCaller("requestInputImageSeries", "Requests image data from a series.")
        , getReferenceCaller("requestReferenceImageSeries", "Requests image data from a series.")
        , getTransformCaller("chainTransform", "Overrides the transformation matrix.")
        , keyTimeInput1Param("Input alignment timestamp 1", "First alignment timestamp for the input image series.")
        , keyTimeReference1Param("Reference timestamp 1", "First alignment timestamp for the reference image series.")
        , keyTimeInput2Param("Input alignment timestamp 2", "Second alignment timestamp for the input image series.")
        , keyTimeReference2Param("Reference timestamp 2", "Second alignment timestamp for the reference image series.")
        , imageRegistrationParam("Transform image", "Applies an affine transformation to the input series.")
        , imageRegistrationAutoParam(
              "Auto image registration", "Auto-adjusts the affine transformation to match the reference series.")
        , cachedTransformMatrix(1, 0, 0, 1, 0, 0)
        , imageCache([](const AsyncImageData2D<>& imageData) { return imageData.getByteSize(); }) {

    getInputCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getInputCaller);
    getReferenceCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getReferenceCaller);
    getTransformCaller.SetCompatibleCall<typename ImageSeries::AffineTransform2DCall::CallDescription>();
    MakeSlotAvailable(&getTransformCaller);

    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetData), &ImageSeriesResampler::getDataCallback);
    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetMetaData),
        &ImageSeriesResampler::getMetaDataCallback);
    MakeSlotAvailable(&getDataCallee);

    getTransformCallee.SetCallback(AffineTransform2DCall::ClassName(),
        AffineTransform2DCall::FunctionName(AffineTransform2DCall::CallGetTransform),
        &ImageSeriesResampler::getTransformCallback);
    MakeSlotAvailable(&getTransformCallee);

    keyTimeInput1Param << new core::param::FloatParam(0);
    keyTimeInput1Param.Parameter()->SetGUIPresentation(Presentation::Slider);
    keyTimeInput1Param.SetUpdateCallback(&ImageSeriesResampler::timestampChangedCallback);
    MakeSlotAvailable(&keyTimeInput1Param);

    keyTimeReference1Param << new core::param::FloatParam(0);
    keyTimeReference1Param.Parameter()->SetGUIPresentation(Presentation::Slider);
    keyTimeReference1Param.SetUpdateCallback(&ImageSeriesResampler::timestampChangedCallback);
    MakeSlotAvailable(&keyTimeReference1Param);

    keyTimeInput2Param << new core::param::FloatParam(1);
    keyTimeInput2Param.Parameter()->SetGUIPresentation(Presentation::Slider);
    keyTimeInput2Param.SetUpdateCallback(&ImageSeriesResampler::timestampChangedCallback);
    MakeSlotAvailable(&keyTimeInput2Param);

    keyTimeReference2Param << new core::param::FloatParam(1);
    keyTimeReference2Param.Parameter()->SetGUIPresentation(Presentation::Slider);
    keyTimeReference2Param.SetUpdateCallback(&ImageSeriesResampler::timestampChangedCallback);
    MakeSlotAvailable(&keyTimeReference2Param);

    imageRegistrationParam << new core::param::BoolParam(0.f);
    imageRegistrationParam.Parameter()->SetGUIPresentation(Presentation::Checkbox);
    imageRegistrationParam.SetUpdateCallback(&ImageSeriesResampler::registrationCallback);
    MakeSlotAvailable(&imageRegistrationParam);

    imageRegistrationAutoParam << new core::param::BoolParam(0.f);
    imageRegistrationAutoParam.Parameter()->SetGUIPresentation(Presentation::Checkbox);
    imageRegistrationAutoParam.SetUpdateCallback(&ImageSeriesResampler::registrationCallback);
    MakeSlotAvailable(&imageRegistrationAutoParam);

    // Set default image cache size to 512 MB
    imageCache.setMaximumSize(512 * 1024 * 1024);
}

ImageSeriesResampler::~ImageSeriesResampler() {
    Release();
}

bool ImageSeriesResampler::create() {
    registrator = std::make_unique<registration::AsyncImageRegistrator>();
    filterRunner = std::make_unique<filter::AsyncFilterRunner<>>();
    return true;
}

void ImageSeriesResampler::release() {
    registrator = nullptr;
    filterRunner = nullptr;
    imageCache.clear();
}

bool ImageSeriesResampler::getDataCallback(core::Call& caller) {
    if (suppressed) {
        updateTransformationMatrix();
        return false;
    }

    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        if (auto* getInput = getInputCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
            auto input = call->GetInput();
            input.time = fromAlignedTimestamp(input.time);
            getInput->SetInput(input);
            if ((*getInput)(ImageSeries::ImageSeries2DCall::CallGetData)) {
                auto output = transformMetadata(getInput->GetOutput());

                if (imageRegistrationParam.Param<core::param::BoolParam>()->Value() && output.imageData) {
                    updateTransformationMatrix();

                    if (cachedTransformMatrix != glm::mat3x2(1, 0, 0, 1, 0, 0)) {
                        output.imageData =
                            imageCache.findOrCreate(output.getHash(), [&](typename AsyncImageData2D<>::Hash) {
                                return filterRunner->run<filter::TransformationFilter>(
                                    output.imageData, cachedTransformMatrix);
                            });
                    }
                }

                call->SetOutput(output);
                return true;
            }
        }
    }
    return false;
}

bool ImageSeriesResampler::getMetaDataCallback(core::Call& caller) {
    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        if (auto* getInput = getInputCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
            if ((*getInput)(ImageSeries::ImageSeries2DCall::CallGetMetaData)) {
                call->SetOutput(transformMetadata(getInput->GetOutput()));
                return true;
            }
        }
    }
    return false;
}

bool ImageSeriesResampler::getTransformCallback(core::Call& caller) {
    if (auto* call = dynamic_cast<AffineTransform2DCall*>(&caller)) {
        updateTransformationMatrix();
        if (!registrator->isActive()) {
            call->SetOutput({cachedTransformMatrix});
            return true;
        }
    }
    return false;
}

bool ImageSeriesResampler::timestampChangedCallback(core::param::ParamSlot& param) {
    return true;
}

bool ImageSeriesResampler::registrationCallback(core::param::ParamSlot& param) {
    registrator->setActive(imageRegistrationAutoParam.Param<core::param::BoolParam>()->Value());
    imageCache.clear();
    return true;
}

double ImageSeriesResampler::toAlignedTimestamp(double timestamp) const {
    return transformTimestamp(timestamp, keyTimeInput1Param.Param<core::param::FloatParam>()->Value(),
        keyTimeInput2Param.Param<core::param::FloatParam>()->Value(),
        keyTimeReference1Param.Param<core::param::FloatParam>()->Value(),
        keyTimeReference2Param.Param<core::param::FloatParam>()->Value());
}

double ImageSeriesResampler::fromAlignedTimestamp(double timestamp) const {
    return transformTimestamp(timestamp, keyTimeReference1Param.Param<core::param::FloatParam>()->Value(),
        keyTimeReference2Param.Param<core::param::FloatParam>()->Value(),
        keyTimeInput1Param.Param<core::param::FloatParam>()->Value(),
        keyTimeInput2Param.Param<core::param::FloatParam>()->Value());
}

ImageSeries2DCall::Output ImageSeriesResampler::transformMetadata(ImageSeries2DCall::Output metadata) const {
    metadata.resultTime = toAlignedTimestamp(metadata.resultTime);
    metadata.minimumTime = toAlignedTimestamp(metadata.minimumTime);
    metadata.maximumTime = toAlignedTimestamp(metadata.maximumTime);
    metadata.framerate = metadata.imageCount / (metadata.maximumTime - metadata.minimumTime);
    return metadata;
}

double ImageSeriesResampler::transformTimestamp(double timestamp, double min1, double max1, double min2, double max2) {
    return (timestamp - min1) / (max1 - min1) * (max2 - min2) + min2;
}

std::shared_ptr<const AsyncImageData2D<>> ImageSeriesResampler::fetchImage(
    core::CallerSlot& caller, double timestamp) const {
    if (auto* seriesCall = caller.CallAs<ImageSeries::ImageSeries2DCall>()) {
        ImageSeries2DCall::Input input;
        input.time = timestamp;
        seriesCall->SetInput(input);
        (*seriesCall)(ImageSeries::ImageSeries2DCall::CallGetData);
        return seriesCall->GetOutput().imageData;
    }
    return nullptr;
}

void ImageSeriesResampler::updateTransformationMatrix() {
    glm::mat3x2 matrix(1, 0, 0, 1, 0, 0);

    if (registrator->isActive()) {
        float keyTimeReference = keyTimeReference1Param.Param<core::param::FloatParam>()->Value();
        float keyTimeInput = keyTimeInput1Param.Param<core::param::FloatParam>()->Value();
        registrator->setReferenceImage(fetchImage(getReferenceCaller, keyTimeReference));
        registrator->setInputImage(fetchImage(getInputCaller, keyTimeInput));

        matrix = registrator->getTransform();

        if (registrator->getStepsSinceLastImprovement() > 50) {
            imageRegistrationAutoParam.Param<core::param::BoolParam>()->SetValue(false);
            registrator->setActive(false);
        }
    } else if (getTransformCaller.GetStatus() == megamol::core::AbstractSlot::STATUS_CONNECTED) {
        if (auto* getTransform = getTransformCaller.CallAs<ImageSeries::AffineTransform2DCall>()) {
            if ((*getTransform)()) {
                matrix = getTransform->GetOutput().matrix;
                suppressed = false;
            } else {
                suppressed = true;
            }
        }
    } else {
        return;
    }

    if (cachedTransformMatrix != matrix) {
        cachedTransformMatrix = matrix;
        imageCache.clear();
    }
}


} // namespace megamol::ImageSeries
