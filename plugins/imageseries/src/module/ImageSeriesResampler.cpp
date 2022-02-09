#include "ImageSeriesResampler.h"
#include "imageseries/ImageSeries2DCall.h"

#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/graphics/BitmapCodecCollection.h"

using Log = megamol::core::utility::log::Log;

namespace megamol::ImageSeries {

ImageSeriesResampler::ImageSeriesResampler()
        : getDataCallee("getData", "Returns data from the image series for the requested timestamp.")
        , getInputCaller("requestInputImageSeries", "Requests image data from a series.")
        , getReferenceCaller("requestReferenceImageSeries", "Requests image data from a series.")
        , keyTimeInput1Param("Input alignment timestamp 1", "First alignment timestamp for the input image series.")
        , keyTimeReference1Param("Reference timestamp 1", "First alignment timestamp for the reference image series.")
        , keyTimeInput2Param("Input alignment timestamp 2", "Second alignment timestamp for the input image series.")
        , keyTimeReference2Param("Reference timestamp 2", "Second alignment timestamp for the reference image series.")
        , imageCache([](const AsyncImageData2D& imageData) { return imageData.getByteSize(); }) {

    getInputCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getInputCaller);
    getReferenceCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getReferenceCaller);

    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetData), &ImageSeriesResampler::getDataCallback);
    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetMetaData),
        &ImageSeriesResampler::getMetaDataCallback);
    MakeSlotAvailable(&getDataCallee);

    keyTimeInput1Param << new core::param::FloatParam(0);
    keyTimeInput1Param.SetUpdateCallback(&ImageSeriesResampler::timestampChangedCallback);
    MakeSlotAvailable(&keyTimeInput1Param);

    keyTimeReference1Param << new core::param::FloatParam(0);
    keyTimeReference1Param.SetUpdateCallback(&ImageSeriesResampler::timestampChangedCallback);
    MakeSlotAvailable(&keyTimeReference1Param);

    keyTimeInput2Param << new core::param::FloatParam(1);
    keyTimeInput2Param.SetUpdateCallback(&ImageSeriesResampler::timestampChangedCallback);
    MakeSlotAvailable(&keyTimeInput2Param);

    keyTimeReference2Param << new core::param::FloatParam(1);
    keyTimeReference2Param.SetUpdateCallback(&ImageSeriesResampler::timestampChangedCallback);
    MakeSlotAvailable(&keyTimeReference2Param);

    // Set default image cache size to 512 MB
    imageCache.setMaximumSize(512 * 1024 * 1024);
}

ImageSeriesResampler::~ImageSeriesResampler() {
    Release();
}

bool ImageSeriesResampler::create() {
    return true;
}

void ImageSeriesResampler::release() {
    imageCache.clear();
}

bool ImageSeriesResampler::getDataCallback(core::Call& caller) {
    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        if (auto* getInput = getInputCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
            auto input = call->GetInput();
            input.time = fromAlignedTimestamp(input.time);
            getInput->SetInput(input);
            if ((*getInput)(ImageSeries::ImageSeries2DCall::CallGetData)) {
                call->SetOutput(transformMetadata(getInput->GetOutput()));
                // TODO: perform spatial resampling here, if necessary
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

bool ImageSeriesResampler::timestampChangedCallback(core::param::ParamSlot& param) {
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


} // namespace megamol::ImageSeries
