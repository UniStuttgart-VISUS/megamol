#include "ImageSeriesTimeDiffFilter.h"
#include "imageseries/ImageSeries2DCall.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"

#include "../filter/TimeOffsetFilter.h"
#include "imageseries/util/ImageUtils.h"

using Log = megamol::core::utility::log::Log;
using Presentation = megamol::core::param::AbstractParamPresentation::Presentation;

namespace megamol::ImageSeries {

ImageSeriesTimeDiffFilter::ImageSeriesTimeDiffFilter()
        : getDataCallee("getData", "Returns data from the image series for the requested timestamp.")
        , getInputCaller("requestInputImageSeries", "Requests image data from a series.")
        , getReferenceCaller("requestReferenceImageSeries", "Requests image data from a series.")
        , frameCountParam("Frame count", "Number of frames to sample forwards + backwards when computing differences.")
        , imageCache([](const AsyncImageData2D& imageData) { return imageData.getByteSize(); }) {

    getInputCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getInputCaller);

    getReferenceCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getReferenceCaller);

    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetData), &ImageSeriesTimeDiffFilter::getDataCallback);
    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetMetaData),
        &ImageSeriesTimeDiffFilter::getMetaDataCallback);
    MakeSlotAvailable(&getDataCallee);

    frameCountParam << new core::param::IntParam(4, 1, 15);
    frameCountParam.Parameter()->SetGUIPresentation(Presentation::Slider);
    frameCountParam.SetUpdateCallback(&ImageSeriesTimeDiffFilter::filterParametersChangedCallback);
    MakeSlotAvailable(&frameCountParam);

    // Set default image cache size to 512 MB
    imageCache.setMaximumSize(512 * 1024 * 1024);
}

ImageSeriesTimeDiffFilter::~ImageSeriesTimeDiffFilter() {
    Release();
}

bool ImageSeriesTimeDiffFilter::create() {
    filterRunner = std::make_unique<filter::AsyncFilterRunner>();
    return true;
}

void ImageSeriesTimeDiffFilter::release() {
    filterRunner = nullptr;
    imageCache.clear();
}

bool ImageSeriesTimeDiffFilter::getDataCallback(core::Call& caller) {
    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        double timestamp = call->GetInput().time;

        auto output = requestFrame(getReferenceCaller, timestamp);

        if (output.imageData) {
            auto reference = output.imageData;
            auto inputMeta = requestFrame(getInputCaller, timestamp);

            // Retrieve cached image or run filter on input data
            auto hash = util::combineHash(reference->getHash(), inputMeta.getHash());
            output.imageData = imageCache.findOrCreate(hash, [=](AsyncImageData2D::Hash) {
                filter::TimeOffsetFilter::Input filterParams;
                filterParams.reference = reference;
                int frameCount = frameCountParam.Param<core::param::IntParam>()->Value();
                for (int i = -frameCount; i <= frameCount; ++i) {
                    filter::TimeOffsetFilter::Input::Frame frame;
                    frame.image = requestFrame(getInputCaller, timestamp + i / inputMeta.framerate).imageData;
                    frame.weight = i < 0 ? -1 : (i > 0 ? 1 : 0);
                    frame.certainty = frameCount - std::abs(i) + 1;
                    frame.primary = (i == 0);

                    if (!filterParams.frames.empty() && filterParams.frames.back().image == frame.image) {
                        // Optimization: merge identical frames
                        filterParams.frames.back().weight += frame.weight;
                        filterParams.frames.back().certainty += frame.certainty;
                        filterParams.frames.back().primary |= frame.primary;
                    } else {
                        filterParams.frames.push_back(frame);
                    }
                }
                return filterRunner->run<filter::TimeOffsetFilter>(filterParams);
            });
            call->SetOutput(output);
            return true;
        }
    }
    return false;
}

bool ImageSeriesTimeDiffFilter::getMetaDataCallback(core::Call& caller) {
    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        if (auto* getReference = getReferenceCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
            if ((*getReference)(ImageSeries::ImageSeries2DCall::CallGetMetaData)) {
                // Pass through reference metadata
                call->SetOutput(getReference->GetOutput());
                return true;
            }
        }
    }
    return false;
}

bool ImageSeriesTimeDiffFilter::filterParametersChangedCallback(core::param::ParamSlot& param) {
    imageCache.clear();
    return true;
}

ImageSeries::ImageSeries2DCall::Output ImageSeriesTimeDiffFilter::requestFrame(
    core::CallerSlot& source, double timestamp) {
    if (auto* call = source.CallAs<ImageSeries::ImageSeries2DCall>()) {
        ImageSeries::ImageSeries2DCall::Input input;
        input.time = timestamp;
        call->SetInput(input);

        if ((*call)(ImageSeries::ImageSeries2DCall::CallGetData)) {
            return call->GetOutput();
        }
    }
    return {};
}

ImageSeries::ImageSeries2DCall::Output ImageSeriesTimeDiffFilter::requestMetadata(core::CallerSlot& source) {
    if (auto* call = source.CallAs<ImageSeries::ImageSeries2DCall>()) {
        if ((*call)(ImageSeries::ImageSeries2DCall::CallGetMetaData)) {
            return call->GetOutput();
        }
    }
    return {};
}


} // namespace megamol::ImageSeries
