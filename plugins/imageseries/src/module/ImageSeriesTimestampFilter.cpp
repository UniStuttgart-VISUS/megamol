/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "ImageSeriesTimestampFilter.h"

#include "../filter/ImageSamplingFilter.h"
#include "../filter/IndexGenerationFilter.h"

#include "imageseries/ImageSeries2DCall.h"
#include "imageseries/util/ImageUtils.h"

#include "mmcore/param/IntParam.h"

#include <memory>

using Log = megamol::core::utility::log::Log;
using Presentation = megamol::core::param::AbstractParamPresentation::Presentation;

namespace megamol::ImageSeries {

ImageSeriesTimestampFilter::ImageSeriesTimestampFilter()
        : getDataCallee("getData", "Returns data from the image series for the requested timestamp.")
        , getInputCaller("requestInputImageSeries", "Requests image data from a series.")
        , denoiseIterations("Denoise iterations", "Number of iterations for denoising.")
        , denoiseNeighborThreshold("Denoise threshold", "Number of equal neighbors required to not count as noise.")
        , imageCache([](const AsyncImageData2D<>& imageData) { return imageData.getByteSize(); }) {

    getInputCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getInputCaller);

    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetData), &ImageSeriesTimestampFilter::getDataCallback);
    getDataCallee.SetCallback(ImageSeries2DCall::ClassName(),
        ImageSeries2DCall::FunctionName(ImageSeries2DCall::CallGetMetaData),
        &ImageSeriesTimestampFilter::getMetaDataCallback);
    MakeSlotAvailable(&getDataCallee);

    denoiseIterations << new core::param::IntParam(10, 0);
    denoiseIterations.SetUpdateCallback(&ImageSeriesTimestampFilter::filterParametersChangedCallback);
    MakeSlotAvailable(&denoiseIterations);

    denoiseNeighborThreshold << new core::param::IntParam(100, 1);
    denoiseNeighborThreshold.SetUpdateCallback(&ImageSeriesTimestampFilter::filterParametersChangedCallback);
    MakeSlotAvailable(&denoiseNeighborThreshold);

    // Set default image cache size to 512 MB
    imageCache.setMaximumSize(512 * 1024 * 1024);
}

ImageSeriesTimestampFilter::~ImageSeriesTimestampFilter() {
    Release();
}

bool ImageSeriesTimestampFilter::create() {
    filterRunner = std::make_unique<filter::AsyncFilterRunner<>>();
    return true;
}

void ImageSeriesTimestampFilter::release() {
    filterRunner = nullptr;
    imageCache.clear();
}

bool ImageSeriesTimestampFilter::getDataCallback(core::Call& caller) {
    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        auto output = requestFrame(getInputCaller, 0);

        if (output.imageData) {
            // Retrieve cached image or run filter on input data
            output.imageData = imageCache.findOrCreate(output.getHash(), [=](typename AsyncImageData2D<>::Hash) {
                filter::IndexGenerationFilter::Input params;
                for (std::size_t i = 0; i < output.imageCount; ++i) {
                    params.frameIndex = i;
                    params.image =
                        requestFrame(getInputCaller, output.minimumTime + (i + 0.5f) / output.framerate).imageData;
                    auto indexMap = filterRunner->run<filter::IndexGenerationFilter>(params);
                    params.indexMap = indexMap;
                }

                filter::ImageSamplingFilter::Input paramsSampling;
                paramsSampling.indexMap = params.indexMap;
                paramsSampling.iterations = denoiseIterations.Param<core::param::IntParam>()->Value();
                paramsSampling.neighborThreshold = denoiseNeighborThreshold.Param<core::param::IntParam>()->Value();
                return filterRunner->run<filter::ImageSamplingFilter>(paramsSampling);
            });
            call->SetOutput(output);
            return true;
        }
    }
    return false;
}

bool ImageSeriesTimestampFilter::getMetaDataCallback(core::Call& caller) {
    if (auto* call = dynamic_cast<ImageSeries2DCall*>(&caller)) {
        if (auto* getInput = getInputCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
            if ((*getInput)(ImageSeries::ImageSeries2DCall::CallGetMetaData)) {
                // Pass through data metadata
                call->SetOutput(getInput->GetOutput());
                return true;
            }
        }
    }
    return false;
}

bool ImageSeriesTimestampFilter::filterParametersChangedCallback(core::param::ParamSlot& param) {
    imageCache.clear();
    return true;
}

ImageSeries::ImageSeries2DCall::Output ImageSeriesTimestampFilter::requestFrame(
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

ImageSeries::ImageSeries2DCall::Output ImageSeriesTimestampFilter::requestMetadata(core::CallerSlot& source) {
    if (auto* call = source.CallAs<ImageSeries::ImageSeries2DCall>()) {
        if ((*call)(ImageSeries::ImageSeries2DCall::CallGetMetaData)) {
            return call->GetOutput();
        }
    }
    return {};
}


} // namespace megamol::ImageSeries
