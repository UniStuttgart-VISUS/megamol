#include "ImageSeriesRenderer.h"

#include "mmcore/CoreInstance.h"

using Log = megamol::core::utility::log::Log;

namespace megamol::ImageSeries::GL {

ImageSeriesRenderer::ImageSeriesRenderer() : getDataCaller("requestImageSeries", "Requests image data from a series.") {
    getDataCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getDataCaller);
}

ImageSeriesRenderer::~ImageSeriesRenderer() {
    Release();
}

bool ImageSeriesRenderer::create() {
    if (megamol::core::CoreInstance* coreInstance = GetCoreInstance()) {
        display = std::make_unique<ImageDisplay2D>(msf::ShaderFactoryOptionsOpenGL(coreInstance->GetShaderPaths()));
        return true;
    } else {
        return false;
    }
}

bool ImageSeriesRenderer::GetExtents(core_gl::view::CallRender2DGL& call) {
    call.SetTimeFramesCount(std::max<unsigned int>(1, metadata.imageCount));

    call.AccessBoundingBoxes().Clear();
    // TODO actual bounds
    call.AccessBoundingBoxes().SetBoundingBox(0.0f, 0.0f, -0.5f, metadata.width, metadata.height, 0.5f);
    call.AccessBoundingBoxes().SetClipBox(call.AccessBoundingBoxes().BoundingBox());

    return true;
}

void ImageSeriesRenderer::release() {}

bool ImageSeriesRenderer::Render(core_gl::view::CallRender2DGL& call) {
    if (display == nullptr) {
        return false;
    }

    if (auto* getData = getDataCaller.CallAs<ImageSeries::ImageSeries2DCall>()) {
        ImageSeries2DCall::Input input;
        input.time = call.Time();
        getData->SetInput(std::move(input));
        (*getData)(ImageSeries::ImageSeries2DCall::CallGetData);
        const auto& output = getData->GetOutput();
        currentImage = output.imageData;
    }

    if (currentImage && currentImage->isValid()) {
        display->updateTexture(*currentImage->getImageData());
    }

    return display->render(call);
}


} // namespace megamol::ImageSeries::GL
