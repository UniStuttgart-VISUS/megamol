#include "ImageSeriesRenderer.h"

#include "OpenGL_Context.h"

#include "imageseries/graph/GraphData2DCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/utility/log/Log.h"

#include <glowl/glowl.h>

using Log = megamol::core::utility::log::Log;

namespace megamol::ImageSeries::GL {

ImageSeriesRenderer::ImageSeriesRenderer()
        : getDataCaller("requestImageSeries", "Requests image data from a series.")
        , getGraphCaller("requestGraph", "Requests graph data to render on top of the image series.")
        , displayModeParam("Display Mode", "Controls how the image should be presented.")
        , graph_hash(-7345) {
    getDataCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getDataCaller);

    getGraphCaller.SetCompatibleCall<typename ImageSeries::GraphData2DCall::CallDescription>();
    MakeSlotAvailable(&getGraphCaller);

    auto* displayMode = new core::param::EnumParam(static_cast<int>(ImageDisplay2D::Mode::Auto));
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::Auto), "Automatic");
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::Color), "Color");
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::Grayscale), "Grayscale");
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::Labels), "Labels");
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::TimeDifference), "Time Difference");

    displayModeParam << displayMode;
    displayModeParam.SetUpdateCallback(&ImageSeriesRenderer::displayModeChangedCallback);
    MakeSlotAvailable(&displayModeParam);
}

ImageSeriesRenderer::~ImageSeriesRenderer() {
    Release();
}

bool ImageSeriesRenderer::create() {
    auto const shader_options = core::utility::make_path_shader_options(
        frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    try {
        display = std::make_unique<ImageDisplay2D>(msf::ShaderFactoryOptionsOpenGL(shader_options));
    }
    catch (const std::exception& ex) {
        core::utility::log::Log::DefaultLog.WriteError(ex.what());

        return false;
    }

    return true;
}

bool ImageSeriesRenderer::GetExtents(mmstd_gl::CallRender2DGL& call) {
    call.SetTimeFramesCount(std::max<unsigned int>(1, metadata.imageCount));

    call.AccessBoundingBoxes().Clear();
    auto size = display ? display->getImageSize() : glm::vec2(1.f, 1.f);
    call.AccessBoundingBoxes().SetBoundingBox(0.0f, 0.0f, 0.0f, size.x, size.y, 0.0f);

    return true;
}

void ImageSeriesRenderer::release() {}

bool ImageSeriesRenderer::Render(mmstd_gl::CallRender2DGL& call) {
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
        metadata = output;

        // Perform read-ahead
        // TODO add option to disable read-ahead
        std::size_t readAheadFrameCount = 5;

        auto performReadAhead = [&](int offset) {
            // Check framerate to avoid division by zero or negative values
            if (output.framerate > 0.001f) {
                input.time = call.Time() + offset / output.framerate;
                // Perform call, but ignore results
                (*getData)(ImageSeries::ImageSeries2DCall::CallGetData);
            }
        };

        if (initialReadAhead) {
            // Initial read-ahead: preload range of frames
            initialReadAhead = false;
            for (std::size_t i = 1; i <= readAheadFrameCount; ++i) {
                performReadAhead(i);
            }
        } else if (std::abs(lastReadAhead - call.Time()) > 0.001f) {
            // Timestamp changed: perform single readahead
            lastReadAhead = call.Time();
            performReadAhead(readAheadFrameCount);
        }
    }

    if (currentImage && currentImage->isValid()) {
        const auto& image = *currentImage->getImageData();

        display->updateTexture(image);

        if (auto* getData = getGraphCaller.CallAs<ImageSeries::GraphData2DCall>()) {
            if ((*getData)(ImageSeries::GraphData2DCall::CallGetData) && getData->DataHash() != graph_hash) {
                graph_hash = getData->DataHash();
                display->updateGraph(*getData->GetOutput().graph->getData());
            }
        }
    }

    return display->render(call);
}

bool ImageSeriesRenderer::displayModeChangedCallback(core::param::ParamSlot& param) {
    if (display) {
        display->setDisplayMode(static_cast<ImageDisplay2D::Mode>(param.Param<core::param::EnumParam>()->Value()));
    }
    return true;
}


} // namespace megamol::ImageSeries::GL
