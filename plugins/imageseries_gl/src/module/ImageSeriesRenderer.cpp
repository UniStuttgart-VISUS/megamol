#include "ImageSeriesRenderer.h"

#include "OpenGL_Context.h"

#include "imageseries/graph/GraphData2DCall.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"

#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"

#include <glowl/glowl.h>

using Log = megamol::core::utility::log::Log;

namespace megamol::ImageSeries::GL {

ImageSeriesRenderer::ImageSeriesRenderer()
        : getDataCaller("requestImageSeries", "Requests image data from a series.")
        , getGraphCaller("requestGraph", "Requests graph data to render on top of the image series.")
        , getTransferFunctionCaller("requestTransferFunction", "Transfer function for mapping values to color.")
        , displayModeParam("Display Mode", "Controls how the image should be presented.")
        , renderGraphParam("Render Graph", "Render the input graph if there is one.")
        , baseRadiusParam("graph::Node radius", "Radius of the nodes.")
        , edgeWidthParam("graph::Edge width", "Width of the edges.")
        , highlight("highlight::Enabled", "Enable highlighting the selected value.")
        , highlightValue("highlight::Selected value", "Value that should be visually highlighted.")
        , highlightColor("highlight::Highlight color", "Color of the highlight.")
        , autoSave("output::Save automatically", "Automatically save files when changes occur.")
        , outputPathParam("output::Image output path", "If set, write resulting images to the selected directory.")
        , image_hash(-9837)
        , graph_hash(-7345) {

    getDataCaller.SetCompatibleCall<typename ImageSeries::ImageSeries2DCall::CallDescription>();
    MakeSlotAvailable(&getDataCaller);

    getGraphCaller.SetCompatibleCall<typename ImageSeries::GraphData2DCall::CallDescription>();
    MakeSlotAvailable(&getGraphCaller);

    getTransferFunctionCaller.SetCompatibleCall<megamol::mmstd_gl::CallGetTransferFunctionGLDescription>();
    MakeSlotAvailable(&getTransferFunctionCaller);

    auto* displayMode = new core::param::EnumParam(static_cast<int>(ImageDisplay2D::Mode::Color));
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::Color), "Image color");
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::TFByte), "Transfer function (byte)");
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::TFWord), "Transfer function (word)");
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::CatByte), "Category lookup (byte)");
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::CatWord), "Category lookup (word)");
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::TFCatByte), "TF category lookup (byte)");
    displayMode->SetTypePair(static_cast<int>(ImageDisplay2D::Mode::TFCatWord), "TF category lookup (word)");

    displayModeParam << displayMode;
    displayModeParam.SetUpdateCallback(&ImageSeriesRenderer::displayModeChangedCallback);
    MakeSlotAvailable(&displayModeParam);

    renderGraphParam << new core::param::BoolParam(true);
    MakeSlotAvailable(&renderGraphParam);

    baseRadiusParam << new core::param::FloatParam(2.0);
    MakeSlotAvailable(&baseRadiusParam);

    edgeWidthParam << new core::param::FloatParam(2.0);
    MakeSlotAvailable(&edgeWidthParam);

    highlight << new core::param::BoolParam(false);
    MakeSlotAvailable(&highlight);

    highlightValue << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&highlightValue);

    highlightColor << new core::param::ColorParam(0.85, 0.17, 0.17, 1.0);
    MakeSlotAvailable(&highlightColor);

    autoSave << new core::param::BoolParam(false);
    MakeSlotAvailable(&autoSave);

    outputPathParam << new core::param::FilePathParam("", core::param::FilePathParam::Flag_Directory_ToBeCreated);
    MakeSlotAvailable(&outputPathParam);
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
        auto input_image_hash = metadata.getHash();

        if (input_image_hash != image_hash) {
            image_hash = input_image_hash;
            display->updateTexture(*currentImage->getImageData());
        }

        if (auto* getData = getGraphCaller.CallAs<ImageSeries::GraphData2DCall>()) {
            if ((*getData)(ImageSeries::GraphData2DCall::CallGetData)) {
                auto input_graph_hash = util::combineHash<util::Hash>(
                    getData->DataHash(), util::computeHash(baseRadiusParam.Param<core::param::FloatParam>()->Value(),
                                             edgeWidthParam.Param<core::param::FloatParam>()->Value()));

                if (input_graph_hash != graph_hash) {
                    graph_hash = input_graph_hash;
                    display->updateGraph(*getData->GetOutput().graph->getData(),
                        baseRadiusParam.Param<core::param::FloatParam>()->Value(),
                        edgeWidthParam.Param<core::param::FloatParam>()->Value());
                }
            }
        }

        const auto tfInfo = getTransferFunction(display->getValueRange());
        display->updateTransferFunction(
            std::get<0>(tfInfo), {std::get<1>(tfInfo), std::get<2>(tfInfo)}, std::get<3>(tfInfo));
    }

    const auto& paramColor = highlightColor.Param<core::param::ColorParam>()->Value();
    display->setHighlight(highlight.Param<core::param::BoolParam>()->Value()
                              ? static_cast<float>(highlightValue.Param<core::param::IntParam>()->Value())
                              : 0.0f,
        glm::vec4(paramColor[0], paramColor[1], paramColor[2], paramColor[3]));

    display->setFilePath(autoSave.Param<core::param::BoolParam>()->Value(),
        outputPathParam.Param<core::param::FilePathParam>()->Value());

    return display->render(call, renderGraphParam.Param<core::param::BoolParam>()->Value());
}

bool ImageSeriesRenderer::displayModeChangedCallback(core::param::ParamSlot& param) {
    if (display) {
        display->setDisplayMode(static_cast<ImageDisplay2D::Mode>(param.Param<core::param::EnumParam>()->Value()));
    }
    return true;
}

std::tuple<unsigned int, float, float, unsigned int> ImageSeriesRenderer::getTransferFunction(
    const std::array<float, 2>& valueRange) {

    mmstd_gl::CallGetTransferFunctionGL* ct =
        this->getTransferFunctionCaller.CallAs<mmstd_gl::CallGetTransferFunctionGL>();

    if (ct != nullptr) {
        ct->SetRange(valueRange);

        if ((*ct)()) {
            return std::make_tuple(ct->OpenGLTexture(), ct->Range()[0], ct->Range()[1], ct->TextureSize());
        }
    }

    return std::make_tuple(0u, 0.0f, 1.0f, 0u);
}

} // namespace megamol::ImageSeries::GL
