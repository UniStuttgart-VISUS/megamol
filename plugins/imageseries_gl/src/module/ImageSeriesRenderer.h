/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "ImageDisplay2D.h"

#include "imageseries/ImageSeries2DCall.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "mmstd_gl/renderer/Renderer2DModuleGL.h"

#include "vislib/Pair.h"
#include "vislib/RawStorage.h"
#include "vislib/SmartPtr.h"
#include "vislib/graphics/AbstractBitmapCodec.h"
#include "vislib/math/Rectangle.h"

#include <array>
#include <memory>
#include <tuple>

namespace megamol::ImageSeries::GL {

class ImageSeriesRenderer : public mmstd_gl::Renderer2DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ImageSeriesRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Displays images from an image series.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    ImageSeriesRenderer();

    /** Dtor. */
    virtual ~ImageSeriesRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender2DGL& call) override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender2DGL& call) override;

    /**
     * Callback for changes to the 'display mode' parameter.
     */
    bool displayModeChangedCallback(core::param::ParamSlot& param);

    /**
     * Update the transfer function and get texture information.
     */
    std::tuple<unsigned int, float, float, unsigned int> getTransferFunction(const std::array<float, 2>& valueRange);

private:
    std::unique_ptr<ImageDisplay2D> display;
    std::shared_ptr<const AsyncImageData2D<>> currentImage;

    core::CallerSlot getDataCaller;
    core::CallerSlot getGraphCaller;
    core::CallerSlot getTransferFunctionCaller;

    core::param::ParamSlot displayModeParam;
    core::param::ParamSlot renderGraphParam;
    core::param::ParamSlot baseRadiusParam;
    core::param::ParamSlot edgeWidthParam;

    core::param::ParamSlot highlight;
    core::param::ParamSlot highlightValue;
    core::param::ParamSlot highlightColor;

    core::param::ParamSlot autoSave;
    core::param::ParamSlot outputPathParam;

    ImageSeries2DCall::Output metadata;

    SIZE_T image_hash, graph_hash;

    bool initialReadAhead = false;
    float lastReadAhead = 0.f;
};

} // namespace megamol::ImageSeries::GL
