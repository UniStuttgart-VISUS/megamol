/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmstd/renderer/CallGetTransferFunction.h"

namespace megamol::core::view {

/**
 * Module defining a transfer function.
 */
class AbstractTransferFunction {
public:
    AbstractTransferFunction()
            : getTFSlot("gettransferfunction", "Provides the transfer function")
            , tfParam("TransferFunction", "The transfer function serialized as JSON string.")
            , texSize(1)
            , tex()
            , texFormat(CallGetTransferFunction::TEXTURE_FORMAT_RGBA)
            , interpolMode(param::TransferFunctionParam::InterpolationMode::LINEAR)
            , range({0.0f, 1.0f})
            , version(0)
            , last_frame_id(0) {}


    virtual ~AbstractTransferFunction() {}

protected:
    /**
     * Callback called when the transfer function is requested.
     *
     * @param call The calling call
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool requestTF(core::Call& call) = 0;

    // VARIABLES ----------------------------------------------------------

    /** The callee slot called on request of a transfer function */
    core::CalleeSlot getTFSlot;

    /** Parameter containing the transfer function data serialized into JSON string */
    core::param::ParamSlot tfParam;

    /** The texture size in texel */
    unsigned int texSize;

    /** The texture data */
    std::vector<float> tex;

    /** The texture format */
    AbstractCallGetTransferFunction::TextureFormat texFormat;

    /** The interpolation mode */
    core::param::TransferFunctionParam::InterpolationMode interpolMode;

    /** The value range */
    std::array<float, 2> range;

    /** Version of texture */
    uint32_t version;

    /** Global frame ID */
    uint32_t last_frame_id;
};

} // namespace megamol::core::view
