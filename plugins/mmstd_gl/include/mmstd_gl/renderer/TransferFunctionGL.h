/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "CallGetTransferFunctionGL.h"
#include "mmstd/renderer/AbstractTransferFunction.h"
#include "mmstd_gl/ModuleGL.h"

namespace megamol::mmstd_gl {

/**
 * Module defining a transfer function.
 */
class TransferFunctionGL : public ModuleGL, public core::view::AbstractTransferFunction {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TransferFunctionGL";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module defining a piecewise linear transfer function";
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
    TransferFunctionGL();

    /** Dtor. */
    ~TransferFunctionGL() override {
        this->Release();
    }

private:
    // FUNCTIONS ----------------------------------------------------------

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * Callback called when the transfer function is requested.
     *
     * @param call The calling call
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool requestTF(core::Call& call) override;

    /** The OpenGL texture object id */
    unsigned int texID;

    /** A flag that signals whether the tf range from the project file should be ignored */
    bool ignore_project_range = true;

    /** Texture interpolation mode */
    core::param::ParamSlot interpolationParam;
};


} // namespace megamol::mmstd_gl
