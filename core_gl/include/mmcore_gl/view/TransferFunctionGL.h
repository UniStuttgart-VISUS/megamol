/*
 * TransferFunctionGL.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "CallGetTransferFunctionGL.h"
#include "mmcore/view/AbstractTransferFunction.h"
#include "mmcore_gl/ModuleGL.h"


namespace megamol {
namespace core_gl {
namespace view {


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
    static const char* ClassName(void) {
        return "TransferFunctionGL";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module defining a piecewise linear transfer function";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    TransferFunctionGL(void);

    /** Dtor. */
    virtual ~TransferFunctionGL(void) {
        this->Release();
    }

private:
    // FUNCTIONS ----------------------------------------------------------

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * Callback called when the transfer function is requested.
     *
     * @param call The calling call
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool requestTF(core::Call& call);

    /** The OpenGL texture object id */
    unsigned int texID;

    /** A flag that signals whether the tf range from the project file should be ignored */
    bool ignore_project_range = true;
};


} /* end namespace view */
} // namespace core_gl
} /* end namespace megamol */
