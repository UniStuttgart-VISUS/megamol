/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Module.h"
#include "mmstd/renderer/AbstractTransferFunction.h"

namespace megamol::core::view {

/**
 * Module defining a transfer function.
 */
class TransferFunction : public Module, public AbstractTransferFunction {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TransferFunction";
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
    TransferFunction();

    /** Dtor. */
    virtual ~TransferFunction() {
        this->Release();
    }

private:
    // FUNCTIONS ----------------------------------------------------------

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create() {
        return true;
    }

    /**
     * Implementation of 'Release'.
     */
    virtual void release() {}

    /**
     * Callback called when the transfer function is requested.
     *
     * @param call The calling call
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool requestTF(core::Call& call);

    /** A flag that signals whether the tf range from the project file should be ignored */
    bool ignore_project_range = true;
};

} // namespace megamol::core::view
