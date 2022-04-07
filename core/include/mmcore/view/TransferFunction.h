/*
 * TransferFunction.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Module.h"
#include "mmcore/view/AbstractTransferFunction.h"


namespace megamol {
namespace core {
namespace view {


/**
 * Module defining a transfer function.
 */
class MEGAMOLCORE_API TransferFunction : public Module, public AbstractTransferFunction {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "TransferFunction";
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
    TransferFunction(void);

    /** Dtor. */
    virtual ~TransferFunction(void) {
        this->Release();
    }

private:
    // FUNCTIONS ----------------------------------------------------------

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void) {
        return true;
    }

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void) {}

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


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */
