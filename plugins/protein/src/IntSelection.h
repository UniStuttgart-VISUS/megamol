/*
 * SIFFDataSource.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_PROTEIN_SELECTION_H_INCLUDED
#define MEGAMOL_PROTEIN_SELECTION_H_INCLUDED
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

namespace megamol::protein {

/**
 * zeug
 */
class IntSelection : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "IntSelection";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module holding a list of selected ints (IDs, ...)";
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
    IntSelection();

    /** Dtor. */
    ~IntSelection() override;

protected:
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

private:
    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getSelectionCallback(core::Call& caller);

    /**
     * Sets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool setSelectionCallback(core::Call& caller);

    /** The slot for requesting data */
    core::CalleeSlot getSelectionSlot;

    /** The data */
    vislib::Array<int> selection;

    /** The data hash */
    SIZE_T datahash;
};

} // namespace megamol::protein

#endif /* MEGAMOL_PROTEIN_SELECTION_H_INCLUDED */
