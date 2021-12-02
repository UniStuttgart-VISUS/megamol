/*
 * SIFFDataSource.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_PROTEIN_SELECTION_H_INCLUDED
#define MEGAMOL_PROTEIN_SELECTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

namespace megamol {
namespace protein {

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
    static const char* ClassName(void) {
        return "IntSelection";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module holding a list of selected ints (IDs, ...)";
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
    IntSelection(void);

    /** Dtor. */
    virtual ~IntSelection(void);

protected:
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

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOL_PROTEIN_SELECTION_H_INCLUDED */
