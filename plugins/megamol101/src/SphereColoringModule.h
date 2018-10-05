/*
 * SphereColoringModule.h
 *
 * Copyright (C) 2016 by Karsten Schatz
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MM101PLG_SPHERECOLORINGMODULE_H_INCLUDED
#define MM101PLG_SPHERECOLORINGMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallSpheres.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace megamol101 {

/**
 * Module that assigns colors to incoming spheres.
 */
class SphereColoringModule : public core::Module {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "SphereColoringModule"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "Assigns colors to spheres contained in a CallSpheres."; }

    /**
     * Answer whether this module is available or not.
     *
     * @return 'true' if this module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Constructor. */
    SphereColoringModule(void);

    /** Destructor. */
    virtual ~SphereColoringModule(void);

private:
    /**
     * Checks whether at least one slot of this module is dirty.
     *
     * @return 'true' if at least one slot is dirty, 'false' otherwise.
     */
    bool areSlotsDirty(void);

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getDataCallback(core::Call& caller);

    /**
     * Gets the data extents from the source.
     *
     * @param caller The calling call.
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getExtentCallback(core::Call& caller);

    /**
     * Modifys the colors in a sphere call.
     *
     * @param cs The sphere call to modify.
     */
    void modifyColors(CallSpheres* cs);

    /**
     * Loads the specified file
     *
     * @param filename The file to load
     * @return 'true' on success, 'false' on failure.
     */
    virtual void release(void);

    /**
     * Resets the dirty state of all slots.
     */
    void resetDirtySlots(void);

    /** The slot for requesting data. */
    core::CalleeSlot outSlot;

    /** The slot for getting data. */
    core::CallerSlot inSlot;

    /** The pointer to the sphere colors */
    float* colors;

    /** The number of stored colors */
    SIZE_T numColor;

    /** The offset from the input hash */
    SIZE_T hashOffset;

    /** The last hash from the source data */
    SIZE_T lastHash;

    /** Internal dirtyness flag */
    bool isDirty;

    /** The slot for the minimum color */
    core::param::ParamSlot minColorSlot;

    /** The slot for the maximum color */
    core::param::ParamSlot maxColorSlot;

    /** Slot for the single color flag */
    core::param::ParamSlot singleColorSlot;

    /** Slot for activation and deactivation of this module */
    core::param::ParamSlot isActiveSlot;
};

} /* end namespace megamol101 */
} /* end namespace megamol */

#endif /* MM101PLG_SPHERECOLORINGMODULE_H_INCLUDED */
