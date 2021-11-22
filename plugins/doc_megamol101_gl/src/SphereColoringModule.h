/**
 * MegaMol
 * Copyright (c) 2016, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOL_MEGAMOL101_SPHERECOLORINGMODULE_H
#define MEGAMOL_MEGAMOL101_SPHERECOLORINGMODULE_H

#include "CallSpheres.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::megamol101_gl {

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
    static const char* ClassName() {
        return "SphereColoringModule";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Assigns colors to spheres contained in a CallSpheres.";
    }

    /**
     * Answer whether this module is available or not.
     *
     * @return 'true' if this module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Constructor. */
    SphereColoringModule();

    /** Destructor. */
    ~SphereColoringModule() override;

private:
    /**
     * Checks whether at least one slot of this module is dirty.
     *
     * @return 'true' if at least one slot is dirty, 'false' otherwise.
     */
    bool areSlotsDirty();

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

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
    void release() override;

    /**
     * Resets the dirty state of all slots.
     */
    void resetDirtySlots();

    /** The slot for requesting data. */
    core::CalleeSlot outSlot;

    /** The slot for getting data. */
    core::CallerSlot inSlot;

    /** The pointer to the sphere colors */
    float* colors;

    /** The number of stored colors */
    std::size_t numColor;

    /** The offset from the input hash */
    std::size_t hashOffset;

    /** The last hash from the source data */
    std::size_t lastHash;

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

} // namespace megamol::megamol101_gl

#endif // MEGAMOL_MEGAMOL101_SPHERECOLORINGMODULE_H
