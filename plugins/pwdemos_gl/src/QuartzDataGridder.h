/*
 * QuartzDataGridder.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "QuartzParticleGridDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/PtrArray.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace demos_gl {

/**
 * Module for loading quartz particle data from binary-fortran files
 */
class DataGridder : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "QuartzDataGridder";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module for gridding quartz particle data";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    DataGridder(void);

    /** Dtor */
    virtual ~DataGridder(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Call callback to get the data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getData(core::Call& c);

    /**
     * Call callback to get the extent
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getExtent(core::Call& c);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:
    /** Clears the data members */
    void clearData(void);

    /** Fills the data members */
    void makeData(void);

    /** Tests if the data needs to be cleared */
    bool needClearData(void);

    /** Tests if the data needs to be (re-)made */
    bool needMakeData(void);

    /** The data callee slot */
    core::CalleeSlot dataOutSlot;

    /** The data caller slot */
    core::CallerSlot dataInSlot;

    /** The crystalite caller slot */
    core::CallerSlot crysInSlot;

    /** The size of the grid in x direction */
    core::param::ParamSlot gridSizeXSlot;

    /** The size of the grid in y direction */
    core::param::ParamSlot gridSizeYSlot;

    /** The size of the grid in z direction */
    core::param::ParamSlot gridSizeZSlot;

    /** The data hash of the incoming particle data */
    SIZE_T partHash;

    /** The data hash of the incoming crystal data */
    SIZE_T crysHash;

    /** The data hash of the outgoing data */
    SIZE_T dataHash;

    /** The cells of the grid */
    ParticleGridDataCall::Cell* cells;

    /** The bounding box */
    vislib::math::Cuboid<float> bbox;

    /** The clipping box */
    vislib::math::Cuboid<float> cbox;

    /** The array of the raw particle data pages */
    vislib::PtrArray<float, vislib::NullLockable, vislib::ArrayAllocator<float>> pdata;

    /** The lists storage pointer */
    ParticleGridDataCall::List* lists;
};

} // namespace demos_gl
} /* end namespace megamol */
