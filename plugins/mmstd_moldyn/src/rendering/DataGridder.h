/*
 * DataGridder.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DATAGRIDDER_H_INCLUDED
#define MEGAMOLCORE_DATAGRIDDER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "ParticleGridDataCall.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/RawStorage.h"
#include "vislib/types.h"


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {


/**
 * Renderer for rendering the vis logo into the unit cube.
 */
class DataGridder : public core::Module {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "DataGridder";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "A data storage module gridding flat particle data";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /**
     * Disallow usage in quickstarts
     *
     * @return false
     */
    static bool SupportQuickstart(void) {
        return false;
    }

    /** Ctor. */
    DataGridder(void);

    /** Dtor. */
    virtual ~DataGridder(void);

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
     * TODO: Document
     */
    static int doSort(const vislib::Pair<SIZE_T, unsigned int> &lhs,
        const vislib::Pair<SIZE_T, unsigned int> &rhs);

    /**
     * Callback publishing the gridded data
     *
     * @param call The call requesting the gridded data
     *
     * @return 'true' on success, 'false' on failure
     */
    bool getData(megamol::core::Call &call);

    /**
     * Callback publishing the extend of the data
     *
     * @param call The call requesting the extend of the data
     *
     * @return 'true' on success, 'false' on failure
     */
    bool getExtent(megamol::core::Call &call);

    /** Slot to fetch flat data */
    core::CallerSlot inDataSlot;

    /** Slot to publicate gridded data */
    core::CalleeSlot outDataSlot;

    /** The grid size in x direction */
    core::param::ParamSlot gridSizeXSlot;

    /** The grid size in y direction */
    core::param::ParamSlot gridSizeYSlot;

    /** The grid size in z direction */
    core::param::ParamSlot gridSizeZSlot;

    /** Flag to quantize the coordinates to shorts */
    core::param::ParamSlot quantizeSlot;

    /** The data hash code */
    SIZE_T datahash;

    /** The current frame id */
    unsigned int frameID;

    /** The grid size informations */
    unsigned int gridSizeX, gridSizeY, gridSizeZ;

    /** The particle types */
    vislib::Array<ParticleGridDataCall::ParticleType> types;

    /** The grid */
    vislib::Array<ParticleGridDataCall::GridCell> grid;

    /** The vert data */
    vislib::Array<vislib::RawStorage> vertData;

    /** The colour data */
    vislib::Array<vislib::RawStorage> colData;

    /** the out-going hash */
    SIZE_T outhash;

};

}
} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DATAGRIDDER_H_INCLUDED */
