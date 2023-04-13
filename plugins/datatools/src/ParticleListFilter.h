/*
 * ParticleListFilter.h
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/types.h"


namespace megamol::datatools {


/**
 * Module to filter calls with multiple particle lists (currently directional and spherical) by list index
 */
class ParticleListFilter : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ParticleListFilter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module to filter calls with multiple particle lists (currently directional and spherical) by particle "
               "type";
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
    ParticleListFilter();

    /** Dtor. */
    ~ParticleListFilter() override;

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
     * Callback publishing the gridded data
     *
     * @param call The call requesting the gridded data
     *
     * @return 'true' on success, 'false' on failure
     */
    bool getDataCallback(core::Call& call);

    /**
     * Callback publishing the extend of the data
     *
     * @param call The call requesting the extend of the data
     *
     * @return 'true' on success, 'false' on failure
     */
    bool getExtentCallback(core::Call& call);

    /**
     * Tokenize includedListsSlot->GetValue into an array of type IDs
     *
     * @return the array of type IDs
     */
    vislib::Array<unsigned int> getSelectedLists();

    core::CallerSlot inParticlesDataSlot;

    core::CalleeSlot outParticlesDataSlot;

    core::param::ParamSlot includedListsSlot;

    core::param::ParamSlot includeAllSlot;

    core::param::ParamSlot globalColorMapComputationSlot;

    core::param::ParamSlot includeHiddenInColorMapSlot;

    SIZE_T datahashParticlesOut;

    SIZE_T datahashParticlesIn;

    unsigned int frameID;
};

} // namespace megamol::datatools
