/*
 * AbstractQuartzModule.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "QuartzCrystalDataCall.h"
#include "QuartzParticleGridDataCall.h"
#include "mmcore/CallerSlot.h"


namespace megamol {
namespace demos_gl {

/**
 * Abstract base class for quartzs data consuming modules
 */
class AbstractQuartzModule {
public:
    /**
     * Ctor
     */
    AbstractQuartzModule();

    /**
     * Dtor
     */
    virtual ~AbstractQuartzModule();

protected:
    /**
     * Answer the particle data from the connected module
     *
     * @return The particle data from the connected module or NULL if no
     *        data could be received
     */
    ParticleGridDataCall* getParticleData();

    /**
     * Answer the crystalite data from the connected module
     *
     * @return The crystalite data from the connected module or NULL if no
     *         data could be received
     */
    virtual CrystalDataCall* getCrystaliteData();

    /** The slot to get the data */
    core::CallerSlot dataInSlot;

    /** The slot to get the types data */
    core::CallerSlot typesInSlot;

    /** The data hash of the types data realized the last time */
    SIZE_T typesDataHash;
};

} // namespace demos_gl
} /* end namespace megamol */
