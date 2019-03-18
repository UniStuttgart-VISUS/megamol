/*
 * DirParticleListSelector.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/AbstractDirParticleManipulator.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

/**
 * Module thinning the number of particles
 *
 * Migrated from SGrottel particle's tool box
 */
class DirParticleListSelector : public AbstractDirParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "DirParticleListSelector"; }

    /** Return module class description */
    static const char* Description(void) {
        return "Selects a single list of particles from a DirectionalParticleDataCall";
    }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    /** Ctor */
    DirParticleListSelector(void);

    /** Dtor */
    virtual ~DirParticleListSelector(void);

protected:
    /**
     * Manipulates the particle data
     *
     * @remarks the default implementation does not changed the data
     *
     * @param outData The call receiving the manipulated data
     * @param inData The call holding the original data
     *
     * @return True on success
     */
    virtual bool manipulateData(megamol::core::moldyn::DirectionalParticleDataCall& outData,
        megamol::core::moldyn::DirectionalParticleDataCall& inData);

private:
    /** The list selection index */
    core::param::ParamSlot listIndexSlot;
};

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */
