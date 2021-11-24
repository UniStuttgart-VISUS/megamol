/*
 * OverrideParticleBBox.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_OVERRIDEPARTICLEBBOX_H_INCLUDED
#define MEGAMOLCORE_OVERRIDEPARTICLEBBOX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace datatools {

/**
 * Module overriding global attributes of particles
 */
class OverrideParticleBBox : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "OverrideParticleBBox";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Module overriding the bounding box of particle data";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    OverrideParticleBBox(void);

    /** Dtor */
    virtual ~OverrideParticleBBox(void);

protected:
    virtual bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData);
    virtual bool manipulateExtent(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData);

private:
    core::param::ParamSlot overrideBBoxSlot;
    core::param::ParamSlot overrideLBBoxSlot;
    core::param::ParamSlot bboxMinSlot;
    core::param::ParamSlot bboxMaxSlot;
    core::param::ParamSlot resetSlot;
    core::param::ParamSlot autocomputeSlot;
    core::param::ParamSlot autocomputeSamplesSlot;
    core::param::ParamSlot autocomputeXSlot;
    core::param::ParamSlot autocomputeYSlot;
    core::param::ParamSlot autocomputeZSlot;
};

} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_OVERRIDEPARTICLEBBOX_H_INCLUDED */
