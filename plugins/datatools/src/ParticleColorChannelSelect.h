/*
 * ParticleColorChannelSelect.h
 *
 * Copyright (C) 2016 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLECOLORCHANNELSELECT_H_INCLUDED
#define MEGAMOLCORE_PARTICLECOLORCHANNELSELECT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"
#include <map>
#include <utility>

namespace megamol {
namespace datatools {

/**
 * Selects one of the RGBA color channels as I color channel
 */
class ParticleColorChannelSelect : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "ParticleColorChannelSelect";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Selects one of the RGBA color channels as I color channel";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    ParticleColorChannelSelect(void);

    /** Dtor */
    virtual ~ParticleColorChannelSelect(void);

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
    virtual bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData);

private:
    core::param::ParamSlot channelSlot;
    size_t dataHash;
    std::map<const void*, std::pair<float, float>> colRange;
};

} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLECOLORCHANNELSELECT_H_INCLUDED */
