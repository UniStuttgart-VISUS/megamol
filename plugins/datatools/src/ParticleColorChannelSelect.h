/*
 * ParticleColorChannelSelect.h
 *
 * Copyright (C) 2016 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLECOLORCHANNELSELECT_H_INCLUDED
#define MEGAMOLCORE_PARTICLECOLORCHANNELSELECT_H_INCLUDED
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"
#include <map>
#include <utility>

namespace megamol::datatools {

/**
 * Selects one of the RGBA color channels as I color channel
 */
class ParticleColorChannelSelect : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "ParticleColorChannelSelect";
    }

    /** Return module class description */
    static const char* Description() {
        return "Selects one of the RGBA color channels as I color channel";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    ParticleColorChannelSelect();

    /** Dtor */
    ~ParticleColorChannelSelect() override;

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
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    core::param::ParamSlot channelSlot;
    size_t dataHash;
    std::map<const void*, std::pair<float, float>> colRange;
};

} // namespace megamol::datatools

#endif /* MEGAMOLCORE_PARTICLECOLORCHANNELSELECT_H_INCLUDED */
