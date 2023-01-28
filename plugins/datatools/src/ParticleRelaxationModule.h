/*
 * ParticleRelaxationModule.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLERELAXATIONMODULE_H_INCLUDED
#define MEGAMOLCORE_PARTICLERELAXATIONMODULE_H_INCLUDED
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/RawStorage.h"


namespace megamol::datatools {

/**
 * Module overriding global attributes of particles
 */
class ParticleRelaxationModule : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "ParticleRelaxationModule";
    }

    /** Return module class description */
    static const char* Description() {
        return "Module relaxing particles to minimize overlapps";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    ParticleRelaxationModule();

    /** Dtor */
    ~ParticleRelaxationModule() override;

protected:
    /**
     * Manipulates the particle data extend information
     *
     * @param outData The call receiving the manipulated information
     * @param inData The call holding the original data
     *
     * @return True on success
     */
    bool manipulateExtent(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

    /**
     * Manipulates the particle data
     *
     * @param outData The call receiving the manipulated data
     * @param inData The call holding the original data
     *
     * @return True on success
     */
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    /** The hash id of the data stored */
    size_t dataHash;

    /** The frame id of the data stored */
    unsigned int frameId;

    /** The generated data */
    vislib::RawStorage data;

    /** The out data hash */
    SIZE_T outDataHash;

    /** The new bounding box */
    vislib::math::Cuboid<float> bbox;

    /** The new clip box */
    vislib::math::Cuboid<float> cbox;
};

} // namespace megamol::datatools

#endif /* MEGAMOLCORE_PARTICLERELAXATIONMODULE_H_INCLUDED */
