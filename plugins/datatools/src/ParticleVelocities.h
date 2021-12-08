/*
 * ParticleVelocities.h
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Vector.h"
#include <map>
#include <vector>


namespace megamol {
namespace datatools {

/**
 * Module computing velocities from a MultiParticleDataCall.
 * Particle numbers must not change between frames. Also particle
 * order must always be the same to allow for identification of
 * particles. Resulting DirectionalParticleDataCall does not have the
 * first frame of the original data, as velocities are computed from
 * frame i-1 to i.
 */
class ParticleVelocities : public megamol::core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "ParticleVelocities";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Computes velocities of (sorted, synchronized) particles across frames i and i-1. Reduces the number of "
               "available time steps by 1.";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    ParticleVelocities(void);

    /** Dtor */
    virtual ~ParticleVelocities(void);

protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);

private:
    /**
     * Called when the data is requested by this module
     *
     * @param c The incoming call
     *
     * @return True on success
     */
    bool getDataCallback(megamol::core::Call& c);

    /**
     * Called when the extend information is requested by this module
     *
     * @param c The incoming call
     *
     * @return True on success
     */
    bool getExtentCallback(megamol::core::Call& c);

    bool assertData(geocalls::MultiParticleDataCall* in, geocalls::MultiParticleDataCall* outMPDC);

    core::param::ParamSlot cyclXSlot;
    core::param::ParamSlot cyclYSlot;
    core::param::ParamSlot cyclZSlot;
    core::param::ParamSlot dtSlot;
    size_t datahash;
    unsigned int time;

    int cachedNumLists;
    unsigned int cachedTime;
    std::vector<float> cachedGlobalRadius;
    std::vector<size_t> cachedListLength;
    std::vector<size_t> cachedStride;
    std::vector<geocalls::MultiParticleDataCall::Particles::VertexDataType> cachedVertexDataType;
    std::vector<void*> cachedVertexData;
    std::vector<float*> cachedDirData;


    /** The slot providing access to the manipulated data */
    megamol::core::CalleeSlot outDataSlot;

    /** The slot accessing the original data */
    megamol::core::CallerSlot inDataSlot;
};

} /* end namespace datatools */
} /* end namespace megamol */
