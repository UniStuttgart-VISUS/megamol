/*
 * ParticleVisibilityFromVolume.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLEVISIBILITYFROMVOLUME_H_INCLUDED
#define MEGAMOLCORE_PARTICLEVISIBILITYFROMVOLUME_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "mmcore/misc/VolumetricDataCall.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

/**
 * Module filtering particles based on volumetric data.
 * Performs trilinear interpolation and applies a user-selected
 * operator to find which particles are kept.
 */
class ParticleVisibilityFromVolume : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "ParticleVisibilityFromVolume"; }

    /** Return module class description */
    static const char* Description(void) { return "Thins number of particles"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    /** Ctor */
    ParticleVisibilityFromVolume(void);

    /** Dtor */
    virtual ~ParticleVisibilityFromVolume(void);

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
    virtual bool manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData, megamol::core::moldyn::MultiParticleDataCall& inData);

private:
    /** What to do with the reference value (smaller, larger, epsilon-equal) */
    core::param::ParamSlot operatorSlot;

    /** reference value for the operation */
    core::param::ParamSlot valueSlot;

    /** epsilon for equality */
    core::param::ParamSlot epsilonSlot;

    core::param::ParamSlot absoluteSlot;

    core::param::ParamSlot minSlot, maxSlot;

    core::CallerSlot volumeSlot;

    std::vector<std::vector<uint8_t>> theVertexData;
    std::vector<std::vector<uint8_t>> theColorData;

    unsigned int lastTime = -1;
    SIZE_T lastParticleHash = 0;
    SIZE_T lastVolumeHash = 0;
};

} /* end namespace datatools */
} /* end namespace stdplugin */
} // namespace datatools

#endif /* MEGAMOLCORE_PARTICLEVISIBILITYFROMVOLUME_H_INCLUDED */
