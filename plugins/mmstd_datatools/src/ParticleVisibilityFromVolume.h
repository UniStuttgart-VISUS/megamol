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

#include "mmcore/CallerSlot.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/AbstractParticleManipulator.h"


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
    static const char* Description(void) { return "Module filtering particles based on volumetric data (in the same place in world space)"; }

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

    /** use absolute values instead */
    core::param::ParamSlot absoluteSlot;

    //core::param::ParamSlot cyclXSlot;
    //core::param::ParamSlot cyclYSlot;
    //core::param::ParamSlot cyclZSlot;

    /** "read-only" slots that show the actual range of values available */
    core::param::ParamSlot minSlot, maxSlot;

    /** the incoming volume */
    core::CallerSlot volumeSlot;

    /** we have to store the filtered data */
    std::vector<std::vector<uint8_t>> theVertexData;
    std::vector<std::vector<uint8_t>> theColorData;

    /** for change tracking: last frameID we pulled */
    unsigned int lastTime = -1;

    /** for change tracking: last hash of the particle source we pulled */
    SIZE_T lastParticleHash = 0;

    /** for change tracking: last hash of the volume source we pulled */
    SIZE_T lastVolumeHash = 0;
};

} /* end namespace datatools */
} /* end namespace stdplugin */
} // namespace datatools

#endif /* MEGAMOLCORE_PARTICLEVISIBILITYFROMVOLUME_H_INCLUDED */
