/*
 * SimpleAstroFilter.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "astro/AstroDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include <set>

namespace megamol {
namespace astro {

class SimpleAstroFilter : public core::Module {
public:
    static const char* ClassName() {
        return "SimpleAstroFilter";
    }
    static const char* Description() {
        return "Filters the filament particles of a AstroParticleDataCall";
    }
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    SimpleAstroFilter();

    /** Dtor. */
    ~SimpleAstroFilter() override;

protected:
    bool create() override;
    void release() override;

private:
    bool getData(core::Call& call);
    bool getExtent(core::Call& call);

    void initFields();
    bool filter(const AstroDataCall& call);
    bool copyContentToOutCall(AstroDataCall& outCall);
    bool copyInCallToContent(const AstroDataCall& inCall, const std::set<uint64_t>& indexSet);
    bool isParamDirty();
    void resetDirtyParams();
    void setDisplayedValues(const AstroDataCall& outCall);

    core::CalleeSlot particlesOutSlot;
    core::CallerSlot particlesInSlot;

    core::param::ParamSlot showOnlyBaryonParam;
    core::param::ParamSlot showOnlyDarkMatterParam;
    core::param::ParamSlot showOnlyStarsParam;
    core::param::ParamSlot showOnlyWindParam;
    core::param::ParamSlot showOnlyStarFormingGasParam;
    core::param::ParamSlot showOnlyAGNsParam;

    core::param::ParamSlot minVelocityMagnitudeParam;
    core::param::ParamSlot maxVelocityMagnitudeParam;
    core::param::ParamSlot filterVelocityMagnitudeParam;

    core::param::ParamSlot minTemperatureParam;
    core::param::ParamSlot maxTemperatureParam;
    core::param::ParamSlot filterTemperatureParam;

    core::param::ParamSlot minMassParam;
    core::param::ParamSlot maxMassParam;
    core::param::ParamSlot filterMassParam;

    core::param::ParamSlot minInternalEnergyParam;
    core::param::ParamSlot maxInternalEnergyParam;
    core::param::ParamSlot filterInternalEnergyParam;

    core::param::ParamSlot minSmoothingLengthParam;
    core::param::ParamSlot maxSmoothingLengthParam;
    core::param::ParamSlot filterSmoothingLengthParam;

    core::param::ParamSlot minMolecularWeightParam;
    core::param::ParamSlot maxMolecularWeightParam;
    core::param::ParamSlot filterMolecularWeightParam;

    core::param::ParamSlot minDensityParam;
    core::param::ParamSlot maxDensityParam;
    core::param::ParamSlot filterDensityParam;

    core::param::ParamSlot minGravitationalPotentialParam;
    core::param::ParamSlot maxGravitationalPotentialParam;
    core::param::ParamSlot filterGravitationalPotentialParam;

    core::param::ParamSlot minEntropyParam;
    core::param::ParamSlot maxEntropyParam;
    core::param::ParamSlot filterEntropyParam;

    core::param::ParamSlot minAgnDistanceParam;
    core::param::ParamSlot maxAgnDistanceParam;
    core::param::ParamSlot filterAgnDistanceParam;

    core::param::ParamSlot fillFilterButtonParam;

    /** Pointer to the position array */
    vec3ArrayPtr positions = nullptr;

    /** Pointer to the velocity array */
    vec3ArrayPtr velocities = nullptr;

    /** Pointer to the temperature array */
    floatArrayPtr temperatures = nullptr;

    /** Pointer to the mass array */
    floatArrayPtr masses = nullptr;

    /** Pointer to the interal energy array */
    floatArrayPtr internalEnergies = nullptr;

    /** Pointer to the smoothing length array */
    floatArrayPtr smoothingLengths = nullptr;

    /** Pointer to the molecular weight array */
    floatArrayPtr molecularWeights = nullptr;

    /** Pointer to the density array */
    floatArrayPtr densities = nullptr;

    /** Pointer to the gravitational potential array */
    floatArrayPtr gravitationalPotentials = nullptr;

    /** Pointer to the entropy array */
    floatArrayPtr entropies = nullptr;

    /** Pointer to the baryon flag array */
    boolArrayPtr isBaryonFlags = nullptr;

    /** Pointer to the star flag array */
    boolArrayPtr isStarFlags = nullptr;

    /** Pointer to the wind flag array */
    boolArrayPtr isWindFlags = nullptr;

    /** Pointer to the star forming gas flag array */
    boolArrayPtr isStarFormingGasFlags = nullptr;

    /** Pointer to the AGN flag array */
    boolArrayPtr isAGNFlags = nullptr;

    /** Pointer to the particle ID array */
    idArrayPtr particleIDs = nullptr;

    /** Pointer to the agn distance array */
    floatArrayPtr agnDistances = nullptr;

    /** flag determining whether the filaments have to be recalculated */
    bool refilter;

    /** Hash of the last calculated dataset */
    uint64_t lastDataHash;

    /** Offset from the hash given by the incoming call */
    uint64_t hashOffset;

    /** ID of the last visualized timestep */
    uint32_t lastTimestep;
};

} // namespace astro
} // namespace megamol
