/*
 * FilamentFilter.h
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
#include "nanoflann.hpp"
#include "utils.h"
#include <set>


namespace megamol {
namespace astro {

class FilamentFilter : public core::Module {
public:
    static const char* ClassName(void) { return "FilamentFilter"; }
    static const char* Description(void) { return "Filters the filament particles of a AstroParticleDataCall"; }
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    FilamentFilter(void);

    /** Dtor. */
    virtual ~FilamentFilter(void);

protected:
    virtual bool create(void);
    virtual void release(void);

private:
    bool getData(core::Call& call);
    bool getExtent(core::Call& call);

    void initFields(void);
    std::pair<float, float> getMinMaxDensity(const AstroDataCall& call) const;
    void retrieveDensityCandidateList(const AstroDataCall& call, std::vector<std::pair<float, uint64_t>>& result);
    bool filterFilaments(const AstroDataCall& call);
    bool copyContentToOutCall(AstroDataCall& outCall);
	bool copyInCallToContent(const AstroDataCall& inCall, const std::set<uint64_t>& indexSet);
    void initSearchStructure(const AstroDataCall& call);

    core::CalleeSlot filamentOutSlot;
    core::CallerSlot particlesInSlot;

    core::param::ParamSlot radiusSlot;
    core::param::ParamSlot minClusterSizeSlot;
    core::param::ParamSlot densitySeedPercentageSlot;
    core::param::ParamSlot isActiveSlot;

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>,
        PointCloud<float>, 3>
        my_kd_tree_t;
    
    std::shared_ptr<my_kd_tree_t> searchIndexPtr = nullptr;
    PointCloud<float> pointCloud;

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

    bool recalculateFilaments;
    uint64_t lastDataHash;
    uint64_t hashOffset;
    uint32_t lastTimestep;
};

} // namespace astro
} // namespace megamol
