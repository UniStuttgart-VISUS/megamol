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
#include <nanoflann.hpp>
#include <set>


namespace megamol {
namespace astro {

/*
 * THIS IS THE APEX OF SHIT and a non-quality copy from nanoflann/examples/utils.h
 * TODO: Replace it with a proper adapter instead of creating a copy to index data!
 */
template<typename T>
struct PointCloud {
    struct Point {
        T x, y, z;
    };

    std::vector<Point> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return pts.size();
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it
    //   again. Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template<class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
        return false;
    }
};

class FilamentFilter : public core::Module {
public:
    static const char* ClassName(void) {
        return "FilamentFilter";
    }
    static const char* Description(void) {
        return "Filters the filament particles of a AstroParticleDataCall";
    }
    static bool IsAvailable(void) {
        return true;
    }

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
    core::param::ParamSlot maxParticlePercentageCuttoff;

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

    /** flag determining whether the filaments have to be recalculated */
    bool recalculateFilaments;

    /** Hash of the last calculated dataset */
    uint64_t lastDataHash;

    /** Offset from the hash given by the incoming call */
    uint64_t hashOffset;

    /** ID of the last visualized timestep */
    uint32_t lastTimestep;
};

} // namespace astro
} // namespace megamol
