/*
 * ParticleNeighborhood.h
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "datatools/PointcloudHelpers.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include <nanoflann.hpp>
#include <vector>

namespace megamol {
namespace datatools {

/**
 * Module overriding global attributes of particles
 */
class ParticleNeighborhood : public megamol::core::Module {
public:
    enum searchTypeEnum { RADIUS, NUM_NEIGHBORS };

    /** Return module class name */
    static const char* ClassName(void) {
        return "ParticleNeighborhood";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Helps track a single particle and its close neighbors.";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    ParticleNeighborhood(void);

    /** Dtor */
    virtual ~ParticleNeighborhood(void);

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

protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);

private:
    bool assertData(megamol::core::AbstractGetData3DCall* in, megamol::core::AbstractGetData3DCall* out);

    core::param::ParamSlot cyclXSlot;
    core::param::ParamSlot cyclYSlot;
    core::param::ParamSlot cyclZSlot;
    core::param::ParamSlot radiusSlot;
    core::param::ParamSlot numNeighborSlot;
    core::param::ParamSlot searchTypeSlot;
    core::param::ParamSlot particleNumberSlot;
    size_t datahash;
    int lastTime;
    std::vector<float> newColors;
    std::vector<size_t> allParts;
    float maxDist;

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, simplePointcloud>, simplePointcloud,
        3 /* dim */, std::size_t>
        my_kd_tree_t;

    std::shared_ptr<my_kd_tree_t> particleTree;
    std::shared_ptr<simplePointcloud> myPts;

    /** The slot providing access to the manipulated data */
    megamol::core::CalleeSlot outDataSlot;

    /** The slot accessing the original data */
    megamol::core::CallerSlot inDataSlot;
};

} /* end namespace datatools */
} /* end namespace megamol */
