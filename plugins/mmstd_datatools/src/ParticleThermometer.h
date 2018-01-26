/*
 * ParticleThermometer.h
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_DATATOOLS_PARTICLETHERMOMETER_H_INCLUDED
#define MMSTD_DATATOOLS_PARTICLETHERMOMETER_H_INCLUDED
#pragma once

#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "PointcloudHelpers.h"
#include <vector>
#include "nanoflann.hpp"

namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module overriding global attributes of particles
     */
    class ParticleThermometer : public megamol::core::Module {
    public:

        enum searchTypeEnum {
            RADIUS,
            NUM_NEIGHBORS
        };

        /** Return module class name */
        static const char *ClassName(void) {
            return "ParticleThermometer";
        }

        /** Return module class description */
        static const char *Description(void) {
            return "Computes an intensity from the kinetic energy of a particle (compared to its surroundings).";
        }

        /** Module is always available */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        ParticleThermometer(void);

        /** Dtor */
        virtual ~ParticleThermometer(void);

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

        bool assertData(core::moldyn::DirectionalParticleDataCall *in,
            core::moldyn::MultiParticleDataCall *outMPDC, core::moldyn::DirectionalParticleDataCall *outDPDC);

        core::param::ParamSlot cyclXSlot;
        core::param::ParamSlot cyclYSlot;
        core::param::ParamSlot cyclZSlot;
        core::param::ParamSlot radiusSlot;
        core::param::ParamSlot numNeighborSlot;
        core::param::ParamSlot searchTypeSlot;
        core::param::ParamSlot minTempSlot;
        core::param::ParamSlot maxTempSlot;
        core::param::ParamSlot massSlot;
        core::param::ParamSlot freedomSlot;
        
        size_t datahash;
        int lastTime;
        std::vector<float> newColors;
        std::vector<size_t> allParts;
        float maxDist;

        typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, directionalPointcloud>,
            directionalPointcloud,
            3 /* dim */
        > my_kd_tree_t;

        std::shared_ptr<my_kd_tree_t> particleTree;
        std::shared_ptr<directionalPointcloud> myPts;

        /** The slot providing access to the manipulated data */
        megamol::core::CalleeSlot outDataSlot;

        /** The slot accessing the original data */
        megamol::core::CallerSlot inDataSlot;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_DATATOOLS_PARTICLETHERMOMETER_H_INCLUDED */
