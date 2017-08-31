/*
 * ParticleIColFilter.h
 *
 * Copyright (C) 2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_PARTICLEICOLFILTER_H_INCLUDED
#define MEGAMOLCORE_PARTICLEICOLFILTER_H_INCLUDED
#pragma once

#include "mmcore/Module.h"
#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "vislib/math/Cuboid.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/ParticleFilterMapDataCall.h"
#include <vector>


namespace megamol {
namespace stdplugin {
namespace datatools {


    /**
     * Removes particles outside a specific interval for I color values
     */
    class ParticleIColFilter : public AbstractParticleManipulator {
    public:
        static const char *ClassName(void) {
            return "ParticleIColFilter";
        }
        static const char *Description(void) {
            return "Removes particles outside a specific interval for I color values";
        }
        static bool IsAvailable(void) {
            return true;
        }
        static bool SupportQuickstart(void) {
            return false;
        }

        ParticleIColFilter(void);
        virtual ~ParticleIColFilter(void);

    protected:

        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:

        bool reset(core::param::ParamSlot&);
        void setData(core::moldyn::MultiParticleDataCall& inDat);
        void setData(core::moldyn::MultiParticleDataCall::Particles& p, vislib::RawStorage& d, const core::moldyn::SimpleSphericalParticles& s, vislib::math::Cuboid<float> bbox, ParticleFilterMapDataCall::index_t& mapOffset);

        bool getParticleMapData(core::Call& c);
        bool getParticleMapExtent(core::Call& c);
        bool getParticleMapHash(core::Call& c);

        core::CalleeSlot particleMapSlot;
        core::param::ParamSlot minValSlot;
        core::param::ParamSlot maxValSlot;
        core::param::ParamSlot staifHackDistSlot;
        size_t dataHash;
        unsigned int frameId;
        std::vector<core::moldyn::MultiParticleDataCall::Particles> parts;
        std::vector<vislib::RawStorage> data;
        std::vector<ParticleFilterMapDataCall::index_t> mapIndex;
        core::param::ParamSlot inValRangeSlot;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLEICOLFILTER_H_INCLUDED */
