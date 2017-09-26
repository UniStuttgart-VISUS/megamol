/*
 * ParticleSortFixHack.h
 *
 * Copyright (C) 2015 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLESORTFIXHACK_H_INCLUDED
#define MEGAMOLCORE_PARTICLESORTFIXHACK_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include <vector>
#include "vislib/math/Dimension.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module overriding global attributes of particles
     */
    class ParticleSortFixHack : public AbstractParticleManipulator {
    public:

        /** Return module class name */
        static const char *ClassName(void) {
            return "ParticleSortFixHack";
        }

        /** Return module class description */
        static const char *Description(void) {
            return "Uses heuristics in an atempt to fixe particle sorting (implicit ids)";
        }

        /** Module is always available */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        ParticleSortFixHack(void);

        /** Dtor */
        virtual ~ParticleSortFixHack(void);

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
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:

        class particle_data {
        public:
            particle_data() : parts(), dat() {}
            particle_data(const particle_data& src) : parts(), dat() {
                throw vislib::Exception("forbidden copy ctor", __FILE__, __LINE__);
            }
            ~particle_data() {}
            particle_data& operator=(const particle_data& src) {
                throw vislib::Exception("forbidden copy ctor", __FILE__, __LINE__);
            }

            megamol::core::moldyn::SimpleSphericalParticles parts;
            vislib::RawStorage dat;
        };

        bool updateIDdata(megamol::core::moldyn::MultiParticleDataCall& inData);
        void updateData(megamol::core::moldyn::MultiParticleDataCall& inData);
        void copyData(particle_data& tar, core::moldyn::SimpleSphericalParticles& src);

        double part_sqdist(const float *p1, const float *p2, const vislib::math::Dimension<float, 3>& bboxsize);

        std::vector<particle_data> data;
        unsigned int inDataTime;
        unsigned int outDataTime;

        std::vector<std::vector<std::vector<unsigned int> > > ids;
        size_t inDataHash;
        size_t outDataHash;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLESORTFIXHACK_H_INCLUDED */
