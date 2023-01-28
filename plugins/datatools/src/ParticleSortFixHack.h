/*
 * ParticleSortFixHack.h
 *
 * Copyright (C) 2015 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "vislib/RawStorage.h"
#include "vislib/math/Dimension.h"
#include <vector>


namespace megamol::datatools {

/**
 * Module overriding global attributes of particles
 */
class ParticleSortFixHack : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "ParticleSortFixHack";
    }

    /** Return module class description */
    static const char* Description() {
        return "Uses heuristics in an atempt to fixe particle sorting (implicit ids)";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    ParticleSortFixHack();

    /** Dtor */
    ~ParticleSortFixHack() override;

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
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

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

        geocalls::SimpleSphericalParticles parts;
        vislib::RawStorage dat;
    };

    bool updateIDdata(geocalls::MultiParticleDataCall& inData);
    void updateData(geocalls::MultiParticleDataCall& inData);
    void copyData(particle_data& tar, geocalls::SimpleSphericalParticles& src);

    double part_sqdist(const float* p1, const float* p2, const vislib::math::Dimension<float, 3>& bboxsize);

    std::vector<particle_data> data;
    unsigned int inDataTime;
    unsigned int outDataTime;

    std::vector<std::vector<std::vector<unsigned int>>> ids;
    size_t inDataHash;
    size_t outDataHash;
};

} // namespace megamol::datatools
