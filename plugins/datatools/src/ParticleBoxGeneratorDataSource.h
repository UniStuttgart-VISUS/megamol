/*
 * ParticleBoxGeneratorDataSource.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLEBOXGENERATORDATASOURCE_H_INCLUDED
#define MEGAMOLCORE_PARTICLEBOXGENERATORDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/RawStorage.h"
#include <cstdint>


namespace megamol {
namespace datatools {


/**
 * Particle data generator
 */
class ParticleBoxGeneratorDataSource : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ParticleBoxGeneratorDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Simple particle data generator filling a box";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    ParticleBoxGeneratorDataSource();

    /** Dtor. */
    ~ParticleBoxGeneratorDataSource() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    typedef geocalls::SimpleSphericalParticles Particles;

    bool reseed(core::param::ParamSlot& p);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getExtentCallback(core::Call& caller);

    void clear();

    /**
     * Ensures that the data file is loaded into memory, if possible
     */
    void assertData();

    core::CalleeSlot dataSlot;

    core::param::ParamSlot randomSeedSlot;
    core::param::ParamSlot randomReseedSlot;
    core::param::ParamSlot particleCountSlot;
    core::param::ParamSlot radiusPerParticleSlot;
    core::param::ParamSlot colorDataSlot;
    core::param::ParamSlot interleavePosAndColorSlot;
    core::param::ParamSlot radiusScaleSlot;
    core::param::ParamSlot positionNoiseSlot;

    size_t dataHash;

    uint64_t cnt;
    vislib::RawStorage data;
    float rad;
    Particles::VertexDataType vdt;
    void* vdp;
    unsigned int vds;
    Particles::ColourDataType cdt;
    void* cdp;
    unsigned int cds;
};

} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLEBOXGENERATORDATASOURCE_H_INCLUDED */

#if 0

    ///// l2 norm
    //template<class T_engine>
    //void makeRandomNormal(float& outx, float& outy, float& outz, std::normal_distribution<float>& dist, T_engine& eng) {
    //    float len = 0.0f;
    //    while (len < 0.001f) {
    //        outx = dist(eng);
    //        outy = dist(eng);
    //        outz = dist(eng);
    //        len = std::sqrt(outx * outx + outy * outy + outz * outz);
    //    }
    //    outx /= len;
    //    outy /= len;
    //    outz /= len;
    //}

    ///// l-inf norm
    //template<class T_engine>
    //void makeRandomNormal_2(float& outx, float& outy, float& outz, std::normal_distribution<float>& dist, T_engine& eng) {
    //    float len = 0.0f;
    //    while (len < 0.001f) {
    //        outx = dist(eng);
    //        outy = dist(eng);
    //        outz = dist(eng);
    //        len = std::max(std::max(std::abs(outx), std::abs(outy)), std::abs(outz));
    //    }
    //    outx /= len;
    //    outy /= len;
    //    outz /= len;
    //}

    template<class T_engine, class T_dist>
    void addNoise(float& x, float& y, float& z, float scale,
        T_dist& dist, T_engine& eng) {
        //float dx, dy, dz;
        //makeRandomNormal_2(dx, dy, dz, dist, eng);
        x += (dist(eng) * 2.0f - 1.0f) * scale;
        y += (dist(eng) * 2.0f - 1.0f) * scale;
        z += (dist(eng) * 2.0f - 1.0f) * scale;
    }

};
#endif
