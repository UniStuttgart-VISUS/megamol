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

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"


namespace megamol {
namespace stdplugin {
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
        static const char *ClassName(void) {
            return "ParticleBoxGeneratorDataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Simple particle data generator filling a box";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
		ParticleBoxGeneratorDataSource(void);

        /** Dtor. */
		virtual ~ParticleBoxGeneratorDataSource(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

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

		void clear(void);

        /**
         * Ensures that the data file is loaded into memory, if possible
         */
        void assertData(void);

        core::param::ParamSlot particleCountSlot;
		core::param::ParamSlot radiusPerParticleSlot;
		core::param::ParamSlot colorDataSlot;
		core::param::ParamSlot interleavePosAndColorSlot;
		core::param::ParamSlot radiusScaleSlot;
		core::param::ParamSlot positionNoiseSlot;

		size_t dataHash;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLEBOXGENERATORDATASOURCE_H_INCLUDED */

#if 0
#pragma once
#include "DataStore.h"
#include <random>

enum class DataScenario {
	Box,
	Line
};

///
/// Particle will not overlap, but touch
///
class DataGenerator {
public:
	DataGenerator(uint32_t c = 1000000u, DataScenario s = DataScenario::Box, DataLayout l = DataLayout::XYZRI, uint32_t r_seed = 5489u);
	~DataGenerator();

	void Generate(DataStore& store);

	inline DataGenerator& SetCount(uint32_t c) { cnt = c; return *this; }
	inline uint32_t Count() const { return cnt; }
	inline DataGenerator& SetScenario(DataScenario s) { scenario = s; return *this; }
	inline DataScenario Scenario() const { return scenario; }
	inline DataGenerator& SetLayout(DataLayout l) { layout = l; return *this; }
	inline DataLayout Layout() const { return layout; }
	inline DataGenerator& SetRandomSeed(uint32_t s) { rnd_seed = s; return *this; }
	inline uint32_t RandomSeed() const { return rnd_seed; }

	///
	/// Radius scale factor
	///
	/// A factor of 1.0 will result in touch spheres if there is no position noise
	///
	inline DataGenerator& SetRadiusScale(float s) { rad_param = s; return *this; }
	inline float RadiusScale() const { return rad_param; }

	///
	/// Position noise scale
	///
	/// A value of 1.0 will displace a particle 1xrad from it's original position
	///
	inline DataGenerator& SetPositionNoiseScale(float s) { pos_param = s;  return *this; }
	inline float PositionNoiseScale() const { return pos_param; }

private:

	uint32_t cnt;
	DataScenario scenario;
	DataLayout layout;
	uint32_t rnd_seed;

	float rad_param;
	float pos_param;

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

	template<class T_engine, class T_dist>
	float makePackedColorRGBA(T_dist& dist, T_engine& eng) {
		float r;
		unsigned char *c = reinterpret_cast<unsigned char *>(&r);
		c[0] = static_cast<unsigned char>(dist(eng)); // red
		c[1] = static_cast<unsigned char>(dist(eng)); // green
		c[2] = static_cast<unsigned char>(dist(eng)); // blue
		c[3] = 255; // alpha
		return r;
	}


};
#endif
