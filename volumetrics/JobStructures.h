#ifndef MEGAMOLCORE_JOBSTRUCTURES_INCLUDED
#define MEGAMOLCORE_JOBSTRUCTURES_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/RawStorage.h"
#include "vislib/Cuboid.h"
#include "vislib/ColourRGBAu8.h"
#include "moldyn/MultiParticleDataCall.h"

namespace megamol {
namespace trisoup {
namespace volumetrics {

	struct FatVoxel {
		float distField;
		float *triangles;
		//unsigned char numTriangles;
        unsigned char mcCase;
		unsigned char consumedTriangles;

        inline bool operator ==(const FatVoxel &rhs) {
            return this->distField == rhs.distField
                && this->triangles == rhs.triangles
                && this->mcCase == rhs.mcCase
                && this->consumedTriangles == rhs.consumedTriangles;
        }
	};

	/**
	 *
	 */
	struct SubJobResult {

		// waer doch echt gut, wenn der schon komplette schalen getrennt rausgeben wuerde oder?

		// schon gleich in nem format, das die trisoup mag.

		// also einfach ueber das ganze volumen iterieren:
		// oberflaeche gefunden?
		//   ja: neu?
		//     ja: grow + tag cells
		//     nein: ignore

		//vislib::RawStorage vertexData;
		//vislib::RawStorage indexData;

		vislib::Array<vislib::math::Point<float, 3> > vertices;
		vislib::Array<vislib::math::Vector<float, 3> > normals;
		vislib::Array<vislib::graphics::ColourRGBAu8> colors;
		vislib::Array<unsigned int> indices;

        vislib::Array<vislib::Array<float *> > surfaces;
        vislib::Array<vislib::Array<FatVoxel> > borderVoxels;

		//float *normals;
		float surface;
		float volume;

		bool done;

		SubJobResult(void) : surface(0.0f), volume(0.0f), done(false) {
			vertices.SetCapacityIncrement(90);
			normals.SetCapacityIncrement(90);
			indices.SetCapacityIncrement(30);
			colors.SetCapacityIncrement(30);
		}
	};

	/**
	 * Structure that hold all parameters so a single thread can work
	 * on his subset of the total volume occupied by a particle list
	 */
	struct SubJobData {
		/**
		 * Volume to work on and rasterize the data of
		 */
		vislib::math::Cuboid<float> Bounds;

		unsigned int resX;

		unsigned int resY;

		unsigned int resZ;

		/**
		 * Maximum radius in the datasource.
		 */
		float MaxRad;

		float RadMult;

		/**
		 * Edge length of a voxel (set in accordance to the radii of the contained particles)
		 */
		float CellSize;

		/**
		 * All particles as taken directly from the respective Datacall
		 */
		//const core::moldyn::MultiParticleDataCall::Particles &Particles;
		
		core::moldyn::MultiParticleDataCall *datacall;

		/**
		 * Here the Job should store its results.
		 */
		SubJobResult Result;

		//SubJobData(const core::moldyn::MultiParticleDataCall::Particles &particles) : Particles(particles) {

		//}
	};


} /* end namespace volumetrics */
} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_JOBSTRUCTURES_INCLUDED */
