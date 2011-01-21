#ifndef MEGAMOLCORE_JOBSTRUCTURES_INCLUDED
#define MEGAMOLCORE_JOBSTRUCTURES_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/RawStorage.h"
#include "vislib/Cuboid.h"
#include "vislib/ColourRGBAu8.h"
#include "moldyn/MultiParticleDataCall.h"
#include "vislib/ShallowShallowTriangle.h"

namespace megamol {
namespace trisoup {
namespace volumetrics {

    class BorderVoxel {
    public:
        vislib::Array<float> triangles;
        unsigned int x;
        unsigned int y;
        unsigned int z;

        inline bool operator ==(const BorderVoxel &rhs) {
            return (this->x == rhs.x) && (this->y == rhs.y) && (this->z == rhs.z);
        }

        inline bool doesTouch(const BorderVoxel &rhs) {
            if ((this->x == rhs.x) || (this->y == rhs.y) || (this->z == rhs.z)) {
                vislib::math::ShallowShallowTriangle<float, 3> sst1(const_cast<float *>(this->triangles.PeekElements()));
                vislib::math::ShallowShallowTriangle<float, 3> sst2(const_cast<float *>(rhs.triangles.PeekElements()));
                for (int i = 0; i < this->triangles.Count() / 9; i++) {
                    for (int j = 0; j < rhs.triangles.Count() / 9; j++) {
                        sst1.SetPointer(const_cast<float *>(this->triangles.PeekElements() + i * 9));
                        sst2.SetPointer(const_cast<float *>(rhs.triangles.PeekElements() + j * 9));
                        if (sst1.HasCommonEdge(sst2)) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        inline bool doesTouch(const BorderVoxel *rhs) {
            return this->doesTouch(*rhs);
        }
    };
    
    struct FatVoxel {
		float distField;
		float *triangles;
		unsigned char numTriangles;
        /** for marching CUBES, just that. for marching tets, the < 0 corners */
        unsigned char mcCase;
		unsigned short consumedTriangles;
        BorderVoxel *borderVoxel;
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

		//vislib::Array<vislib::math::Point<float, 3> > vertices;
		//vislib::Array<vislib::math::Vector<float, 3> > normals;
		//vislib::Array<vislib::graphics::ColourRGBAu8> colors;
		//vislib::Array<unsigned int> indices;

        vislib::Array<vislib::Array<float> > surfaces;
        vislib::Array<vislib::Array<BorderVoxel *> > borderVoxels;
        vislib::Array<float> surfaceSurfaces;
        //vislib::Array<float> volumes;

		//float *normals;
		//float surface;
		//float volume;

		bool done;

		SubJobResult(void) : done(false) {
			//vertices.SetCapacityIncrement(90);
			//normals.SetCapacityIncrement(90);
			//indices.SetCapacityIncrement(30);
			//colors.SetCapacityIncrement(30);
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
