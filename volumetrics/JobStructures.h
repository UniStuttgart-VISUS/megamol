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
#include "vislib/forceinline.h"

namespace megamol {
namespace trisoup {
namespace volumetrics {

    /** typdef steering the arithmetic precision of the voxelizer. */
    typedef float VoxelizerFloat;

    /**
     * Utility class for less precise (larger-epsilon) comparison between doubles.
     * Employed for stitching.
     */
    class Dowel {
    private:
        /**
         * Test for equality of <double, 3> points. This operation uses the
         * IsEqual with a hard-coded, large, epsilon.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this point are equal, false otherwise.
         */
        template <class Tp>
        VISLIB_FORCEINLINE static bool IsEqual(const vislib::math::AbstractPoint<double, 3, Tp> &a,
            const vislib::math::AbstractPoint<double, 3, Tp> &b) {
                return a == b;
                // TODO ZOMG OMGWTF BUG FIXME
                //return (vislib::math::IsEqual(a.GetX(), b.GetX(), 0.00001))
                //    && (vislib::math::IsEqual(a.GetY(), b.GetY(), 0.00001))
                //    && (vislib::math::IsEqual(a.GetZ(), b.GetZ(), 0.00001));
        }

        /**
         * Test for equality of <float, 3> points. This operation is a pass-through to
         * Point's == operator.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this point are equal, false otherwise.
         */
        template <class Tp>
        VISLIB_FORCEINLINE static bool IsEqual(const vislib::math::AbstractPoint<float, 3, Tp> &a,
            const vislib::math::AbstractPoint<float, 3, Tp> &b) {
                return a == b;
        }

    public:
		/**
         * Answer whether this and rhs have at least one edge in common. Uses Dowel::IsEqual.
		 *
		 * @param rhs the other triangle
		 *
		 * @return true if this and rhs have one edge in common
		 */
        VISLIB_FORCEINLINE static bool HaveCommonEdge(const vislib::math::ShallowShallowTriangle<double, 3> &a,
            const vislib::math::ShallowShallowTriangle<double, 3> &b) {
                return ((Dowel::IsEqual(a.PeekCoordinates()[0], b.PeekCoordinates()[0])
                    || Dowel::IsEqual(a.PeekCoordinates()[0], b.PeekCoordinates()[1])
                    || Dowel::IsEqual(a.PeekCoordinates()[0], b.PeekCoordinates()[2])) ? 1U : 0U)
				+ ((Dowel::IsEqual(a.PeekCoordinates()[1], b.PeekCoordinates()[0])
				|| Dowel::IsEqual(a.PeekCoordinates()[1], b.PeekCoordinates()[1])
				|| Dowel::IsEqual(a.PeekCoordinates()[1], b.PeekCoordinates()[2])) ? 1U : 0U)
				+ ((Dowel::IsEqual(a.PeekCoordinates()[2], b.PeekCoordinates()[0])
				|| Dowel::IsEqual(a.PeekCoordinates()[2], b.PeekCoordinates()[1])
				|| Dowel::IsEqual(a.PeekCoordinates()[2], b.PeekCoordinates()[2])) ? 1U : 0U) >= 2U;
		}

		/**
         * Answer whether this and rhs have at least one edge in common. This operation is a pass-through to
         * Triangle's == operator.
         * 
		 * @param rhs the other triangle
		 *
		 * @return true if this and rhs have one edge in common
		 */
        VISLIB_FORCEINLINE static bool HaveCommonEdge(const vislib::math::ShallowShallowTriangle<float, 3> &a,
            const vislib::math::ShallowShallowTriangle<float, 3> &b) {
                return a.HasCommonEdge(b);
		}
    };

    /**
     * Class used for storing information about voxels that potentially touch neighboring
     * sub-volumes and thus need to be checked for stitching. A copy of the resulting
     * geometry is retained such that the FatVoxel volume can be safely discarded.
     */
    class BorderVoxel {
    public:
        /** Array of triangles in this voxel */
        vislib::Array<VoxelizerFloat> triangles;

        /** ABSOLUTE x coordinate in the virtual volume encompassing the whole dataset */
        unsigned int x;

        /** ABSOLUTE y coordinate in the virtual volume encompassing the whole dataset */
        unsigned int y;

        /** ABSOLUTE z coordinate in the virtual volume encompassing the whole dataset */
        unsigned int z;

        /** 
         * Test for equality.
         *
         * @param rhs The right side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        inline bool operator ==(const BorderVoxel &rhs) {
            return (this->x == rhs.x) && (this->y == rhs.y) && (this->z == rhs.z);
        }

        /**
         * Test whether the geometry in two BorderVoxels does touch, i.e. have a common
         * edge. For stitching surfaces. If double-precision arithmetic is used, the Dowel
         * utility class is employed for a more forgiving epsilon.
         *
         * @param rhs The right side operand.
         *
         * @return true, if *this and rhs contain geometry that can be stitched.
         */
        inline bool doesTouch(const BorderVoxel &rhs) {
            // this one can only work for marching cubes since there are only edges on cube faces! (I think)
            //if ((this->x == rhs.x) || (this->y == rhs.y) || (this->z == rhs.z)) {

            // TODO this is slow. do something inline-ey
            //if (vislib::math::Point<float, 3>(this->x, this->y, this->z).Distance(
            //            vislib::math::Point<float, 3>(rhs.x, rhs.y, rhs.z)) < 2) {
            if (((this->x - rhs.x) * (this->x - rhs.x) + (this->y - rhs.y) * (this->y - rhs.y)
                    + (this->z - rhs.z) * (this->z - rhs.z)) <= 2) {
                vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> 
                    sst1(const_cast<VoxelizerFloat *>(this->triangles.PeekElements()));
                vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> 
                    sst2(const_cast<VoxelizerFloat *>(rhs.triangles.PeekElements()));
                for (int i = 0; i < this->triangles.Count() / 9; i++) {
                    for (int j = 0; j < rhs.triangles.Count() / 9; j++) {
                        sst1.SetPointer(const_cast<VoxelizerFloat *>(this->triangles.PeekElements() + i * 9));
                        sst2.SetPointer(const_cast<VoxelizerFloat *>(rhs.triangles.PeekElements() + j * 9));
                        if (Dowel::HaveCommonEdge(sst1, sst2)) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        /**
         * Test whether the geometry in two BorderVoxels does touch, i.e. have a common
         * edge. For stitching surfaces. If double-precision arithmetic is used, the Dowel
         * utility class is employed for a more forgiving epsilon.
         *
         * @param rhs The right side operand.
         *
         * @return true, if *this and *rhs contain geometry that can be stitched.
         */
        inline bool doesTouch(const BorderVoxel *rhs) {
            return this->doesTouch(*rhs);
        }
    };
    
    /**
     * Data structure for the sampling of a glyph-based dataset into a distance field.
     * Currently the whole code assumes the lowest-value (lower-left-behind) corner
     * of the voxel as the point sampled.
     */
    struct FatVoxel {
        /**
         * Distance to the nearest glyph surface. Negative distances indicate
         * we are inside the geometry.
         */
		VoxelizerFloat distField;

        /** Pointer to the memory holding numTriangles * 3 * 3 VoxelizerFloats. */
		VoxelizerFloat *triangles;

        /**
         * Pointer to the memory holding numTriangles VoxelizerFloats containing the
         * volume associated with each triangle.
         */
        VoxelizerFloat *volumes;

        /** Number of triangles contained in this FatVoxel */
		unsigned char numTriangles;

        /**
         * For marching CUBES, just that: the case from the table. For marching tets,
         * the Number of < 0, i.e. inside-the-volume corners
         */
        unsigned char mcCase;

        /**
         * bit field to remember the triangles already collected whe stitching
         * surfaces.
         */
		unsigned short consumedTriangles;

        /**
         * BorderVoxel containing a copy of the geometry for those FatVoxels that
         * potentially touch other subvolumes. This memory must NOT be deleted alongside
         * the FatVoxel since it is linked for fast access/deduplication, but needs to
         * survive the volume itself.
         */
        BorderVoxel *borderVoxel;
	};

    /**
     * represents a contiguous surface inside a subvolume and associated
     * metrics.
     */
    struct Surface {
        /** array of coordinates forming triangles making up the surface */
        vislib::Array<VoxelizerFloat> mesh;

        /** array of Bordervoxels */
        vislib::Array<BorderVoxel *> border;

        /** surface area */
        VoxelizerFloat surface;

        /** volume encompassed by this surface */
        VoxelizerFloat volume;

        /**
         * bit field indicating on which subvolume faces the attachment of a
         * full subvolume is indicated.
         */
        unsigned char fullFaces;

        /**
         * Answer the equality of this and rhs. Equality actually means
         * equality of surface and volume.
         *
         * @param rhs the right hand side operand
         *
         * @return whether this and rhs are equal
         */
        inline bool operator ==(const Surface &rhs) {
            return (this->surface == rhs.surface) && (this->volume == rhs.volume);
        }
    };
    
    /**
	 * Struct containing the results of a marching (whatever) on an instance of SubJubData.
	 */
	struct SubJobResult {
        ///** (outer) array of triangles per surface, (inner) array of coordinates per triangle */
        //vislib::Array<vislib::Array<VoxelizerFloat> > surfaces;

        ///** (outer) array of BorderVoxels corresponding to each of the surfaces */
        //vislib::Array<vislib::Array<BorderVoxel *> > borderVoxels;

        ///** computed surface area per surface */
        //vislib::Array<VoxelizerFloat> surfaceSurfaces;

        ///** computed volume corresponding to each of the surfaces */
        //vislib::Array<VoxelizerFloat> volumes;
        vislib::Array<Surface> surfaces;

        /** whether this job has completed runnning */
		bool done;

        /** ctor (yuck). mostly sets done to false just to be sure */
		SubJobResult(void) : done(false) {
		}
	};

	/**
	 * Structure that hold all parameters so a single thread can work
	 * on his subset of the total volume occupied by a particle list
	 */
	struct SubJobData {
		/** Volume to work on and rasterize the data of */
        vislib::math::Cuboid<VoxelizerFloat> Bounds;

        /** volume resolution, i.e. number of subdivisions with respect to bounds */
		int resX;

        /** volume resolution, i.e. number of subdivisions with respect to bounds */
		int resY;

        /** volume resolution, i.e. number of subdivisions with respect to bounds */
		int resZ;

        /** global voxel position offset (for absolute voxel positions) */
        int offsetX;

        /** global voxel position offset (for absolute voxel positions) */
        int offsetY;

        /** global voxel position offset (for absolute voxel positions) */
        int offsetZ;

		/** Maximum radius in the datasource. */
        VoxelizerFloat MaxRad;

        /** radius multiplication factor for calibration */
		VoxelizerFloat RadMult;

		/**
		 * Edge length of a voxel (set in accordance to the radii of
         * the contained particles)
		 */
        VoxelizerFloat CellSize;

        /** datacall that gives access to the particles */		
		core::moldyn::MultiParticleDataCall *datacall;

		/** here the Job should store its results */
		SubJobResult Result;
	};


} /* end namespace volumetrics */
} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_JOBSTRUCTURES_INCLUDED */
