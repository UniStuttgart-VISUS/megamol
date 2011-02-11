#ifndef MEGAMOLCORE_TETRAVOXELIZER_H_INCLUDED
#define MEGAMOLCORE_TETRAVOXELIZER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/ThreadPool.h"
#include "JobStructures.h"
#include "TagVolume.h"
#include "vislib/ShallowShallowTriangle.h"

namespace megamol {
namespace trisoup {
namespace volumetrics {

	class TetraVoxelizer : public vislib::sys::Runnable {
	public:

		TetraVoxelizer(void);
		~TetraVoxelizer(void);

		VoxelizerFloat GetOffset(VoxelizerFloat fValue1, VoxelizerFloat fValue2, VoxelizerFloat fValueDesired);

		void growSurfaceFromTriangle(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z,
							 unsigned char triIndex, Surface &surf);

        /**
         * also does compute fullFaces
         */
        VoxelizerFloat growVolume(FatVoxel *theVolume, unsigned char &fullFaces,
            unsigned int x, unsigned int y, unsigned int z);

		bool CellHasNoGeometry(FatVoxel *theVolume, unsigned x, unsigned y, unsigned z);

        bool CellFull(FatVoxel *theVolume, unsigned x, unsigned y, unsigned z);

		void MarchCell(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z);

		void CollectCell(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z);

        static vislib::math::Point<signed char, 3> cornerNeighbors[8][7];
        static vislib::math::Point<signed char, 3> moreNeighbors[6];

		/**
		 * Thread entry point.
		 *
		 * @param userData Pointer to a 'JobDescription' object holding the range
		 *                 of clusters to work on.
		 *
		 * @return 0 on success, positive on abort, negative on error.
		 */
		virtual DWORD Run(void *userData);

		/**
		 * Asks the thread to terminate after the calculation of the current
		 * cluster is finished
		 *
		 * @return Always 'true'
		 */
		virtual bool Terminate(void);

        /**
         * Test for equality. Actually this checks for equality of the
         * associated SubJobData.
         *
         * @param rhs the right hand set operand
         *
         * @return whether this->sjd and rhs.sjd are equal
         */
		inline bool operator ==(const TetraVoxelizer& rhs) const {
			return this->sjd == rhs.sjd;
		}

	private:

        void debugPrintTriangle(vislib::math::ShallowShallowTriangle<float, 3> &tri);
        void debugPrintTriangle(vislib::math::ShallowShallowTriangle<double, 3> &tri);

        inline bool isBorder(unsigned int x, unsigned int y, unsigned int z) {
            return (x == 0) || (x == sjd->resX - 2)
                || (y == 0) || (y == sjd->resY - 2)
                || (z == 0) || (z == sjd->resZ - 2);
        }

		bool terminate;

		SubJobData *sjd;

        vislib::SingleLinkedList<vislib::math::Point<unsigned int, 4> > cellFIFO;
};

} /* end namespace volumetrics */
} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TETRAVOXELIZER_H_INCLUDED */
