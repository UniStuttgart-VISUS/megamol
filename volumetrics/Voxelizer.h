#ifndef MEGAMOLCORE_VOXELIZER_H_INCLUDED
#define MEGAMOLCORE_VOXELIZER_H_INCLUDED
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

	class Voxelizer : public vislib::sys::Runnable {
	public:

		Voxelizer(void);
		~Voxelizer(void);

		float getOffset(float fValue1, float fValue2, float fValueDesired);

		void growSurfaceFromTriangle(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z,
							 unsigned char triIndex, 
							 vislib::Array<double> &surf, vislib::Array<BorderVoxel *> &border,
                             double &surfSurf);

		bool CellEmpty(FatVoxel *theVolume, unsigned x, unsigned y, unsigned z);

		void marchCell(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z);

		void collectCell(FatVoxel *theVolume, unsigned int x, unsigned int y, unsigned int z);

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

		inline bool operator ==(const Voxelizer& rhs) const {
			// TODO I cannot think of anything better right now
			return this->sjd == rhs.sjd;
		}

	private:

        inline bool isBorder(unsigned int x, unsigned int y, unsigned int z) {
            return (x == 0) || (x == sjd->resX - 2)
                || (y == 0) || (y == sjd->resY - 2)
                || (z == 0) || (z == sjd->resZ - 2);
        }

		bool terminate;

		SubJobData *sjd;

		//unsigned int fifoLen, fifoEnd, fifoCur;
		
        vislib::SingleLinkedList<vislib::math::Point<unsigned int, 4> > cellFIFO;

		//vislib::Array<vislib::math::Point<float, 3> > triangleSoup;
};

} /* end namespace volumetrics */
} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VOXELIZER_H_INCLUDED */
