#ifndef MEGAMOLCORE_VOXELIZER_H_INCLUDED
#define MEGAMOLCORE_VOXELIZER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/ThreadPool.h"
#include "JobStructures.h"
#include "TagVolume.h"

namespace megamol {
namespace trisoup {

	class Voxelizer : public vislib::sys::Runnable {
	public:

		struct FatVoxel {
			float distField;

		};

		Voxelizer(void);
		~Voxelizer(void);

		float getOffset(float fValue1, float fValue2, float fValueDesired);

		bool isCellNotEmpty(FatVoxel *theVolume, unsigned x, unsigned y, unsigned z);

		void marchCell(FatVoxel *theVolume, TagVolume &markedCells, unsigned int x, unsigned int y, unsigned int z);
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
		bool terminate;

		SubJobData *sjd;

		unsigned int fifoLen, fifoEnd, fifoCur;
		
		vislib::math::Point<unsigned int, 3> *cellFIFO;

		vislib::Array<vislib::math::Point<float, 3> > triangleSoup;
};

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VOXELIZER_H_INCLUDED */