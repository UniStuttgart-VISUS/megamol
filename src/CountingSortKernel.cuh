#ifndef _COUNTINGSORTKERNEL_CUH_
#define _COUNTINGSORTKERNEL_CUH_

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

typedef unsigned int uint;

typedef struct GPUSortHandle {
	float *ParticleOrigin;
	float *GridDim;
	float *GridDelta;
	float *d_ParticlePos;
	float *d_ParticlePosOut;
	float *d_ParticleColor;
	float *d_ParticleColorOut;
	uint   *d_GridCount;
	uint   *d_GridIDX;
	uint   *d_CellIDX;
	size_t  ParticleCount;
	size_t  CellCount;
} GPUSortHandle;

namespace megamol {
namespace protein_cuda {

	/*
	 * Calculate BinIndex of each particle.
	 * Particle-centered approach, grid is in global memory and will be altered via "atomicAdd"
	 */
	template<unsigned int D, typename T>
	__global__ void BinIndexing(unsigned int ParticleCount, T *d_ParticlePos, T *ParticleOrigin, T *GridDim, T *GridDelta, uint *d_GridCount, uint *d_GridIDX, uint *d_CellIDX) {
		uint pidx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (pidx >= ParticleCount) return;

		//float3 gridCoord; // = floor((handle->d_ParticlePos[pidx] - handle->ParticleOrigin).xyz*handle->GridDelta);
		//gridCoord.x = floor((d_ParticlePos[pidx].x - ParticleOrigin.x)*GridDelta.x);
		//gridCoord.y = floor((d_ParticlePos[pidx].y - ParticleOrigin.y)*GridDelta.y);
		//gridCoord.z = floor((d_ParticlePos[pidx].z - ParticleOrigin.z)*GridDelta.z);
		uint gidx = 0;
		uint gridCoord[D];
		for (uint d = 0; d < D; d++) {
			gridCoord[d] = static_cast<uint>(floor((d_ParticlePos[pidx*D + d] - ParticleOrigin[d])*GridDelta[d]));
			uint factor = 1;
			if (d != 0)
			for (int i = 0; i < d; i++) {
				factor *= GridDim[i];
			}
			gidx += gridCoord[d] * factor;
		}

		//uint gidx = gridCoord.x + GridDim.x*(gridCoord.y + gridCoord.z*GridDim.y);

		d_CellIDX[pidx] = atomicAdd(&d_GridCount[gidx], 1);

		d_GridIDX[pidx] = gidx;
	}

	__global__ void CountingSort(unsigned int ParticleCount, uint *d_GridCount, uint *d_GridIDX, uint *d_CellIDX, float3 *d_ParticlePos, float3 *d_ParticlePosOut) {
		uint pidx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (pidx >= ParticleCount) return;

		uint sidx = d_GridCount[d_GridIDX[pidx]] + d_CellIDX[pidx];

		d_ParticlePosOut[sidx] = d_ParticlePos[pidx];

		/*for (uint d = 0; d < TD; d++) {
			d_ParticlePosOut[sidx*TD + d] = d_ParticlePos[pidx*TD + d];
			}*/

		/*if (d_ParticleColorOut != NULL && d_ParticleColor != NULL) {
			for (uint d = 0; d < CD; d++) {
			d_ParticleColorOut[sidx*CD + d] = d_ParticleColor[pidx*CD + d];
			}
			}*/
	}
} /* end namespace protein_cuda */
} /* end namespace megamol */


#endif