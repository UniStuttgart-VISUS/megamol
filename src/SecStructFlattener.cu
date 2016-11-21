#ifndef _SECSTRUCTFLATTENER_KERNEL_CU_
#define _SECSTRUCTFLATTENER_KERNEL_CU_

#include "CountingSortKernel.cuh"

#include <helper_cuda.h>
#include <helper_math.h>
#include <vector>
#include <thrust/device_vector.h>

struct Spring {
	uint source;
	uint target;
	float length;
	float springConstant;
	float friction;

	Spring() : source(0), target(0), length(0.0f), springConstant(0.0f), friction(0.0f) {}

	Spring(const uint src, const uint trgt, const float l, const float sc, const float f) {
		this->source = src;
		this->target = trgt;
		this->length = l;
		this->springConstant = sc;
		this->friction = f;
	}
};

thrust::device_vector<float> d_atomPositions;
thrust::device_vector<uint> d_cAlphaIndices;
thrust::device_vector<uint> d_oIndices;

__device__ float3 d_PositionBoundingBoxMin;
__device__ float3 d_PositionBoundingBoxMax;
__device__ float3 d_CAlphaBoundingBoxMin;
__device__ float3 d_CAlphaBoundingBoxMax;

thrust::device_vector<Spring> d_springs;

/**
 *	Computes the bounding boxes for the uploaded data
 */
void computeBoundingBoxes() {

	float3 pbmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	float3 pbmax = make_float3(FLT_MIN, FLT_MIN, FLT_MIN);
	float3 cabmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	float3 cabmax = make_float3(FLT_MIN, FLT_MIN, FLT_MIN);

	for (uint i = 0; i < d_atomPositions.size() / 3; i++) {
		if (d_atomPositions[i * 3 + 0] < pbmin.x) {
			pbmin.x = d_atomPositions[i * 3 + 0];
		}
		if (d_atomPositions[i * 3 + 1] < pbmin.y) {
			pbmin.y = d_atomPositions[i * 3 + 1];
		}
		if (d_atomPositions[i * 3 + 2] < pbmin.z) {
			pbmin.z = d_atomPositions[i * 3 + 2];
		}
		if (d_atomPositions[i * 3 + 0] > pbmax.x) {
			pbmax.x = d_atomPositions[i * 3 + 0];
		}
		if (d_atomPositions[i * 3 + 1] > pbmax.y) {
			pbmax.y = d_atomPositions[i * 3 + 1];
		}
		if (d_atomPositions[i * 3 + 2] > pbmax.z) {
			pbmax.z = d_atomPositions[i * 3 + 2];
		}
	}

	for (uint i = 0; i < d_cAlphaIndices.size(); i++) {

		uint j = d_cAlphaIndices[i];

		if (d_atomPositions[j * 3 + 0] < cabmin.x) {
			cabmin.x = d_atomPositions[j * 3 + 0];
		}
		if (d_atomPositions[j * 3 + 1] < cabmin.y) {
			cabmin.y = d_atomPositions[j * 3 + 1];
		}
		if (d_atomPositions[j * 3 + 2] < cabmin.z) {
			cabmin.z = d_atomPositions[j * 3 + 2];
		}
		if (d_atomPositions[j * 3 + 0] > cabmax.x) {
			cabmax.x = d_atomPositions[j * 3 + 0];
		}
		if (d_atomPositions[j * 3 + 1] > cabmax.y) {
			cabmax.y = d_atomPositions[j * 3 + 1];
		}
		if (d_atomPositions[j * 3 + 2] > cabmax.z) {
			cabmax.z = d_atomPositions[j * 3 + 2];
		}
	}

	checkCudaErrors(cudaMemcpyToSymbol(d_PositionBoundingBoxMin, &pbmin, sizeof(float3), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_PositionBoundingBoxMax, &pbmax, sizeof(float3), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_CAlphaBoundingBoxMin, &cabmin, sizeof(float3), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_CAlphaBoundingBoxMax, &cabmin, sizeof(float3), 0, cudaMemcpyHostToDevice));
}

/**
 *	Recomputes the spatial grid based on the currently available particle data 
 */
void recomputeGrid() {

}

/**
 *	Performs a single particle system timestep on the particle data.
 *
 *	@param timestepSize The duration of the timestep.
 */
extern "C"
void performTimestep(float timestepSize) {
	recomputeGrid();
	printf("timestep with %f performed\n", timestepSize);
}

/**
 *	Transfers new atom data to the GPU.
 *
 *	@param h_atomPositions Pointer to the atom positions (order: xyzxyzxyz...).
 *	@param numPositions Number of different atom positions (number of values in h_atomPositions / 3).
 *	@param h_cAlphaIndices Pointer to the indices of the c alpha atoms in the position vector.
 *	@param numCAlphas The number of available c alpha atoms / indices.
 */
extern "C"
void transferAtomData(float * h_atomPositions, uint numPositions, uint * h_cAlphaIndices, uint numCAlphas) {

	// copy necessary values from host to device
	d_atomPositions = thrust::device_vector<float>(h_atomPositions, h_atomPositions + numPositions * 3);
	d_cAlphaIndices = thrust::device_vector<uint>(h_cAlphaIndices, h_cAlphaIndices + numCAlphas);

	checkCudaErrors(cudaDeviceSynchronize());
	computeBoundingBoxes();
}

/**
 *	Transfers new atom connection (spring) data to the GPU.
 *
 *	@param h_atomPositions Pointer to the atom positions (order: xyzxyzxyz...).
 *	@param numPositions Number of different atom positions (number of values in h_atomPositions / 3).
 *	@param h_cAlphaIndices Pointer to the indices of the c alpha atoms in the position vector.
 *	@param numCAlphas The number of available c alpha atoms / indices.
 *	@param h_oIndices Pointer to the indices of the o atom of the carbonyl group.
 *	@param numOs number of available o atoms / indices.
 */
extern "C"
void transferSpringData(const float * h_atomPositions, uint numPositions, uint * h_cAlphaIndices, uint numCAlphas, uint * h_oIndices, uint numOs, float conFriction, float conConstant, float hFriction, float hConstant) {
	d_springs.clear();

	const float3 * h_atomPositions3D = reinterpret_cast<const float3 *>(h_atomPositions);

	for (uint i = 1; i < numCAlphas; i++) {
		float3 lastPos = h_atomPositions3D[h_cAlphaIndices[i - 1]];
		float3 thisPos = h_atomPositions3D[h_cAlphaIndices[i]];
		thisPos = thisPos - lastPos;
		float dist = sqrt(thisPos.x * thisPos.x + thisPos.y * thisPos.y + thisPos.z * thisPos.z);

		// TODO adjust spring constant and friction
		Spring newSpring = Spring(h_cAlphaIndices[i - 1], h_cAlphaIndices[i], dist, conConstant, conFriction);
		d_springs.push_back(newSpring);
	}

	// TODO H-Bonds ?
	// correct or fake?
	// 


}

#endif // #ifndef _SECSTRUCTFLATTENER_KERNEL_CU_