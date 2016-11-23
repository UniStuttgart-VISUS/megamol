#ifndef _SECSTRUCTFLATTENER_KERNEL_CU_
#define _SECSTRUCTFLATTENER_KERNEL_CU_

#include "CountingSortKernel.cuh"

#include <helper_cuda.h>
#include <helper_math.h>
#include <vector>
#include <thrust/device_vector.h>

#include "vislib/math/Matrix.h"
#include "vislib/math/Plane.h"

__host__ __device__ struct Spring {
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

__host__ __device__ bool operator<(const Spring &lhs, const Spring &rhs) {
	return lhs.source < rhs.source;
}

thrust::device_vector<float> d_atomPositions;
thrust::device_vector<float> d_atomPositionsSave;
thrust::device_vector<uint> d_cAlphaIndices;
thrust::device_vector<uint> d_oIndices;

__device__ float3 d_PositionBoundingBoxMin;
__device__ float3 d_PositionBoundingBoxMax;
__device__ float3 d_CAlphaBoundingBoxMin;
__device__ float3 d_CAlphaBoundingBoxMax;

__device__ float3 d_planeNormal;
__device__ float3 d_planeOrigin;

thrust::device_vector<Spring> d_springs;

typedef struct {
	float4 m[4];
} float4x4;

float4x4 h_transMat;
float4x4 h_invTransMat;

// number of threads per block
uint TpB = 256;

/**
 *	Performs a matrix vector multiplication
 *	
 *	@param M The 4x4 matrix.
 *	@param v The vector with 4 entrys.
 */
__device__ float4 mul(const float4x4 &M, const float4 &v) {
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = dot(v, M.m[3]);
	return r;
}

/**
 *	Kernel for applying a transformation matrix to all positions stored in the given pointer.
 *
 *	@param d_pos The given position array.
 *	@param sizePos The size of the position array
 *	@param mat The transformation matrix
 */
__global__ void applyMatrixToPositionsKernel(float * d_pos, uint sizePos, float4x4 mat) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= sizePos) {
		return;
	}

	float4 pos = make_float4(d_pos[idx * 3 + 0], d_pos[idx * 3 + 1], d_pos[idx * 3 + 2], 1.0f);
	pos = mul(mat, pos);
	d_pos[idx * 3 + 0] = pos.x;
	d_pos[idx * 3 + 1] = pos.y;
	d_pos[idx * 3 + 2] = pos.z;
}

/**
 *	Kernel for running the force directed layouting of the atoms
 *
 *	@param d_pos The position array.
 *	@param sizePos The number of available positions
 *	@param d_ca Array of c alpha atom indices.
 *	@param caSize Number of available c alpha indices.
 *	@param d_springs Array of available spring connections between c alpha atoms
 *	@param springSize The number of avaialable springs.
 *	@param timestepSize The duration of a single timestep
 */
__global__ void runSimulationKernel(float * d_pos, uint sizePos, uint * d_ca, uint caSize, Spring * d_springs, uint springSize, float timestepSize) {
	
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= caSize) {
		return;
	}

	uint caIdx = d_ca[idx];
	float3 bbCenter = (d_CAlphaBoundingBoxMin + d_CAlphaBoundingBoxMax) * 0.5f;
	float3 atPos = make_float3(d_pos[caIdx * 3 + 0], d_pos[caIdx * 3 + 1], d_pos[caIdx * 3 + 2]);
	float3 vector = normalize(atPos - bbCenter) * timestepSize;
	d_pos[caIdx * 3 + 0] += timestepSize;
	d_pos[caIdx * 3 + 1] += 0.0f;
	d_pos[caIdx * 3 + 2] += 0.0f;
}

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
	float * d_pos = thrust::raw_pointer_cast(d_atomPositions.data().get());
	uint * d_ca = thrust::raw_pointer_cast(d_cAlphaIndices.data().get());
	Spring * d_spring = thrust::raw_pointer_cast(d_springs.data().get());
	uint N = static_cast<uint>(d_atomPositions.size());
	uint NCa = static_cast<uint>(d_cAlphaIndices.size());
	uint NSpring = static_cast <uint>(d_springs.size());
	runSimulationKernel <<<(NCa + TpB - 1) / TpB, TpB >>>(d_pos, N, d_ca, NCa, d_spring, NSpring, timestepSize);
	checkCudaErrors(cudaDeviceSynchronize());
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

	float * d_pos = thrust::raw_pointer_cast(d_atomPositions.data().get());
	uint N = static_cast<uint>(d_atomPositions.size());
	applyMatrixToPositionsKernel << <(N + TpB - 1) / TpB, TpB >> >(d_pos, N, h_transMat);

	checkCudaErrors(cudaDeviceSynchronize());
	computeBoundingBoxes();
}

/**
 *	Transfers the computed atom positions back to the host.
 *
 *	@param h_atomPositions The pointer the position gets written to. The memory has to be allocated beforehand
 *	@param numPositions The number of allocated atom positions.
 */
extern "C"
void getPositions(float * h_atomPositions, uint numPositions) {
	if (numPositions != d_atomPositions.size() / 3) {
		printf("ERROR: Mismatching vector sizes in SecStructFlattener\n");
		exit(-1);
	}

	if (d_atomPositionsSave.size() != d_atomPositions.size()) {
		d_atomPositionsSave.resize(d_atomPositions.size());
	}

	// copy the values on the device to the vector the conversion happens on
	thrust::copy(d_atomPositions.begin(), d_atomPositions.end(), d_atomPositionsSave.begin());

	// transform the positions back to 3d space
	float * d_pos = thrust::raw_pointer_cast(d_atomPositionsSave.data().get());
	uint N = static_cast<uint>(d_atomPositionsSave.size()) / 3;
	applyMatrixToPositionsKernel << <(N + TpB - 1) / TpB, TpB >> >(d_pos, N, h_invTransMat);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(h_atomPositions, d_atomPositionsSave.data().get(), d_atomPositionsSave.size() * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
}

extern "C"
void transferPlane(vislib::math::Plane<float>& thePlane) {
	/**
	*	http://math.stackexchange.com/questions/1167717/transform-a-plane-to-the-xy-plane
	*/

	// translate plane to origin
	vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> transMat;
	transMat.SetIdentity();

	if (abs(thePlane.C()) > 0.000000001) {
		transMat.SetAt(2, 3, -(thePlane.D() / thePlane.C()));
	}
	auto transMatInverse = transMat;
	transMatInverse.SetAt(2, 3, -transMat.GetAt(2, 3));

	// rotate plane to the xy plane
	vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> rotMat;
	rotMat.SetIdentity();

	vislib::math::Vector<float, 3> v(thePlane.A(), thePlane.B(), thePlane.C());
	vislib::math::Vector<float, 3> k(0.0f, 0.0f, 1.0f);
	float theta = acos(thePlane.C() / v.Length());

	vislib::math::Vector<float, 3> u = v.Cross(k) / v.Length();
	float ct = cos(theta);
	float st = sin(theta);

	// set matrix values
	float val = ct + u[0] * u[0] * (1.0f - ct);
	rotMat.SetAt(0, 0, val);
	val = u[0] * u[1] * (1 - ct);
	rotMat.SetAt(0, 1, val);
	val = u[1] * st;
	rotMat.SetAt(0, 2, val);
	val = u[0] * u[1] * (1 - ct);
	rotMat.SetAt(1, 0, val);
	val = ct + u[1] * u[1] * (1.0f - ct);
	rotMat.SetAt(1, 1, val);
	val = -u[0] * st;
	rotMat.SetAt(1, 2, val);
	val = -u[1] * st;
	rotMat.SetAt(2, 0, val);
	val = u[0] * st;
	rotMat.SetAt(2, 1, val);
	val = ct;
	rotMat.SetAt(2, 2, val);

	// perform the translation to origin first
	auto result = rotMat * transMat;
	auto rotMatInverse = rotMat;
	bool q = rotMatInverse.Invert();
	auto resultInverse = transMatInverse * rotMatInverse;
	
	if (!q) {
		printf("ERROR: Matrix inversion not possible\n");
		exit(-1);
	}

	// transfer matrices to global memory
	result.Transpose();
	memcpy(&h_transMat, result.PeekComponents(), 16 * sizeof(float));

	resultInverse.Transpose();
	memcpy(&h_invTransMat, resultInverse.PeekComponents(), 16 * sizeof(float));
}

/**
 *	Transfers new atom connection (spring) data to the GPU.
 *
 *	@param h_atomPositions Pointer to the atom positions (order: xyzxyzxyz...). This has to be a pointer to the unflattened positions.
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