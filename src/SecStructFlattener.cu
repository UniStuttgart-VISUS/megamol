#ifndef _SECSTRUCTFLATTENER_KERNEL_CU_
#define _SECSTRUCTFLATTENER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>
#include <vector>
#include <thrust/device_vector.h>

typedef unsigned int uint;

//struct Spring {
//	uint source;
//	uint target;
//	float length;
//	float springConstant;
//	float friction;
//	Spring(uint src, uint trgt, float l, float sc, float f) : 
//		source(src), target(trgt), length(l), springConstant(sc), friction(f) {}
//};

thrust::device_vector<float3> d_atomPositions;
thrust::device_vector<uint> d_cAlphaIndices;
thrust::device_vector<uint> d_oIndices;

//thrust::device_vector<Spring> d_springs;

extern "C"
void performTimestep(float timestepSize) {

}

extern "C"
void transferAtomData(float * h_atomPositions, uint numPositions, uint * h_cAlphaIndices, uint numCAlphas, uint * h_oIndices, uint numOs) {
	// copy necessary values from host to device
	/*d_atomPositions = thrust::device_vector<float3>(h_atomPositions, h_atomPositions + numPositions);
	d_cAlphaIndices = thrust::device_vector<uint>(h_cAlphaIndices, h_cAlphaIndices + numCAlphas);
	d_oIndices = thrust::device_vector<uint>(h_oIndices, h_oIndices + numOs);*/

	std::vector<float3> h_atomPositions3D(reinterpret_cast<float3 *>(h_atomPositions), reinterpret_cast<float3 *>(h_atomPositions) + numPositions);
	d_atomPositions = h_atomPositions3D;

	std::vector<uint> h_cAlphaVector(h_cAlphaIndices, h_cAlphaIndices + numCAlphas);
	d_cAlphaIndices = h_cAlphaVector;

	std::vector<uint> h_oVector(h_oIndices, h_oIndices + numOs);
	d_oIndices = h_oVector;
}

extern "C"
void transferSpringData(float * h_atomPositions, uint numPositions, uint * h_cAlphaIndices, uint numCAlphas, uint * h_oIndices, uint numOs) {
	//d_springs.clear();

	float3 * h_atomPositions3D = reinterpret_cast<float3 *>(h_atomPositions);

	for (uint i = 1; i < numCAlphas; i++) {
		float3 lastPos = h_atomPositions3D[h_cAlphaIndices[i - 1]];
		float3 thisPos = h_atomPositions3D[h_cAlphaIndices[i]];
		thisPos = thisPos - lastPos;
		float dist = sqrt(thisPos.x * thisPos.x + thisPos.y * thisPos.y + thisPos.z * thisPos.z);

		// TODO adjust spring constant and friction
		/*Spring newSpring(h_cAlphaIndices[i - 1], h_cAlphaIndices[i], dist, (float)1.0f, (float)1.0f);
		d_springs.push_back(newSpring);*/
	}

	// TODO H-Bonds ?
	// correct or fake?
	// 
}

#endif // #ifndef _SECSTRUCTFLATTENER_KERNEL_CU_