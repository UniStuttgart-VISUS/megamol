/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

// Simple 3D volume renderer

#ifndef _CUDARAYCASTER_KERNEL_CU_
#define _CUDARAYCASTER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <vector>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaEvent_t evtStart, evtStop;

cudaArray *d_volumeArray = 0;
cudaArray *d_customTransferFuncArray;

cudaArray *d_isoValArray = 0;
__device__ int d_numIsoVals = 0;

std::vector<float> fpsVec;

typedef float VolumeType;

texture<VolumeType, 3, cudaReadModeElementType> tex;
texture<float4, 1, cudaReadModeElementType> customTransferTex;
float minVal, maxVal;

typedef struct {
	float4 m[3];
} float3x4;

#ifdef MAT4
typedef struct {
	float4 m[4];
} float4x4;
#endif

#ifndef MAT4
__constant__ float3x4 c_invViewMatrix;  // inverse view matrix
#else
__constant__ float4x4 c_invViewMatrix;  // inverse view matrix
#endif

struct Ray {
	float3 o;   // origin
	float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__ int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar) {
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f) / r.d;
	float3 tbot = invR * (boxmin - r.o);
	float3 ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = fminf(ttop, tbot);
	float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__ float3 mul(const float3x4 &M, const float3 &v) {
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

#ifdef MAT4
// transform vector by matrix with translation
__device__ float4 mul(const float4x4 &M, const float4 &v) {
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = dot(v, M.m[3]);
	return r;
}

// transform vector by matrix (no translation)
__device__ float3 mul(const float4x4 &M, const float3 &v) {
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}
#endif

// transform vector by matrix with translation
__device__ float4 mul(const float3x4 &M, const float4 &v) {
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

__device__ uint rgbaFloatToInt(float4 rgba) {
	rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

__global__ void
d_render(uint *d_output, uint imageW, uint imageH, float fovx, float fovy, float3 camPos, float3 camDir, float3 camUp, float3 camRight, float zNear,
float density, float brightness, float transferOffset, float transferScale, float minVal, float maxVal,
const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f), const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f), cudaExtent = make_cudaExtent(1, 1, 1))
{
	const int maxSteps = 512;
	//const float tstep = 0.0009765625f;

	//const float tstep = (boxMax.x - boxMin.x) / (float)maxSteps;
	const float tstep = length(boxMax - boxMin) / (float)maxSteps;
	const float opacityThreshold = 0.95f;

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	float u = (x / (float)imageW)*2.0f - 1.0f;
	float v = (y / (float)imageH)*2.0f - 1.0f;

	float3 oL = (tan(fovx * 0.5f) * zNear) * (-camRight) + (tan(fovy * 0.5) * zNear) * camUp + camDir * zNear + camPos;
	float3 uL = (tan(fovx * 0.5f) * zNear) * (-camRight) + (tan(fovy * 0.5) * zNear) * (-camUp) + camDir * zNear + camPos;
	float3 oR = (tan(fovx * 0.5f) * zNear) * camRight + (tan(fovy * 0.5) * zNear) * camUp + camDir * zNear + camPos;
	float3 uR = (tan(fovx * 0.5f) * zNear) * camRight + (tan(fovy * 0.5) * zNear) * (-camUp) + camDir * zNear + camPos;

	float3 targetL = lerp(uL, oL, (v + 1.0f) * 0.5f);
	float3 targetR = lerp(uR, oR, (v + 1.0f) * 0.5f);

	float3 target = lerp(targetL, targetR, (u + 1.0f) * 0.5f);

	// calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = camPos;
	eyeRay.d = normalize(target - camPos);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit) {
		d_output[y*imageW + x] = rgbaFloatToInt(make_float4(0.0f));
		return;
	} 
	/*else {
		d_output[y*imageW + x] = rgbaFloatToInt(make_float4(1.0f));
		return;
	}*/

	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;

	float3 diff = boxMax - boxMin;

	for (int i = 0; i<maxSteps; i++) {
		// read from 3D texture
		// remap position to [0, 1] coordinates
		//float sample = tex3D(tex, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f);
		float3 samplePos;
		samplePos.x = (pos.x - boxMin.x) / diff.x;
		samplePos.y = (pos.y - boxMin.y) / diff.y;
		samplePos.z = (pos.z - boxMin.z) / diff.z;
		float sample = tex3D(tex, samplePos.x, samplePos.y, samplePos.z);

		sample /= maxVal;

		//sample *= 2.0f;

		float sampleCamDist = length(eyeRay.o - pos);

		// lookup in transfer function texture
		//float4 col = tex1D(customTransferTex, (sample - transferOffset)*transferScale);
		float4 col = make_float4(0.0);

		col = make_float4(sample);

		/*float4 col = make_float4(0.0);
		if( sample > 0.00001 )
			col = make_float4(1.0);*/

		col.w *= density;

		// "under" operator for back-to-front blending
		//sum = lerp(sum, col, col.w);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;

		// "over" operator for front-to-back blending
		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold)
			break;

		t += tstep;

		if (t > tfar) break;

		pos += step;
	}

	sum *= brightness;

	// write output color
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
	//d_output[y*imageW + x] = sum;
}

extern "C"
void setTextureFilterMode(bool bLinearFilter) {
	tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C"
void initCudaDevice(void *h_volume, cudaExtent volumeSize) {
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

	// get min and max value from volume
	float *volptr = static_cast<float*>(h_volume);
	minVal = FLT_MAX;
	maxVal = FLT_MIN;
	for (unsigned int i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++) {
		if (minVal > volptr[i])
			minVal = volptr[i];
		if (maxVal < volptr[i])
			maxVal = volptr[i];
	}
	printf("min = %f, max = %f\n", minVal, maxVal);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	//std::cout << "Using 32-bit float volume data." << std::endl;
	copyParams.srcPtr = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tex.normalized = true;                      // access with normalized texture coordinates
	tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	tex.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

	float4 customTransFunc[256];
	for (int i = 0; i < 256; i++)
		customTransFunc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	cudaChannelFormatDesc channelDesc3 = cudaCreateChannelDesc<float4>();
	cudaArray *d_customTransferFuncArray;
	checkCudaErrors(cudaMallocArray(&d_customTransferFuncArray, &channelDesc3, 256, 1));
	checkCudaErrors(cudaMemcpyToArray(d_customTransferFuncArray, 0, 0, customTransFunc, sizeof(customTransFunc), cudaMemcpyHostToDevice));

	customTransferTex.filterMode = cudaFilterModeLinear;
	customTransferTex.normalized = true;
	customTransferTex.addressMode[0] = cudaAddressModeClamp;

	checkCudaErrors(cudaBindTextureToArray(customTransferTex, d_customTransferFuncArray, channelDesc3));

	checkCudaErrors(cudaEventCreate(&evtStart));
	checkCudaErrors(cudaEventCreate(&evtStop));
}

extern "C"
void freeCudaBuffers() {
	checkCudaErrors(cudaFreeArray(d_volumeArray));
	checkCudaErrors(cudaFreeArray(d_customTransferFuncArray));
	checkCudaErrors(cudaFreeArray(d_isoValArray));
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float fovx, float fovy, float3 camPos, float3 camDir, 
	float3 camUp, float3 camRight, float zNear, float density, float brightness, float transferOffset, float transferScale,
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f), const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f), cudaExtent volSize = make_cudaExtent(1, 1, 1)) {

	d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, fovx, fovy, camPos, camDir, camUp, camRight, zNear, density,
		brightness, transferOffset, transferScale, minVal, maxVal, boxMin, boxMax, volSize);
}

extern "C"
void renderArray_kernel(cudaArray* renderArray, dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float fovx, float fovy, float3 camPos, float3 camDir,
	float3 camUp, float3 camRight, float zNear, float density, float brightness, float transferOffset, float transferScale,
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f), const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f), cudaExtent volSize = make_cudaExtent(1,1,1)) {
	
	float* horst = new float[volSize.width * volSize.height * volSize.depth];
	horst[0] = 1.0f;
	checkCudaErrors(cudaDeviceSynchronize());
	//checkCudaErrors(cudaMemcpyFromArray(horst, renderArray, 0, 0, sizeof(float) * volSize.x * volSize.y, cudaMemcpyDeviceToHost));
	
	cudaExtent volExt = volSize;

	cudaPitchedPtr pitchedHorst = make_cudaPitchedPtr(horst, sizeof(float) * volSize.width, volSize.height, volSize.height);

	cudaMemcpy3DParms myParms = { 0 };
	myParms.extent = volExt;
	myParms.srcArray = renderArray;
	myParms.dstPtr = pitchedHorst;
	myParms.kind = cudaMemcpyDeviceToHost;

	checkCudaErrors(cudaMemcpy3D(&myParms));

	checkCudaErrors(cudaDeviceSynchronize());
	for (unsigned int i = 0; i < volSize.width * volSize.depth * volSize.height; i++) {
		if (horst[i] > 0.000001)
			printf("%i - %.3f\n", i, horst[i]);
	}
	delete[] horst;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	checkCudaErrors(cudaBindTextureToArray(tex, renderArray, channelDesc));

	checkCudaErrors(cudaDeviceSynchronize());

	d_render << <gridSize, blockSize >> >(d_output, imageW, imageH, fovx, fovy, camPos, camDir, camUp, camRight, zNear, density,
		brightness, transferOffset, transferScale, minVal, maxVal, boxMin, boxMax, volSize);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaUnbindTexture(tex));
}

extern "C"
void copyLUT(float4* myLUT, int lutSize = 256)
{
	checkCudaErrors(cudaFreeArray(d_customTransferFuncArray));

	cudaChannelFormatDesc channelDesc3 = cudaCreateChannelDesc<float4>();
	cudaArray *d_customTransferFuncArray;
	checkCudaErrors(cudaMallocArray(&d_customTransferFuncArray, &channelDesc3, 256, 1));
	checkCudaErrors(cudaMemcpyToArray(d_customTransferFuncArray, 0, 0, myLUT, sizeof(float4)* lutSize, cudaMemcpyHostToDevice));

	customTransferTex.filterMode = cudaFilterModeLinear;
	customTransferTex.normalized = true;
	customTransferTex.addressMode[0] = cudaAddressModeClamp;

	checkCudaErrors(cudaBindTextureToArray(customTransferTex, d_customTransferFuncArray, channelDesc3));
}

extern "C"
void transferIsoValues(float* h_isoVals, int h_numIsos) {

	if (d_isoValArray) {
		checkCudaErrors(cudaFreeArray(d_isoValArray));
		d_isoValArray = 0;
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	checkCudaErrors(cudaMallocArray(&d_isoValArray, &channelDesc, h_numIsos));
	checkCudaErrors(cudaMemcpyToArray(d_isoValArray, 0, 0, h_isoVals, sizeof(VolumeType)*h_numIsos, cudaMemcpyHostToDevice));

	cudaMemset(&d_numIsoVals, h_numIsos, sizeof(int));
}

extern "C"
void transferNewVolume(void* h_volume, cudaExtent volumeSize) {

	if (d_volumeArray) {
		checkCudaErrors(cudaFreeArray(d_volumeArray));
		d_volumeArray = 0;

		checkCudaErrors(cudaUnbindTexture(tex));
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

	// get min and max value from volume
	float *volptr = static_cast<float*>(h_volume);
	thrust::device_ptr<float> ptr = thrust::device_ptr<float>(volptr);

	auto res = thrust::minmax_element(ptr, ptr + (volumeSize.width * volumeSize.depth * volumeSize.height));
	minVal = (float)*res.first;
	maxVal = (float)*res.second;

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);

	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tex.normalized = true;                      // access with normalized texture coordinates
	tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	tex.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
