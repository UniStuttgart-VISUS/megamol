#ifndef VOLUME_CUDA_CUDAVOLUMERAYCASTER_KERNEL_H_INCLUDED
#define VOLUME_CUDA_CUDAVOLUMERAYCASTER_KERNEL_H_INCLUDED

#include "cuda.h"
#include "helper_cuda.h"
#include "helper_math.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

cudaArray * d_volumeArray = 0;
cudaArray * d_customTransferFuncArray = 0;

typedef float VolumeType;

texture<VolumeType, 3, cudaReadModeElementType> tex;
texture<float4, 1, cudaReadModeElementType> customTransferTex;
float minVal, maxVal;

typedef struct {
	float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray {
	float3 o;   // origin
	float3 d;   // direction
};

/** 
 *	Intersect ray with a box
 *	http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
 *	
 *	@param r The ray itself.
 *	@param boxmin The minimum values for all three box dimensions.
 *	@param boxmax The maximum values for all three box dimensions.
 *	@param tnear OUT: Pointer to the distance of the nearest intersection point.
 *	@param tfar OUT: Pointer to the distance of the furthest intersection point.
 *	@return Value greater 0 if a intersection happened, 0 otherwise.
 */
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

/**
 *	Transform vector by matrix (no translation)
 *
 *	@param M The 3x4 matrix
 *	@param v The vector to be transformed
 *	@return The transformed vector.
 */
__device__ float3 mul(const float3x4 &M, const float3 &v) {
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

/**
 *	Transform vector by matrix (with translation)
 *
 *	@param M The 3x4 matrix
 *	@param v The vector to be transformed
 *	@return The transformed vector.
 */
__device__ float4 mul(const float3x4 &M, const float4 &v) {
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

/**
 *	Converts a rgba color to a colour represented by an unsigned int
 *
 *	@param rgba The rgba colour.
 *	@return The colour as an unsigned int
 */
__device__ uint rgbaFloatToInt(float4 rgba) {
	rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

/**
 *	The CUDA Kernel for the rendering process
 */
__global__ void d_render(uint * d_output, uint imageW, uint imageH, float fovx, float fovy, float3 camPos, float3 camDir, float3 camUp, float3 camRight, float zNear,
	float density, float brightness, float transferOffset, float transferScale, float minVal, float maxVal,
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f), const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f), cudaExtent volSize = make_cudaExtent(1, 1, 1)) {

	const int maxSteps = 450;

	const float tstep = (boxMax.x - boxMin.x) / static_cast<float>(maxSteps);
	const float opacityThreshold = 0.95f;

	// pixel coordinates
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	// texture coordinates
	float u = (x / static_cast<float>(imageW)) * 2.0f - 1.0f;
	float v = (y / static_cast<float>(imageH)) * 2.0f - 1.0f;

	// calculate intersection with near plane in world space
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

	if (tnear < 0.0f) tnear = 0.0f; // clamp to near plane

	// march along ray from front to back, accumulating colour
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d * tnear;
	float3 step = eyeRay.d * tstep;
	float3 diff = boxMax - boxMin;

	for (int i = 0; i < maxSteps; i++) {
		// remap position to [0, 1] coordinates
		float3 samplePos;
		samplePos.x = (pos.x - boxMin.x) / diff.x;
		samplePos.y = (pos.y - boxMin.y) / diff.y;
		samplePos.z = (pos.z - boxMin.z) / diff.z;

		// read from 3D texture
		float sample = tex3D(tex, samplePos.x, samplePos.y, samplePos.z);
		
		// normalize the sample
		sample /= maxVal;

		float sampleCamDist = length(eyeRay.o - pos);

		// lookup in transfer function texture
		float4 col = tex1D(customTransferTex, (sample - transferOffset) * transferScale);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;

		// "over" operator for front-to-back blending
		sum = sum + col * (1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold) {
			break;
		}

		t += tstep;

		if (t > tfar) break;

		pos += step;
	}

	sum *= brightness;

	// write output color
	d_output[y * imageW + x] = rgbaFloatToInt(sum);
}

/**
 *	Sets the filtering mode of the CUDA interpolation
 *	
 *	@param bLinearFilter True for trilinear filtering, false for nearest neighbour
 */
extern "C"
void setTextureFilterMode(bool bLinearFilter) {
	tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

/**
 *	Frees all initialized CUDA buffers.
 */
extern "C" 
void freeCudaBuffers(void) {
	checkCudaErrors(cudaFreeArray(d_volumeArray));
	checkCudaErrors(cudaFreeArray(d_customTransferFuncArray));
}

/**
 *	Renders the scene using the given parameters.
 *	
 *	@param gridSize The CUDA grid size.
 *	@param blockSize The CUDA block size.
 *	@param d_output Pointer to the output image.
 *	@param imageW The width of the output image.
 *	@param imageH The height of the output image.
 *	@param fovx The camera field of view in x direction (Radians).
 *	@param fovy The camera field of view in y direction (Radians).
 *	@param camPos The position of the camera.
 *	@param camDir The view direction of the camera.
 *	@param camUp The up vector of the camera.
 *	@param camRight The right vector of the camera.
 *	@param zNear The distance of the near plane to the camera position.
 *	@param density The density scaling factor.
 *	@param brightness The brightness scaling factor.
 *	@param transferOffset The offset for the transfer function.
 *	@param transferScale The scaling factor for the transfer function.
 *	@param boxMin The minimum values of the bounding box.
 *	@param boxMax The maximum values of the bounding box.
 *	@param volSize The size of the rendered volume.
 */
extern "C" 
void render_kernel(dim3 gridSize, dim3 blockSize, uint * d_output, uint imageW, uint imageH, float fovx, float fovy, float3 camPos, float3 camDir,
	float3 camUp, float3 camRight, float zNear, float density, float brightness, float transferOffset, float transferScale,
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f), const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f), cudaExtent volSize = make_cudaExtent(1, 1, 1)) {

	d_render <<<gridSize, blockSize >>>(d_output, imageW, imageH, fovx, fovy, camPos, camDir, camUp, camRight, zNear, density, brightness,
		transferOffset, transferScale, minVal, maxVal, boxMin, boxMax, volSize);
}

/**
 *	Transfers a new transfer function to the GPU.
 *	
 *	@param transferFunction Pointer to the transfer function values.
 *	@param functionSize number of transfer function values.
 */
extern "C" 
void copyTransferFunction(float4 * transferFunction, int functionSize = 256) {
	checkCudaErrors(cudaFreeArray(d_customTransferFuncArray));

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaArray * d_customTransferFuncArray;
	checkCudaErrors(cudaMallocArray(&d_customTransferFuncArray, &channelDesc, functionSize, 1));
	checkCudaErrors(cudaMemcpyToArray(d_customTransferFuncArray, 0, 0, transferFunction, sizeof(float4)*functionSize, cudaMemcpyHostToDevice));

	customTransferTex.filterMode = cudaFilterModeLinear;
	customTransferTex.normalized = true;
	customTransferTex.addressMode[0] = cudaAddressModeClamp;
	
	checkCudaErrors(cudaBindTextureToArray(customTransferTex, d_customTransferFuncArray, channelDesc));
}

/**
 *	Transfers a new volume to the GPU
 *
 *	@param h_volume Pointer to the float volume data.
 *	@param volumeSize The extents of the volume.
 */
extern "C"
void transferNewVolume(void * h_volume, cudaExtent volumeSize) {
	if (d_volumeArray) {
		checkCudaErrors(cudaFreeArray(d_volumeArray));
		d_volumeArray = 0;
		checkCudaErrors(cudaUnbindTexture(tex));
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

	// compute min and max values of the volume
	float * volptr = static_cast<float*>(h_volume);
	auto res = thrust::minmax_element(volptr, volptr + (volumeSize.width * volumeSize.depth * volumeSize.height));
	minVal = (float)*res.first;
	maxVal = (float)*res.second;

	// copy the data
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(h_volume, volumeSize.width * sizeof(VolumeType), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tex.normalized = true;						// access with normalized texture coordinates
	tex.filterMode = cudaFilterModeLinear;		// linear interpolation
	tex.addressMode[0] = cudaAddressModeClamp;	// we want 0s outside for beautiful isosurfaces at the borders
	tex.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));
}

/**
 *	Initializes the CUDA device
 *	
 *	@param h_volume Pointer to the initial float volume.
 *	@param volumeSize The extents of the transferred volume.
 *	@param h_transferFunction Pointer to the transfer function values.
 *	@param functionSize Number of transfer function values.
 */
extern "C"
void initCudaDevice(void * h_volume, cudaExtent volumeSize, float4 * transferFunction, int functionSize = 256) {
	transferNewVolume(h_volume, volumeSize);
	copyTransferFunction(transferFunction, functionSize);
}

#endif /* defined VOLUME_CUDA_CUDAVOLUMERAYCASTER_KERNEL_H_INCLUDED */