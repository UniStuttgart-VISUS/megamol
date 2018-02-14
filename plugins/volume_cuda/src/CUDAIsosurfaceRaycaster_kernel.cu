#ifndef VOLUME_CUDA_CUDAISOSURFACERAYCASTER_KERNEL_H_INCLUDED
#define VOLUME_CUDA_CUDAISOSURFACERAYCASTER_KERNEL_H_INCLUDED

#include "CUDAIsosurfaceRaycaster_kernel.cuh"
#include "CUDAGenericFunctions.cuh"

using namespace megamol;
using namespace megamol::volume_cuda;

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

/**
 * CUDAIsosurfaceRaycaster_kernel::d_render
 */
__global__ void d_renderIso(uint * d_output, float * d_depth, cudaTextureObject_t tex, cudaTextureObject_t transferTex,
    float * d_isovalues, int numisovalues,
    uint imageW, uint imageH, float fovx, float fovy,
    float3 camPos, float3 camDir, float3 camUp, float3 camRight, float zNear, float zFar,
    float density, float brightness, float transferOffset, float transferScale, float minVal, float maxVal,
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f), const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f),
    cudaExtent volSize = make_cudaExtent(1, 1, 1)) {

	int maxSteps = 450;

	float tstep = (boxMax.x - boxMin.x) / static_cast<float>(maxSteps);
	if ((boxMax.y - boxMin.y) / static_cast<float>(maxSteps) > tstep) {
		tstep = (boxMax.y - boxMin.y) / static_cast<float>(maxSteps);
	}
	if ((boxMax.z - boxMin.z) / static_cast<float>(maxSteps) > tstep) {
		tstep = (boxMax.z - boxMin.z) / static_cast<float>(maxSteps);
	}
	maxSteps *= 2; // security factor, could be sqrt(2)

	const float opacityThreshold = 0.95f;

	// pixel coordinates
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	// read the depth value and transform it to world coordinates
	float dv = 1.0f;
	if (d_depth != NULL) {
		dv = d_depth[y * imageW + x];
	}

	// TODO correct depth value
	//float depthVal = (2.0f * zNear) / (zFar + zNear - dv * (zFar - zNear));
	//depthVal = zNear + depthVal * (zFar - zNear);

	//printf("%f %f\n", dv, depthVal);

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

    if (numisovalues < 1) {
        d_output[y*imageW + x] = rgbaFloatToInt(make_float4(0.0f));
        return;
    }

    float3 sP;
    sP.x = (pos.x - boxMin.x) / diff.x;
    sP.y = (pos.y - boxMin.y) / diff.y;
    sP.z = (pos.z - boxMin.z) / diff.z;
    float val = (tex3D<float>(tex, sP.x, sP.y, sP.z) - minVal / (maxVal - minVal));

    //float isoDiff = 0;
    //float isoDiffOld = val - d_isovalues[0];

    float * isoDiffs = new float[numisovalues];
    float * isoDiffsOld = new float[numisovalues];

    for (int i = 0; i < numisovalues; i++) {
        isoDiffs[i] = 0.0f;
        isoDiffsOld[i] = val - d_isovalues[i];
    }

    float3 voxelSize = make_float3(1.0f / (float)volSize.width, 1.0f / (float)volSize.height, 1.0f / (float)volSize.depth);

    //float alpha = 1.0f / (float)numisovalues;

    float4 colors = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    // TODO make colors changeable

    //bool firstHit = true;
    //float3 firstHitPos;

	float projA = -(zFar + zNear) / (zFar - zNear);
	float projB = -2.0f * zNear * zFar / (zFar - zNear);

	for (int i = 0; i < maxSteps; i++) {
		// remap position to [0, 1] coordinates
		float3 samplePos;
		samplePos.x = (pos.x - boxMin.x) / diff.x;
		samplePos.y = (pos.y - boxMin.y) / diff.y;
		samplePos.z = (pos.z - boxMin.z) / diff.z;

		// read from 3D texture
		float sample = tex3D<float>(tex, samplePos.x, samplePos.y, samplePos.z);
		
		// normalize the sample
		sample = (sample - minVal) / (maxVal - minVal);

        for (int isoIndex = 0; isoIndex < numisovalues; isoIndex++) {
            isoDiffs[isoIndex] = sample - d_isovalues[isoIndex];

            if ((isoDiffs[isoIndex] * isoDiffsOld[isoIndex] <= 0.0f)) {
                // interpolated exact position of the isosurface
                float3 isoPos = lerp(pos - step, pos, isoDiffsOld[isoIndex] / (isoDiffsOld[isoIndex] - isoDiffs[isoIndex]));
                
                //if (firstHit) {
                //    firstHitPos = isoPos;
                //    firstHit = false;
                //}

                float3 isoSamplePos;
                isoSamplePos.x = (isoPos.x - boxMin.x) / diff.x;
                isoSamplePos.y = (isoPos.y - boxMin.y) / diff.y;
                isoSamplePos.z = (isoPos.z - boxMin.z) / diff.z;

                float3 gradient = make_float3(1, 0, 0);
                gradient.x = ((tex3D<float>(tex, isoSamplePos.x + voxelSize.x, isoSamplePos.y, isoSamplePos.z) - minVal) / (maxVal - minVal))
                    - ((tex3D<float>(tex, isoSamplePos.x - voxelSize.x, isoSamplePos.y, isoSamplePos.z) - minVal) / (maxVal - minVal));
                gradient.y = ((tex3D<float>(tex, isoSamplePos.x, isoSamplePos.y + voxelSize.y, isoSamplePos.z) - minVal) / (maxVal - minVal))
                    - ((tex3D<float>(tex, isoSamplePos.x, isoSamplePos.y - voxelSize.y, isoSamplePos.z) - minVal) / (maxVal - minVal));
                gradient.z = ((tex3D<float>(tex, isoSamplePos.x, isoSamplePos.y, isoSamplePos.z + voxelSize.z) - minVal) / (maxVal - minVal))
                    - ((tex3D<float>(tex, isoSamplePos.x, isoSamplePos.y, isoSamplePos.z - voxelSize.z) - minVal) / (maxVal - minVal));
                gradient = normalize(gradient);

                float4 col = make_float4(0.0);

                col = colors;

                // TODO make this adjustable
                float3 lightDir = make_float3(0.0f, 0.0f, 1.0f);
                float4 lightParams = make_float4(0.1f, 0.3f, 0.2f, 10.0f);
                float4 mycol = CUDAIsosurfaceRaycaster_kernel::performLighting(gradient, -eyeRay.d, lightDir, col, lightParams);

                mycol *= mycol.w;
                mycol.w = col.w;

                sum = sum + (mycol * (1.0f - sum.w));

                // exit early if opaque
                if (sum.w > opacityThreshold) {
                    break;
                }
            }
            isoDiffsOld[isoIndex] = isoDiffs[isoIndex];
        }

		float sampleCamDist = length(eyeRay.o - pos);

		// depth correction if another image is already present
		float localdepth = 0.5f * (-projA * sampleCamDist + projB) / sampleCamDist + 0.5f;
		if (localdepth >= dv) {
			break;
		}

		// lookup in transfer function texture
		float4 col = tex1D<float4>(transferTex, (sample - transferOffset) * transferScale);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;

		// "over" operator for front-to-back blending
		sum = sum + col * (1.0f - sum.w);

		t += tstep;
		if (t > tfar) break;
		pos += step;
	}
    
	sum *= brightness;

	// write output color
	d_output[y * imageW + x] = rgbaFloatToInt(sum);

    free(isoDiffs);
    free(isoDiffsOld);
}

/**
 * CUDAIsosurfaceRaycaster_kernel::performLighting
 */
__device__ float4 CUDAIsosurfaceRaycaster_kernel::performLighting(float3 normal, float3 camDirection, float3 lightDirection, float4 surfaceColor, float4 lightParams) {
    return make_float4(1.0f, 0.0f, 0.0f, 1.0f);
}
 
/**
 * CUDAIsosurfaceRaycaster_kernel::freeCudaBuffers
 */
void CUDAIsosurfaceRaycaster_kernel::freeCudaBuffers(void) {

    if (d_volumeArray) {
        checkCudaErrors(cudaDestroyTextureObject(this->texObj));
        checkCudaErrors(cudaFreeArray(d_volumeArray));
    }

    if (d_customTransferFuncArray) {
        checkCudaErrors(cudaDestroyTextureObject(this->customTransferTexObj));
        checkCudaErrors(cudaFreeArray(d_customTransferFuncArray));
    }
}

/**
 * CUDAIsosurfaceRaycaster_kernel::render_kernel
 */
void CUDAIsosurfaceRaycaster_kernel::render_kernel(dim3 gridSize, dim3 blockSize, uint * d_output, float * d_depth, uint imageW, uint imageH, float fovx, float fovy, float3 camPos, float3 camDir,
	float3 camUp, float3 camRight, float zNear, float zFar, float density, float brightness, float transferOffset, float transferScale,
	const float3 boxMin, const float3 boxMax, cudaExtent volSize) {

    int numIsovalues = static_cast<int>(this->isovalues.size());
    float * isovalptr = thrust::raw_pointer_cast(this->isovalues.data());

	d_renderIso <<<gridSize, blockSize >>>(d_output, d_depth, this->texObj, this->customTransferTexObj, isovalptr, numIsovalues, imageW, imageH, fovx, fovy, camPos, camDir, camUp, camRight, zNear, zFar, density, brightness,
		transferOffset, transferScale, minVal, maxVal, boxMin, boxMax, volSize);
}


/**
 * CUDAIsosurfaceRaycaster_kernel::copyColorValues
 */
void CUDAIsosurfaceRaycaster_kernel::copyColorValues(float4 * cvals, int functionSize) {
    if (this->d_customTransferFuncArray) {
        checkCudaErrors(cudaDestroyTextureObject(this->customTransferTexObj));
        checkCudaErrors(cudaFreeArray(this->d_customTransferFuncArray));
        this->d_customTransferFuncArray = 0;
    }

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaMallocArray(&this->d_customTransferFuncArray, &channelDesc, functionSize, 1));
	checkCudaErrors(cudaMemcpyToArray(this->d_customTransferFuncArray, 0, 0, cvals, sizeof(float4)*functionSize, cudaMemcpyHostToDevice));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = this->d_customTransferFuncArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&this->customTransferTexObj, &texRes, &texDescr, NULL));
}

/**
 * CUDAIsosurfaceRaycaster_kernel::transferNewVolume
 */
void CUDAIsosurfaceRaycaster_kernel::transferNewVolume(void * h_volume, cudaExtent volumeSize) {
	if (d_volumeArray) {
        checkCudaErrors(cudaDestroyTextureObject(this->texObj));
		checkCudaErrors(cudaFreeArray(this->d_volumeArray));
		d_volumeArray = 0;
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaMalloc3DArray(&this->d_volumeArray, &channelDesc, volumeSize));

	// compute min and max values of the volume
	float * volptr = static_cast<float*>(h_volume);
	thrust::pair<float*, float*> res = thrust::minmax_element(volptr, volptr + (volumeSize.width * volumeSize.depth * volumeSize.height));
	this->minVal = (float)*res.first;
	this->maxVal = (float)*res.second;

	// copy the data
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(h_volume, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = this->d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = this->d_volumeArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&this->texObj, &texRes, &texDescr, NULL));
}


/**
 * CUDAIsosurfaceRaycaster_kernel::initCudaDevice
 */
void CUDAIsosurfaceRaycaster_kernel::initCudaDevice(void * h_volume, cudaExtent volumeSize, float4 * transferFunction, int functionSize) {
	transferNewVolume(h_volume, volumeSize);
	//copyTransferFunction(transferFunction, functionSize);
}

/**
 * CUDAIsosurfaceRaycaster_kernel::initCudaDevice
 */
void CUDAIsosurfaceRaycaster_kernel::setIsoValues(const std::vector<float>& h_isovalues, int h_numisovalues) {
    this->isovalues = h_isovalues;
}

/**
 * CUDAIsosurfaceRaycaster_kernel::CUDAIsosurfaceRaycaster_kernel
 */
CUDAIsosurfaceRaycaster_kernel::CUDAIsosurfaceRaycaster_kernel(void) {
    this->d_volumeArray = 0;
    this->d_customTransferFuncArray = 0;
    this->texObj = 0;
    this->customTransferTexObj = 0;
}

/**
 * CUDAIsosurfaceRaycaster_kernel::~CUDAIsosurfaceRaycaster_kernel
 */
CUDAIsosurfaceRaycaster_kernel::~CUDAIsosurfaceRaycaster_kernel(void) {
}

#endif /* defined VOLUME_CUDA_CUDAISOSURFACERAYCASTER_KERNEL_H_INCLUDED */