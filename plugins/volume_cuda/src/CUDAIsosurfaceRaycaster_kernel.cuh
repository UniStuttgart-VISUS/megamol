/*
 * CUDAIsosurfaceRaycaster_kernel.cuh
 * Copyright (C) 2009-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMMOLMAPPLG_CUDAISOSURFACERAYCASTER_KERNEL_CUH_INCLUDED
#define MMMOLMAPPLG_CUDAISOSURFACERAYCASTER_KERNEL_CUH_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "helper_includes/helper_cuda.h"
#include "helper_includes/exception.h"
#include "helper_includes/helper_math.h"

#include "CUDAAdditionalTypedefs.cuh"

#include "stdafx.h"

namespace megamol {
namespace volume_cuda {
    class CUDAIsosurfaceRaycaster_kernel {
    public:

        /** 
         * Constructor
         */
        CUDAIsosurfaceRaycaster_kernel(void);

        /**
         * Destructor
         */
        virtual ~CUDAIsosurfaceRaycaster_kernel(void);

        /**
         * Transfers a new transfer function to the GPU.
         * 
         * @param transferFunction Pointer to the transfer function values.
         * @param functionSize number of transfer function values.
         */
        void copyTransferFunction(float4 * transferFunction, int functionSize = 256);

        /**
         * Initializes the CUDA device
         * 
         * @param h_volume Pointer to the initial float volume.
         * @param volumeSize The extents of the transferred volume.
         * @param h_transferFunction Pointer to the transfer function values.
         * @param functionSize Number of transfer function values.
         */
        void initCudaDevice(void * h_volume, cudaExtent volumeSize, float4 * transferFunction, int functionSize = 256);

        /**
         * Frees all initialized CUDA buffers.
         */
        void freeCudaBuffers(void);

        /**
         * Renders the scene using the given parameters.
         * 
         * @param gridSize The CUDA grid size.
         * @param blockSize The CUDA block size.
         * @param d_output Pointer to the output image.
         *  @param d_depth Pointer to the already existing depth buffer image to test against.
         * @param imageW The width of the output image.
         * @param imageH The height of the output image.
         * @param fovx The camera field of view in x direction (Radians).
         * @param fovy The camera field of view in y direction (Radians).
         * @param camPos The position of the camera.
         * @param camDir The view direction of the camera.
         * @param camUp The up vector of the camera.
         * @param camRight The right vector of the camera.
         * @param zNear The distance of the near plane to the camera position.
         * @param density The density scaling factor.
         * @param brightness The brightness scaling factor.
         * @param transferOffset The offset for the transfer function.
         * @param transferScale The scaling factor for the transfer function.
         * @param boxMin The minimum values of the bounding box.
         * @param boxMax The maximum values of the bounding box.
         * @param volSize The size of the rendered volume.
         */
        void render_kernel(dim3 gridSize, dim3 blockSize, uint * d_output, float * d_depth, uint imageW, uint imageH, float fovx, float fovy, float3 camPos, float3 camDir,
            float3 camUp, float3 camRight, float zNear, float zFar, float density, float brightness, float transferOffset, float transferScale,
            const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f), const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f), cudaExtent volSize = make_cudaExtent(1, 1, 1));

        /**
         * Transfers a new volume to the GPU
         * 
         * @param h_volume Pointer to the float volume data.
         * @param volumeSize The extents of the volume.
         */
        void transferNewVolume(void * h_volume, cudaExtent volumeSize);

    private:

        /** Pointer to the cuda volume array */
        cudaArray * d_volumeArray;

        /** Pointer to the transfer function array */
        cudaArray * d_customTransferFuncArray;

        /**  */
        //static texture<float, 3, cudaReadModeElementType> tex;
        //static texture<float4, 1, cudaReadModeElementType> customTransferTex;

        /** The texture object containing the volume texture */
        cudaTextureObject_t texObj;

        /** The texture object containing the transfer function texture */
        cudaTextureObject_t customTransferTexObj;

        /** The minimum volume value */
        float minVal;

        /** The maximum volume value */
        float maxVal;
    };

} /* end namespace volume_cuda */
} /* end namespace megamol */

#endif