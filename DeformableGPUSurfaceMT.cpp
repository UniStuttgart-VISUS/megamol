//
// DeformableGPUSurfaceMT.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 17, 2013
// Author     : scharnkn
//

#include "stdafx.h"

#include <glh/glh_extensions.h>
#include "DeformableGPUSurfaceMT.h"
#include "ogl_error_check.h"
#include "cuda_error_check.h"
#include "ComparativeSurfacePotentialRenderer.cuh"
#include "HostArr.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifdef WITH_CUDA
#define USE_TIMER

using namespace megamol;
using namespace megamol::protein;


/*
 * DeformableGPUSurfaceMT::DeformableGPUSurfaceMT
 */
DeformableGPUSurfaceMT::DeformableGPUSurfaceMT() : GPUSurfaceMT(),
        neighboursReady(false) {
}


/*
 * DeformableGPUSurfaceMT::DeformableGPUSurfaceMT
 */
DeformableGPUSurfaceMT::DeformableGPUSurfaceMT(const DeformableGPUSurfaceMT& other) :
    GPUSurfaceMT(other) {

    CudaSafeCall(this->vertexExternalForcesScl_D.Validate(other.vertexExternalForcesScl_D.GetSize()));
    CudaSafeCall(cudaMemcpy(
            this->vertexExternalForcesScl_D.Peek(),
            other.vertexExternalForcesScl_D.PeekConst(),
            this->vertexExternalForcesScl_D.GetSize()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->volGradient_D.Validate(other.volGradient_D.GetSize()));
    CudaSafeCall(cudaMemcpy(
            this->volGradient_D.Peek(),
            other.volGradient_D.PeekConst(),
            this->volGradient_D.GetSize()*sizeof(float4),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->laplacian_D.Validate(other.laplacian_D.GetSize()));
    CudaSafeCall(cudaMemcpy(
            this->laplacian_D.Peek(),
            other.laplacian_D.PeekConst(),
            this->laplacian_D.GetSize()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->displLen_D.Validate(other.displLen_D.GetSize()));
    CudaSafeCall(cudaMemcpy(
            this->displLen_D.Peek(),
            other.displLen_D.PeekConst(),
            this->displLen_D.GetSize()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    /// Flag whether the neighbors have been computed
    this->neighboursReady = other.neighboursReady;
}


/*
 * DeformableGPUSurfaceMT::~DeformableGPUSurfaceMT
 */
DeformableGPUSurfaceMT::~DeformableGPUSurfaceMT() {
}


/*
 * DeformableGPUSurfaceMT::MorphToVolume
 */
bool DeformableGPUSurfaceMT::MorphToVolume(float *volume_D, size_t volDim[3],
        float volWSOrg[3], float volWSDelta[3], float isovalue,
        InterpolationMode interpMode, size_t maxIt, float surfMappedMinDisplScl,
        float springStiffness, float forceScl, float externalForcesWeight) {

    CheckForCudaError();

    using vislib::sys::Log;

    if ((!this->triangleIdxReady)||(!this->neighboursReady)) {
        return false;
    }

    if (volume_D == NULL) {
        return false;
    }

    size_t volSize = volDim[0]*volDim[1]*volDim[2];


    /* Init grid parameters for all files */

    if (!CudaSafeCall(InitVolume(
            make_uint3(volDim[0], volDim[1], volDim[2]),
            make_float3(volWSOrg[0], volWSOrg[1], volWSOrg[2]),
            make_float3(volWSDelta[0], volWSDelta[1], volWSDelta[2])))) {
        return false;
    }

    if (!CudaSafeCall(InitVolume_surface_mapping(
            make_uint3(volDim[0], volDim[1], volDim[2]),
            make_float3(volWSOrg[0], volWSOrg[1], volWSOrg[2]),
            make_float3(volWSDelta[0], volWSDelta[1], volWSDelta[2])))) {
        return false;
    }

    // Compute gradient
    if (!CudaSafeCall(this->volGradient_D.Validate(volSize))) {
        return false;
    }
    if (!CudaSafeCall(this->volGradient_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(), volume_D,
            this->volGradient_D.GetSize()))) {
        return false;
    }

//    // DEBUG Print gradient field
//    HostArr<float4> gradFieldTest;
//    gradFieldTest.Validate(this->volGradient_D.GetSize());
//    if (!CudaSafeCall(this->volGradient_D.CopyToHost(gradFieldTest.Peek()))) {
//        return false;
//    }
//    for (int i = 0; i < this->volGradient_D.GetSize(); ++i) {
//        if (gradFieldTest.Peek()[i].x || gradFieldTest.Peek()[i].y|| gradFieldTest.Peek()[i].z) {
//        printf("%i: Gradient: %f %f %f\n", i,
//                gradFieldTest.Peek()[i].x,
//                gradFieldTest.Peek()[i].y,
//                gradFieldTest.Peek()[i].z);
//        }
//    }
//    // END DEBUG

    //    printf("Create VBO of size %u\n", activeVertexCount*this->vertexDataStride*sizeof(float));
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0))) {
        return false;
    }


    // Get mapped pointers to the vertex data buffers
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            this->vertexDataResource))) {                 // The mapped resource
        return false;
    }

    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
        return false;
    }

    // Init forces scale factor with -1 or 1, depending on whether they start
    // outside or inside the isosurface
    if (!CudaSafeCall (InitExternalForceScl(
            this->vertexExternalForcesScl_D.Peek(),
            volume_D,
            vboPt,
            static_cast<uint>(this->vertexExternalForcesScl_D.GetCount()),
            isovalue,
            this->vertexDataOffsPos,
            this->vertexDataStride))) {
        return false;
    }

    if (!CudaSafeCall(this->laplacian_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->laplacian_D.Set(0))) {
        return false;
    }

    if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Set(0xff))) {
        return false;
    }

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

//    // DEBUG Print mapped positions
//    printf("Mapped positions before\n");
//    HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*this->vertexDataStride);
//    cudaMemcpy(vertexPos.Peek(), vboPt, sizeof(float)*vertexCnt*this->vertexDataStride, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 10; ++i) {
//        printf("%i: Vertex position (%f %f %f)\n", i,
//                vertexPos.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+0],
//                vertexPos.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+1],
//                vertexPos.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+2]);
//
//    }
//    printf("vertexCnt %u, externalForcesWeight %f, forcesScl %f, springStiffness %f, isovalue %f, surfMappedMinDisplScl %f\n",
//            this->vertexCnt, externalForcesWeight, forceScl,
//            springStiffness, isovalue, surfMappedMinDisplScl);
//    // End DEBUG

    if (interpMode == INTERP_LINEAR) {
        for (uint i = 0; i < maxIt; i += UPDATE_VTX_POS_ITERATIONS_PER_KERNEL) {

            // Update position for all vertices
            // Use no distance field
            if (!CudaSafeCall(UpdateVertexPositionTrilinear(
                    volume_D,
                    vboPt,
                    this->vertexExternalForcesScl_D.Peek(),
                    this->vertexNeighbours_D.Peek(),
                    this->volGradient_D.Peek(),
                    this->laplacian_D.Peek(),
                    this->vertexCnt,
                    externalForcesWeight,
                    forceScl,
                    springStiffness,
                    isovalue,
                    surfMappedMinDisplScl,
                    this->vertexDataOffsPos,
                    this->vertexDataStride))) {
                return false;
            }
        }
        CheckForCudaErrorSync();
    } else {
        for (uint i = 0; i < maxIt; i += UPDATE_VTX_POS_ITERATIONS_PER_KERNEL) {

            // Update position for all vertices
            if (!CudaSafeCall(UpdateVertexPositionTricubic(
                    volume_D,
                    vboPt,
                    this->vertexExternalForcesScl_D.Peek(),
                    this->vertexNeighbours_D.Peek(),
                    this->volGradient_D.Peek(),
                    this->laplacian_D.Peek(),
                    this->vertexCnt,
                    externalForcesWeight,
                    forceScl,
                    springStiffness,
                    isovalue,
                    surfMappedMinDisplScl,
                    this->vertexDataOffsPos,
                    this->vertexDataStride))) {
                return false;
            }
        }
    }

//    // DEBUG Print mapped positions
//    printf("Mapped positions after\n");
//    //HostArr<float> vertexPos;
//    vertexPos.Validate(vertexCnt*this->vertexDataStride);
//    cudaMemcpy(vertexPos.Peek(), vboPt, sizeof(float)*vertexCnt*this->vertexDataStride, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 10; ++i) {
//        printf("%i: Vertex position (%f %f %f)\n", i,
//                vertexPos.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+0],
//                vertexPos.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+1],
//                vertexPos.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+2]);
//
//    }
//    // End DEBUG

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "%s: Time for mapping (%u iterations, %u vertices): %f sec\n",
            "DeformableGPUSurfaceMT",
            maxIt, this->vertexCnt, dt_ms/1000.0f);
#endif

//    // Flag vertices adjacent to corrupt triangles
//    if (!CudaSafeCall(FlagVerticesInCorruptTriangles(
//            vboPt,
//            this->vertexDataMappedStride,
//            this->vertexDataMappedOffsPosNew,
//            this->vertexDataMappedOffsCorruptTriangleFlag,
//            vboTriangleIdxPt,
//            volume_D,
//            this->vertexExternalForcesScl_D.Peek(),
//            triangleCnt,
//            surfMappedMinDisplScl,
//            isovalue))) {
//        return false;
//    } // TODO Separate function?

    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }


    return true;
}


/*
 * DeformableGPUSurfaceMT::operator=
 */
DeformableGPUSurfaceMT& DeformableGPUSurfaceMT::operator=(const DeformableGPUSurfaceMT &rhs) {
    GPUSurfaceMT::operator =(rhs);


    CudaSafeCall(this->vertexExternalForcesScl_D.Validate(rhs.vertexExternalForcesScl_D.GetSize()));
    CudaSafeCall(cudaMemcpy(
            this->vertexExternalForcesScl_D.Peek(),
            rhs.vertexExternalForcesScl_D.PeekConst(),
            this->vertexExternalForcesScl_D.GetSize()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->volGradient_D.Validate(rhs.volGradient_D.GetSize()));
    CudaSafeCall(cudaMemcpy(
            this->volGradient_D.Peek(),
            rhs.volGradient_D.PeekConst(),
            this->volGradient_D.GetSize()*sizeof(float4),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->laplacian_D.Validate(rhs.laplacian_D.GetSize()));
    CudaSafeCall(cudaMemcpy(
            this->laplacian_D.Peek(),
            rhs.laplacian_D.PeekConst(),
            this->laplacian_D.GetSize()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->displLen_D.Validate(rhs.displLen_D.GetSize()));
    CudaSafeCall(cudaMemcpy(
            this->displLen_D.Peek(),
            rhs.displLen_D.PeekConst(),
            this->displLen_D.GetSize()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    /// Flag whether the neighbors have been computed
    this->neighboursReady = rhs.neighboursReady;

    return *this;
}

#endif // WITH_CUDA

