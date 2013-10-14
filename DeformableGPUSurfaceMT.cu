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
#ifdef WITH_CUDA
#include "ogl_error_check.h"
#include "cuda_error_check.h"
#include "ComparativeSurfacePotentialRenderer.cuh"
#include "HostArr.h"
#include "DiffusionSolver.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#define USE_TIMER

using namespace megamol;
using namespace megamol::protein;


/*
 * DeformableGPUSurfaceMT::DeformableGPUSurfaceMT
 */
DeformableGPUSurfaceMT::DeformableGPUSurfaceMT() : GPUSurfaceMT() {
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

    CudaSafeCall(this->externalForces_D.Validate(other.externalForces_D.GetSize()));
    CudaSafeCall(cudaMemcpy(
            this->externalForces_D.Peek(),
            other.externalForces_D.PeekConst(),
            this->externalForces_D.GetSize()*sizeof(float4),
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

    if (!CudaSafeCall(InitVolume_surface_generation(
            make_uint3(volDim[0], volDim[1], volDim[2]),
            make_float3(volWSOrg[0], volWSOrg[1], volWSOrg[2]),
            make_float3(volWSDelta[0], volWSDelta[1], volWSDelta[2])))) {
        return false;
    }

    // Compute gradient
    if (!CudaSafeCall(this->externalForces_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->externalForces_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(CalcVolGradient((float4*)this->externalForces_D.Peek(), volume_D,
            this->externalForces_D.GetSize()))) {
        return false;
    }

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

    if (interpMode == INTERP_LINEAR) {
        for (uint i = 0; i < maxIt; i += UPDATE_VTX_POS_ITERATIONS_PER_KERNEL) {

            // Update position for all vertices
            // Use no distance field
            if (!CudaSafeCall(UpdateVertexPositionTrilinear(
                    volume_D,
                    vboPt,
                    this->vertexExternalForcesScl_D.Peek(),
                    this->vertexNeighbours_D.Peek(),
                    (float4*)this->externalForces_D.Peek(),
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
    } else {
        for (uint i = 0; i < maxIt; i += UPDATE_VTX_POS_ITERATIONS_PER_KERNEL) {

            // Update position for all vertices
            if (!CudaSafeCall(UpdateVertexPositionTricubic(
                    volume_D,
                    vboPt,
                    this->vertexExternalForcesScl_D.Peek(),
                    this->vertexNeighbours_D.Peek(),
                    (float4*)this->externalForces_D.Peek(),
                    this->laplacian_D.Peek(),
                    this->vertexCnt,
                    externalForcesWeight,
                    forceScl,
                    springStiffness,
                    isovalue,
                    surfMappedMinDisplScl,
                    this->vertexDataOffsPos,
                    this->vertexDataOffsNormal,
                    this->vertexDataStride))) {
                return false;
            }
        }
    }

    CheckForCudaErrorSync();

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

    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::MorphToVolumeDistfield
 */
bool DeformableGPUSurfaceMT::MorphToVolumeDistfield(float *volume_D, size_t volDim[3],
        float volWSOrg[3], float volWSDelta[3], float isovalue,
        InterpolationMode interpMode, size_t maxIt, float surfMappedMinDisplScl,
        float springStiffness, float forceScl, float externalForcesWeight, float distfieldDist) {

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

    if (!CudaSafeCall(InitVolume_surface_generation(
            make_uint3(volDim[0], volDim[1], volDim[2]),
            make_float3(volWSOrg[0], volWSOrg[1], volWSOrg[2]),
            make_float3(volWSDelta[0], volWSDelta[1], volWSDelta[2])))) {
        return false;
    }


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


    // Compute distance field
    if (!CudaSafeCall(this->distField_D.Validate(volSize))) {
        return false;
    }
    if (!CudaSafeCall(ComputeDistField(
            vboPt,
            this->distField_D.Peek(),
            volSize,
            this->vertexCnt,
            this->vertexDataOffsPos,
            this->vertexDataStride))) {
        return false;
    }

    // Compute gradient
    if (!CudaSafeCall(this->externalForces_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->externalForces_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(CalcVolGradientWithDistField(
            (float4*)this->externalForces_D.Peek(),
            volume_D,
            this->distField_D.Peek(),
            distfieldDist,
            isovalue,
            volSize))) {
        return false;
    }
//
//    if (!CudaSafeCall(CalcVolGradient(this->volGradient_D.Peek(), volume_D,
//            this->volGradient_D.GetSize()))) {
//        return false;
//    }

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
                    (float4*)this->externalForces_D.Peek(),
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
                    (float4*)this->externalForces_D.Peek(),
                    this->laplacian_D.Peek(),
                    this->vertexCnt,
                    externalForcesWeight,
                    forceScl,
                    springStiffness,
                    isovalue,
                    surfMappedMinDisplScl,
                    this->vertexDataOffsPos,
                    this->vertexDataOffsNormal,
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
 * DeformableGPUSurfaceMT::MorphToVolumeGVF
 */
bool DeformableGPUSurfaceMT::MorphToVolumeGVF(float *volumeSource_D,
        float *volumeTarget_D,
        const unsigned int *targetCubeStates_D,
        size_t volDim[3],
        float volWSOrg[3], float volWSDelta[3], float isovalue,
        InterpolationMode interpMode, size_t maxIt, float surfMappedMinDisplScl,
        float springStiffness, float forceScl, float externalForcesWeight,
        float gvfScl, unsigned int gvfIt) {

    CheckForCudaError();

    using vislib::sys::Log;

    if ((!this->triangleIdxReady)||(!this->neighboursReady)) {
        return false;
    }

    if (volumeTarget_D == NULL) {
        return false;
    }

    size_t volSize = volDim[0]*volDim[1]*volDim[2];
    size_t gridCellCnt = (volDim[0]-1)*(volDim[1]-1)*(volDim[2]-1);


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

    // Compute external forces
    if (!CudaSafeCall(this->externalForces_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->externalForces_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfTmp_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfTmp_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfConstData_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfConstData_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(this->grad_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->grad_D.Set(0))) {
        return false;
    }

    // Use GVF
    if (!DiffusionSolver::CalcGVF(
            volumeTarget_D,
            this->gvfConstData_D.Peek(),
            targetCubeStates_D,
            this->grad_D.Peek(),
            volDim,
            isovalue,
            this->externalForces_D.Peek(),
            this->gvfTmp_D.Peek(),
            gvfIt,
            gvfScl)) {
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
            volumeTarget_D,
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
                    volumeTarget_D,
                    vboPt,
                    this->vertexExternalForcesScl_D.Peek(),
                    this->vertexNeighbours_D.Peek(),
                    (float4*)this->externalForces_D.Peek(),
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
                    volumeTarget_D,
                    vboPt,
                    this->vertexExternalForcesScl_D.Peek(),
                    this->vertexNeighbours_D.Peek(),
                    (float4*)this->externalForces_D.Peek(),
                    this->laplacian_D.Peek(),
                    this->vertexCnt,
                    externalForcesWeight,
                    forceScl,
                    springStiffness,
                    isovalue,
                    surfMappedMinDisplScl,
                    this->vertexDataOffsPos,
                    this->vertexDataOffsNormal,
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
 * DeformableGPUSurfaceMT::MorphToVolumeTwoWayGVF
 */
bool DeformableGPUSurfaceMT::MorphToVolumeTwoWayGVF(
        float *volumeSource_D,
        float *volumeTarget_D,
        const unsigned int *cellStatesSource_D,
        const unsigned int *cellStatesTarget_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue,
        InterpolationMode interpMode,
        size_t maxIt,
        float surfMappedMinDisplScl,
        float springStiffness,
        float forceScl,
        float externalForcesWeight,
        float gvfScl,
        unsigned int gvfIt) {

    using vislib::sys::Log;

    if ((!this->triangleIdxReady)||(!this->neighboursReady)) {
        return false;
    }

    if ((volumeTarget_D == NULL)||(volumeSource_D == NULL)) {
        return false;
    }

    size_t volSize = volDim.x*volDim.y*volDim.z;
    size_t gridCellCnt = (volDim.x-1)*(volDim.y-1)*(volDim.z-1);


    /* Init grid parameters for all files */

    if (!CudaSafeCall(InitVolume(
            make_uint3(volDim.x, volDim.y, volDim.z),
            make_float3(volOrg.x, volOrg.y, volOrg.z),
            make_float3(volDelta.x, volDelta.y, volDelta.z)))) {
        return false;
    }

    if (!CudaSafeCall(InitVolume_surface_mapping(
            make_uint3(volDim.x, volDim.y, volDim.z),
            make_float3(volOrg.x, volOrg.y, volOrg.z),
            make_float3(volDelta.x, volDelta.y, volDelta.z)))) {
        return false;
    }

    // Compute external forces
    if (!CudaSafeCall(this->externalForces_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->externalForces_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfTmp_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfTmp_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfConstData_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->gvfConstData_D.Set(0))) {
        return false;
    }
    if (!CudaSafeCall(this->grad_D.Validate(volSize*4))) {
        return false;
    }
    if (!CudaSafeCall(this->grad_D.Set(0))) {
        return false;
    }

    // Initialize device constants
    DiffusionSolver::grid grid_H;
    grid_H.size = volDim;
    grid_H.delta = volDelta;
    grid_H.org = volOrg;
    if (!CudaSafeCall(DiffusionSolver::InitDevConstants(grid_H, isovalue))) {
        return false;
    }

    // Calculate two way gvf by using isotropic diffusion
    if (!DiffusionSolver::CalcTwoWayGVF(
           volumeSource_D,
           volumeTarget_D,
           cellStatesSource_D,
           cellStatesTarget_D,
           volDim,
           volOrg,
           volDelta,
           isovalue,
           this->gvfConstData_D.Peek(),
           this->externalForces_D.Peek(),
           this->gvfTmp_D.Peek(),
           gvfIt,
           gvfScl)) {
        return false;
    }

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
            volumeTarget_D,
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

    if (interpMode == INTERP_LINEAR) {
        for (uint i = 0; i < maxIt; i += UPDATE_VTX_POS_ITERATIONS_PER_KERNEL) {

            // Update position for all vertices
            // Use no distance field
            if (!CudaSafeCall(UpdateVertexPositionTrilinear(
                    volumeTarget_D,
                    vboPt,
                    this->vertexExternalForcesScl_D.Peek(),
                    this->vertexNeighbours_D.Peek(),
                    (float4*)this->externalForces_D.Peek(),
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
                    volumeTarget_D,
                    vboPt,
                    this->vertexExternalForcesScl_D.Peek(),
                    this->vertexNeighbours_D.Peek(),
                    (float4*)this->externalForces_D.Peek(),
                    this->laplacian_D.Peek(),
                    this->vertexCnt,
                    externalForcesWeight,
                    forceScl,
                    springStiffness,
                    isovalue,
                    surfMappedMinDisplScl,
                    this->vertexDataOffsPos,
                    this->vertexDataOffsNormal,
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

    CudaSafeCall(this->externalForces_D.Validate(rhs.externalForces_D.GetSize()));
    CudaSafeCall(cudaMemcpy(
            this->externalForces_D.Peek(),
            rhs.externalForces_D.PeekConst(),
            this->externalForces_D.GetSize()*sizeof(float),
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

