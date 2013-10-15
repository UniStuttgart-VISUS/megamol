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
#include "ComparativeSurfacePotentialRenderer_inline_device_functions.cuh"
#include "HostArr.h"
#include "DiffusionSolver.h"
#include "constantGridParams.cuh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#define USE_TIMER

using namespace megamol;
using namespace megamol::protein;


/**
 * Writes a flag for every vertex that is adjacent to a corrupt triangles.
 *
 * @param[in,out] vertexData_D              The buffer with the vertex data
 * @param[in]     vertexDataStride          The stride for the vertex data
 *                                          buffer
 * @param[in]     vertexDataOffsPos         The position offset in the vertex
 *                                          data buffer
 * @param[in]     vertexDataOffsCorruptFlag The corruption flag offset in the
 *                                          vertex data buffer
 * @param[in]     triangleVtxIdx_D          Array with triangle vertex indices
 * @param[in]     volume_D                  The target volume defining the
 *                               iso-surface
 * @param[in]     externalForcesScl_D       Array with the scale factor for the external force
 * @param[in]     triangleCnt               The number of triangles
 * @param[in]     minDispl                  Minimum force scale to keep going
 * @param[in]     isoval                    The iso-value defining the iso-surface
 *
 * TODO
 */
__global__ void FlagCorruptTriangleVertices_D(
        float *vertexFlag_D,
        float *vertexData_D,
        uint vertexDataStride,
        uint vertexDataOffsPos,
        uint vertexDataOffsNormal,
        uint *triangleVtxIdx_D,
        float *targetVol_D,
        uint triangleCnt,
        float isoval) {

    const uint idx = GetThreadIndex();
    if (idx >= triangleCnt) {
        return;
    }

    /* Alternative 1: Sample volume at trianglemidpoint */

    const uint baseIdx0 = vertexDataStride*triangleVtxIdx_D[3*idx+0];
    const uint baseIdx1 = vertexDataStride*triangleVtxIdx_D[3*idx+1];
    const uint baseIdx2 = vertexDataStride*triangleVtxIdx_D[3*idx+2];
    const float3 p0 = make_float3(vertexData_D[baseIdx0+vertexDataOffsPos+0],
                                  vertexData_D[baseIdx0+vertexDataOffsPos+1],
                                  vertexData_D[baseIdx0+vertexDataOffsPos+2]);
    const float3 p1 = make_float3(vertexData_D[baseIdx1+vertexDataOffsPos+0],
                                  vertexData_D[baseIdx1+vertexDataOffsPos+1],
                                  vertexData_D[baseIdx1+vertexDataOffsPos+2]);
    const float3 p2 = make_float3(vertexData_D[baseIdx2+vertexDataOffsPos+0],
                                  vertexData_D[baseIdx2+vertexDataOffsPos+1],
                                  vertexData_D[baseIdx2+vertexDataOffsPos+2]);
    // Sample volume at midpoint
    const float3 midPoint = (p0+p1+p2)/3.0;
    const float volSampleMidPoint = ::SampleFieldAtPosTricub_D<float>(midPoint, targetVol_D);
    float flag = float(::fabs(volSampleMidPoint-isoval) > 0.3);
    vertexFlag_D[triangleVtxIdx_D[3*idx+0]] = flag;
    vertexFlag_D[triangleVtxIdx_D[3*idx+1]] = flag;
    vertexFlag_D[triangleVtxIdx_D[3*idx+2]] = flag;

    /* Alternative 2: calc variance of angle between normals */

//    const uint baseIdx0 = vertexDataStride*triangleVtxIdx_D[3*idx+0];
//    const uint baseIdx1 = vertexDataStride*triangleVtxIdx_D[3*idx+1];
//    const uint baseIdx2 = vertexDataStride*triangleVtxIdx_D[3*idx+2];
//    const float3 n0 = make_float3(vertexData_D[baseIdx0+vertexDataOffsNormal+0],
//                                  vertexData_D[baseIdx0+vertexDataOffsNormal+1],
//                                  vertexData_D[baseIdx0+vertexDataOffsNormal+2]);
//    const float3 n1 = make_float3(vertexData_D[baseIdx1+vertexDataOffsNormal+0],
//                                  vertexData_D[baseIdx1+vertexDataOffsNormal+1],
//                                  vertexData_D[baseIdx1+vertexDataOffsNormal+2]);
//    const float3 n2 = make_float3(vertexData_D[baseIdx2+vertexDataOffsNormal+0],
//                                  vertexData_D[baseIdx2+vertexDataOffsNormal+1],
//                                  vertexData_D[baseIdx2+vertexDataOffsNormal+2]);
//    // Sample volume at midpoint
//    const float3 avgNormal = (n0+n1+n2)/3.0;
//    float dot0 = clamp(dot(n0, avgNormal), 0.0, 1.0);
//    float dot1 = clamp(dot(n1, avgNormal), 0.0, 1.0);
//    float dot2 = clamp(dot(n2, avgNormal), 0.0, 1.0);
//    float maxDot = max(dot0, max(dot1, dot2));
//    float flag = float(maxDot > 0.9);
//    vertexFlag_D[triangleVtxIdx_D[3*idx+0]] = flag;
//    vertexFlag_D[triangleVtxIdx_D[3*idx+1]] = flag;
//    vertexFlag_D[triangleVtxIdx_D[3*idx+2]] = flag;
}


/*
 * DeformableGPUSurfaceMT::DeformableGPUSurfaceMT
 */
DeformableGPUSurfaceMT::DeformableGPUSurfaceMT() : GPUSurfaceMT(),
        vboCorruptTriangleVertexFlag(0) {

}


/*
 * DeformableGPUSurfaceMT::DeformableGPUSurfaceMT
 */
DeformableGPUSurfaceMT::DeformableGPUSurfaceMT(const DeformableGPUSurfaceMT& other) :
    GPUSurfaceMT(other) {

    CudaSafeCall(this->vertexExternalForcesScl_D.Validate(other.vertexExternalForcesScl_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexExternalForcesScl_D.Peek(),
            other.vertexExternalForcesScl_D.PeekConst(),
            this->vertexExternalForcesScl_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->externalForces_D.Validate(other.externalForces_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->externalForces_D.Peek(),
            other.externalForces_D.PeekConst(),
            this->externalForces_D.GetCount()*sizeof(float4),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->laplacian_D.Validate(other.laplacian_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->laplacian_D.Peek(),
            other.laplacian_D.PeekConst(),
            this->laplacian_D.GetCount()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->displLen_D.Validate(other.displLen_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->displLen_D.Peek(),
            other.displLen_D.PeekConst(),
            this->displLen_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    /* Make deep copy of corrupt triangle flag buffer */

    if (other.vboCorruptTriangleVertexFlag) {
        // Destroy if necessary
        if (this->vboCorruptTriangleVertexFlag) {
            glBindBufferARB(GL_ARRAY_BUFFER, this->vboCorruptTriangleVertexFlag);
            glDeleteBuffersARB(1, &this->vboCorruptTriangleVertexFlag);
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            this->vboCorruptTriangleVertexFlag = 0;
        }

        // Create vertex buffer object for triangle indices
        glGenBuffersARB(1, &this->vboCorruptTriangleVertexFlag);

        CheckForGLError();

        // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, other.vboCorruptTriangleVertexFlag);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboCorruptTriangleVertexFlag);
        glBufferDataARB(GL_COPY_WRITE_BUFFER,
                sizeof(int)*this->vertexCnt*3, 0, GL_DYNAMIC_DRAW);
        // Copy data
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                sizeof(int)*this->vertexCnt*3);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, 0);
        glBindBufferARB(GL_COPY_READ_BUFFER, 0);
        CheckForGLError();
    }
}


/*
 * DeformableGPUSurfaceMT::~DeformableGPUSurfaceMT
 */
DeformableGPUSurfaceMT::~DeformableGPUSurfaceMT() {
}


/*
 * DeformableGPUSurfaceMT::FlagCorruptTriangleVertices
 */
bool DeformableGPUSurfaceMT::FlagCorruptTriangleVertices(
        float *targetVol_D,
        int3 volDim,
        float volWSOrg[3],
        float volWSDelta[3],
        float isovalue) {

    if (!this->InitCorruptFlagVBO(this->vertexCnt)) {
        return false;
    }

    // Init grid constants
    if (!this->InitGridParams(
            make_uint3(volDim.x, volDim.y, volDim.z),
            make_float3(volWSOrg[0], volWSOrg[1], volWSOrg[2]),
            make_float3(volWSDelta[0], volWSDelta[1], volWSDelta[2]))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    cudaGraphicsResource* cudaTokens[3];

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[0],
            this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[1],
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &cudaTokens[2],
            this->vboCorruptTriangleVertexFlag,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    // Map cuda ressource handles
    if (!CudaSafeCall(cudaGraphicsMapResources(3, cudaTokens, 0))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    /* Get mapped pointers to the vertex data buffer */

    float *vboFlagPt;
    float *vboPt;
    size_t vboSize;
    unsigned int *vboTriangleIdxPt;

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt),
            &vboSize,
            cudaTokens[0]))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriangleIdxPt),
            &vboSize,
            cudaTokens[1]))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboFlagPt),
            &vboSize,
            cudaTokens[2]))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    // Call kernel
    FlagCorruptTriangleVertices_D <<< this->Grid(this->triangleCnt, 256), 256 >>> (
            vboFlagPt,
            vboPt,
            AbstractGPUSurface::vertexDataStride,
            AbstractGPUSurface::vertexDataOffsPos,
            AbstractGPUSurface::vertexDataOffsNormal,
            vboTriangleIdxPt,
            targetVol_D,
            this->triangleCnt,
            isovalue);

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGetLastError())) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnmapResources(3, cudaTokens, 0))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[0]))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[1]))) {
        return false;
    }

//    ::CheckForCudaErrorSync();

    if (!CudaSafeCall(cudaGraphicsUnregisterResource(cudaTokens[2]))) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::InitCorruptFlagVBO
 */
bool DeformableGPUSurfaceMT::InitCorruptFlagVBO(size_t vertexCnt) {

    // Destroy if necessary
    if (this->vboCorruptTriangleVertexFlag) {
        glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, this->vboCorruptTriangleVertexFlag);
        glDeleteBuffersARB(1, &this->vboCorruptTriangleVertexFlag);
        this->vboCorruptTriangleVertexFlag = 0;
    }

    // Create vertex buffer object for corrupt vertex flag
    glGenBuffersARB(1, &this->vboCorruptTriangleVertexFlag);
    glBindBufferARB(GL_ARRAY_BUFFER, this->vboCorruptTriangleVertexFlag);
    glBufferDataARB(GL_ARRAY_BUFFER, sizeof(float)*3*vertexCnt, 0, GL_DYNAMIC_DRAW);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

    return CheckForGLError();
}

/*
 * DeformableGPUSurfaceMT::InitGridParams
 */
bool DeformableGPUSurfaceMT::InitGridParams(uint3 gridSize, float3 org, float3 delta) {
    cudaMemcpyToSymbol(gridSize_D, &gridSize, sizeof(uint3));
    cudaMemcpyToSymbol(gridOrg_D, &org, sizeof(float3));
    cudaMemcpyToSymbol(gridDelta_D, &delta, sizeof(float3));
//    printf("Init grid with org %f %f %f, delta %f %f %f, dim %u %u %u\n", org.x,
//            org.y, org.z, delta.x, delta.y, delta.z, gridSize.x, gridSize.y,
//            gridSize.z);
    return CudaSafeCall(cudaGetLastError());
}


/*
 * DeformableGPUSurfaceMT::MorphToVolume
 */
bool DeformableGPUSurfaceMT::MorphToVolume(float *volume_D, size_t volDim[3],
        float volWSOrg[3], float volWSDelta[3], float isovalue,
        InterpolationMode interpMode, size_t maxIt, float surfMappedMinDisplScl,
        float springStiffness, float forceScl, float externalForcesWeight) {

    using vislib::sys::Log;

    if (!this->triangleIdxReady) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: triangles not ready",
                this->ClassName());
        return false;
    }
    if (!this->neighboursReady) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: neighbours not ready",
                this->ClassName());
        return false;
    }

    if (volume_D == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: volume is zero",
                this->ClassName());
        return false;
    }

    size_t volSize = volDim[0]*volDim[1]*volDim[2];


    /* Init grid parameters for all files */

    if (!CudaSafeCall(InitVolume(
            make_uint3(volDim[0], volDim[1], volDim[2]),
            make_float3(volWSOrg[0], volWSOrg[1], volWSOrg[2]),
            make_float3(volWSDelta[0], volWSDelta[1], volWSDelta[2])))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init device constants",
                this->ClassName());
        return false;
    }

    if (!CudaSafeCall(InitVolume_surface_mapping(
            make_uint3(volDim[0], volDim[1], volDim[2]),
            make_float3(volWSOrg[0], volWSOrg[1], volWSOrg[2]),
            make_float3(volWSDelta[0], volWSDelta[1], volWSDelta[2])))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init device constants",
                this->ClassName());
        return false;
    }

    if (!CudaSafeCall(InitVolume_surface_generation(
            make_uint3(volDim[0], volDim[1], volDim[2]),
            make_float3(volWSOrg[0], volWSOrg[1], volWSOrg[2]),
            make_float3(volWSDelta[0], volWSDelta[1], volWSDelta[2])))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init device constants",
                this->ClassName());
        return false;
    }

    // Compute gradient
    if (!CudaSafeCall(this->externalForces_D.Validate(volSize*4))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not allocate memory",
                this->ClassName());
        return false;
    }
    if (!CudaSafeCall(this->externalForces_D.Set(0))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init memory",
                this->ClassName());
        return false;
    }
    if (!CudaSafeCall(CalcVolGradient((float4*)this->externalForces_D.Peek(), volume_D,
            this->externalForces_D.GetCount()))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not calc volume gradient",
                this->ClassName());
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not register buffer",
                this->ClassName());
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not map resources",
                this->ClassName());
        return false;
    }


    // Get mapped pointers to the vertex data buffers
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,              // The size of the accessible data
            this->vertexDataResource))) {                 // The mapped resource
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not acquire mapped pointer",
                this->ClassName());
        return false;
    }

    if (!CudaSafeCall(this->vertexExternalForcesScl_D.Validate(this->vertexCnt))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not allocate memory",
                this->ClassName());
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
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init external forces",
                this->ClassName());
        return false;
    }

    if (!CudaSafeCall(this->laplacian_D.Validate(this->vertexCnt))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not allocate memory",
                this->ClassName());
        return false;
    }
    if (!CudaSafeCall(this->laplacian_D.Set(0))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init memory",
                this->ClassName());
        return false;
    }

    if (!CudaSafeCall(this->displLen_D.Validate(this->vertexCnt))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not allocate memory",
                this->ClassName());
        return false;
    }
    if (!CudaSafeCall(this->displLen_D.Set(0xff))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init memory",
                this->ClassName());
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
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not update position with trilinear interpolation",
                        this->ClassName());
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
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: could not update position with tricubic interpolation",
                        this->ClassName());
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
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not unmap resources",
                this->ClassName());
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not unregister buffer",
                this->ClassName());
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
//            this->volGradient_D.GetCount()))) {
//        return false;
//    }

//    // DEBUG Print gradient field
//    HostArr<float4> gradFieldTest;
//    gradFieldTest.Validate(this->volGradient_D.GetCount());
//    if (!CudaSafeCall(this->volGradient_D.CopyToHost(gradFieldTest.Peek()))) {
//        return false;
//    }
//    for (int i = 0; i < this->volGradient_D.GetCount(); ++i) {
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
//    gradFieldTest.Validate(this->volGradient_D.GetCount());
//    if (!CudaSafeCall(this->volGradient_D.CopyToHost(gradFieldTest.Peek()))) {
//        return false;
//    }
//    for (int i = 0; i < this->volGradient_D.GetCount(); ++i) {
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


    CudaSafeCall(this->vertexExternalForcesScl_D.Validate(rhs.vertexExternalForcesScl_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexExternalForcesScl_D.Peek(),
            rhs.vertexExternalForcesScl_D.PeekConst(),
            this->vertexExternalForcesScl_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->externalForces_D.Validate(rhs.externalForces_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->externalForces_D.Peek(),
            rhs.externalForces_D.PeekConst(),
            this->externalForces_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->laplacian_D.Validate(rhs.laplacian_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->laplacian_D.Peek(),
            rhs.laplacian_D.PeekConst(),
            this->laplacian_D.GetCount()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->displLen_D.Validate(rhs.displLen_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->displLen_D.Peek(),
            rhs.displLen_D.PeekConst(),
            this->displLen_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->gvfTmp_D.Validate(rhs.gvfTmp_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->gvfTmp_D.Peek(),
            rhs.gvfTmp_D.PeekConst(),
            this->gvfTmp_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->gvfConstData_D.Validate(rhs.gvfConstData_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->gvfConstData_D.Peek(),
            rhs.gvfConstData_D.PeekConst(),
            this->gvfConstData_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->grad_D.Validate(rhs.grad_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->grad_D.Peek(),
            rhs.grad_D.PeekConst(),
            this->grad_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->distField_D.Validate(rhs.distField_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->distField_D.Peek(),
            rhs.distField_D.PeekConst(),
            this->distField_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    /* Make deep copy of corrupt triangle flag buffer */

    if (rhs.vboCorruptTriangleVertexFlag) {
        // Destroy if necessary
        if (this->vboCorruptTriangleVertexFlag) {
            glBindBufferARB(GL_ARRAY_BUFFER, this->vboCorruptTriangleVertexFlag);
            glDeleteBuffersARB(1, &this->vboCorruptTriangleVertexFlag);
            glBindBufferARB(GL_ARRAY_BUFFER, 0);
            this->vboCorruptTriangleVertexFlag = 0;
        }

        // Create vertex buffer object for triangle indices
        glGenBuffersARB(1, &this->vboCorruptTriangleVertexFlag);

        CheckForGLError();

        // Map as copy buffer
        glBindBufferARB(GL_COPY_READ_BUFFER, rhs.vboCorruptTriangleVertexFlag);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, this->vboCorruptTriangleVertexFlag);
        glBufferDataARB(GL_COPY_WRITE_BUFFER,
                sizeof(int)*this->vertexCnt*3, 0, GL_DYNAMIC_DRAW);
        // Copy data
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                sizeof(int)*this->vertexCnt*3);
        glBindBufferARB(GL_COPY_WRITE_BUFFER, 0);
        glBindBufferARB(GL_COPY_READ_BUFFER, 0);
        CheckForGLError();
    }

    return *this;
}


/*
 * DeformableGPUSurfaceMT::Release
 */
void DeformableGPUSurfaceMT::Release() {
    GPUSurfaceMT::Release();
    CudaSafeCall(this->vertexExternalForcesScl_D.Release());
    CudaSafeCall(this->gvfTmp_D.Release());
    CudaSafeCall(this->gvfConstData_D.Release());
    CudaSafeCall(this->grad_D.Release());
    CudaSafeCall(this->laplacian_D.Release());
    CudaSafeCall(this->displLen_D.Release());
    CudaSafeCall(this->distField_D.Release());
    CudaSafeCall(this->externalForces_D.Release());
}

#endif // WITH_CUDA

