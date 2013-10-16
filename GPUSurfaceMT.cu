//
// GPUSurfaceMT.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 17, 2013
// Author     : scharnkn
//

#include "stdafx.h"

#include "GPUSurfaceMT.h"

#ifdef WITH_CUDA

#include "cuda_error_check.h"

#include "ComparativeSurfacePotentialRenderer.cuh"
#include "HostArr.h"
#include "sort_triangles.cuh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace megamol;
using namespace megamol::protein;


/*
 * GPUSurfaceMT::GPUSurfaceMT
 */
GPUSurfaceMT::GPUSurfaceMT() : AbstractGPUSurface() , neighboursReady(false) {
}


/*
 * GPUSurfaceMT::GPUSurfaceMT
 */
GPUSurfaceMT::GPUSurfaceMT(const GPUSurfaceMT& other) : AbstractGPUSurface(other) {

    // Copy GPU memory

    CudaSafeCall(this->cubeStates_D.Validate(other.cubeStates_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeStates_D.Peek(),
            other.cubeStates_D.PeekConst(),
            this->cubeStates_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->cubeOffsets_D.Validate(other.cubeOffsets_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeOffsets_D.Peek(),
            other.cubeOffsets_D.PeekConst(),
            this->cubeOffsets_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->cubeMap_D.Validate(other.cubeMap_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeMap_D.Peek(),
            other.cubeMap_D.PeekConst(),
            this->cubeMap_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->cubeMapInv_D.Validate(other.cubeMapInv_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeMapInv_D.Peek(),
            other.cubeMapInv_D.PeekConst(),
            this->cubeMapInv_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexStates_D.Validate(other.vertexStates_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexStates_D.Peek(),
            other.vertexStates_D.PeekConst(),
            this->vertexStates_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->activeVertexPos_D.Validate(other.activeVertexPos_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->activeVertexPos_D.Peek(),
            other.activeVertexPos_D.PeekConst(),
            this->activeVertexPos_D.GetCount()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexIdxOffs_D.Validate(other.vertexIdxOffs_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexIdxOffs_D.Peek(),
            other.vertexIdxOffs_D.PeekConst(),
            this->vertexIdxOffs_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexMap_D.Validate(other.vertexMap_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexMap_D.Peek(),
            other.vertexMap_D.PeekConst(),
            this->vertexMap_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexMapInv_D.Validate(other.vertexMapInv_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexMapInv_D.Peek(),
            other.vertexMapInv_D.PeekConst(),
            this->vertexMapInv_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexNeighbours_D.Validate(other.vertexNeighbours_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexNeighbours_D.Peek(),
            other.vertexNeighbours_D.PeekConst(),
            this->vertexNeighbours_D.GetCount()*sizeof(int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->verticesPerTetrahedron_D.Validate(other.verticesPerTetrahedron_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->verticesPerTetrahedron_D.Peek(),
            other.verticesPerTetrahedron_D.PeekConst(),
            this->verticesPerTetrahedron_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->tetrahedronVertexOffsets_D.Validate(other.tetrahedronVertexOffsets_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->tetrahedronVertexOffsets_D.Peek(),
            other.tetrahedronVertexOffsets_D.PeekConst(),
            this->tetrahedronVertexOffsets_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->triangleCamDistance_D.Validate(other.triangleCamDistance_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->triangleCamDistance_D.Peek(),
            other.triangleCamDistance_D.PeekConst(),
            this->triangleCamDistance_D.GetCount()*sizeof(float),
            cudaMemcpyDeviceToDevice));

    // The number of active cells
    this->activeCellCnt = other.activeCellCnt;

    // Check whether neighbors have been computed
    this->neighboursReady = other.neighboursReady;
}


/*
 * GPUSurfaceMT::~GPUSurfaceMT
 */
GPUSurfaceMT::~GPUSurfaceMT() {
}


/*
 * DeformableGPUSurfaceMT::ComputeVertexPositions
 */
bool GPUSurfaceMT::ComputeVertexPositions(
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue) {

    using vislib::sys::Log;

    size_t gridCellCnt = (volDim.x-1)*(volDim.y-1)*(volDim.z-1);


    /* Init grid parameters */

    if (!CudaSafeCall(InitVolume(
            make_uint3(volDim.x, volDim.y, volDim.z),
            volOrg,
            volDelta))) {
        return false;
    }

    if (!CudaSafeCall(InitVolume_surface_generation(
            make_uint3(volDim.x, volDim.y, volDim.z),
            volOrg,
            volDelta))) {
        return false;
    }

//    printf("Grid dims %u %u %u\n", volDim[0], volDim[1], volDim[2]);
//    printf("cell count %u\n", gridCellCnt);


    /* Find active grid cells */

    if (!CudaSafeCall(this->cubeStates_D.Validate(gridCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->cubeOffsets_D.Validate(gridCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->cubeStates_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(this->cubeOffsets_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(FindActiveGridCells(
            this->cubeStates_D.Peek(),
            this->cubeOffsets_D.Peek(),
            gridCellCnt,
            isovalue,
            volume_D))) {
        return false;
    }

//    // DEBUG Print Cube states and offsets
//    HostArr<unsigned int> cubeStates;
//    HostArr<unsigned int> cubeOffsets;
//    cubeStates.Validate(gridCellCnt);
//    cubeOffsets.Validate(gridCellCnt);
//    this->cubeStates_D.CopyToHost(cubeStates.Peek());
//    this->cubeOffsets_D.CopyToHost(cubeOffsets.Peek());
//    for (int i = 0; i < gridCellCnt; ++i) {
//        printf ("Cell %i: state %u, offs %u\n", i, cubeStates.Peek()[i],
//                cubeOffsets.Peek()[i]);
//    }
//    // END DEBUG


    /* Get number of active grid cells */

    this->activeCellCnt =
            this->cubeStates_D.GetAt(gridCellCnt-1) +
            this->cubeOffsets_D.GetAt(gridCellCnt-1);
    if (!CheckForCudaError()) {
        return false;
    }


//    printf("Active cell count %u\n", activeCellCnt); // DEBUG
//    printf("Reduction %f\n", 1.0 - static_cast<float>(activeCellCnt)/
//            static_cast<float>(gridCellCnt)); // DEBUG


    /* Prepare cube map */

    if (!CudaSafeCall(this->cubeMapInv_D.Validate(gridCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->cubeMapInv_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(this->cubeMap_D.Validate(this->activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(CalcCubeMap(
            this->cubeMap_D.Peek(),
            this->cubeMapInv_D.Peek(),
            this->cubeOffsets_D.Peek(),
            this->cubeStates_D.Peek(),
            gridCellCnt))) {
        return false;
    }

//
//    // DEBUG Cube map
//    HostArr<unsigned int> cubeMap;
//    HostArr<unsigned int> cubeMapInv;
//    cubeMap.Validate(activeCellCnt);
//    cubeMapInv.Validate(gridCellCnt);
//    cubeMapInv_D.CopyToHost(cubeMapInv.Peek());
//    cubeMap_D.CopyToHost(cubeMap.Peek());
//    for (int i = 0; i < gridCellCnt; ++i) {
//        printf ("Cell %i: cubeMapInv %u\n", i, cubeMapInv.Peek()[i]);
//    }
//    for (int i = 0; i < activeCellCnt; ++i) {
//        printf ("Cell %i: cubeMap %u\n", i, cubeMap.Peek()[i]);
//    }
//    // END DEBUG


    /* Get vertex positions */

    if (!CudaSafeCall(this->vertexStates_D.Validate(7*this->activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->activeVertexPos_D.Validate(7*this->activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexIdxOffs_D.Validate(7*this->activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexStates_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(this->activeVertexPos_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexIdxOffs_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(CalcVertexPositions(
            this->vertexStates_D.Peek(),
            this->activeVertexPos_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            this->cubeMap_D.Peek(),
            this->activeCellCnt,
            isovalue,
            volume_D))) {
        return false;
    }

//    // DEBUG Print active vertex positions
//    HostArr<float3> activeVertexPos;
//    HostArr<unsigned int> vertexStates;
//    HostArr<unsigned int> vertexIdxOffsets;
//    activeVertexPos.Validate(7*this->activeCellCnt);
//    vertexIdxOffsets.Validate(7*this->activeCellCnt);
//    vertexStates.Validate(7*activeCellCnt);
//    cudaMemcpy(vertexStates.Peek(), this->vertexStates_D.Peek(), 7*activeCellCnt*sizeof(unsigned int),
//            cudaMemcpyDeviceToHost);
//    cudaMemcpy(activeVertexPos.Peek(), this->activeVertexPos_D.Peek(), 7*activeCellCnt*sizeof(unsigned int),
//            cudaMemcpyDeviceToHost);
//    cudaMemcpy(vertexIdxOffsets.Peek(), this->vertexIdxOffs_D.Peek(), 7*activeCellCnt*sizeof(unsigned int),
//            cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 7*this->activeCellCnt; ++i) {
//        printf("#%i: active vertexPos %f %f %f (state = %u)\n", i,
//                activeVertexPos.Peek()[i].x,
//                activeVertexPos.Peek()[i].y,
//                activeVertexPos.Peek()[i].z,
//                vertexStates.Peek()[i]);
//    }

//    for (int i = 0; i < 7*this->activeCellCnt; ++i) {
//        printf("#%i: vertex index offset %u (state %u)\n",i,
//                vertexIdxOffsets.Peek()[i],
//                vertexStates.Peek()[i]);
//    }
    // END DEBUG


    /* Get number of active vertices */

    this->vertexCnt =
            this->vertexStates_D.GetAt(7*this->activeCellCnt-1) +
            this->vertexIdxOffs_D.GetAt(7*this->activeCellCnt-1);
    if (!CheckForCudaError()) {
        return false;
    }

//    printf("Vertex Cnt %u\n", this->vertexCnt);

    /* Create vertex buffer object and register with CUDA */

    // Create empty vbo to hold vertex data for the surface
    if (!this->InitVertexDataVBO(this->vertexCnt)) {
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            this->vertexDataResource))) {                   // The mapped resource
        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
            return false;
        }
        return false;
    }

    // Init with zeros
    if (!CudaSafeCall(cudaMemset(vboPt, 0, vboSize))) {
        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
            return false;
        }
        return false;
    }

//    printf("Got VBO of size %u\n", vboSize);


    /* Compact list of vertex positions (keep only active vertices) */

    if (!CudaSafeCall(CompactActiveVertexPositions(
            vboPt,
            this->vertexStates_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            this->activeVertexPos_D.Peek(),
            this->activeCellCnt,
            this->vertexDataOffsPos,  // Array data byte offset
            this->vertexDataStride    // Array data element size
            ))) {
        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
            return false;
        }
        return false;
    }

//    // DEBUG Print vertex positions
//    HostArr<float> vertexPos;
//    vertexPos.Validate(this->vertexCnt*this->vertexDataStride);
//    cudaMemcpy(vertexPos.Peek(), vboPt, this->vertexCnt*this->vertexDataStride*sizeof(float),
//            cudaMemcpyDeviceToHost);
//    for (int i = 0; i < this->vertexCnt; ++i) {
//        printf("#%i: vertexPos %f %f %f\n", i, vertexPos.Peek()[9*i+0],
//                vertexPos.Peek()[9*i+1], vertexPos.Peek()[9*i+2]);
//    }
//    // END DEBUG

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::computeTriangles
 */
bool GPUSurfaceMT::ComputeTriangles(
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue) {

    if (!this->vertexDataReady) { // We need vertex data to generate triangles
        return false;
    }

    size_t triangleVtxCnt;

    /* Calc vertex index map */

    if (!CudaSafeCall(this->vertexMap_D.Validate(this->vertexCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexMapInv_D.Validate(7*this->activeCellCnt))) {
        return false;
    }
    if (!CudaSafeCall(this->vertexMapInv_D.Set(0xff))) {
        return false;
    }
    if (!CudaSafeCall(CalcVertexMap(
            this->vertexMap_D.Peek(),
            this->vertexMapInv_D.Peek(),
            this->vertexIdxOffs_D.Peek(),
            this->vertexStates_D.Peek(),
            this->activeCellCnt))) {
        return false;
    }

//    // DEBUG Print vertex map
//    HostArr<unsigned int> vertexMap;
//    vertexMap.Validate(this->vertexCnt);
//    vertexMap_D.CopyToHost(vertexMap.Peek());
//    for (int i = 0; i < this->vertexMap_D.GetCount(); ++i) {
//        printf("Vertex mapping %i: %u\n", i, vertexMap.Peek()[i]);
//    }
//    // END DEBUG
//
//    // DEBUG Print vertex map
//    HostArr<unsigned int> vertexMapInv;
//    vertexMapInv.Validate(this->vertexMapInv_D.GetCount());
//    vertexMapInv_D.CopyToHost(vertexMapInv.Peek());
//    for (int i = 0; i < this->vertexMapInv_D.GetCount(); ++i) {
//        printf("Inverse Vertex mapping %i: %u\n", i, vertexMapInv.Peek()[i]);
//    }
//    // END DEBUG


    /* Flag tetrahedrons */

    if (!CudaSafeCall(this->verticesPerTetrahedron_D.Validate(6*this->activeCellCnt))) return false;
    if (!CudaSafeCall(FlagTetrahedrons(
            this->verticesPerTetrahedron_D.Peek(),
            this->cubeMap_D.Peek(),
            isovalue,
            this->activeCellCnt,
            volume_D))) {
        return false;
    }


    /* Scan tetrahedrons */

    if (!CudaSafeCall(this->tetrahedronVertexOffsets_D.Validate(6*activeCellCnt))) return false;
    if (!CudaSafeCall(GetTetrahedronVertexOffsets(
            this->tetrahedronVertexOffsets_D.Peek(),
            this->verticesPerTetrahedron_D.Peek(),
            activeCellCnt*6))) {
        return false;
    }


    /* Get triangle vertex count */

    triangleVtxCnt =
            this->tetrahedronVertexOffsets_D.GetAt(activeCellCnt*6-1) +
            this->verticesPerTetrahedron_D.GetAt(activeCellCnt*6-1);
    if (!CheckForCudaError()) {
        return false;
    }

//    printf("Triangle cnt %u\n", triangleVtxCnt);

    this->triangleCnt = triangleVtxCnt/3;

    /* Create vertex buffer object and register with CUDA */

    // Create empty vbo to hold the triangle indices
    if (!this->InitTriangleIdxVBO(this->triangleCnt)) {
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->triangleIdxResource,
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    unsigned int *vboTriangleIdxPt;
    size_t vboTriangleIdxSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->triangleIdxResource, 0))) {

        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
            return false;
        }
        return false;
    }

    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriangleIdxPt), // The mapped pointer
            &vboTriangleIdxSize,             // The size of the accessible data
            this->triangleIdxResource))) {                   // The mapped resource

        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->triangleIdxResource))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
            return false;
        }
        return false;
    }


    /* Generate triangles */

    if (!CudaSafeCall(cudaMemset(vboTriangleIdxPt, 0x00, vboTriangleIdxSize))) {

        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->triangleIdxResource))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
            return false;
        }
        return false;
    }

    if (!CudaSafeCall(GetTrianglesIdx(
            this->tetrahedronVertexOffsets_D.Peek(),
            this->cubeMap_D.Peek(),
            this->cubeMapInv_D.Peek(),
            isovalue,
            this->activeCellCnt*6,
            this->activeCellCnt,
            vboTriangleIdxPt,
            this->vertexMapInv_D.Peek(),
            volume_D))) {

        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->triangleIdxResource))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
            return false;
        }
        return false;
    }

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->triangleIdxResource))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
        return false;
    }
    return true;
}


/*
 * GPUSurfaceMT::computeVertexNormals
 */
bool GPUSurfaceMT::ComputeNormals(
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue) {

    using vislib::sys::Log;

    if (!this->triangleIdxReady) { // We need the triangles mesh info
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: triangles not computed",
                this->ClassName());
        return false;
    }

    CheckForCudaErrorSync();

    /* Init grid parameters */

    if (!CudaSafeCall(InitVolume_surface_generation(
            make_uint3(volDim.x, volDim.y, volDim.z),
            volOrg,
            volDelta))) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not init device constants",
                this->ClassName());

        return false;
    }

//        printf("Init volume surface generation\n");
//        printf("grid size  %u %u %u\n", volDim[0], volDim[1], volDim[2]);
//        printf("grid org   %f %f %f\n", volWSOrg[0], volWSOrg[1], volWSOrg[2]);
//        printf("grid delta %f %f %f\n", volWSDelta[0], volWSDelta[1], volWSDelta[2]);

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not register vertex buffer",
                this->ClassName());

        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not map resources",
                this->ClassName());
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            this->vertexDataResource))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not acquire mapped pointer",
                this->ClassName());
        return false;
    }



//    int cnt = 0;
//    // DEBUG Print vertex map
//    HostArr<unsigned int> vertexMap;
//    vertexMap.Validate(this->vertexCnt);
//    if (!CudaSafeCall(vertexMap_D.CopyToHost(vertexMap.Peek()))) {
//        return false;
//    }
//    for (int i = 0; i < this->vertexMap_D.GetCount(); ++i) {
//        printf("Vertex mapping %i: %u\n", i, vertexMap.Peek()[i]);
////        cnt += vertexMap.Peek()[i];
//    }
//    // END DEBUG
//
//    // DEBUG Print vertex map
//    HostArr<unsigned int> vertexMapInv;
//    vertexMapInv.Validate(this->vertexMapInv_D.GetCount());
//    if (!CudaSafeCall(vertexMapInv_D.CopyToHost(vertexMapInv.Peek()))) {
//        return false;
//    }
//    for (int i = 0; i < this->vertexMapInv_D.GetCount(); ++i) {
//        printf("Inverse Vertex mapping %i: %u\n", i, vertexMapInv.Peek()[i]);
////        cnt += vertexMapInv.Peek()[i];
//    }
//    // END DEBUG

//    printf("active vertex count %u\n", this->vertexCnt);
//    printf("active cube count %u\n", this->activeCellCnt);
//    printf("normals vbo %u\n", vboSize);
//    printf("vertexMap size %u\n", this->vertexMap_D.GetCount());
//    printf("vertexMapInv size %u\n", this->vertexMapInv_D.GetCount());
//    printf("cubeMap_D size %u\n", this->cubeMap_D.GetCount());
//    printf("cubeMapInv_D size %u\n", this->cubeMapInv_D.GetCount());

//        // DEBUG Print buffer content
//        HostArr<float> vertexBuffer;
//        vertexBuffer.Validate(this->vertexDataStride*this->vertexCnt*sizeof(float));
//        if (!CudaSafeCall(cudaMemcpy(vertexBuffer.Peek(), vboPt,
//                this->vertexDataStride*this->vertexCnt*sizeof(float), cudaMemcpyDeviceToHost))) {
//            return false;
//        }
//        for (int i = 0; i < this->vertexCnt; ++i) {
//    //        if (uint(abs(vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsNormal+0]))>= this->vertexCnt) {
//            printf("%i: pos %f %f %f, normal %f %f %f, texcoord %f %f %f\n", i,
//                    vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+0],
//                    vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+1],
//                    vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+2],
//                    vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsNormal+0],
//                    vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsNormal+1],
//                    vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsNormal+2],
//                    vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsTexCoord+0],
//                    vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsTexCoord+1],
//                    vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsTexCoord+2],
//                    this->vertexCnt);
//    //        }
//        }
//        vertexBuffer.Release();
//        // end DEBUG

    if (!CudaSafeCall(ComputeVertexNormals(
            vboPt,
            this->vertexMap_D.Peek(),
            this->vertexMapInv_D.Peek(),
            this->cubeMap_D.Peek(),
            this->cubeMapInv_D.Peek(),
            volume_D,
            isovalue,
            this->vertexCnt,
            this->vertexDataOffsPos,
            this->vertexDataOffsNormal,
            this->vertexDataStride))) {

        return false;
    }

//    // DEBUG Print normals
//    HostArr<float> vertexBuffer;
//    vertexBuffer.Validate(this->vertexDataStride*this->vertexCnt*sizeof(float));
//    if (!CudaSafeCall(cudaMemcpy(vertexBuffer.Peek(), vboPt,
//            this->vertexDataStride*this->vertexCnt*sizeof(float), cudaMemcpyDeviceToHost))) {
//        return false;
//    }
//    for (int i = 0; i < this->vertexCnt; i+=3) {
////        if (uint(abs(vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsNormal+0]))>= this->vertexCnt) {
//                    printf("%i: pos %f %f %f, normal %f %f %f, texcoord %f %f %f\n", i,
//                            vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+0],
//                            vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+1],
//                            vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsPos+2],
//                            vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsNormal+0],
//                            vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsNormal+1],
//                            vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsNormal+2],
//                            vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsTexCoord+0],
//                            vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsTexCoord+1],
//                            vertexBuffer.Peek()[this->vertexDataStride*i+this->vertexDataOffsTexCoord+2],
//                            this->vertexCnt);
////        }
//    }
//    vertexBuffer.Release();
//    // end DEBUG

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not unmap resources",
                this->ClassName());
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: could not unregister buffers",
                this->ClassName());
        return false;
    }

    return true;
}


/*
 * DeformableGPUSurfaceMT::ComputeTexCoords
 */
bool GPUSurfaceMT::ComputeTexCoords(float minCoords[3], float maxCoords[3]) {
    if (!this->triangleIdxReady) { // We need the triangles mesh info
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0));
    CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,             // The size of the accessible data
            this->vertexDataResource));                   // The mapped resource

    if (!CudaSafeCall(ComputeVertexTexCoords(
            vboPt,
            minCoords[0],
            minCoords[1],
            minCoords[2],
            maxCoords[0],
            maxCoords[1],
            maxCoords[2],
            this->vertexCnt,
            this->vertexDataOffsPos,
            this->vertexDataOffsTexCoord,
            this->vertexDataStride))) {

        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
            return false;
        }

        return false;
    }

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }

    return true;
}


/*
 * GPUSurfaceMT::Rotate
 */
bool GPUSurfaceMT::Rotate(float rotMat[9]) {
    CudaDevArr<float> rotate_D;

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,                         // The size of the accessible data
            this->vertexDataResource))) {     // The mapped resource
        return false;
    }

    // Rotate for best fit
    rotate_D.Validate(9);
    if (!CudaSafeCall(cudaMemcpy((void *)rotate_D.Peek(), &rotMat[0],
            9*sizeof(float), cudaMemcpyHostToDevice))) {
        return false;
    }
    if (!CudaSafeCall(RotatePos(
            vboPt,
            this->vertexDataStride,
            this->vertexDataOffsPos,
            rotate_D.Peek(),
            vertexCnt))) {
        return false;
    }

    // Clean up
    rotate_D.Release();

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }

    return true;
}


/*
 * GPUSurfaceMT::SortTrianglesByCamDist
 */
bool GPUSurfaceMT::SortTrianglesByCamDist(float camPos[3]) {

    if (!CudaSafeCall(this->triangleCamDistance_D.Validate(triangleCnt))) {
        return false;
    }

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }
    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->triangleIdxResource,
            this->vboTriangleIdx,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // We need both cuda graphics resources to be mapped at the same time
    cudaGraphicsResource *cudaToken[2];
    cudaToken[0] = this->vertexDataResource;
    cudaToken[1] = this->triangleIdxResource;
    if (!CudaSafeCall(cudaGraphicsMapResources(2, cudaToken, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data and the triangle indices
    float *vboPt;
    uint *vboTriangleIdxPt;
    size_t vboSize, vboTriangleIdxSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,                         // The size of the accessible data
            cudaToken[0]))) {                 // The mapped resource
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboTriangleIdxPt), // The mapped pointer
            &vboTriangleIdxSize,              // The size of the accessible data
            cudaToken[1]))) {                 // The mapped resource
        return false;
    }

    if (!CudaSafeCall(SortTrianglesByCamDistance(
            vboPt,
            this->vertexDataStride,
            this->vertexDataOffsPos,
            make_float3(camPos[0], camPos[1], camPos[2]),
            vboTriangleIdxPt,
            this->triangleCnt,
            this->triangleCamDistance_D.Peek()))) {

        // Unmap CUDA graphics resource
        if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaToken, 0))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
            return false;
        }
        if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
            return false;
        }

        return false;
    }

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, cudaToken, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->triangleIdxResource))) {
        return false;
    }


    return true;
}


/*
 * GPUSurfaceMT::Translate
 */
bool GPUSurfaceMT::Translate(float transVec[3]) {

    // Register memory with CUDA
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->vertexDataResource, this->vboVtxData,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    // Get mapped pointer to the vbo
    float *vboPt;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->vertexDataResource, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&vboPt), // The mapped pointer
            &vboSize,                         // The size of the accessible data
            this->vertexDataResource))) {     // The mapped resource
        return false;
    }

    // Move vertex positions to origin (with respect to centroid)
    if (!CudaSafeCall(TranslatePos(
            vboPt,
            this->vertexDataStride,
            this->vertexDataOffsPos,
            make_float3(transVec[0], transVec[0], transVec[0]),
            this->vertexCnt))) {
        return false;
    }

    // Unmap CUDA graphics resource
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->vertexDataResource, 0))) {
        return false;
    }
    if (!CudaSafeCall(cudaGraphicsUnregisterResource(this->vertexDataResource))) {
        return false;
    }

    return true;
}


/*
 * GPUSurfaceMT::operator=
 */
GPUSurfaceMT& GPUSurfaceMT::operator=(const GPUSurfaceMT &rhs) {
    AbstractGPUSurface::operator=(rhs);

    // Copy GPU memory

    CudaSafeCall(this->cubeStates_D.Validate(rhs.cubeStates_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeStates_D.Peek(),
            rhs.cubeStates_D.PeekConst(),
            this->cubeStates_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->cubeOffsets_D.Validate(rhs.cubeOffsets_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeOffsets_D.Peek(),
            rhs.cubeOffsets_D.PeekConst(),
            this->cubeOffsets_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->cubeMap_D.Validate(rhs.cubeMap_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeMap_D.Peek(),
            rhs.cubeMap_D.PeekConst(),
            this->cubeMap_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->cubeMapInv_D.Validate(rhs.cubeMapInv_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->cubeMapInv_D.Peek(),
            rhs.cubeMapInv_D.PeekConst(),
            this->cubeMapInv_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexStates_D.Validate(rhs.vertexStates_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexStates_D.Peek(),
            rhs.vertexStates_D.PeekConst(),
            this->vertexStates_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->activeVertexPos_D.Validate(rhs.activeVertexPos_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->activeVertexPos_D.Peek(),
            rhs.activeVertexPos_D.PeekConst(),
            this->activeVertexPos_D.GetCount()*sizeof(float3),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexIdxOffs_D.Validate(rhs.vertexIdxOffs_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexIdxOffs_D.Peek(),
            rhs.vertexIdxOffs_D.PeekConst(),
            this->vertexIdxOffs_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexMap_D.Validate(rhs.vertexMap_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexMap_D.Peek(),
            rhs.vertexMap_D.PeekConst(),
            this->vertexMap_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexMapInv_D.Validate(rhs.vertexMapInv_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexMapInv_D.Peek(),
            rhs.vertexMapInv_D.PeekConst(),
            this->vertexMapInv_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->vertexNeighbours_D.Validate(rhs.vertexNeighbours_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->vertexNeighbours_D.Peek(),
            rhs.vertexNeighbours_D.PeekConst(),
            this->vertexNeighbours_D.GetCount()*sizeof(int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->verticesPerTetrahedron_D.Validate(rhs.verticesPerTetrahedron_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->verticesPerTetrahedron_D.Peek(),
            rhs.verticesPerTetrahedron_D.PeekConst(),
            this->verticesPerTetrahedron_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

    CudaSafeCall(this->tetrahedronVertexOffsets_D.Validate(rhs.tetrahedronVertexOffsets_D.GetCount()));
    CudaSafeCall(cudaMemcpy(
            this->tetrahedronVertexOffsets_D.Peek(),
            rhs.tetrahedronVertexOffsets_D.PeekConst(),
            this->tetrahedronVertexOffsets_D.GetCount()*sizeof(unsigned int),
            cudaMemcpyDeviceToDevice));

//    CudaSafeCall(this->triangleCamDistance_D.Validate(rhs.triangleCamDistance_D.GetCount()));
//    CudaSafeCall(cudaMemcpy(
//            this->triangleCamDistance_D.Peek(),
//            rhs.triangleCamDistance_D.PeekConst(),
//            this->triangleCamDistance_D.GetCount()*sizeof(float),
//            cudaMemcpyDeviceToDevice));

    // The number of active cells
    this->activeCellCnt = rhs.activeCellCnt;

    /// Flag whether the neighbors have been computed
    this->neighboursReady = rhs.neighboursReady;

    return *this;

}


/*
 * GPUSurfaceMT::ComputeConnectivity
 */
bool GPUSurfaceMT::ComputeConnectivity(
        float *volume_D,
        int3 volDim,
        float3 volOrg,
        float3 volDelta,
        float isovalue) {

    CheckForCudaErrorSync();

    using namespace vislib::sys;
    /* Init grid parameters for all files */

    if (!CudaSafeCall(InitVolume(
            make_uint3(volDim.x, volDim.y, volDim.z),
            volOrg,
            volDelta))) {

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: could not init device constants",
                this->ClassName());
        return false;
    }

    CheckForCudaErrorSync();

    if (!CudaSafeCall(InitVolume_surface_generation(
            make_uint3(volDim.x, volDim.y, volDim.z),
            volOrg,
            volDelta))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: could not init device constants",
                this->ClassName());
        return false;
    }

    /* Compute neighbours */

    CheckForCudaErrorSync();

    if (!CudaSafeCall(vertexNeighbours_D.Validate(this->vertexCnt*18))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: could not allocate device memory",
                this->ClassName());
        return false;
    }
    CheckForCudaErrorSync();
    //if (!CudaSafeCall(vertexNeighbours_D.Set(-1))) {
    if (!CudaSafeCall(vertexNeighbours_D.Set(0xff))) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: could not init device memory",
                this->ClassName());
        return false;
    }
    CheckForCudaErrorSync();
    if (!CudaSafeCall(ComputeVertexConnectivity(
            this->vertexNeighbours_D.Peek(),
            this->vertexStates_D.Peek(),
            this->vertexMap_D.Peek(),
            this->vertexMapInv_D.Peek(),
            this->cubeMap_D.Peek(),
            this->cubeMapInv_D.Peek(),
            this->cubeStates_D.Peek(),
            this->vertexCnt,
            volume_D,
            isovalue))) {

//        // DEBUG Print neighbour indices
//        HostArr<int> vertexNeighbours;
//        vertexNeighbours.Validate(vertexNeighbours_D.GetCount());
//        vertexNeighbours_D.CopyToHost(vertexNeighbours.Peek());
//        for (int i = 0; i < vertexNeighbours_D.GetCount()/18; ++i) {
//            printf("Neighbours vtx #%i: ", i);
//            for (int j = 0; j < 18; ++j) {
//                printf("%i ", vertexNeighbours.Peek()[i*18+j]);
//            }
//            printf("\n");
//        }
//        // END DEBUG

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "%s: could not compute neighbors",
                this->ClassName());

        return false;
    }

    CheckForCudaErrorSync();

    this->neighboursReady = true;
    return true;
}


/*
 * GPUSurfaceMT::Release
 */
void GPUSurfaceMT::Release() {
    CudaSafeCall(this->cubeStates_D.Release());
    CudaSafeCall(this->cubeOffsets_D.Release());
    CudaSafeCall(this->cubeMap_D.Release());
    CudaSafeCall(this->cubeMapInv_D.Release());
    CudaSafeCall(this->vertexStates_D.Release());
    CudaSafeCall(this->activeVertexPos_D.Release());
    CudaSafeCall(this->vertexIdxOffs_D.Release());
    CudaSafeCall(this->vertexMap_D.Release());
    CudaSafeCall(this->vertexMapInv_D.Release());
    CudaSafeCall(this->vertexNeighbours_D.Release());
    CudaSafeCall(this->verticesPerTetrahedron_D.Release());
    CudaSafeCall(this->tetrahedronVertexOffsets_D.Release());
    CudaSafeCall(this->triangleCamDistance_D.Release());
}

/**
 * Returns a 1D grid definition based on the given threadsPerBlock value.
 *
 * @param size             The minimum number of threads
 * @param threadsPerBlock  The number of threads per block
 * @return The grid dimensions
 */
extern "C" dim3 GPUSurfaceMT::Grid(const unsigned int size, const int threadsPerBlock) {
    //TODO: remove hardcoded hardware capabilities :(
    // see: http://code.google.com/p/thrust/source/browse/thrust/detail/backend/cuda/arch.inl
    //   and http://code.google.com/p/thrust/source/browse/thrust/detail/backend/cuda/detail/safe_scan.inl
    //   for refactoring.
    // Get maximum grid size of CUDA device.
    //CUdevice device;
    //cuDeviceGet(&device, 0);
    //CUdevprop deviceProps;
    //cuDeviceGetProperties(&deviceProps, device);
    //this->gridSize = dim3(deviceProps.maxGridSize[0],
    //  deviceProps.maxGridSize[1],
    //  deviceProps.maxGridSize[2]);
    const dim3 maxGridSize(65535, 65535, 0);
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

    return grid;
}

#endif // WITH_CUDA
