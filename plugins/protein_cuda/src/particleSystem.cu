/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#ifdef _WIN32
#include <Windows.h>
#endif
#include <cstdlib>
#include <cstdio>
#include <string.h>

#include "vislib/graphics/gl/IncludeAllGL.h"
#define WGL_NV_gpu_affinity
#include <cuda_gl_interop.h>

#include "thrust/device_ptr.h"
#include "thrust/sort.h"
#include "thrust/scan.h"

#include "helper_cuda.h"

#include "particles_kernel.cu"

extern "C"
{

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles) {
    thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                        thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                        thrust::device_ptr<uint>(dGridParticleIndex));
}

void scanParticles(uint *dInput, uint *dOutput, uint count) {
    thrust::exclusive_scan(thrust::device_ptr<uint>(dInput), 
                           thrust::device_ptr<uint>(dInput + count),
                           thrust::device_ptr<uint>(dOutput));
}

void cudaInit(int argc, char **argv) {
    // use CUDA device with highest Gflops/s
    cudaSetDevice( gpuGetMaxGflopsDeviceId() );
}

void cudaGLInit(int argc, char **argv) {
    // use CUDA device with highest Gflops/s
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
}

void allocateArray(void **devPtr, size_t size) {
    checkCudaErrors(cudaMalloc(devPtr, size));
    //cudaMalloc(devPtr, size);
    //cudaError e;
    //e = cudaGetLastError();
    //int a = 0;
}

void freeArray(void *devPtr) {
	checkCudaErrors(cudaFree(devPtr));
}

void threadSync() {
	checkCudaErrors(cudaThreadSynchronize());
}

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size) {   
    if (vbo)
		checkCudaErrors(cudaGLMapBufferObject((void**)&device, vbo));

	checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    
    if (vbo)
		checkCudaErrors(cudaGLUnmapBufferObject(vbo));
}

void copyArrayToDevice(void* device, const void* host, int offset, int size) {
	checkCudaErrors(cudaMemcpy((char *)device + offset, host, size, cudaMemcpyHostToDevice));
}

void registerGLBufferObject(uint vbo) {
	checkCudaErrors(cudaGLRegisterBufferObject(vbo));
}

void unregisterGLBufferObject(uint vbo) {
	checkCudaErrors(cudaGLUnregisterBufferObject(vbo));
}

void *mapGLBufferObject(uint vbo) {
    void *ptr;
	checkCudaErrors(cudaGLMapBufferObject(&ptr, vbo));
    return ptr;
}

void unmapGLBufferObject(uint vbo) {
	checkCudaErrors(cudaGLUnmapBufferObject(vbo));
}

void setParameters(SimParams *hostParams) {
    // copy parameters to constant memory
	checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
}

void setRSParameters(RSParams *hostParams) {
    // copy parameters to constant memory
	checkCudaErrors(cudaMemcpyToSymbol(rsParams, hostParams, sizeof(RSParams)));
}

//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads) {
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}

void calcHash(uint*  gridParticleHash,
              uint*  gridParticleIndex,
              float* pos, 
              int    numParticles) {
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    // execute the kernel
    calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                           gridParticleIndex,
                                           (float4 *) pos,
                                           numParticles);
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
}

void reorderDataAndFindCellStart(uint*  cellStart,
                                 uint*  cellEnd,
                                 float* sortedPos,
                                 uint*  gridParticleHash,
                                 uint*  gridParticleIndex,
                                 float* oldPos,
                                 uint   numParticles,
                                 uint   numCells) {
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    // set all cells to empty
	checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

    //checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));

    uint smemSize = sizeof(uint)*(numThreads+1);
    reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
        cellStart,
        cellEnd,
        (float4 *) sortedPos,
        gridParticleHash,
        gridParticleIndex,
        (float4 *) oldPos,
        numParticles);
    getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

    //checkCudaErrors(cudaUnbindTexture(oldPosTex));
}

void countNeighbors( uint*  neighborCount,
                     uint*  neighbors,
                     float* smallCircles,
                     float* sortedPos,
                     uint*  gridParticleIndex,
                     uint*  cellStart,
                     uint*  cellEnd,
                     uint   numAtoms,
                     uint   numNeighbors,
                     uint   numCells) {
	checkCudaErrors(cudaBindTexture(0, atomPosTex, sortedPos, numAtoms*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, neighborCountTex, neighborCount, numAtoms*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, neighborsTex, neighbors, numAtoms*numNeighbors*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, smallCirclesTex, smallCircles, numAtoms*numNeighbors*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));

    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize( numAtoms, 64, numBlocks, numThreads);

    // execute the kernel
    countNeighbors<<< numBlocks, numThreads >>>( neighborCount,
                                                 neighbors,
                                                 (float4*)smallCircles,
                                                 (float4*)sortedPos,
                                                 gridParticleIndex,
                                                 cellStart,
                                                 cellEnd,
                                                 numAtoms);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

	checkCudaErrors(cudaUnbindTexture(atomPosTex));
	checkCudaErrors(cudaUnbindTexture(neighborCountTex));
	checkCudaErrors(cudaUnbindTexture(neighborsTex));
	checkCudaErrors(cudaUnbindTexture(smallCirclesTex));
	checkCudaErrors(cudaUnbindTexture(cellStartTex));
	checkCudaErrors(cudaUnbindTexture(cellEndTex));
}

void countNeighbors2( uint*  neighborCount,
                     uint*  neighbors,
                     float* sortedPos,
                     uint*  gridParticleIndex,
                     uint*  cellStart,
                     uint*  cellEnd,
                     uint   numAtoms,
                     uint   numNeighbors,
                     uint   numCells) {
	checkCudaErrors(cudaBindTexture(0, atomPosTex, sortedPos, numAtoms*sizeof(float4)));
    //checkCudaErrors( cudaBindTexture( 0, neighborCountTex, neighborCount, numAtoms*sizeof(uint)));
    //checkCudaErrors( cudaBindTexture( 0, neighborsTex, neighbors, numAtoms*numNeighbors*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));

    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize( numAtoms, 64, numBlocks, numThreads);

    // execute the kernel
    countNeighbors<<< numBlocks, numThreads >>>( neighborCount,
                                                 neighbors,
                                                 (float4*)sortedPos,
                                                 gridParticleIndex,
                                                 cellStart,
                                                 cellEnd,
                                                 numAtoms);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

	checkCudaErrors(cudaUnbindTexture(atomPosTex));
    //checkCudaErrors( cudaUnbindTexture( neighborCountTex));
    //checkCudaErrors( cudaUnbindTexture( neighborsTex));
	checkCudaErrors(cudaUnbindTexture(cellStartTex));
	checkCudaErrors(cudaUnbindTexture(cellEndTex));
}

void countProbeNeighbors( //uint*  probeNeighborCount,
                     float3* probeNeighborCount,
                     //uint*  probeNeighbors,
                     float3* probeNeighbors,
                     float* sortedProbePos,
                     uint*  gridParticleIndex,
                     uint*  cellStart,
                     uint*  cellEnd,
                     uint   numProbes,
                     uint   numNeighbors,
                     uint   numCells) {
    // bind textures
    // TODO!!
	checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));

    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize( numProbes, 64, numBlocks, numThreads);

    // execute the kernel
    countProbeNeighbors<<< numBlocks, numThreads >>>( probeNeighborCount,
        probeNeighbors, (float4*)sortedProbePos, gridParticleIndex,
        cellStart, cellEnd, numProbes);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution (countProbeNeighbors) failed");

    // unbind textures
    // TODO!!
	checkCudaErrors(cudaUnbindTexture(cellStartTex));
	checkCudaErrors(cudaUnbindTexture(cellEndTex));
}

void computeArcsCUDA( float*  arcs,
                      uint*   neighborCount,
                      uint*   neighbors,
                      float*  smallCircles,
                      float*  sortedPos,
                      uint*   gridParticleIndex,
                      uint    numAtoms,
                      uint    numNeighbors) {
    //checkCudaErrors( cudaBindTexture( 0, arcsTex, arcs, numAtoms*numNeighbors*4*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, atomPosTex, sortedPos, numAtoms*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, neighborCountTex, neighborCount, numAtoms*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, neighborsTex, neighbors, numAtoms*numNeighbors*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, smallCirclesTex, smallCircles, numAtoms*numNeighbors*sizeof(float4)));

    // execute the kernel
    computeArcs<<< numAtoms, numNeighbors >>>( (float4*)arcs,
                                      neighborCount,
                                      neighbors,
                                      (float4*)smallCircles,
                                      (float4*)sortedPos,
                                      gridParticleIndex,
                                      numAtoms);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

    //checkCudaErrors( cudaUnbindTexture( arcsTex));
	checkCudaErrors(cudaUnbindTexture(atomPosTex));
	checkCudaErrors(cudaUnbindTexture(neighborCountTex));
	checkCudaErrors(cudaUnbindTexture(neighborsTex));
	checkCudaErrors(cudaUnbindTexture(smallCirclesTex));
}

void computeReducedSurfaceCuda( uint* point1, 
        //float* point2, float* point3, 
        float* probePos, uint* neighborCount, uint* neighbors, float* atomPos,
        uint* gridParticleIndex, float* visibleAtoms, uint* visibleAtomsId,
        uint numAtoms, uint numVisibleAtoms, uint numNeighbors) {
    // texture bindings
	checkCudaErrors(cudaBindTexture(0, neighborCountTex, neighborCount, numAtoms*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, neighborsTex, neighbors, numAtoms*numNeighbors*sizeof(uint)));
	checkCudaErrors(cudaBindTexture(0, atomPosTex, atomPos, numAtoms*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, visibleAtomsTex, visibleAtoms, numAtoms*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, visibleAtomsIdTex, visibleAtomsId, numAtoms*sizeof(uint)));

    dim3 numThreads;
    numThreads.x = 8;
    numThreads.y = 8;
    numThreads.z = 1;
    dim3 numBlocks;
    numBlocks.x = ( numNeighbors * numNeighbors) / numThreads.x  
        + ( ( numNeighbors * numNeighbors) % numThreads.x == 0 ? 0:1);
    numBlocks.y = numVisibleAtoms / numThreads.y  
        + ( numVisibleAtoms % numThreads.y == 0 ? 0:1);
    numBlocks.z = 1;
    // execute the kernel
    computeReducedSurface<<< numBlocks, numThreads >>>( 
        //(float4*)point1, (float4*)point2, (float4*)point3, (float4*)probePos,
        (uint4*)point1, (float4*)probePos,
        neighborCount, neighbors, (float4*)atomPos, gridParticleIndex,
        (float4*)visibleAtoms, visibleAtomsId);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution (computeReducedSurface) failed");

	checkCudaErrors(cudaUnbindTexture(neighborCountTex));
	checkCudaErrors(cudaUnbindTexture(neighborsTex));
	checkCudaErrors(cudaUnbindTexture(atomPosTex));
	checkCudaErrors(cudaUnbindTexture(visibleAtomsTex));
	checkCudaErrors(cudaUnbindTexture(visibleAtomsIdTex));
}

void computeTriangleVBOCuda( float3* vbo, uint4* point1, 
        //float* point2, float* point3,
        float* atomPos, float* visibleAtoms,
        uint numAtoms, uint numVisibleAtoms, uint numNeighbors, uint offset) {
    // texture bindings
    // TODO!
	checkCudaErrors(cudaBindTexture(0, atomPosTex, atomPos, numAtoms*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, visibleAtomsTex, visibleAtoms, numAtoms*sizeof(float4)));
    //checkCudaErrors( cudaBindTexture( 0, point1Tex, point1, numAtoms*numNeighbors*numNeighbors*sizeof(uint4)));

    dim3 numThreads;
    numThreads.x = 8;
    numThreads.y = 8;
    numThreads.z = 1;
    dim3 numBlocks;
    numBlocks.x = ( numNeighbors * numNeighbors) / numThreads.x  
        + ( ( numNeighbors * numNeighbors) % numThreads.x == 0 ? 0:1);
    numBlocks.y = numVisibleAtoms / numThreads.y 
        + ( ( numVisibleAtoms % numThreads.y ) == 0 ? 0:1);
    numBlocks.z = 1;
    // execute the kernel
    computeTriangleVBO<<< numBlocks, numThreads >>>( 
        //vbo, (float4*)point1, (float4*)point2, (float4*)point3);
        vbo, (uint4*)point1, (float4*)atomPos, (float4*)visibleAtoms, offset);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution (computeTriangleVBO) failed");

    // unbind textures
    // TODO!
	checkCudaErrors(cudaUnbindTexture(atomPosTex));
	checkCudaErrors(cudaUnbindTexture(visibleAtomsTex));
    //checkCudaErrors( cudaUnbindTexture( point1Tex));
}

void computeVisibleTriangleVBOCuda( float3* vbo, uint4* point1, cudaArray* visibility,
        //float* point2, float* point3,
        float* atomPos, float* visibleAtoms,
        uint numAtoms, uint numVisibleAtoms, uint numNeighbors, uint offset) {
    // texture bindings
    // TODO!
	checkCudaErrors(cudaBindTexture(0, atomPosTex, atomPos, numAtoms*sizeof(float4)));
	checkCudaErrors(cudaBindTexture(0, visibleAtomsTex, visibleAtoms, numAtoms*sizeof(float4)));
    //checkCudaErrors( cudaBindTexture( 0, point1Tex, point1, numAtoms*numNeighbors*numNeighbors*sizeof(uint4)));
	checkCudaErrors(cudaBindTextureToArray(inVisibilityTex, visibility));

    struct cudaChannelFormatDesc desc; 
    checkCudaErrors(cudaGetChannelDesc(&desc, visibility));
    
    dim3 numThreads;
    numThreads.x = 8;
    numThreads.y = 8;
    numThreads.z = 1;
    dim3 numBlocks;
    numBlocks.x = ( numNeighbors * numNeighbors) / numThreads.x  
        + ( ( numNeighbors * numNeighbors) % numThreads.x == 0 ? 0:1);
    numBlocks.y = numVisibleAtoms / numThreads.y 
        + ( ( numVisibleAtoms % numThreads.y ) == 0 ? 0:1);
    numBlocks.z = 1;
    // execute the kernel
    computeVisibleTriangleVBO<<< numBlocks, numThreads >>>( 
        //vbo, (float4*)point1, (float4*)point2, (float4*)point3);
        vbo, (uint4*)point1, (float4*)atomPos, (float4*)visibleAtoms, offset);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution (computeVisibleTriangleVBO) failed");

    // unbind textures
    // TODO!
    checkCudaErrors( cudaUnbindTexture( atomPosTex));
    checkCudaErrors( cudaUnbindTexture( visibleAtomsTex));
    //checkCudaErrors( cudaUnbindTexture( point1Tex));
    checkCudaErrors( cudaUnbindTexture( inVisibilityTex));
}

void computeSESPrimiticesVBOCuda(
        float4* outTorusVBO,    // the output VBO (positions + attributes for torus drawing)
        float4* outSTriaVBO,    // the output VBO (positions + attributes for spherical triangle drawing)
        float4* inVBO,          // the input VBO (indices of visible triangles)
        float* atomPos,         // the sorted atom positions (for neighboring atoms)
        float* visibleAtoms,    // the visible atoms' positions
        uint4* point1,          // the atom index array
        float* probePos,        // the probe position array
        uint numAtoms,          // the total number of atoms
        uint numVisibleAtoms,   // the number of visible atoms
        uint numNeighbors,      // the maximum number of neighbors per atom
        uint numVisibleTria) {  // the number of visible triangles
    // check number of visible triangles
    if( numVisibleTria == 0 ) return;
    // texture bindings
    checkCudaErrors( cudaBindTexture( 0, atomPosTex, atomPos, numAtoms*sizeof(float4)));
    checkCudaErrors( cudaBindTexture( 0, visibleAtomsTex, visibleAtoms, numAtoms*sizeof(float4)));

    uint numThreads, numBlocks;
    numThreads = 8;
    numBlocks = numVisibleTria / numThreads
        + ( numVisibleTria % numThreads == 0 ? 0:1);
    // execute the kernel
    computeTorusVBO<<< numBlocks, numThreads >>>( outTorusVBO, outSTriaVBO, inVBO, 
        (float4*)atomPos, (float4*)visibleAtoms, (uint4*)point1, (float4*)probePos );

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution (computeSESPrimiticesVBO) failed");
    
    // unbind textures
    checkCudaErrors( cudaUnbindTexture( atomPosTex));
    checkCudaErrors( cudaUnbindTexture( visibleAtomsTex));
}

void writeProbePositionsCuda(
    float* probePos,    // output (probe positions)
    float4* sTriaVbo,   // the VBO containing the probe positions
    uint numProbes ) {  // the number of probes
    // check number of probes
    if( numProbes == 0 ) return;

    uint numThreads, numBlocks;
    numThreads = 8;
    numBlocks = numProbes / numThreads
        + ( numProbes % numThreads == 0 ? 0:1);
    // execute the kernel
    writeProbePositions<<< numBlocks, numThreads >>>( (float4*)probePos, sTriaVbo, numProbes );

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution (writeProbePositions) failed");
}

void writeSingularitiesCuda(
        float3* outArray,
        uint*  probeNeighbors,
        float* probePos,
        uint numProbes,
        uint numNeighbors ) {
    // check the number of probes
    if( numProbes == 0 ) return;
    
    dim3 numThreads;
    numThreads.x = 8;
    numThreads.y = 8;
    numThreads.z = 1;
    dim3 numBlocks;
    numBlocks.x = ( numNeighbors) / numThreads.x  
        + ( numNeighbors % numThreads.x == 0 ? 0:1);
    numBlocks.y = numProbes / numThreads.y 
        + ( ( numProbes % numThreads.y ) == 0 ? 0:1);
    numBlocks.z = 1;
    // execute the kernel
    writeSingularities<<< numBlocks, numThreads >>>( 
        outArray, probeNeighbors, (float4*)probePos);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution (writeSingularities) failed");
}

void findAdjacentTrianglesCuda(
        float* outPbo,
        cudaArray* visibility,
        uint* point1,
        float* probePos, 
        uint* neighborCount, 
        uint* neighbors, 
        float* atomPos,
        float* visibleAtoms, 
        uint* visibleAtomsId,
        uint numAtoms, 
        uint numVisibleAtoms, 
        uint numNeighbors ) {
    // check the number of visible atoms
    if( numVisibleAtoms == 0 ) return;
    // texture bindings
    checkCudaErrors( cudaBindTexture( 0, neighborCountTex, neighborCount, numAtoms*sizeof(uint)));
    checkCudaErrors( cudaBindTexture( 0, neighborsTex, neighbors, numAtoms*numNeighbors*sizeof(uint)));
    checkCudaErrors( cudaBindTexture( 0, atomPosTex, atomPos, numAtoms*sizeof(float4)));
    checkCudaErrors( cudaBindTexture( 0, visibleAtomsTex, visibleAtoms, numAtoms*sizeof(float4)));
    checkCudaErrors( cudaBindTexture( 0, visibleAtomsIdTex, visibleAtomsId, numAtoms*sizeof(uint)));
    checkCudaErrors( cudaBindTextureToArray( inVisibilityTex, visibility));

    struct cudaChannelFormatDesc desc; 
    checkCudaErrors(cudaGetChannelDesc(&desc, visibility));

    dim3 numThreads;
    numThreads.x = 8;
    numThreads.y = 8;
    numThreads.z = 1;
    dim3 numBlocks;
    numBlocks.x = numNeighbors / numThreads.x  
        + ( numNeighbors % numThreads.x == 0 ? 0:1);
    numBlocks.y = numVisibleAtoms / numThreads.y  
        + ( numVisibleAtoms % numThreads.y == 0 ? 0:1);
    numBlocks.z = 1;
    // execute the kernel
    findAdjacentTriangles<<< numBlocks, numThreads >>>( 
        outPbo, (uint4*)point1, (float4*)probePos, neighborCount, neighbors, 
        (float4*)atomPos, (float4*)visibleAtoms, visibleAtomsId);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution (findAdjacentTriangles) failed");

    checkCudaErrors( cudaUnbindTexture( neighborCountTex));
    checkCudaErrors( cudaUnbindTexture( neighborsTex));
    checkCudaErrors( cudaUnbindTexture( atomPosTex));
    checkCudaErrors( cudaUnbindTexture( visibleAtomsTex));
    checkCudaErrors( cudaUnbindTexture( visibleAtomsIdTex));
    checkCudaErrors( cudaUnbindTexture( inVisibilityTex));
}

void findNeighborsCB(
        uint*   neighborCount,
        uint*   neighbors,
        float*  smallCircles,
        float*  sortedPos,
        uint*   cellStart,
        uint*   cellEnd,
        uint    numAtoms,
        uint    numNeighbors,
        uint    numCells) {
    checkCudaErrors( cudaBindTexture( 0, atomPosTex, sortedPos, numAtoms*sizeof(float4)));
    checkCudaErrors( cudaBindTexture( 0, neighborCountTex, neighborCount, numAtoms*sizeof(uint)));
    checkCudaErrors( cudaBindTexture( 0, neighborsTex, neighbors, numAtoms*numNeighbors*sizeof(uint)));
    checkCudaErrors( cudaBindTexture( 0, smallCirclesTex, smallCircles, numAtoms*numNeighbors*sizeof(float4)));
    checkCudaErrors( cudaBindTexture( 0, cellStartTex, cellStart, numCells*sizeof(uint)));
    checkCudaErrors( cudaBindTexture( 0, cellEndTex, cellEnd, numCells*sizeof(uint)));

    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize( numAtoms, 64, numBlocks, numThreads);

    // execute the kernel
    findNeighborsCBCuda<<< numBlocks, numThreads >>>( neighborCount, neighbors, 
        (float4*)smallCircles, (float4*)sortedPos, cellStart, cellEnd, numAtoms);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors( cudaUnbindTexture( atomPosTex));
    checkCudaErrors( cudaUnbindTexture( neighborCountTex));
    checkCudaErrors( cudaUnbindTexture( neighborsTex));
    checkCudaErrors( cudaUnbindTexture( smallCirclesTex));
    checkCudaErrors( cudaUnbindTexture( cellStartTex));
    checkCudaErrors( cudaUnbindTexture( cellEndTex));
}

void removeCoveredSmallCirclesCB(
        float* smallCircles,
        uint*  smallCircleVisible,
        uint*  neighborCount,
        uint*  neighbors,
        float* sortedPos,
        uint   numAtoms,
        uint   numNeighbors) {
    checkCudaErrors( cudaBindTexture( 0, atomPosTex, sortedPos, numAtoms*sizeof(float4)));
    checkCudaErrors( cudaBindTexture( 0, neighborsTex, neighbors, numAtoms*numNeighbors*sizeof(uint)));

    // one thread per particle neighbor
    dim3 numThreads;
    numThreads.x = 64;
    numThreads.y = 4;
    numThreads.z = 1;
    dim3 numBlocks;
    numBlocks.x = numNeighbors / numThreads.x + ( numNeighbors % numThreads.x == 0 ? 0:1);
    numBlocks.y = numAtoms / numThreads.y + ( numAtoms % numThreads.y == 0 ? 0:1);
    numBlocks.z = 1;

    // execute the kernel
    removeCoveredSmallCirclesCBCuda<<< numBlocks, numThreads >>>(
        (float4*)smallCircles, smallCircleVisible, neighborCount, neighbors, (float4*)sortedPos, numAtoms);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors( cudaUnbindTexture( atomPosTex));
    checkCudaErrors( cudaUnbindTexture( neighborsTex));
}

void computeArcsCB(
        float* smallCircles,
        uint*  smallCircleVisible,
        uint*  neighborCount,
        uint*  neighbors,
        float* sortedPos,
        float* arcs,
        uint*  arcCount,
        uint   numAtoms,
        uint   numNeighbors) {
    checkCudaErrors( cudaBindTexture( 0, atomPosTex, sortedPos, numAtoms*sizeof(float4)));
    checkCudaErrors( cudaBindTexture( 0, neighborCountTex, neighborCount, numAtoms*sizeof(uint)));
    checkCudaErrors( cudaBindTexture( 0, neighborsTex, neighbors, numAtoms*numNeighbors*sizeof(uint)));
    checkCudaErrors( cudaBindTexture( 0, smallCirclesTex, smallCircles, numAtoms*numNeighbors*sizeof(float4)));

    // one thread per particle neighbor
    dim3 numThreads;
    numThreads.x = 64;
    numThreads.y = 4;
    numThreads.z = 1;
    dim3 numBlocks;
    numBlocks.x = numNeighbors / numThreads.x + ( numNeighbors % numThreads.x == 0 ? 0:1);
    numBlocks.y = numAtoms / numThreads.y + ( numAtoms % numThreads.y == 0 ? 0:1);
    numBlocks.z = 1;

    // execute the kernel
    computeArcsCBCuda<<< numBlocks, numThreads >>>(
        smallCircleVisible, (float4*)arcs, arcCount, numAtoms);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors( cudaUnbindTexture( atomPosTex));
    checkCudaErrors( cudaUnbindTexture( neighborCountTex));
    checkCudaErrors( cudaUnbindTexture( neighborsTex));
    checkCudaErrors( cudaUnbindTexture( smallCirclesTex));
}

void writeProbePositionsCB(
        float*	probePos,
        float*	sphereTriaVec1,
        float*	sphereTriaVec2,
        float*	sphereTriaVec3,
        float*	torusPos,
        float*	torusVS,
        float*	torusAxis,
        uint*   neighborCount,
        uint*   neighbors,
        float*  sortedAtomPos,
        float*  arcs,
        uint*	arcCount,
        uint*	arcCountScan,
        uint*	scCount,
        uint*	scCountScan,
        float*	smallCircles,
        uint    numAtoms,
        uint    numNeighbors) {
    checkCudaErrors( cudaBindTexture( 0, atomPosTex, sortedAtomPos, numAtoms*sizeof(float4)));
    checkCudaErrors( cudaBindTexture( 0, neighborCountTex, neighborCount, numAtoms*sizeof(uint)));
    checkCudaErrors( cudaBindTexture( 0, neighborsTex, neighbors, numAtoms*numNeighbors*sizeof(uint)));
    checkCudaErrors( cudaBindTexture( 0, smallCirclesTex, smallCircles, numAtoms*numNeighbors*sizeof(float4)));

    // one thread per particle neighbor
    dim3 numThreads;
    numThreads.x = 64;
    numThreads.y = 4;
    numThreads.z = 1;
    dim3 numBlocks;
    numBlocks.x = numNeighbors / numThreads.x + ( numNeighbors % numThreads.x == 0 ? 0:1);
    numBlocks.y = numAtoms / numThreads.y + ( numAtoms % numThreads.y == 0 ? 0:1);
    numBlocks.z = 1;

    // execute the kernel
    writeProbePositionsCBCuda<<< numBlocks, numThreads >>>( (float4*)probePos, 
        (float4*)sphereTriaVec1, (float4*)sphereTriaVec2, (float4*)sphereTriaVec3,
        (float4*)torusPos, (float4*)torusVS, (float4*)torusAxis,
        (float4*)arcs, arcCount, arcCountScan, scCount, scCountScan, numAtoms);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors( cudaUnbindTexture( atomPosTex));
    checkCudaErrors( cudaUnbindTexture( neighborCountTex));
    checkCudaErrors( cudaUnbindTexture( neighborsTex));
    checkCudaErrors( cudaUnbindTexture( smallCirclesTex));
}

void writeSingularityTextureCB(
        float*  texCoord,
        float*  singTex,
        float*  sortedProbePos,
        uint*   gridProbeIndex,
        uint*   cellStart,
        uint*   cellEnd,
        uint    numProbes,
        uint    numNeighbors,
        uint    numCells) {
    checkCudaErrors( cudaBindTexture( 0, cellStartTex, cellStart, numCells*sizeof(uint)));
    checkCudaErrors( cudaBindTexture( 0, cellEndTex, cellEnd, numCells*sizeof(uint)));

    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize( numProbes, 64, numBlocks, numThreads);

    //// execute the kernel
    findProbeNeighborsCBCuda<<< numBlocks, numThreads >>>( (float3*)texCoord, (float3*)singTex, 
        (float4*)sortedProbePos, gridProbeIndex, cellStart, cellEnd, numProbes, numNeighbors);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors( cudaUnbindTexture( cellStartTex));
    checkCudaErrors( cudaUnbindTexture( cellEndTex));
}


}   // extern "C"
