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

extern "C"
{
void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

void scanParticles(uint *dInput, uint *dOutput, uint count);

void cudaInit(int argc, char **argv);

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void registerGLBufferObject(unsigned int vbo);
void unregisterGLBufferObject(unsigned int vbo);
void *mapGLBufferObject(uint vbo);
void unmapGLBufferObject(uint vbo);

void setParameters(SimParams *hostParams);

void setRSParameters(RSParams *hostParams);

void calcHash(uint*  gridParticleHash,
              uint*  gridParticleIndex,
              float* pos, 
              int    numParticles);

void reorderDataAndFindCellStart(uint*  cellStart,
                                 uint*  cellEnd,
                                 float* sortedPos,
                                 uint*  gridParticleHash,
                                 uint*  gridParticleIndex,
                                 float* oldPos,
                                 uint   numParticles,
                                 uint   numCells);

void countNeighbors( uint*  neighborCount,
                     uint*  neighbors,
                     float* smallCircles,
                     float* sortedPos,
                     uint*  gridParticleIndex,
                     uint*  cellStart,
                     uint*  cellEnd,
                     uint   numAtoms,
                     uint   numNeighbors,
                     uint   numCells);

void countNeighbors2( uint*  neighborCount,
                     uint*  neighbors,
                     float* sortedPos,
                     uint*  gridParticleIndex,
                     uint*  cellStart,
                     uint*  cellEnd,
                     uint   numAtoms,
                     uint   numNeighbors,
                     uint   numCells);

void countProbeNeighbors( //uint*  probeNeighborCount,
                     float3* probeNeighborCount,
                     //uint*  probeNeighbors,
                     float3*  probeNeighbors,
                     float* sortedProbePos,
                     uint*  gridParticleIndex,
                     uint*  cellStart,
                     uint*  cellEnd,
                     uint   numProbes,
                     uint   numNeighbors,
                     uint   numCells);

void computeArcsCUDA( float*  arcs,
                      uint*   neighborCount,
                      uint*   neighbors,
                      float*  smallCircles,
                      float*  sortedPos,
                      uint*   gridParticleIndex,
                      uint    numAtoms,
                      uint    numNeighbors);

void computeReducedSurfaceCuda(
    uint* point1, 
    //float* point2, 
    //float* point3, 
    float* probePos, 
    uint* neighborCount, 
    uint* neighbors, 
    float* atomPos,
    uint* gridParticleIndex, 
    float* visibleAtoms, 
    uint* visibleAtomsId,
    uint numAtoms, 
    uint numVisibleAtoms, 
    uint numNeighbors);

void computeTriangleVBOCuda( 
    float3* vbo, 
    uint* point1, 
    //float* point2, 
    //float* point3,
    float* atomPos,
    float* visibleAtoms, 
    uint numAtoms, 
    uint numVisibleAtoms, 
    uint numNeighbors,
    uint offset);

void computeVisibleTriangleVBOCuda( 
    float3* vbo, 
    uint* point1,
    cudaArray* visiblity,
    float* atomPos,
    float* visibleAtoms, 
    uint numAtoms, 
    uint numVisibleAtoms, 
    uint numNeighbors,
    uint offset);

void computeSESPrimiticesVBOCuda(
    float4* outTorusVBO,
    float4* outSTriaVBO,
    float4* inVBO,
    float* atomPos,
    float* visibleAtoms,
    uint* point1,
    float* probePos,
    uint numAtoms,
    uint numVisibleAtoms,
    uint numNeighbors,
    uint numVisibleTria);

void writeProbePositionsCuda(
    float* probePos,
    float4* sTriaVbo,
    uint numProbes);

void writeSingularitiesCuda(
    float3* outArray,
    uint*  probeNeighbors,
    float* probePos,
    uint numProbes,
    uint numNeighbors );

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
    uint numNeighbors);

void findNeighborsCB(
    uint*  neighborCount,
    uint*  neighbors,
    float* smallCircles,
    float* sortedPos,
    uint*  cellStart,
    uint*  cellEnd,
    uint   numAtoms,
    uint   numNeighbors,
    uint   numCells);

void removeCoveredSmallCirclesCB(
    float* smallCircles,
    uint*  smallCircleVisible,
    uint*  neighborCount,
    uint*  neighbors,
    float* sortedPos,
    uint   numAtoms,
    uint   numNeighbors);

void computeArcsCB(
    float* smallCircles,
    uint*  smallCircleVisible,
    uint*  neighborCount,
    uint*  neighbors,
    float* sortedPos,
    float* arcs,
    uint*  arcCount,
    uint   numAtoms,
    uint   numNeighbors);

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
    uint    numNeighbors);

void writeSingularityTextureCB(
    float*  texCoord,
    float*  singTex,
    float*  sortedProbePos,
    uint*   gridProbeIndex,
    uint*   cellStart,
    uint*   cellEnd,
    uint    numProbes,
    uint    numNeighbors,
    uint    numCells);

}
