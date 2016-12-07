//
// ComparativeSurfacePotentialRenderer.cuh
//
// Contains CUDA functionality used by the 'ComparativeSurfacePotentialRenderer'
// and the 'ProteinVariantMatch' class.
//
// Note: The source code is distributed amongst several files to prevent very
// long compilation times during development.
// The source files are
// 'ComparativeSurfacePotentialRenderer.cu'
// 'ComparativeSurfacePotentialRenderer_thrust_sort_code.cu'
// 'ComparativeSurfacePotentialRenderer_surface_mapping.cu'
// 'ComparativeSurfacePotentialRenderer_surface_generation.cu'
// 'ComparativeSurfacePotentialRenderer_metric.cu'
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 13, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_POTENTIALVOLUMERENDERERCUDA_CUH_INCLUDED
#define MMPROTEINCUDAPLUGIN_POTENTIALVOLUMERENDERERCUDA_CUH_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

typedef unsigned int uint;

// The number of possible neighbours for every vertex when using the
// Freudenthal subdivision scheme
#define MT_FREUDENTHAL_SUBDIV_MAXNEIGHBOURS 18

// The number of iterations during surface mapping performed in one kernel call
#define UPDATE_VTX_POS_ITERATIONS_PER_KERNEL 1

extern "C" {

cudaError_t InitVolume(uint3 gridSize, float3 org, float3 delta);

cudaError_t InitVolume_surface_generation(uint3 gridSize, float3 org,
        float3 delta);

cudaError_t InitVolume_metric(uint3 gridSize, float3 org,
        float3 delta);


// Wrapper functions for Marching Tetrahedra with Freudenthal subdivision and
// other surface initialization stuff (normals, tex coords etc)

cudaError_t CalcCubeMap(uint *cubeMap_D, uint *cubeMapInv_D, uint *cubeOffs_D,
        uint *cubeStates_D, uint cubeCount);

cudaError_t CalcVertexMap(uint *vertexMap_D, uint *vertexMapInv_D,
        uint *vertexIdxOffs_D, uint *vertexStates_D, uint activeCellsCount);

cudaError_t CalcVertexPositions(uint *vertexStates_D, float3 *activeVertexPos_D,
        uint *vertexIdxOffs_D, uint *cubeMap_D, uint activeCubeCount,
        float isoval, float *volume_D);

cudaError_t ComputeVertexConnectivity(int *vertexNeighbours_D,
        uint *vertexStates_D, uint *vertexMap_D, uint *vertexMapInv_D,
        uint *cubeMap_D, uint *cubeMapInv_D, uint *cubeStates_D,
        uint activeVertexCnt, float *volume_D, float isoval);

cudaError_t ComputeVertexNormals(float *dataBuffer_D, uint *vertexMap_D,
        uint *vertexMapInv_D, uint *cubeMap_D, uint *cubeMapInv_D,
        float *volume_D, float isoval, uint activeVertexCnt,
        uint arrDataOffsPos, uint arrDataOffsNormals, uint arrDataSize);

cudaError_t ComputeVertexTexCoords(float *dataBuff_D, float volMinX,
        float volMinY, float volMinZ, float volMaxX, float volMaxY,
        float volMaxZ, uint activeVertexCnt, uint arrDataOffsPos,
        uint arrDataOffsTexCoords, uint arrDataSize);

cudaError_t CompactActiveVertexPositions(float *vertexPos_D,
        uint *vertexStates_D, uint *vertexIdxOffs_D, float3 *activeVertexPos_D,
        uint activeCellCount, uint outputArrOffs, uint outputArrDataSize);

cudaError_t FindActiveGridCells(uint *cubeStates_D, uint *cubeOffs_D,
        uint cubeCount, float isoval, float *volume_D);

cudaError_t FlagTetrahedrons(uint *verticesPerTetrahedron_D, uint *cubeMap_D,
        float isoval, uint activeCellCount, float *volume_D);

cudaError_t GetTrianglesIdx(uint *tetrahedronVertexOffsets_D, uint *cubeMap_D,
        uint *cubeMapInv_D, float isoval, uint tetrahedronCount,
        uint activeCellCount, uint *triangleVertexIdx_D, uint *vertexMapInv_D,
        float *volume_D);

cudaError_t GetTetrahedronVertexOffsets(uint *tetrahedronVertexOffsets_D,
        uint *verticesPerTetrahedron_D, uint tetrahedronCount);


// Wrapper functions for surface mapping (uses deformable models)

cudaError_t CalcVolGradient(float4 *grad_D, float *field_D, uint gridSize);

cudaError_t CalcVolGradientWithDistField(float4 *grad_D, float *field_D,
        float *distField_D, float minDist, float isovalue, uint gridSize);

cudaError_t ComputeDistField(float *vertexPos_D, float *distField_D,
        uint volSize, uint vertexCnt,
        uint dataArrOffs, uint dataArrSize);

cudaError_t ComputeHausdorffDistance(
        float *vertexPos_D,
        float *vertexPosOld_D,
        float *hausdorffDist_D,
        uint vertexCnt,
        uint vertexCntOld,
        uint dataArrOffsPos,
        uint dataArrSize);

cudaError_t FlagVerticesInCorruptTriangles(float *vertexData_D, uint vertexDataStride,
        uint vertexDataOffsPos, uint vertexDataOffsCorruptFlag,
        uint *triangleVtxIdx_D, float *volume_D, float *externalForcesScl_D,
        uint triangleCnt, float minDispl, float isoval);

cudaError_t InitExternalForceScl (float *arr_D, float *volume_D,
        float *vertexPos_D, uint nElements, float isoval, uint dataArrOffs,
        uint dataArrSize);

cudaError_t InitVertexData3(float *dataOut_D, uint dataOutStride,
        uint dataOutOffs, float *dataIn_D, uint dataInStride, uint dataInOffs,
        uint vertexCnt);

cudaError_t InitVolume_surface_mapping(uint3 gridSize, float3 org,
        float3 delta);

cudaError_t UpdateVertexPositionTricubic(
        float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        uint vertexCount,
        float externalWeight,
        float forcesScl, float stiffness, float isoval,
        float minDispl,
        uint dataArrOffsPos,
        uint dataArrOffsNormal,
        uint dataArrSize);

cudaError_t UpdateVertexPositionTrilinear(float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        uint vertexCount,
        float externalWeight,
        float forcesScl,
        float stiffness,
        float isoval,
        float minDispl,
        uint dataArrOffs,
        uint dataArrSize);

// DEBUG versions that keep track of displacement length
cudaError_t UpdateVertexPositionTricubicWithDispl(float *targetVolume_D,
        float *vertexPosMapped_D, float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D, float4 *gradient_D, float3 *laplacian_D,
        float *displLen_D,
        uint vertexCount, float externalWeight,
        float forcesScl, float stiffness, float isoval,
        float minDispl, uint dataArrOffs, uint dataArrSize);

cudaError_t UpdateVertexPositionTrilinearWithDispl(float *targetVolume_D,
        float *vertexPosMapped_D,
        float *vertexExternalForcesScl_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        float *displLen_D,
        uint vertexCount,
        float externalWeight,
        float forcesScl,
        float stiffness,
        float isoval,
        float minDispl,
        uint dataArrOffs,
        uint dataArrSize);
// END DEBUG


// Wrapper functions for vertex transformations (needed to apply RMS fitting)

cudaError_t RotatePos(float *vertexData_D, uint vertexDataSize,
        uint vertexDataPosOffs, float *rotation_D, uint vertexCnt);

cudaError_t TranslatePos(float *vertexData_D, uint vertexDataSize,
        uint vertexDataPosOffs, float3 translation, uint vertexCnt);


// Wrapper functions for thrust stuff

cudaError_t ComputePrefixSumExclusiveScan(uint *flagArray_D, uint *offsArray_D,
        uint cnt);

cudaError_t AccumulateFloat(float &res, float *begin_D, uint cnt);

cudaError_t ReduceToMax(float &res, float *begin_D, uint cnt, float init);


// Wrapper functions for the metric to quantify the potential difference

cudaError_t InitPotentialTexParams(int idx, int3 dim, float3 minC, float3 delta);

cudaError_t ComputeVertexPotentialDiff(
        float *vertexPotentialDiff_D,
        float *vertexPosNew_D,
        float *vertexPosOld_D,
        float *potentialTex0_D,
        float *potentialTex1_D,
        float volMinXOld, float volMinYOld, float volMinZOld,
        float volMaxXOld, float volMaxYOld, float volMaxZOld,
        float volMinXNew, float volMinYNew, float volMinZNew,
        float volMaxXNew, float volMaxYNew, float volMaxZNew,
        uint vertexCnt);

cudaError_t ComputeVertexPotentialDiffSqr(
        float *vertexPotentialDiff_D,
        float *vertexPosNew_D,
        float *vertexPosOld_D,
        float *potentialTex0_D,
        float *potentialTex1_D,
        float volMinXOld, float volMinYOld, float volMinZOld,
        float volMaxXOld, float volMaxYOld, float volMaxZOld,
        float volMinXNew, float volMinYNew, float volMinZNew,
        float volMaxXNew, float volMaxYNew, float volMaxZNew,
        uint vertexCnt);

cudaError_t ComputeVertexPotentialSignDiff(
        float *vertexPotentialSignDiff_D,
        float *vertexPosNew_D,
        float *vertexPosOld_D,
        float *potentialTex0_D,
        float *potentialTex1_D,
        float volMinXOld, float volMinYOld, float volMinZOld,
        float volMaxXOld, float volMaxYOld, float volMaxZOld,
        float volMinXNew, float volMinYNew, float volMinZNew,
        float volMaxXNew, float volMaxYNew, float volMaxZNew,
        uint vertexCnt);

// Non-corrupt triangles
cudaError_t ComputeTriangleArea(
        float *trianglesArea_D,
        float *corruptTriangleFlag_D,
        float *vertexPos_D,
        uint *triangleIdx_D,
        uint triangleCnt);

// All triangles
cudaError_t ComputeTriangleAreaAll(
        float *trianglesArea_D,
        float *vertexPos_D,
        uint *triangleIdx_D,
        uint triangleCnt);

// Corrupt triangles
cudaError_t ComputeTriangleAreaCorrupt(
        float *trianglesArea_D,
        float *corruptTriangleFlag_D,
        float *vertexPos_D,
        uint *triangleIdx_D,
        uint triangleCnt);

cudaError_t IntegrateScalarValueOverTriangles(
        float *trianglesAreaWeightedVertexVals_D, float *corruptTriangleFlag_D,
        float *trianglesArea_D, uint *triangleIdx_D, float *scalarValue_D,
        uint triangleCnt);

cudaError_t FlagCorruptTriangles(float *corruptTriangleFlag_D,
        float *vertexPos_D, uint *triangleVtxIdx_D,
        float *volume_D, uint triangleCnt, uint vertexCnt, float isoval);

cudaError_t BindTexRef0ToArray(cudaArray *texArray);

cudaError_t BindTexRef1ToArray(cudaArray *texArray);

cudaError_t MultTriangleAreaWithWeight(float *corruptTriangleFlag_D,
        float *trianglesArea_D, uint triangleCnt);

cudaError_t CalcInternalForceLen(
        float *internalForceLen_D,
        float *vertexPosMapped_D,
        int *vertexNeighbours_D,
        float4 *gradient_D,
        float3 *laplacian_D,
        uint vertexCount,
        float stiffness,
        float forcesScl,
        uint dataArrOffs,
        uint dataArrSize);

} // extern "C"

#endif // MMPROTEINCUDAPLUGIN_POTENTIALVOLUMERENDERERCUDA_H_INCLUDED
