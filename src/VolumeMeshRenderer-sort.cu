/*
* VolumeMeshRenderer-sort.cu
*
* Copyright (C) 2012 by Universitaet Stuttgart (VIS).
* Alle Rechte vorbehalten.
*/
#ifndef MEGAMOLPROTEIN_VOLUMEMESHRENDERER_SORT_CU_INCLUDED
#define MEGAMOLPROTEIN_VOLUMEMESHRENDERER_SORT_CU_INCLUDED

#include "VolumeMeshRenderer.cuh"
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include "helper_math.h"

/*
 * Note:
 * This is a VERY slow compiling piece of code (because of thrust::sort).
 * Its in an extra file so that small changes on other parts of VolumeMeshRenderer
 * wont lead to huge compilation times.
 */
extern "C"
cudaError CentroidReduce(uint* centroidLabelsCount, uint* centroidLabels, float4* centroidSums, uint* centroidCounts, uint* vertexLabels, float4* vertices, uint vertexCount)
{
    uint* vertexLabelsEnd = vertexLabels + vertexCount;
    // Sort (reduce needs consecutive keys).
    thrust::sort_by_key(thrust::device_ptr<uint>(vertexLabels), thrust::device_ptr<uint>(vertexLabelsEnd), 
        thrust::device_ptr<float4>(vertices));
    // Count.
    thrust::reduce_by_key(thrust::device_ptr<uint>(vertexLabels), thrust::device_ptr<uint>(vertexLabelsEnd),
        thrust::constant_iterator<uint>(1), thrust::device_ptr<uint>(centroidLabels),
        thrust::device_ptr<uint>(centroidCounts));
    // Sum.
    uint* centroidLabelsEnd = thrust::reduce_by_key(thrust::device_ptr<uint>(vertexLabels), thrust::device_ptr<uint>(vertexLabelsEnd),
        thrust::device_ptr<float4>(vertices), thrust::device_ptr<uint>(centroidLabels),
        thrust::device_ptr<float4>(centroidSums)).first.get();
    if (centroidLabelsEnd >= centroidLabels) {
        *centroidLabelsCount = centroidLabelsEnd - centroidLabels;
    } else{
        *centroidLabelsCount = 0;
    }
    return cudaGetLastError();
}

extern "C"
cudaError ComputeFeatureBBox( float* fBBoxMinX, float* fBBoxMinY, float* fBBoxMinZ, float* fBBoxMaxX, float* fBBoxMaxY, float* fBBoxMaxZ,
        uint* triaLabelsMinX, uint* triaLabelsMinY, uint* triaLabelsMinZ, uint* triaLabelsMaxX, uint* triaLabelsMaxY, uint* triaLabelsMaxZ,
        uint triaCount) {
    // compute the bboxes of all features
    
    // sort the min values
    thrust::sort_by_key( thrust::device_ptr<float>(fBBoxMinX), thrust::device_ptr<float>(fBBoxMinX + triaCount), thrust::device_ptr<uint>(triaLabelsMinX));
    thrust::sort_by_key( thrust::device_ptr<float>(fBBoxMinY), thrust::device_ptr<float>(fBBoxMinY + triaCount), thrust::device_ptr<uint>(triaLabelsMinY));
    thrust::sort_by_key( thrust::device_ptr<float>(fBBoxMinZ), thrust::device_ptr<float>(fBBoxMinZ + triaCount), thrust::device_ptr<uint>(triaLabelsMinZ));
    // sort the max values
    thrust::sort_by_key( thrust::device_ptr<float>(fBBoxMaxX), thrust::device_ptr<float>(fBBoxMaxX + triaCount), thrust::device_ptr<uint>(triaLabelsMaxX), thrust::greater<float>());
    thrust::sort_by_key( thrust::device_ptr<float>(fBBoxMaxY), thrust::device_ptr<float>(fBBoxMaxY + triaCount), thrust::device_ptr<uint>(triaLabelsMaxY), thrust::greater<float>());
    thrust::sort_by_key( thrust::device_ptr<float>(fBBoxMaxZ), thrust::device_ptr<float>(fBBoxMaxZ + triaCount), thrust::device_ptr<uint>(triaLabelsMaxZ), thrust::greater<float>());
    // sort the min values by label
    thrust::stable_sort_by_key( thrust::device_ptr<uint>(triaLabelsMinX), thrust::device_ptr<uint>(triaLabelsMinX + triaCount), thrust::device_ptr<float>(fBBoxMinX));
    thrust::stable_sort_by_key( thrust::device_ptr<uint>(triaLabelsMinY), thrust::device_ptr<uint>(triaLabelsMinY + triaCount), thrust::device_ptr<float>(fBBoxMinY));
    thrust::stable_sort_by_key( thrust::device_ptr<uint>(triaLabelsMinZ), thrust::device_ptr<uint>(triaLabelsMinZ + triaCount), thrust::device_ptr<float>(fBBoxMinZ));
    // sort the max values by label
    thrust::stable_sort_by_key( thrust::device_ptr<uint>(triaLabelsMaxX), thrust::device_ptr<uint>(triaLabelsMaxX + triaCount), thrust::device_ptr<float>(fBBoxMaxX));
    thrust::stable_sort_by_key( thrust::device_ptr<uint>(triaLabelsMaxY), thrust::device_ptr<uint>(triaLabelsMaxY + triaCount), thrust::device_ptr<float>(fBBoxMaxY));
    thrust::stable_sort_by_key( thrust::device_ptr<uint>(triaLabelsMaxZ), thrust::device_ptr<uint>(triaLabelsMaxZ + triaCount), thrust::device_ptr<float>(fBBoxMaxZ));
    // get the min/max x/y/z-value per feature
    thrust::unique_by_key( thrust::device_ptr<uint>(triaLabelsMinX), thrust::device_ptr<uint>(triaLabelsMinX + triaCount), thrust::device_ptr<float>(fBBoxMinX));
    thrust::unique_by_key( thrust::device_ptr<uint>(triaLabelsMinY), thrust::device_ptr<uint>(triaLabelsMinY + triaCount), thrust::device_ptr<float>(fBBoxMinY));
    thrust::unique_by_key( thrust::device_ptr<uint>(triaLabelsMinZ), thrust::device_ptr<uint>(triaLabelsMinZ + triaCount), thrust::device_ptr<float>(fBBoxMinZ));
    thrust::unique_by_key( thrust::device_ptr<uint>(triaLabelsMaxX), thrust::device_ptr<uint>(triaLabelsMaxX + triaCount), thrust::device_ptr<float>(fBBoxMaxX));
    thrust::unique_by_key( thrust::device_ptr<uint>(triaLabelsMaxY), thrust::device_ptr<uint>(triaLabelsMaxY + triaCount), thrust::device_ptr<float>(fBBoxMaxY));
    thrust::unique_by_key( thrust::device_ptr<uint>(triaLabelsMaxZ), thrust::device_ptr<uint>(triaLabelsMaxZ + triaCount), thrust::device_ptr<float>(fBBoxMaxZ));

    return cudaGetLastError();
}

extern "C"
cudaError SortPrevTetraLabel( int2* labelPair, uint tetrahedronCount, int &labelCount) {
    thrust::sort( thrust::device_ptr<int2>(labelPair), thrust::device_ptr<int2>(labelPair + tetrahedronCount), lessInt2X());
    const int numberOfUniqueValues = thrust::unique( thrust::device_ptr<int2>(labelPair), thrust::device_ptr<int2>(labelPair + tetrahedronCount), equalInt2()) - thrust::device_ptr<int2>(labelPair);
    labelCount = numberOfUniqueValues;

    return cudaGetLastError();
}

extern "C"
cudaError TriangleVerticesToIndexList( float4* featureVertices, float4* featureVerticesOut, uint* featureVertexIdx, uint* featureVertexCnt, uint* featureVertexCntOut, 
                                       uint* featureVertexStartIdx, uint* featureVertexIdxOut, uint triaVertexCnt, uint &vertexCnt) {
    thrust::sequence( thrust::device_ptr<uint>(featureVertexIdx), thrust::device_ptr<uint>(featureVertexIdx + triaVertexCnt));
    thrust::fill_n( thrust::device_ptr<uint>(featureVertexCnt), triaVertexCnt, 1);
    cudaDeviceSynchronize();
	thrust::sort_by_key( thrust::device_ptr<float4>(featureVertices), 
        thrust::device_ptr<float4>(featureVertices + triaVertexCnt), 
        thrust::device_ptr<uint>(featureVertexIdx), less_float4());
	cudaDeviceSynchronize();
	float4* new_end = thrust::reduce_by_key( thrust::device_ptr<float4>(featureVertices), 
        thrust::device_ptr<float4>(featureVertices + triaVertexCnt), 
        thrust::device_ptr<uint>(featureVertexCnt),
        thrust::device_ptr<float4>(featureVerticesOut), 
        thrust::device_ptr<uint>(featureVertexCntOut), equal_float4()).first.get();
	cudaDeviceSynchronize();
    vertexCnt = (new_end - featureVerticesOut);
    thrust::exclusive_scan( thrust::device_ptr<uint>(featureVertexCntOut), thrust::device_ptr<uint>(featureVertexCntOut + vertexCnt), thrust::device_ptr<uint>(featureVertexStartIdx));
    WriteTriangleVertexIndexList( featureVertexIdx, featureVertexCntOut, featureVertexStartIdx, featureVertexIdxOut, triaVertexCnt, vertexCnt);
    return cudaGetLastError();
}

extern "C"
cudaError TriangleEdgeList( uint* featureVertexIdxOut, uint* featureEdgeCnt, uint* featureEdgeCntOut, uint triaCnt, uint2 *featureEdges, uint2 *featureEdgesOut, uint &edgeCnt) {
    thrust::fill_n( thrust::device_ptr<uint>(featureEdgeCnt), triaCnt * 3, 1);
    // write edges
    WriteTriangleEdgeList( featureVertexIdxOut, triaCnt, featureEdges);
    cudaDeviceSynchronize();
	thrust::sort( thrust::device_ptr<uint2>(featureEdges), 
        thrust::device_ptr<uint2>(featureEdges + (triaCnt * 3)), less_uint2());
	cudaDeviceSynchronize();
	uint2* new_end = thrust::reduce_by_key( thrust::device_ptr<uint2>(featureEdges), 
        thrust::device_ptr<uint2>(featureEdges + (triaCnt * 3)), 
        thrust::device_ptr<uint>(featureEdgeCnt),
        thrust::device_ptr<uint2>(featureEdgesOut), 
        thrust::device_ptr<uint>(featureEdgeCntOut), equal_uint2()).first.get();
	cudaDeviceSynchronize();
    edgeCnt = (new_end - featureEdgesOut);
    return cudaGetLastError();
}

// global variable for camera position
__device__ __constant__ float3 camPosition;

cudaError_t copyCamPosToDevice( float3 camPos) {
    cudaError_t error = cudaMemcpyToSymbol( camPosition, (void*)&camPos, sizeof(float3));
    cudaDeviceSynchronize();
    return error;
}

/*
 * greater than comparison
 */
__device__ bool greater_float4x3::operator()(const float4x3& lhs, const float4x3& rhs) const {
    // use midpoint
    float3 trianglePos1 = make_float3(lhs.v1 + lhs.v2 + lhs.v3)/3.0f;
    float3 trianglePos2 = make_float3(rhs.v1 + rhs.v2 + rhs.v3)/3.0f;
    float dist1 = length(trianglePos1 - camPosition);
    float dist2 = length(trianglePos2 - camPosition);

    return (dist1 > dist2);
}

extern "C"
cudaError SortTrianglesDevice( uint triaCnt, float4x3 *vertices, float4x3 *verticesCopy, float4x3 *colors, float4x3 *normals) {
    thrust::sort_by_key( thrust::device_ptr<float4x3>(vertices), 
        thrust::device_ptr<float4x3>(vertices + triaCnt), 
        thrust::device_ptr<float4x3>(colors), greater_float4x3());
    thrust::sort_by_key( thrust::device_ptr<float4x3>(verticesCopy), 
        thrust::device_ptr<float4x3>(verticesCopy + triaCnt), 
        thrust::device_ptr<float4x3>(normals), greater_float4x3());
	cudaDeviceSynchronize();
    return cudaGetLastError();
}

#endif // MEGAMOLPROTEIN_VOLUMEMESHRENDERER_SORT_CU_INCLUDED
