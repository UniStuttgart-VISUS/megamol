/*
 * VolumeMeshRenderer.cuh
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLPROTEIN_VOLUMEMESHRENDERER_CUH_INCLUDED
#define MEGAMOLPROTEIN_VOLUMEMESHRENDERER_CUH_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/functional.h>

typedef unsigned int uint;

cudaError_t copyCamPosToDevice( float3 camPos);
cudaError_t copyVolSizeToDevice( uint3 volSize);
cudaError_t copyVolSizeFromDevice( uint3 &volSize);

// structure for triangle sorting
struct float4x3{
    float4 v1;
    float4 v2;
    float4 v3;
};

/*
 * greater than comparison for int2 (compares x values)
 */
struct greater_float4x3 {
    __device__
    bool operator()(const float4x3& lhs, const float4x3& rhs) const;
};


/*
 * greater than comparison for int2 (compares x values)
 */
struct greaterInt2X {
    __host__ __device__
    bool operator()(const int2& lhs, const int2& rhs) const {
        return ( lhs.x > rhs.x) || ( ( lhs.x == rhs.x ) && ( lhs.y > rhs.y) );
    }
};

/*
 * less than comparison for int2 (compares x values)
 */
struct lessInt2X {
    __host__ __device__
    bool operator()(const int2& lhs, const int2& rhs) const {
        return ( lhs.x < rhs.x) || ( ( lhs.x == rhs.x ) && ( lhs.y < rhs.y) );
    }
};

/*
 * less than comparison for uint2 (compares x values)
 */
struct less_uint2 {
    __host__ __device__
    bool operator()(const uint2& lhs, const uint2& rhs) const {
        return ( lhs.x < rhs.x) || ( ( lhs.x == rhs.x ) && ( lhs.y < rhs.y) );
    }
};

/*
 * greater than comparison for int2 (compares x values)
 */
struct greaterInt2Y {
    __host__ __device__
    bool operator()(const int2& lhs, const int2& rhs) const {
        return lhs.y > rhs.y;
    }
};

/*
 * equal_to comparison adapted from thrust for int2
 */
struct equalInt2 {
    __host__ __device__
    bool operator()(const int2& lhs, const int2& rhs) const {
        return (lhs.x == rhs.x && lhs.y == rhs.y);
    }
};
  
/*
 * equal_to comparison adapted from thrust for uint2
 */
struct equal_uint2 {
    __host__ __device__
    bool operator()(const uint2& lhs, const uint2& rhs) const {
        return (lhs.x == rhs.x && lhs.y == rhs.y);
    }
};
  
/**
 * Less-function for float4
 */
struct less_float4 : public thrust::binary_function<float4,float4,bool> {
    __host__ __device__ bool operator()(const float4 &l, const float4 &r) const {
        if( l.x < r.x ) return true;
        if( l.x > r.x ) return false;
        // l.x == r.x
        if( l.y < r.y ) return true;
        if( l.y > r.y ) return false;
        // l.x == r.x && l.y == r.y
        if( l.z < r.z ) return true;
        if( l.z > r.z ) return false;
        // l.x == r.x && l.y == r.y && l.z == r.z
        if( l.w < r.w ) return true;
        if( l.w > r.w ) return false;
        return false;
    }
};
        
/**
 * equal-function for float4
 */
struct equal_float4 : public thrust::binary_function<float4,float4,bool> {
    __host__ __device__ bool operator()(const float4 &l, const float4 &r) const {
		const float eps = 1e-4f;
        return (( fabsf( l.x - r.x) < eps) && ( fabsf( l.y - r.y) < eps) && ( fabsf( l.z - r.z) < eps) && ( fabsf( l.w - r.w) < eps));
    }
};
      
extern "C"
cudaError BindVolumeTexture(float* textureArray);
//cudaError BindVolumeTexture(cudaArray* textureArray);

extern "C"
cudaError BindNeighborAtomTexture(int* textureArray);

extern "C"
cudaError UnbindVolumeTexture();

extern "C"
cudaError BindAOVolumeTexture(cudaArray* textureArray);

extern "C"
cudaError UnbindAOVolumeTexture();

extern "C"
cudaError ClassifyCubes(uint* cubeStates, float thresholdValue, uint cubeCount);

extern "C"
cudaError ScanCubes(uint* cubeOffsets, uint* cubeStates, uint cubeCount);

extern "C"
cudaError CompactCubes(uint* cubeMap, uint* cubeOffsets, uint* cubeStates, uint cubeCount);

extern "C"
cudaError ClassifyTetrahedronsInACube(uint* verticesPerTetrahedron, uint* cubeMap, float thresholdValue, uint activeCubeCount);

extern "C"
cudaError ScanTetrahedrons(uint* vertexOffsets, uint* verticesPerTetrahedron, uint tetrahedronCount);

extern "C"
cudaError GenerateTriangles(float4* vertices, int* neighborAtoms, float4* normals, float translateX, float translateY, float translateZ,
    float scaleX, float scaleY, float scaleZ, uint* vertexOffsets, uint* cubeMap, float thresholdValue, uint tetrahedronCount);

extern "C"
cudaError MeshReset(uint* eqList, uint* refList, uint tetrahedronCount, uint* vertexOffsets, float* triangleAO);

extern "C"
cudaError MeshScan(float4* vertices, float4* normals, uint* vertexOffsets, float* triangleAO, float aoThreshold, uint* eqList, uint* refList, bool* modified, 
    uint* cubeStates, uint* cubeOffsets, uint* cubeMap, float thresholdValue, uint tetrahedronCount);

extern "C"
cudaError MeshAnalysis(uint* eqList, uint* refList, uint tetrahedronCount);

extern "C"
cudaError MeshLabeling(uint* eqList, uint* refList, uint tetrahedronCount);

extern "C"
cudaError CentroidMap(uint* vertexLabels, uint* vertexOffsets, uint* verticesPerTetrahedron, uint* eqList, uint tetrahedronCount);

extern "C"
cudaError CentroidReduce(uint* centroidCount, uint* centroidLabels, float4* centroidSums, uint* centroidCounts, uint* vertexLabels, float4* vertices, uint vertexCount);

extern "C"
cudaError CentroidFinalize(float4* centroids, float4* centroidSums, uint* centroidCounts, uint centroidCount);

extern "C"
cudaError ColorizeByCentroid(float4* colors, float4* centroidColors, uint* centroidLabels, uint centroidCount, uint* vertexLabels, uint vertexCount);

extern "C"
cudaError ColorizeByAO(float4* colors, float* triangleAO, float aoThreashold, uint vertexCount);

extern "C"
cudaError ComputeSurfaceArea(float4* verts, float* areas, unsigned int triaCount);

extern "C"
cudaError ComputeCentroidArea(float* centroidAreas, uint2* featureStartEnd, float* triaAreas, uint* vertexLabels, uint* centroidLabels, uint triaCount, uint centroidCount);

extern "C"
cudaError ComputeTriangleAO(float4* verts, float4* normals, float* ao, unsigned int triaCount);

extern "C"
cudaError RemoveSmallSegments(float* centroidAreas, float* triangleAO, uint* vertexLabels, uint* centroidLabels, float areaThreshold, float aoThreshold, uint triaCount, uint centroidCount, bool* segmentsRemoved);

extern "C"
cudaError ComputeTriangleBBox(float* fBBoxMinX, float* fBBoxMinY, float* fBBoxMinZ, float* fBBoxMaxX, float* fBBoxMaxY, float* fBBoxMaxZ, uint* triangleLabels, float4* verts, uint* vertexLabels, uint triaCount);

extern "C"
cudaError ComputeFeatureBBox( float* fBBoxMinX, float* fBBoxMinY, float* fBBoxMinZ, float* fBBoxMaxX, float* fBBoxMaxY, float* fBBoxMaxZ,
    uint* triaLabelsMinX, uint* triaLabelsMinY, uint* triaLabelsMinZ, uint* triaLabelsMaxX, uint* triaLabelsMaxY, uint* triaLabelsMaxZ, uint triaCount);

extern "C"
cudaError WritePrevTetraLabel( int2* labelPair, uint* cubeStatesOld, uint* cubeOffsetsOld, //uint* cubeMapOld,
    uint* verticesPerTetrahedronOld, uint* eqListOld, uint* cubeMap, uint* verticesPerTetrahedron, 
    uint* eqList, uint tetrahedronCount);

extern "C"
cudaError SortPrevTetraLabel( int2* labelPair, uint tetrahedronCount, int &labelCount);

extern "C"
cudaError WriteTriangleVertexIndexList( uint* featureVertexIdx, uint* featureVertexCnt, uint* featureVertexStartIdx, uint* featureVertexIdxOut, uint triaVertexCnt, uint vertexCnt);

extern "C"
cudaError TriangleVerticesToIndexList( float4* featureVertices, float4* featureVerticesOut, uint* featureVertexIdx, uint* featureVertexCnt, uint* featureVertexCntOut, 
                                       uint* featureVertexStartIdx, uint* featureVertexIdxOut, uint triaVertexCnt, uint &vertexCnt);

extern "C"
cudaError WriteTriangleEdgeList( uint* featureVertexIdxOut, uint triaCnt, uint2 *featureEdges);

extern "C"
cudaError TriangleEdgeList( uint* featureVertexIdxOut, uint* featureEdgeCnt, uint* featureEdgeCntOut, uint triaCnt, uint2 *featureEdges, uint2 *featureEdgesOut, uint &edgeCnt);

extern "C"
cudaError SortTrianglesDevice( uint triaCnt, float4x3 *vertices, float4x3 *verticesCopy, float4x3 *colors, float4x3 *normals);

#endif // MEGAMOLPROTEIN_VOLUMEMESHRENDERER_CUH_INCLUDED
