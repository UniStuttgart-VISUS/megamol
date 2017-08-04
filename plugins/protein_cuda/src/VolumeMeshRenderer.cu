/*
* VolumeMeshRenderer.cu
*
* Copyright (C) 2012 by Universitaet Stuttgart (VIS).
* Alle Rechte vorbehalten.
*/
#ifndef MEGAMOLPROTEIN_VOLUMEMESHRENDERER_CU_INCLUDED
#define MEGAMOLPROTEIN_VOLUMEMESHRENDERER_CU_INCLUDED

#include "VolumeMeshRenderer.cuh"
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_ptr.h>
#include "helper_math.h"

#ifdef CUDA_NO_SM_11_ATOMIC_INTRINSICS
#error "Atomic intrinsics are missing (nvcc -arch=sm_11)"
#endif

#define GT_THREADS 128

/*
 * Variables.
 */
//texture<float4, cudaTextureType3D, cudaReadModeElementType> volumeTex;
//texture<float, cudaTextureType1D, cudaReadModeElementType> volumeTex;
__device__ float* volumeTex;
__device__ int* neighborAtomTex;
texture<float, cudaTextureType3D, cudaReadModeElementType> aoVolumeTex;

__device__ __constant__ uint3 dVolumeSize;

cudaError_t copyVolSizeToDevice( uint3 volSize) {
    cudaError_t error = cudaMemcpyToSymbol( dVolumeSize, (void*)&volSize, sizeof(uint3));
    cudaDeviceSynchronize();
    return error;
}

cudaError_t copyVolSizeFromDevice( uint3 &volSize) {
    volSize = make_uint3( 0, 0, 0);
    cudaError_t error = cudaMemcpyFromSymbol( (void*)&volSize, dVolumeSize, sizeof(uint3));
    cudaDeviceSynchronize();
    return error;
}

/*
 * Voxel index and access functions.
 */
inline __device__
uint Index()
{
    return __umul24(__umul24(blockIdx.y, gridDim.x) + blockIdx.x, blockDim.x) + threadIdx.x;
}

inline __device__
float VoxelValue(float3 index)
{
    uint idx = index.z * dVolumeSize.x * dVolumeSize.y + index.y * dVolumeSize.x + index.x;
    return volumeTex[idx];//tex1Dfetch(volumeTex, idx);
}

inline __device__
int NeighborAtom(float3 index)
{
    uint idx = index.z * dVolumeSize.x * dVolumeSize.y + index.y * dVolumeSize.x + index.x;
    return neighborAtomTex[idx];
}

inline __device__
float3 VoxelGradient(float3 index)
{
    float3 result;
    // Compute central difference.
    result.x = VoxelValue(make_float3(index.x - 1.0f, index.y, index.z)) 
        - VoxelValue(make_float3(index.x + 1.0f, index.y, index.z));
    result.y = VoxelValue(make_float3(index.x, index.y - 1.0f, index.z)) 
        - VoxelValue(make_float3(index.x, index.y + 1.0f, index.z));
    result.z = VoxelValue(make_float3(index.x, index.y, index.z - 1.0f))
        - VoxelValue(make_float3(index.x, index.y, index.z + 1.0f));
    return result;
}

/*
 * Marching tetrahedrons.
 */
inline __device__
float3 CubeVertex0(uint index) 
{
    return make_float3(index % dVolumeSize.x, (index / dVolumeSize.x) % dVolumeSize.y, 
        index / (dVolumeSize.x * dVolumeSize.y));
}

__global__
void ClassifyCubes_kernel(uint* cubeStates, float thresholdValue, uint cubeCount)
{
    const uint cubeIndex = min(Index(), cubeCount - 1);
    const float3 cubeVertex0 = CubeVertex0(cubeIndex);
    const float cubeVertexOffsetsL[8][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {1, 1, 0},
        {0, 1, 0}, 
        {0, 0, 1}, 
        {1, 0, 1}, 
        {1, 1, 1}, 
        {0, 1, 1}
    };
    // Add "vertex states" of a cube (0: inactive, 1: active).
    unsigned char cubeFlags = static_cast<uint>(VoxelValue(cubeVertex0) <= thresholdValue);
    for (int cubeVertexIndex = 1; cubeVertexIndex < 8; ++cubeVertexIndex) {
        const float3 cubeVertex = make_float3(
            cubeVertex0.x + cubeVertexOffsetsL[cubeVertexIndex][0],
            cubeVertex0.y + cubeVertexOffsetsL[cubeVertexIndex][1],
            cubeVertex0.z + cubeVertexOffsetsL[cubeVertexIndex][2]);
        cubeFlags |= static_cast<uint>(VoxelValue(cubeVertex) <= thresholdValue) * (1 << cubeVertexIndex);
    }
    // Reduce "vertex states" to a "cube state".
    cubeStates[cubeIndex] = min(cubeFlags % 255, 1);
}

__global__
void CompactCubes_kernel(uint* cubeMap, uint* cubeOffsets, uint* cubeStates, uint cubeCount)
{
    const uint cubeIndex = min(Index(), cubeCount - 1);
    if (cubeStates[cubeIndex] != 0) {
        // Map from active cubes list to cube index.
        cubeMap[cubeOffsets[cubeIndex]] = cubeIndex;
    }
}

__device__ __shared__ float3 cubeVertexOffsets[8];
__device__ __constant__ float cubeVertexOffsetsC[8][3] = {
    {0, 0, 0},
    {1, 0, 0},
    {1, 1, 0},
    {0, 1, 0}, 
    {0, 0, 1}, 
    {1, 0, 1}, 
    {1, 1, 1}, 
    {0, 1, 1}
};

inline __device__
void LoadCubeOffsets()
{
    // Load cube vertex offsets into shared memory.
    if (threadIdx.x < 8) {
        cubeVertexOffsets[threadIdx.x] = make_float3(cubeVertexOffsetsC[threadIdx.x][0], 
            cubeVertexOffsetsC[threadIdx.x][1], cubeVertexOffsetsC[threadIdx.x][2]);
    }
}

__device__ __shared__ uint tetrahedronsInACube[6][4];
__device__ __constant__ uint tetrahedronsInACubeC[6][4] = {
    {0, 5, 1, 6},
    {0, 1, 2, 6},
    {0, 2, 3, 6},
    {0, 3, 7, 6},
    {0, 7, 4, 6},
    {0, 4, 5, 6}
};

inline __device__
void LoadTetrahedronsInACube()
{
    // Load tetrahedron vertex index to cube index map into shared memory.
    if (threadIdx.x < 6) {
        for (int i = 0; i < 4; ++i) {
            tetrahedronsInACube[threadIdx.x][i] = tetrahedronsInACubeC[threadIdx.x][i];
        }
    }
}

__device__ __shared__ uint tetrahedronVertexCount[16];
__device__ __constant__ uint tetrahedronVertexCountC[16] = {
    0, 3, 3, 6, 3, 6, 6, 3,
    3, 6, 6, 3, 6, 3, 3, 0
}; 

inline __device__
void LoadTetrahedronVertexCount()
{
    // Load tetrahedron vertex count into shared memory.
    if (threadIdx.x < 16) {
        tetrahedronVertexCount[threadIdx.x] = tetrahedronVertexCountC[threadIdx.x];
    }
}

inline __device__
unsigned char TetrahedronFlags(float3 cubeVertex0, int tetrahedronIndex, float thresholdValue)
{
    unsigned char flags = 0;
    for (int tetrahedronVertexIndex = 0; tetrahedronVertexIndex < 4; ++tetrahedronVertexIndex) {
        const float3 cubeVertexOffset = cubeVertexOffsets[tetrahedronsInACube[tetrahedronIndex][tetrahedronVertexIndex]];
        if (VoxelValue(cubeVertex0 + cubeVertexOffset) <= thresholdValue) {
            flags |= 1 << static_cast<unsigned char>(tetrahedronVertexIndex);
        }
    }
    return flags;
}

__global__
void ClassifyTetrahedronsInACube_kernel(uint* verticesPerTetrahedron, uint* cubeMap, float thresholdValue, uint activeCubeCount)
{
    const uint activeCubeIndex = Index();
    const float3 cubeVertex0 = CubeVertex0(cubeMap[activeCubeIndex]);
    LoadCubeOffsets();
    LoadTetrahedronsInACube();
    LoadTetrahedronVertexCount();
    __syncthreads();
    // Prevent non-power of two writes.
    if (activeCubeIndex >= activeCubeCount) {
        return;
    }
    // Classify all tetrahedrons in a cube.
    for (int tetrahedronIndex = 0; tetrahedronIndex < 6; ++tetrahedronIndex) {
        // Compute tetrahedron flags.
        unsigned char tetrahedronFlags = TetrahedronFlags(cubeVertex0, tetrahedronIndex, thresholdValue);
        // Store number of vertices.
        verticesPerTetrahedron[activeCubeIndex * 6 + tetrahedronIndex] = tetrahedronVertexCount[tetrahedronFlags];
    }
}

__device__ __shared__ unsigned char tetrahedronEdgeFlags[16];
__device__ __constant__ unsigned char tetrahedronEdgeFlagsC[16] = {
    0x00, 0x0d, 0x13, 0x1e, 0x26, 0x2b, 0x35, 0x38,
    0x38, 0x35, 0x2b, 0x26, 0x1e, 0x13, 0x0d, 0x00
}; 

__device__ __shared__ char tetrahedronEdgeConnections[6][2];
__device__ __constant__ char tetrahedronEdgeConnectionsC[6][2] = {
    {0, 1},  {1, 2},  {2, 0},  {0, 3},  {1, 3},  {2, 3}
};

inline __device__
void LoadTetrahedronEdgeFlagsAndConnections()
{
    // Load tetrahedron edge flags into shared memory.
    if (threadIdx.x < 16) {
        tetrahedronEdgeFlags[threadIdx.x] = tetrahedronEdgeFlagsC[threadIdx.x];
    }
    // Load tetrahedron edge connection table into shared memory.
    if (threadIdx.x < 6) {
        tetrahedronEdgeConnections[threadIdx.x][0] = tetrahedronEdgeConnectionsC[threadIdx.x][0];
        tetrahedronEdgeConnections[threadIdx.x][1] = tetrahedronEdgeConnectionsC[threadIdx.x][1];
    }
}

__device__ __shared__ char tetrahedronTriangles[16][6];
__device__ __constant__ char tetrahedronTrianglesC[16][6] = {
    {-1, -1, -1, -1, -1, -1},
    { 0,  3,  2, -1, -1, -1},
    { 0,  1,  4, -1, -1, -1},
    { 1,  4,  2,  2,  4,  3},
    { 1,  2,  5, -1, -1, -1},
    { 0,  3,  5,  0,  5,  1},
    { 0,  2,  5,  0,  5,  4},
    { 5,  4,  3, -1, -1, -1},
    { 3,  4,  5, -1, -1, -1},
    { 4,  5,  0,  5,  2,  0},
    { 1,  5,  0,  5,  3,  0},
    { 5,  2,  1, -1, -1, -1},
    { 3,  4,  2,  2,  4,  1},
    { 4,  1,  0, -1, -1, -1},
    { 2,  3,  0, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1}
};

inline __device__
void LoadTetrahedronTriangles()
{
    // Load tetrahedron triangle table into shared memory.
    if (threadIdx.x < 16) {
        for (int i = 0; i < 6; ++i) {
            tetrahedronTriangles[threadIdx.x][i] = tetrahedronTrianglesC[threadIdx.x][i];
        }
    }
}

__global__
void GenerateTriangles_kernel(float4* vertices, int* neighborAtom, float4* normals, float translateX, float translateY, float translateZ,
    float scaleX, float scaleY, float scaleZ, uint* vertexOffsets, uint* cubeMap, float thresholdValue, uint tetrahedronCount)
{
    const uint id = Index();
    const uint activeCubeIndex = id / 6;
    const int tetrahedronIndex = id % 6;
    const float3 cubeVertex0 = CubeVertex0(cubeMap[activeCubeIndex]);
    LoadCubeOffsets();
    LoadTetrahedronsInACube();
    LoadTetrahedronEdgeFlagsAndConnections();
    LoadTetrahedronTriangles();
    __syncthreads();
    // Prevent non-power of two writes.
    if (id >= tetrahedronCount) {
        return;
    }
    unsigned char tetrahedronFlags = TetrahedronFlags(cubeVertex0, tetrahedronIndex, thresholdValue);
    // Skip inaktive tetrahedrons.
    if (tetrahedronFlags == 0x00 || tetrahedronFlags == 0x0F) {
        return;
    }
    __shared__ float3 edgeVertex[6 * GT_THREADS];
    // temporary storage for edge vertex neighbor atom
    __shared__ int edgeVertexNeighborAtom[6 * GT_THREADS];
    __shared__ float3 edgeNormal[6 * GT_THREADS];
    // Find intersection of the surface with each edge.
    for (int edgeIndex = 0; edgeIndex < 6; edgeIndex++) {
        // Test if edge intersects with surface.
        if (tetrahedronEdgeFlags[tetrahedronFlags] & (1 << static_cast<unsigned char>(edgeIndex)))  {
            // Interpolate vertex.
            const float3 v0 = cubeVertex0 + cubeVertexOffsets[tetrahedronsInACube[tetrahedronIndex][tetrahedronEdgeConnections[edgeIndex][0]]];		
            const float3 v1 = cubeVertex0 + cubeVertexOffsets[tetrahedronsInACube[tetrahedronIndex][tetrahedronEdgeConnections[edgeIndex][1]]];	
            const float f0 = VoxelValue(v0);
            const float interpolator = (thresholdValue - f0) / (VoxelValue(v1) - f0);
            float3 vertex = lerp(make_float3(v0.x, v0.y, v0.z), make_float3(v1.x, v1.y, v1.z), interpolator);
            edgeVertex[threadIdx.x * 6 + edgeIndex] = vertex;
            // store nearest atom per edge vertex
            edgeVertexNeighborAtom[threadIdx.x * 6 + edgeIndex] = (interpolator < 0.5) ?  NeighborAtom(v0) : NeighborAtom(v1);
            // Compute normal from gradient.
            edgeNormal[threadIdx.x * 6 + edgeIndex] = normalize(lerp(VoxelGradient(v0), VoxelGradient(v1), interpolator));
        }
    }
    // Write vertices.
    for (int triangleIndex = 0; triangleIndex < 2; triangleIndex++) {
        if (tetrahedronTriangles[tetrahedronFlags][3 * triangleIndex] >= 0) {	 
            //int edgeIndex0 = threadIdx.x * 6 + tetrahedronTriangles[tetrahedronFlags][3 * triangleIndex + 0];
            //int edgeIndex1 = threadIdx.x * 6 + tetrahedronTriangles[tetrahedronFlags][3 * triangleIndex + 1];
            //int edgeIndex2 = threadIdx.x * 6 + tetrahedronTriangles[tetrahedronFlags][3 * triangleIndex + 2];
            //float3 faceNormal = cross(edgeVertex[edgeIndex1] - edgeVertex[edgeIndex0], edgeVertex[edgeIndex2] - edgeVertex[edgeIndex0]);

            for (int cornerIndex = 0; cornerIndex < 3; cornerIndex++) {
                int edgeIndex = threadIdx.x * 6 + tetrahedronTriangles[tetrahedronFlags][3 * triangleIndex + cornerIndex];
                uint vertexOffset = vertexOffsets[id] + 3 * triangleIndex + cornerIndex;
                //vertices[vertexOffset] = make_float4(translateX + edgeVertex[edgeIndex].x * scaleX, 
                //	translateY + edgeVertex[edgeIndex].y * scaleY, 
                //	translateZ + edgeVertex[edgeIndex].z * scaleZ, 1.0f);
                //vertices[vertexOffset] = make_float4( edgeVertex[edgeIndex].x / 128.0f, 
                //	 edgeVertex[edgeIndex].y / 128.0f , 
                //	 edgeVertex[edgeIndex].z / 128.0f, 1.0f);
                vertices[vertexOffset] = make_float4( edgeVertex[edgeIndex].x, 
                     edgeVertex[edgeIndex].y, edgeVertex[edgeIndex].z, 1.0f);
                // store nearest atom per output vertex
                neighborAtom[vertexOffset] = edgeVertexNeighborAtom[edgeIndex];
                //normals[vertexOffset] = make_float4(faceNormal.x, faceNormal.y, faceNormal.z, 0.0f);
                normals[vertexOffset] = make_float4(edgeNormal[edgeIndex].x, 
                    edgeNormal[edgeIndex].y, edgeNormal[edgeIndex].z, 0.0f);
            }
        }
    }
}

/*
 * Mesh scanning.
 */
__global__
void MeshReset_kernel(uint* eqList, uint* refList, uint tetrahedronCount, uint* vertexOffsets, float* triangleAO)
{
    const uint id = Index();
    // Prevent non-power of two writes.
    if (id >= tetrahedronCount) {
        return;
    }

    // TODO: fix this! -> MeshScan should work correctly for different AO factors in one tetrahedron!
    // Set the ambient occlusion values for all triangles to the same value (average of all individual triangle AO factors)
    float aoValue = 0.0;
    for (uint vertexOffset = vertexOffsets[id]; vertexOffset < vertexOffsets[id + 1]; vertexOffset += 3) {
        aoValue += triangleAO[vertexOffset/3];
    }
    aoValue /= float(( vertexOffsets[id + 1] - vertexOffsets[id]) / 3);
    for (uint vertexOffset = vertexOffsets[id]; vertexOffset < vertexOffsets[id + 1]; vertexOffset += 3) {
        triangleAO[vertexOffset/3] = aoValue;
    }

    eqList[id] = id;
    refList[id] = id;
}

__device__ __shared__ unsigned char tetrahedronFaceMask[4];
__device__ __constant__ unsigned char tetrahedronFaceMaskC[4] = {
    0x07, 0x0b, 0x0d, 0x0e
};

// This macro greatly simplifies the adjacent neighbour lookup by computing an
// offset from the cube's index (not the tetrahedron's global index) using a
// cube offsets and a local tetrahedron index.
#define TFN(t, cX, cY, cZ) {t, cX, cY, cZ}

__device__ __shared__ int tetrahedronFaceNeighbours[6][4][4];
__device__ __constant__ int tetrahedronFaceNeighboursC[6][4][4] = {
    {TFN(2, 0, -1, 0), TFN(5, 0, 0, 0), TFN(1, 0, 0, 0), TFN(4, 1, 0, 0)},
    {TFN(5, 0, 0, -1), TFN(0, 0, 0, 0), TFN(2, 0, 0, 0), TFN(3, 1, 0, 0)},
    {TFN(4, 0, 0, -1), TFN(1, 0, 0, 0), TFN(3, 0, 0, 0), TFN(0, 0, 1, 0)},
    {TFN(1, -1, 0, 0), TFN(2, 0, 0, 0), TFN(4, 0, 0, 0), TFN(5, 0, 1, 0)},
    {TFN(0, -1, 0, 0), TFN(3, 0, 0, 0), TFN(5, 0, 0, 0), TFN(2, 0, 0, 1)},
    {TFN(3, 0, -1, 0), TFN(4, 0, 0, 0), TFN(0, 0, 0, 0), TFN(1, 0, 0, 1)}
};

#undef TFN

inline __device__
void LoadTetrahedronNeighbours()
{
    // Load tetrahedron face masks into shared memory.
    if (threadIdx.x < 4) {
        tetrahedronFaceMask[threadIdx.x] = tetrahedronFaceMaskC[threadIdx.x];		
    }
    // Load tetrahedron face neighbours table into shared memory.
    if (threadIdx.x < 6) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                tetrahedronFaceNeighbours[threadIdx.x][i][j] = tetrahedronFaceNeighboursC[threadIdx.x][i][j];
            }
        }
    }
}

inline __device__
float AOValue(float3 position, float3 normal) {
    //TODO: double check this scale * 4 hack.
    float3 scale = make_float3(1.0f / dVolumeSize.x,
        1.0f / dVolumeSize.y, 1.0f / dVolumeSize.z);
    float x = position.x * scale.x + normal.x * scale.x;
    float y = position.y * scale.y + normal.y * scale.y;
    float z = position.z * scale.z + normal.z * scale.z;
    return tex3D(aoVolumeTex, x, y, z);
    //float aofac = 0.0f;
    //for ( uint i = 0; i < 10; i++ ) {
    //    x = position.x * scale.x + normal.x * scale.x * ( 1.0f / 10.0f);
    //    y = position.y * scale.y + normal.y * scale.y * ( 1.0f / 10.0f);
    //    z = position.z * scale.z + normal.z * scale.z * ( 1.0f / 10.0f);
    //    aofac = max( aofac, tex3D(aoVolumeTex, x, y, z));
    //}
    //return aofac;
}

__global__
void MeshScan_kernel(float4* vertices, float4* normals, uint* vertexOffsets, float* triangleAO, float aoThreshold, uint* eqList, uint* refList,
    bool* modified, uint* cubeStates, uint* cubeOffsets, uint* cubeMap, float thresholdValue, uint tetrahedronCount)
{
    const uint id = Index();
    const uint activeCubeIndex = id / 6;
    const int tetrahedronIndex = id % 6;
    const float3 cubeVertex0 = CubeVertex0(cubeMap[activeCubeIndex]);
    const uint label1 = eqList[id];
    uint label2 = ~0;
    LoadCubeOffsets();
    LoadTetrahedronsInACube();
    LoadTetrahedronNeighbours();
    __syncthreads();
    // Prevent non-power of two writes.
    if (id >= tetrahedronCount) {
        return;
    }
    // Search for minimum label among neighbours.
    unsigned char tetrahedronFlags = TetrahedronFlags(cubeVertex0, tetrahedronIndex, thresholdValue);
    const uint volumeCubeSize = dVolumeSize.x * dVolumeSize.y * dVolumeSize.z;
    for(int faceIndex = 0; faceIndex < 4; faceIndex++) {
        unsigned char tetrahedronFaceFlags = tetrahedronFlags & tetrahedronFaceMask[faceIndex];
        // Test if this face is connected to a neigbour (using identity).
        if (tetrahedronFaceFlags != 0 && tetrahedronFaceFlags != tetrahedronFaceMask[faceIndex])  {
            uint tfn = tetrahedronFaceNeighbours[tetrahedronIndex][faceIndex][0] 
                + (tetrahedronFaceNeighbours[tetrahedronIndex][faceIndex][1] 
                + (tetrahedronFaceNeighbours[tetrahedronIndex][faceIndex][2] * dVolumeSize.x)
                + (tetrahedronFaceNeighbours[tetrahedronIndex][faceIndex][3] * dVolumeSize.y * dVolumeSize.x)) * 6;
            int neighbourId = cubeMap[activeCubeIndex] * 6 + tfn;
            int neighbourCubeId = neighbourId / 6;
            // Test if cube is active.
            if (neighbourCubeId >= 0 && neighbourCubeId < volumeCubeSize && cubeStates[neighbourCubeId] == 1) {
                neighbourId = cubeOffsets[neighbourCubeId] * 6 + neighbourId % 6;
                // For each triangle.
                bool aoState1 = true;
                for (uint vertexOffset = vertexOffsets[id]; vertexOffset < vertexOffsets[id + 1]; vertexOffset += 3) {
                    //float3 v0 = make_float3(vertices[vertexOffset].x, vertices[vertexOffset].y, vertices[vertexOffset].z);
                    //float3 v1 = make_float3(vertices[vertexOffset + 1].x, vertices[vertexOffset + 1].y, vertices[vertexOffset + 1].z);
                    //float3 v2 = make_float3(vertices[vertexOffset + 2].x, vertices[vertexOffset + 2].y, vertices[vertexOffset + 2].z);
                    //float3 n0 = make_float3(normals[vertexOffset].x, normals[vertexOffset].y, normals[vertexOffset].z);
                    //float3 n1 = make_float3(normals[vertexOffset + 1].x, normals[vertexOffset + 1].y, normals[vertexOffset + 1].z);
                    //float3 n2 = make_float3(normals[vertexOffset + 2].x, normals[vertexOffset + 2].y, normals[vertexOffset + 2].z);
                    //float aoValue = (AOValue(v0, n0) + AOValue(v1, n1) + AOValue(v2, n2)) / 3.0f;
                    float aoValue = triangleAO[vertexOffset/3];
                    if (aoValue > aoThreshold) {
                        aoState1 = false;
                        //break;
                    }
                }
                // For each neighbour triangle.
                bool aoState2 = true;
                for (uint vertexOffset = vertexOffsets[neighbourId]; vertexOffset < vertexOffsets[neighbourId + 1]; vertexOffset += 3) {
                    //float3 v0 = make_float3(vertices[vertexOffset].x, vertices[vertexOffset].y, vertices[vertexOffset].z);
                    //float3 v1 = make_float3(vertices[vertexOffset + 1].x, vertices[vertexOffset + 1].y, vertices[vertexOffset + 1].z);
                    //float3 v2 = make_float3(vertices[vertexOffset + 2].x, vertices[vertexOffset + 2].y, vertices[vertexOffset + 2].z);
                    //float3 n0 = make_float3(normals[vertexOffset].x, normals[vertexOffset].y, normals[vertexOffset].z);
                    //float3 n1 = make_float3(normals[vertexOffset + 1].x, normals[vertexOffset + 1].y, normals[vertexOffset + 1].z);
                    //float3 n2 = make_float3(normals[vertexOffset + 2].x, normals[vertexOffset + 2].y, normals[vertexOffset + 2].z);
                    //float aoValue = (AOValue(v0, n0) + AOValue(v1, n1) + AOValue(v2, n2)) / 3.0f;
                    float aoValue = triangleAO[vertexOffset/3];
                    if (aoValue > aoThreshold) {
                        aoState2 = false;
                        //break;
                    }
                }
                // Test for AO-Shading connectedness.
                if (aoState1 == aoState2) {
                    // Store connectedness.
                    uint neighbourLabel = eqList[neighbourId];
                    if (neighbourLabel < label2) {
                        label2 = neighbourLabel;
                    }
                }
            }
        }
    }
    if (label2 < label1) {
        // Write out minimum (atomic, thus no synchronization).
        atomicMin(&refList[label1], label2);
        *modified = true;
    }
}

__global__
void MeshAnalysis_kernel(uint* eqList, uint* refList, uint tetrahedronCount)
{
    const uint id = Index();
    // Prevent non-power of two writes.
    if (id >= tetrahedronCount) {
        return;
    }
    // Test if thread is the "label owner".
    if (eqList[id] == id) {
        uint label = id;
        uint ref = refList[label];
        // Expand equivalence chain.
        while (ref != label) {
            label = ref;
            ref = refList[label];
        }
        refList[id] = ref;
    }
}

__global__
void MeshLabeling_kernel(uint* eqList, uint* refList, uint tetrahedronCount)
{
    const uint id = Index();
    // Prevent non-power of two writes.
    if (id >= tetrahedronCount) {
        return;
    }
    eqList[id] = refList[eqList[id]];
}

/*
 * Mesh centroid handling.
 */
__global__
void CentroidMap_kernel(uint* vertexLabels, uint* vertexOffsets, uint* verticesPerTetrahedron, uint* eqList, uint tetrahedronCount)
{
    const uint id = Index();
    // Prevent non-power of two access.
    if (id >= tetrahedronCount) {
        return;
    }
    const uint label = eqList[id];
    const uint offset = vertexOffsets[id];
    const uint nextOffset = offset + verticesPerTetrahedron[id];
    for (uint index = offset; index < nextOffset; ++index) {
        vertexLabels[index] = label;
    }
}

__global__
void CentroidFinalize_kernel(float4* centroids, float4* centroidSums, uint* centroidCounts, uint centroidCount)
{
    const uint id = Index();
    // Prevent non-power of two writes.
    if (id >= centroidCount) {
        return;
    }
    centroids[id].x = centroidSums[id].x / centroidCounts[id];
    centroids[id].y = centroidSums[id].y / centroidCounts[id];
    centroids[id].z = centroidSums[id].z / centroidCounts[id];
    centroids[id].w = 1.0f;
}

__global__
void ColorizeByCentroid_kernel(float4* colors, float4* centroidColors, uint* centroidLabels, uint centroidCount, uint* vertexLabels, uint vertexCount)
{
    const uint id = Index();
    // Prevent non-power of two writes.
    if (id >= vertexCount) {
        return;
    }
    for (int i = 0; i < centroidCount; ++i) {
        if (vertexLabels[id] == centroidLabels[i]) {
            colors[id] = centroidColors[i];
            return;
        }
    }
}

__global__
void ColorizeByAO_kernel(float4* colors, float* triaAO, float aoThreshold, uint vertexCount)
{
    const uint id = Index();
    // Prevent non-power of two writes.
    if (id >= vertexCount) {
        return;
    }

    if( triaAO[id/3] > aoThreshold )
        colors[id] = lerp( colors[id], make_float4( 0.0, 1.0, 0.25, 1.0), 0.5);
    else
        colors[id] = lerp( colors[id], make_float4( 1.0, 0.25, 0.0, 1.0), 0.5);
}

__global__ 
void ComputeTriangleAreas_kernel(float4 *pos, float *area, unsigned int maxTria) 
{
    unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

    // prevent overrunning of array boundary
    if (i >= maxTria)
      return;

    // get all three triangle vertices
    float4 v0 = pos[3*i];
    float4 v1 = pos[3*i+1];
    float4 v2 = pos[3*i+2];

    // compute edge lengths
    float a = length( v0 - v1);
    float b = length( v0 - v2);
    float c = length( v1 - v2);

    // compute area (Heron's formula)
    float rad = ( a + b + c) * ( a + b - c) * ( b + c - a) * ( c + a - b);
    // make sure radicand is not negative
    rad = rad > 0.0f ? rad : 0.0f;
    area[i] = 0.25f * sqrt( rad);
}

///*
// * Utility functions.
// */
//dim3 Grid(const uint size, const int threadsPerBlock) {
//    //TODO: remove hardcoded hardware capabilities :(
//    // see: http://code.google.com/p/thrust/source/browse/thrust/detail/backend/cuda/arch.inl
//    //   and http://code.google.com/p/thrust/source/browse/thrust/detail/backend/cuda/detail/safe_scan.inl
//    //   for refactoring.
//    // Get maximum grid size of CUDA device.
//    //CUdevice device;
//    //cuDeviceGet(&device, 0);
//    //CUdevprop deviceProps;
//    //cuDeviceGetProperties(&deviceProps, device);
//    //this->gridSize = dim3(deviceProps.maxGridSize[0],
//    //	deviceProps.maxGridSize[1],
//    //	deviceProps.maxGridSize[2]);
//    const dim3 maxGridSize(65535, 65535, 0);
//    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
//    dim3 grid(blocksPerGrid, 1, 1);
//    // Test if grid needs to be extended to 2D.
//    while (grid.x > maxGridSize.x) {
//        grid.x /= 2;
//        grid.y *= 2;
//    }
//    return grid;
//}

__global__
void SetVol_kernel(float* vol)
{
    volumeTex = vol;
}

__global__
void SetNeighborAtomVol_kernel(int* vol)
{
    neighborAtomTex = vol;
}

/*
 * Wrappers.
 */
extern "C"
cudaError BindVolumeTexture(float* textureArray) {
    //return cudaBindTextureToArray(volumeTex, textureArray);
    //return cudaBindTexture(0, volumeTex, textureArray, cudaCreateChannelDesc<float>());
    
    SetVol_kernel<<<Grid(1, 1), 1>>>(textureArray);
    return cudaGetLastError();
}

extern "C"
cudaError BindNeighborAtomTexture(int* textureArray) {
    SetNeighborAtomVol_kernel<<<Grid(1, 1), 1>>>(textureArray);
    return cudaGetLastError();
}

extern "C"
cudaError UnbindVolumeTexture()
{
    return cudaSuccess;//cudaUnbindTexture(volumeTex);
}

extern "C"
cudaError BindAOVolumeTexture(cudaArray* textureArray)
{
    // set texture parameters
    aoVolumeTex.normalized = 1;
    aoVolumeTex.filterMode = cudaFilterModeLinear;
    aoVolumeTex.addressMode[0] = cudaAddressModeClamp;
    aoVolumeTex.addressMode[1] = cudaAddressModeClamp;
    aoVolumeTex.addressMode[2] = cudaAddressModeClamp;
    // bind array to 3D texture
    return cudaBindTextureToArray(aoVolumeTex, textureArray, cudaCreateChannelDesc<float>());
}

extern "C"
cudaError UnbindAOVolumeTexture()
{
    return cudaUnbindTexture(aoVolumeTex);
}

extern "C"
cudaError ClassifyCubes(uint* cubeStates, float thresholdValue, uint cubeCount)
{
    const int threads = 256;
    ClassifyCubes_kernel<<<Grid(cubeCount, threads), threads>>>(cubeStates, thresholdValue, cubeCount);
    return cudaGetLastError();
}

extern "C"
cudaError ScanCubes(uint* cubeOffsets, uint* cubeStates, uint cubeCount)
{
    uint* cubeStatesLast = cubeStates + cubeCount;
    thrust::exclusive_scan(thrust::device_ptr<uint>(cubeStates), 
        thrust::device_ptr<uint>(cubeStatesLast),
        thrust::device_ptr<uint>(cubeOffsets));
    return cudaGetLastError();
}

extern "C"
cudaError CompactCubes(uint* cubeMap, uint* cubeOffsets, uint* cubeStates, uint cubeCount)
{
    const int threads = 256;
    CompactCubes_kernel<<<Grid(cubeCount, threads), threads>>>(cubeMap, cubeOffsets, cubeStates, cubeCount);
    return cudaGetLastError();
}

extern "C"
cudaError ClassifyTetrahedronsInACube(uint* verticesPerTetrahedron, uint* cubeMap, float thresholdValue, uint activeCubeCount)
{
    const int threads = 64;
    ClassifyTetrahedronsInACube_kernel<<<Grid(activeCubeCount, threads), threads>>>(verticesPerTetrahedron, cubeMap, thresholdValue, activeCubeCount);
    return cudaGetLastError();
}

extern "C"
cudaError ScanTetrahedrons(uint* vertexOffsets, uint* verticesPerTetrahedron, uint tetrahedronCount)
{
    uint* verticesPerTetrahedronLast = verticesPerTetrahedron + tetrahedronCount;
    thrust::exclusive_scan(thrust::device_ptr<uint>(verticesPerTetrahedron), 
        thrust::device_ptr<uint>(verticesPerTetrahedronLast),
        thrust::device_ptr<uint>(vertexOffsets));
    return cudaGetLastError();
}

extern "C"
cudaError GenerateTriangles(float4* vertices, int* neighborAtoms, float4* normals, float translateX, float translateY, float translateZ,
    float scaleX, float scaleY, float scaleZ, uint* vertexOffsets, uint* cubeMap, float thresholdValue, uint tetrahedronCount)
{
    const int threads = GT_THREADS;
    GenerateTriangles_kernel<<<Grid(tetrahedronCount, threads), threads>>>(vertices, neighborAtoms, normals, translateX, translateY, translateZ, 
        scaleX, scaleY, scaleZ, vertexOffsets, cubeMap, thresholdValue, tetrahedronCount);
    return cudaGetLastError();
}

extern "C"
cudaError MeshReset(uint* eqList, uint* refList, uint tetrahedronCount, uint* vertexOffsets, float* triangleAO)
{
    const int threads = 128;
    MeshReset_kernel<<<Grid(tetrahedronCount, threads), threads>>>(eqList, refList, tetrahedronCount, vertexOffsets, triangleAO);
    return cudaGetLastError();
}

extern "C"
cudaError MeshScan(float4* vertices, float4* normals, uint* vertexOffsets, float* triangleAO, float aoThreshold, uint* eqList, uint* refList, bool* modified, uint* cubeStates, uint* cubeOffsets, uint* cubeMap, float thresholdValue, uint tetrahedronCount)
{
    const int threads = 128;
    MeshScan_kernel<<<Grid(tetrahedronCount, threads), threads>>>(vertices, normals, vertexOffsets, triangleAO, aoThreshold, eqList, refList, modified, cubeStates, cubeOffsets, cubeMap, thresholdValue, tetrahedronCount);
    return cudaGetLastError();
}

extern "C"
cudaError MeshAnalysis(uint* eqList, uint* refList, uint tetrahedronCount)
{
    const int threads = 128;
    MeshAnalysis_kernel<<<Grid(tetrahedronCount, threads), threads>>>(eqList, refList, tetrahedronCount);
    return cudaGetLastError();
}

extern "C"
cudaError MeshLabeling(uint* eqList, uint* refList, uint tetrahedronCount)
{
    const int threads = 128;
    MeshLabeling_kernel<<<Grid(tetrahedronCount, threads), threads>>>(eqList, refList, tetrahedronCount);
    return cudaGetLastError();
}

extern "C"
cudaError CentroidMap(uint* vertexLabels, uint* vertexOffsets, uint* verticesPerTetrahedron, uint* eqList, uint tetrahedronCount)
{
    const int threads = 128;
    CentroidMap_kernel<<<Grid(tetrahedronCount, threads), threads>>>(vertexLabels, vertexOffsets, verticesPerTetrahedron, eqList, tetrahedronCount);
    return cudaGetLastError();
}

extern "C"
cudaError CentroidFinalize(float4* centroids, float4* centroidSums, uint* centroidCounts, uint centroidCount)
{
    const int threads = 128;
    CentroidFinalize_kernel<<<Grid(centroidCount, threads), threads>>>(centroids, centroidSums, centroidCounts, centroidCount);
    return cudaGetLastError();
}

extern "C"
cudaError ColorizeByCentroid(float4* colors, float4* centroidColors, uint* centroidLabels, uint centroidCount, uint* vertexLabels, uint vertexCount)
{
    const int threads = 128;
    ColorizeByCentroid_kernel<<<Grid(vertexCount, threads), threads>>>(colors, centroidColors, centroidLabels, centroidCount, vertexLabels, vertexCount);
    return cudaGetLastError();
}

extern "C"
cudaError ColorizeByAO(float4* colors, float* triangleAO, float aoThreashold, uint vertexCount)
{
    const int threads = 128;
    ColorizeByAO_kernel<<<Grid(vertexCount, threads), threads>>>(colors, triangleAO, aoThreashold, vertexCount);
    return cudaGetLastError();
}

extern "C"
cudaError ComputeSurfaceArea(float4* verts, float* areas, unsigned int triaCount)
{
    // do nothing for zero triangles
    if (triaCount <= 0) 
        return cudaSuccess;

    // compute area for each triangle
    int threads = 256;
    dim3 grid(int(ceil(float(triaCount) / float(threads))), 1, 1);

    // get around maximum grid size of 65535 in each dimension
    while(grid.x > 65535) {
        grid.x = (unsigned int) (ceil(float(grid.x) / 2.0f));
        grid.y *= 2;
    }
    while(grid.y > 65535) {
        grid.y = (unsigned int) (ceil(float(grid.y) / 2.0f));
        grid.z *= 2;
    }

    ComputeTriangleAreas_kernel<<<grid, threads>>>(verts, areas, triaCount);
    return cudaGetLastError();
}

__global__
void ComputeCentroidArea_kernel(float* centroidAreas, uint2* featureStartEnd, float* triaAreas, uint* vertexLabels, uint* centroidLabels, uint triaCount, uint centroidCount)
{
    const uint id = Index();
    // Prevent overrunning of array
    if (id >= triaCount) {
        return;
    }

    uint labelId = vertexLabels[id*3];
    for (int i = 0; i < centroidCount; ++i) {
        if ( labelId == centroidLabels[i]) {
            atomicAdd( &centroidAreas[i], triaAreas[id]);
            
            if( id > 0 ) {
                // check if the label of the previous triangle is different
                if( vertexLabels[(id-1)*3] != labelId ) {
                    featureStartEnd[i].x = id;
                    featureStartEnd[i-1].y = id-1;
                }
            } else {
                featureStartEnd[0].x = 0;
                featureStartEnd[centroidCount-1].y = triaCount-1;
            }

        }
    }
}

extern "C"
cudaError ComputeCentroidArea(float* centroidAreas, uint2* featureStartEnd, float* triaAreas, uint* vertexLabels, uint* centroidLabels, uint triaCount, uint centroidCount)
{
    // do nothing for zero triangles
    if (triaCount <= 0) 
        return cudaSuccess;
    // do nothing for zero centroids
    if (centroidCount <= 0) 
        return cudaSuccess;

    // compute area for each triangle
    int threads = 256;
    dim3 grid(int(ceil(float(triaCount) / float(threads))), 1, 1);

    // get around maximum grid size of 65535 in each dimension
    while(grid.x > 65535) {
        grid.x = (unsigned int) (ceil(float(grid.x) / 2.0f));
        grid.y *= 2;
    }
    while(grid.y > 65535) {
        grid.y = (unsigned int) (ceil(float(grid.y) / 2.0f));
        grid.z *= 2;
    }

    cudaError err = cudaMemset( centroidAreas, 0, centroidCount * sizeof(float));
    if( err != cudaSuccess ) return err;

    ComputeCentroidArea_kernel<<<grid, threads>>>(centroidAreas, featureStartEnd, triaAreas, vertexLabels, centroidLabels, triaCount, centroidCount);
    return cudaGetLastError();
}

__global__ 
void ComputeTriangleAO_kernel(float4 *vertices, float4* normals, float *ao, unsigned int maxTria) 
{
    unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

    // prevent overrunning of array boundary
    if (i >= maxTria)
      return;

    // get all three triangle vertices and normals
    float3 v0 = make_float3(vertices[3*i+0].x, vertices[3*i+0].y, vertices[3*i+0].z);
    float3 v1 = make_float3(vertices[3*i+1].x, vertices[3*i+1].y, vertices[3*i+1].z);
    float3 v2 = make_float3(vertices[3*i+2].x, vertices[3*i+2].y, vertices[3*i+2].z);
    float3 n0 = make_float3(normals[3*i+0].x, normals[3*i+0].y, normals[3*i+0].z);
    float3 n1 = make_float3(normals[3*i+1].x, normals[3*i+1].y, normals[3*i+1].z);
    float3 n2 = make_float3(normals[3*i+2].x, normals[3*i+2].y, normals[3*i+2].z);
    ao[i] = (AOValue(v0, n0) + AOValue(v1, n1) + AOValue(v2, n2)) / 3.0f;
}

extern "C"
cudaError ComputeTriangleAO(float4* verts, float4* normals, float* ao, unsigned int triaCount)
{
    // do nothing for zero triangles
    if (triaCount <= 0) 
        return cudaSuccess;

    // compute area for each triangle
    int threads = 256;
    dim3 grid(int(ceil(float(triaCount) / float(threads))), 1, 1);

    // get around maximum grid size of 65535 in each dimension
    while(grid.x > 65535) {
        grid.x = (unsigned int) (ceil(float(grid.x) / 2.0f));
        grid.y *= 2;
    }
    while(grid.y > 65535) {
        grid.y = (unsigned int) (ceil(float(grid.y) / 2.0f));
        grid.z *= 2;
    }

    ComputeTriangleAO_kernel<<<grid, threads>>>(verts, normals, ao, triaCount);
    return cudaGetLastError();
}

__global__
void RemoveSmallSegments_kernel(float* centroidAreas, float* triangleAO, uint* vertexLabels, uint* centroidLabels, float areaThreshold, float aoThreshold, 
    uint triaCount, uint centroidCount, bool* segmentsRemoved)
{
    const uint id = Index();
    // Prevent overrunning of array
    if (id >= triaCount) {
        return;
    }

    for (int i = 0; i < centroidCount; ++i) {
        if (vertexLabels[id*3] == centroidLabels[i]) {
            // check centroid area
            if( centroidAreas[i] < areaThreshold ) {
                // "flip" ambient occlusion factor
                if( triangleAO[id] <= aoThreshold ) {
                    triangleAO[id] = aoThreshold + 1.0f;
                } else {
                    triangleAO[id] = 0.0f;
                }
                // mark as modified
                *segmentsRemoved = true;
            }
        }
    }
}

extern "C"
cudaError RemoveSmallSegments(float* centroidAreas, float* triangleAO, uint* vertexLabels, uint* centroidLabels, float areaThreshold, float aoThreshold, 
    uint triaCount, uint centroidCount, bool* segmentsRemoved)
{
    // do nothing for zero triangles
    if (triaCount <= 0) 
        return cudaSuccess;
    // do nothing for zero centroids
    if (centroidCount <= 0) 
        return cudaSuccess;

    // compute area for each triangle
    int threads = 256;
    dim3 grid(int(ceil(float(triaCount) / float(threads))), 1, 1);

    // get around maximum grid size of 65535 in each dimension
    while(grid.x > 65535) {
        grid.x = (unsigned int) (ceil(float(grid.x) / 2.0f));
        grid.y *= 2;
    }
    while(grid.y > 65535) {
        grid.y = (unsigned int) (ceil(float(grid.y) / 2.0f));
        grid.z *= 2;
    }
    
    RemoveSmallSegments_kernel<<<grid, threads>>>(centroidAreas, triangleAO, vertexLabels, centroidLabels, areaThreshold, aoThreshold, triaCount, centroidCount, segmentsRemoved);
    return cudaGetLastError();
}


__global__
void ComputeTriangleBBox_kernel(float* fBBoxMinX, float* fBBoxMinY, float* fBBoxMinZ, 
        float* fBBoxMaxX, float* fBBoxMaxY, float* fBBoxMaxZ, uint* triangleLabels, 
        float4* verts, uint* vertexLabels, uint triaCount)
{
    const uint id = Index();
    // Prevent overrunning of array
    if (id >= triaCount) {
        return;
    }
    
    // get min/max x/y/z of all three triangle vertices (i.e. triangle bbox)
    float4 v0 = verts[3*id];
    float4 v1 = verts[3*id+1];
    float4 v2 = verts[3*id+2];
    float3 minVec = make_float3(
        min( min( v0.x, v1.x), v2.x),
        min( min( v0.y, v1.y), v2.y),
        min( min( v0.z, v1.z), v2.z));
    float3 maxVec = make_float3(
        max( max( v0.x, v1.x), v2.x),
        max( max( v0.y, v1.y), v2.y),
        max( max( v0.z, v1.z), v2.z));
    // write result
    fBBoxMinX[id] = minVec.x;
    fBBoxMinY[id] = minVec.y;
    fBBoxMinZ[id] = minVec.z;
    fBBoxMaxX[id] = maxVec.x;
    fBBoxMaxY[id] = maxVec.y;
    fBBoxMaxZ[id] = maxVec.z;
    triangleLabels[id] = vertexLabels[3*id];
}

extern "C"
cudaError ComputeTriangleBBox(float* fBBoxMinX, float* fBBoxMinY, float* fBBoxMinZ, 
        float* fBBoxMaxX, float* fBBoxMaxY, float* fBBoxMaxZ, uint* triangleLabels,
        float4* verts, uint* vertexLabels, uint triaCount)
{
    // do nothing for zero triangles
    if (triaCount <= 0) 
        return cudaSuccess;

    // compute area for each triangle
    int threads = 256;
    dim3 grid(int(ceil(float(triaCount) / float(threads))), 1, 1);

    // get around maximum grid size of 65535 in each dimension
    while(grid.x > 65535) {
        grid.x = (unsigned int) (ceil(float(grid.x) / 2.0f));
        grid.y *= 2;
    }
    while(grid.y > 65535) {
        grid.y = (unsigned int) (ceil(float(grid.y) / 2.0f));
        grid.z *= 2;
    }
    
    ComputeTriangleBBox_kernel<<<grid, threads>>>( fBBoxMinX, fBBoxMinY, fBBoxMinZ, 
        fBBoxMaxX, fBBoxMaxY, fBBoxMaxZ, triangleLabels, verts, vertexLabels, triaCount);
    return cudaGetLastError();
}


__global__
void WritePrevTetraLabel_kernel( int2* labelPair, uint* cubeStatesOld, uint* cubeOffsetsOld, //uint* cubeMapOld,
    uint* verticesPerTetrahedronOld, uint* eqListOld, uint* cubeMap, uint* verticesPerTetrahedron, 
    uint* eqList, uint tetrahedronCount)
{
    const uint activeTetraIdx = Index();
    const uint activeCubeIdx = activeTetraIdx / 6;
    // Prevent non-power of two writes.
    if( activeTetraIdx >= tetrahedronCount ) {
        return;
    }
    const uint globalCubeIdx = cubeMap[activeCubeIdx];
    const uint tetraIdx = activeTetraIdx % 6;
    //const uint globalTetraIdx = globalCubeIdx * 6 + tetraIdx;
    
    uint cnt1 = verticesPerTetrahedron[activeTetraIdx];

    // check if cube was active in previous time step
    if( cubeStatesOld[globalCubeIdx] > 0 && cnt1 > 0 ) {
        // get the index of the current cube in the previous time step
        uint activeCubeIdxOld = cubeOffsetsOld[globalCubeIdx];
        // Test if the mapped global indes is equal --> TEST PASSED!
        //if( cubeMapOld[activeCubeIdxOld] != globalCubeIdx ) {
        //    printf( "Greetings from thread %i!\n", activeTetraIdx);
        //}
        uint cnt0 = verticesPerTetrahedronOld[activeCubeIdxOld * 6 + tetraIdx];
        if( cnt0 > 0 ) {
            labelPair[activeTetraIdx].x = eqList[activeTetraIdx];
            labelPair[activeTetraIdx].y = eqListOld[activeCubeIdxOld * 6 + tetraIdx];
        } else {
            labelPair[activeTetraIdx].x = -1;
            labelPair[activeTetraIdx].y = -1;
        }
    } else {
        // cube was not active previously
        labelPair[activeTetraIdx].x = -1;
        labelPair[activeTetraIdx].y = -1;
    }
}

extern "C"
cudaError WritePrevTetraLabel( int2* labelPair, uint* cubeStatesOld, uint* cubeOffsetsOld, //uint* cubeMapOld,
    uint* verticesPerTetrahedronOld, uint* eqListOld, uint* cubeMap, uint* verticesPerTetrahedron, 
    uint* eqList, uint tetrahedronCount)
{
    const int threads = GT_THREADS;
    WritePrevTetraLabel_kernel<<<Grid(tetrahedronCount, threads), threads>>>( labelPair, cubeStatesOld, cubeOffsetsOld, //cubeMapOld,
        verticesPerTetrahedronOld, eqListOld, cubeMap, verticesPerTetrahedron, eqList, tetrahedronCount);
    return cudaGetLastError();
}

__global__
void WriteTriangleVertexIndexList_kernel( uint* featureVertexIdx, uint* featureVertexCnt, uint* featureVertexStartIdx, uint* featureVertexIdxOut, uint triaVertexCnt, uint vertexCnt)
{
    // vertex index
    const uint vertexIdx = Index();
    // check bounds
    if( vertexIdx >= vertexCnt ) {
        return;
    }
    
    // the number of duplications for vertex 'vertexIdx'
    const uint fvCnt = featureVertexCnt[vertexIdx];
    // the start index of the duplicates for vertex 'vertexIdx'
    const uint fvStartIdx = featureVertexStartIdx[vertexIdx];

    // temp variable for original vertex index
    uint origIdx;
    // loop over all duplicates
    for( uint i = 0; i < fvCnt; i++ ) {
        // get original vertex index
        origIdx = featureVertexIdx[fvStartIdx + i];
        // write new vertex index
        featureVertexIdxOut[origIdx] = vertexIdx;
    }

}

extern "C"
cudaError WriteTriangleVertexIndexList( uint* featureVertexIdx, uint* featureVertexCnt, uint* featureVertexStartIdx, uint* featureVertexIdxOut, uint triaVertexCnt, uint vertexCnt) {
    const int threads = GT_THREADS;
    WriteTriangleVertexIndexList_kernel<<<Grid(vertexCnt, threads), threads>>>( featureVertexIdx, featureVertexCnt, featureVertexStartIdx, featureVertexIdxOut, triaVertexCnt, vertexCnt);
    return cudaGetLastError();
}

__global__
void WriteTriangleEdgeList_kernel( uint* featureVertexIdxOut, uint triaCnt, uint2 *featureEdges)
{
    // feature index
    const uint triangleIdx = Index() * 3;
    // check bounds
    if( triangleIdx >= (triaCnt * 3)) {
        return;
    }
    
    // get the three vertex indices of the current triangle
    uint idx0 = featureVertexIdxOut[triangleIdx];
    uint idx1 = featureVertexIdxOut[triangleIdx+1];
    uint idx2 = featureVertexIdxOut[triangleIdx+2];
    
    // write three edges of the current feature triangle (vertex indices sorted)
    featureEdges[triangleIdx]   = make_uint2( min( idx0, idx1), max( idx0, idx1));
    featureEdges[triangleIdx+1] = make_uint2( min( idx0, idx2), max( idx0, idx2));
    featureEdges[triangleIdx+2] = make_uint2( min( idx1, idx2), max( idx1, idx2));
}

extern "C"
cudaError WriteTriangleEdgeList( uint* featureVertexIdxOut, uint triaCnt, uint2 *featureEdges) {
    const int threads = GT_THREADS;
    WriteTriangleEdgeList_kernel<<<Grid(triaCnt, threads), threads>>>( featureVertexIdxOut, triaCnt, featureEdges);
    return cudaGetLastError();
}

#endif
