//
// VecField3DCUDA.cu
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//

#include "helper_cuda.h"
#include "helper_functions.h"
#include "helper_math.h"


// Shut up eclipse syntax error highlighting
#ifdef __CDT_PARSER__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __host__
#endif

// Toggle performance measurement and respective messages
#define USE_TIMER

__constant__ __device__ uint3 dim;
__constant__ __device__ float3 org;
__constant__ __device__ float3 maxCoord;
__constant__ __device__ float3 spacing;
__constant__ __device__ float streamlinesStep;
__constant__ __device__ uint streamlinesStepCnt;
__constant__ __device__ uint nPos;
__constant__ __device__ uint maxStackSize;

/* isValidGridPos_D */
inline __device__
bool isValidGridPos_D(float3 pos) {
    return (pos.x < maxCoord.x)&&
           (pos.y < maxCoord.y)&&
           (pos.z < maxCoord.z)&&
           (pos.x >= org.x)&&
           (pos.y >= org.y)&&
           (pos.z >= org.z);
}

/* sampleVecFieldLin_D */
inline __device__
float3 sampleVecFieldLin_D(float3 v0, float3 v1, float alpha) {
    return v0+alpha*(v1-v0);
}

/* sampleVecFieldBilin_D */
inline __device__
float3 sampleVecFieldBilin_D(float3 v0, float3 v1, float3 v2, float3 v3,
        float alpha, float beta) {
    return sampleVecFieldLin_D(sampleVecFieldLin_D(v0, v1, alpha), 
    sampleVecFieldLin_D(v2, v3, alpha), beta);    
}

/* sampleVecFieldTrilin_D */
inline __device__
float3 sampleVecFieldTrilin_D(float3 v[8], float alpha, float beta,
        float gamma) {
    return sampleVecFieldLin_D(
        sampleVecFieldBilin_D(v[0], v[1], v[2], v[3], alpha, beta), 
        sampleVecFieldBilin_D(v[4], v[5], v[6], v[7], alpha, beta), gamma);
 /*   float3 a, b, c, d, e, f, g, h;
    a = v[0];
    b = v[1] - v[0];
    c = v[2] - v[0];
    d = v[3] - v[1] - v[2] + v[0];
    e = v[4] - v[0];
    f = v[5] - v[1] - v[4] + v[0];
    g = v[6] - v[2] - v[4] + v[0];
    h = v[7] - v[3] - v[5] - v[6] + v[1] + v[2] + v[4] - v[0];
    return a + b*alpha + c*beta + d*alpha*beta + e*gamma + f*alpha*gamma
            + g*beta*gamma + h*alpha*beta*gamma;    */
}

/* sampleVecFieldAtTrilinNorm_D */
inline __device__
float3 sampleVecFieldAtTrilinNorm_D(float3 pos, const float3 *vecField_D) {
    float3 f;
    uint3 c;
    
    // Get id of the cell containing the given position and interpolation
    // coefficients
    
    f.x = (pos.x-org.x)/spacing.x;
    f.y = (pos.y-org.y)/spacing.y;
    f.z = (pos.z-org.z)/spacing.z;
    c.x = (uint)(f.x);
    c.y = (uint)(f.y);
    c.z = (uint)(f.z);
    f.x = f.x-(float)c.x; // alpha
    f.y = f.y-(float)c.y; // beta
    f.z = f.z-(float)c.z; // gamma
    
    // Get vector field at corners of current cell
    float3 v[8];
    v[0] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+0))+c.x+0]);
    v[1] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+0))+c.x+1]);
    v[2] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+1))+c.x+0]);
    v[3] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+1))+c.x+1]);
    v[4] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+0))+c.x+0]);
    v[5] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+0))+c.x+1]);
    v[6] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+1))+c.x+0]);
    v[7] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+1))+c.x+1]);
    
    return sampleVecFieldTrilin_D(v, f.x, f.y, f.z);   
}

/* UpdatePositionRK4_D */
__global__
void UpdatePositionRK4_D(const float3 *vecField_D, float3 *pos_D) {

    // Get thread idx
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nPos) return;

    //float3 posAlt = pos_D[idx];
    
    float3 x0, x1, x2, x3, v0, v1, v2, v3;
    v0 = make_float3(0.0, 0.0, 0.0);
    v1 = make_float3(0.0, 0.0, 0.0);
    v2 = make_float3(0.0, 0.0, 0.0);
    v3 = make_float3(0.0, 0.0, 0.0);
    
    x0 = pos_D[idx];
    v0 = normalize(sampleVecFieldAtTrilinNorm_D(x0, vecField_D));
    v0 *= streamlinesStep;

    x1 = x0 + 0.5*v0;
    if(isValidGridPos_D(x1)) {
        v1 = normalize(sampleVecFieldAtTrilinNorm_D(x1, vecField_D));
        v1 *= streamlinesStep;
    }

    x2 = x0 + 0.5f*v1;
    if(isValidGridPos_D(x2)) {
        v2 = normalize(sampleVecFieldAtTrilinNorm_D(x2, vecField_D));
        v2 *= streamlinesStep;
    }

    x3 = x0 + v2;
    if(isValidGridPos_D(x3)) {
        v3 = normalize(sampleVecFieldAtTrilinNorm_D(x3, vecField_D));
        v3 *= streamlinesStep;
    }

    x0 += (1.0f/6.0f)*(v0+2.0f*v1+2.0f*v2+v3);

    if(isValidGridPos_D(x0)) {
        pos_D[idx] = x0;
    }
    
    /*//pos_D[idx] = (1.0f/6.0f)*(v0+2.0f*v1+2.0f*v2+v3); // DEBUG
    //pos_D[idx] = v0; // DEBUG
    float3 f;
    uint3 c;
    f.x = (posAlt.x-org.x)/spacing.x;
    f.y = (posAlt.y-org.y)/spacing.y;
    f.z = (posAlt.z-org.z)/spacing.z;
    c.x = (uint)(f.x);
    c.y = (uint)(f.y);
    c.z = (uint)(f.z);
    f.x = f.x-(float)c.x; // alpha
    f.y = f.y-(float)c.y; // beta
    f.z = f.z-(float)c.z; // gamma
        // Get vector field at corners of current cell
    float3 v[8];
    v[0] = vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+0))+c.x+0];
    v[1] = vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+0))+c.x+1];
    v[2] = vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+1))+c.x+0];
    v[3] = vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+1))+c.x+1];
    v[4] = vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+0))+c.x+0];
    v[5] = vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+0))+c.x+1];
    v[6] = vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+1))+c.x+0];
    v[7] = vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+1))+c.x+1];
    float3 v_test = sampleVecFieldAtTrilinNorm_D(posAlt, vecField_D);
    //pos_D[idx] = make_float3(f.x, f.y, f.z); // DEBUG
    //pos_D[idx] = make_float3((float)c.x, (float)c.y, (float)c.z); // DEBUG
    //pos_D[idx] = make_float3(org.x, org.y, org.z); // DEBUG
    //pos_D[idx] = make_float3(spacing.x, spacing.y, spacing.z); // DEBUG
    //pos_D[idx] = make_float3(v[7].x, v[7].y, v[7].z); // DEBUG
    //pos_D[idx] = make_float3(v_test.x, v_test.y, v_test.z); // DEBUG*/
}


/* UpdatePositionRK4_D */
__global__
void UpdatePositionBackwardRK4_D(const float3 *vecField_D, float3 *pos_D) {

    // Get thread idx
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nPos) return;

    //float3 posAlt = pos_D[idx];
    
    float3 x0, x1, x2, x3, v0, v1, v2, v3;
    v0 = make_float3(0.0, 0.0, 0.0);
    v1 = make_float3(0.0, 0.0, 0.0);
    v2 = make_float3(0.0, 0.0, 0.0);
    v3 = make_float3(0.0, 0.0, 0.0);
    
    x0 = pos_D[idx];
    v0 = normalize(sampleVecFieldAtTrilinNorm_D(x0, vecField_D));
    v0 *= streamlinesStep;

    x1 = x0 - 0.5*v0;
    if(isValidGridPos_D(x1)) {
        v1 = normalize(sampleVecFieldAtTrilinNorm_D(x1, vecField_D));
        v1 *= streamlinesStep;
    }

    x2 = x0 - 0.5f*v1;
    if(isValidGridPos_D(x2)) {
        v2 = normalize(sampleVecFieldAtTrilinNorm_D(x2, vecField_D));
        v2 *= streamlinesStep;
    }

    x3 = x0 - v2;
    if(isValidGridPos_D(x3)) {
        v3 = normalize(sampleVecFieldAtTrilinNorm_D(x3, vecField_D));
        v3 *= streamlinesStep;
    }

    x0 -= (1.0f/6.0f)*(v0+2.0f*v1+2.0f*v2+v3);

    if(isValidGridPos_D(x0)) {
        pos_D[idx] = x0;
    }
    
    /*//pos_D[idx] = (1.0f/6.0f)*(v0+2.0f*v1+2.0f*v2+v3); // DEBUG
    //pos_D[idx] = v0; // DEBUG
    float3 f;
    uint3 c;
    f.x = (posAlt.x-org.x)/spacing.x;
    f.y = (posAlt.y-org.y)/spacing.y;
    f.z = (posAlt.z-org.z)/spacing.z;
    c.x = (uint)(f.x);
    c.y = (uint)(f.y);
    c.z = (uint)(f.z);
    f.x = f.x-(float)c.x; // alpha
    f.y = f.y-(float)c.y; // beta
    f.z = f.z-(float)c.z; // gamma
        // Get vector field at corners of current cell
    float3 v[8];
    v[0] = vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+0))+c.x+0];
    v[1] = vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+0))+c.x+1];
    v[2] = vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+1))+c.x+0];
    v[3] = vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+1))+c.x+1];
    v[4] = vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+0))+c.x+0];
    v[5] = vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+0))+c.x+1];
    v[6] = vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+1))+c.x+0];
    v[7] = vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+1))+c.x+1];
    float3 v_test = sampleVecFieldAtTrilinNorm_D(posAlt, vecField_D);
    //pos_D[idx] = make_float3(f.x, f.y, f.z); // DEBUG
    //pos_D[idx] = make_float3((float)c.x, (float)c.y, (float)c.z); // DEBUG
    //pos_D[idx] = make_float3(org.x, org.y, org.z); // DEBUG
    //pos_D[idx] = make_float3(spacing.x, spacing.y, spacing.z); // DEBUG
    //pos_D[idx] = make_float3(v[7].x, v[7].y, v[7].z); // DEBUG
    //pos_D[idx] = make_float3(v_test.x, v_test.y, v_test.z); // DEBUG*/
}


/* isFieldVanishingInCell */
inline __device__
bool isFieldVanishingInCell_D(float3 v[8]) {
    return (!(((v[0].x > 0)&&(v[1].x > 0)&&(v[2].x > 0)&&
            (v[3].x > 0)&&(v[4].x > 0)&&(v[5].x > 0)&&
            (v[6].x > 0)&&(v[7].x > 0))||
            ((v[0].x < 0)&&(v[1].x < 0)&&(v[2].x < 0)&&
            (v[3].x < 0)&&(v[4].x < 0)&&(v[5].x < 0)&&
            (v[6].x < 0)&&(v[7].x < 0))||
            ((v[0].y > 0)&&(v[1].y > 0)&&(v[2].y > 0)&&
            (v[3].y > 0)&&(v[4].y > 0)&&(v[5].y > 0)&&
            (v[6].y > 0)&&(v[7].y > 0))||
            ((v[0].y < 0)&&(v[1].y < 0)&&(v[2].y < 0)&&
            (v[3].y < 0)&&(v[4].y < 0)&&(v[5].y < 0)&&
            (v[6].y < 0)&&(v[7].y < 0))||
            ((v[0].z > 0)&&(v[1].z > 0)&&(v[2].z > 0)&&
            (v[3].z > 0)&&(v[4].z > 0)&&(v[5].z > 0)&&
            (v[6].z > 0)&&(v[7].z > 0))||
            ((v[0].z < 0)&&(v[1].z < 0)&&(v[2].z < 0)&&
            (v[3].z < 0)&&(v[4].z < 0)&&(v[5].z < 0)&&
            (v[6].z < 0)&&(v[7].z < 0))));
}

/* calcCellCoords_D */
__global__
void calcCellCoords_D(const float3 *vecField_D,     // dim.x*dim.y*dim.z
                      float3 *cellCoords_D) {

    // Get thread index
    uint nCells = (dim.x-1)*(dim.y-1)*(dim.z-1);
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nCells) return;

    // Get coordinates of the lower/left/back corner
    uint3 c;
    c.x = idx%(dim.x-1);
    c.y = (idx/(dim.x-1))%(dim.y-1);
    c.z = (idx/(dim.x-1))/(dim.y-1);


    // Init stack

    const uint maxStackSize = 6;
    uint currStackPos = 0;
    uint currSubCell[maxStackSize]; // 0 ... 7
    currSubCell[0] = 0;
    
    float cellSize = 1.0f;

    float3 stackCorners[maxStackSize*8];
    stackCorners[0] = make_float3(0.0, 0.0, 0.0);
    stackCorners[1] = make_float3(1.0, 0.0, 0.0);
    stackCorners[2] = make_float3(0.0, 1.0, 0.0);
    stackCorners[3] = make_float3(1.0, 1.0, 0.0);
    stackCorners[4] = make_float3(0.0, 0.0, 1.0);
    stackCorners[5] = make_float3(1.0, 0.0, 1.0);
    stackCorners[6] = make_float3(0.0, 1.0, 1.0);
    stackCorners[7] = make_float3(1.0, 1.0, 1.0);

    float3 stackV[8*maxStackSize]; // Vector field at corners of current (sub-)cell
    stackV[0] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+0))+c.x+0]);
    stackV[1] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+0))+c.x+1]);
    stackV[2] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+1))+c.x+0]);
    stackV[3] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+1))+c.x+1]);
    stackV[4] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+0))+c.x+0]);
    stackV[5] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+0))+c.x+1]);
    stackV[6] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+1))+c.x+0]);
    stackV[7] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+1))+c.x+1]);
    
    float3 stackSubgrid[19*maxStackSize];
    
    // Init cell coords
    cellCoords_D[idx].x = -1.0;
    cellCoords_D[idx].y = -1.0;
    cellCoords_D[idx].z = -1.0;
    
    bool done = false;
    while(!done) {
        if((isFieldVanishingInCell_D(&stackV[currStackPos*8]))&&(currSubCell[currStackPos] < 8)) {
            
            if(currStackPos < maxStackSize-1) {
                
                if(currSubCell[currStackPos] == 0) {
                    // Compute subgrid values and put them on the stack
                    // Edges
                    stackSubgrid[19*currStackPos+0] = sampleVecFieldLin_D(stackV[(currStackPos)*8+0], stackV[(currStackPos)*8+1], 0.5);
                    stackSubgrid[19*currStackPos+1] = sampleVecFieldLin_D(stackV[(currStackPos)*8+0], stackV[(currStackPos)*8+2], 0.5);
                    stackSubgrid[19*currStackPos+2] = sampleVecFieldLin_D(stackV[(currStackPos)*8+1], stackV[(currStackPos)*8+3], 0.5);
                    stackSubgrid[19*currStackPos+3] = sampleVecFieldLin_D(stackV[(currStackPos)*8+2], stackV[(currStackPos)*8+3], 0.5);
                    stackSubgrid[19*currStackPos+4] = sampleVecFieldLin_D(stackV[(currStackPos)*8+0], stackV[(currStackPos)*8+4], 0.5);
                    stackSubgrid[19*currStackPos+5] = sampleVecFieldLin_D(stackV[(currStackPos)*8+1], stackV[(currStackPos)*8+5], 0.5);
                    stackSubgrid[19*currStackPos+6] = sampleVecFieldLin_D(stackV[(currStackPos)*8+2], stackV[(currStackPos)*8+6], 0.5);
                    stackSubgrid[19*currStackPos+7] = sampleVecFieldLin_D(stackV[(currStackPos)*8+3], stackV[(currStackPos)*8+7], 0.5);
                    stackSubgrid[19*currStackPos+8] = sampleVecFieldLin_D(stackV[(currStackPos)*8+4], stackV[(currStackPos)*8+5], 0.5);
                    stackSubgrid[19*currStackPos+9] = sampleVecFieldLin_D(stackV[(currStackPos)*8+4], stackV[(currStackPos)*8+6], 0.5);
                    stackSubgrid[19*currStackPos+10] = sampleVecFieldLin_D(stackV[(currStackPos)*8+5], stackV[(currStackPos)*8+7], 0.5);
                    stackSubgrid[19*currStackPos+11] = sampleVecFieldLin_D(stackV[(currStackPos)*8+6], stackV[(currStackPos)*8+7], 0.5);
                    // Faces
                    // Back
                    stackSubgrid[19*currStackPos+12] = sampleVecFieldLin_D(stackSubgrid[19*currStackPos+0], stackSubgrid[19*currStackPos+3], 0.5);
                    // Front
                    stackSubgrid[19*currStackPos+13] = sampleVecFieldLin_D(stackSubgrid[19*currStackPos+9], stackSubgrid[19*currStackPos+10], 0.5);
                    // Bottom
                    stackSubgrid[19*currStackPos+14] = sampleVecFieldLin_D(stackSubgrid[19*currStackPos+4], stackSubgrid[19*currStackPos+5], 0.5);
                    // Top
                    stackSubgrid[19*currStackPos+15] = sampleVecFieldLin_D(stackSubgrid[19*currStackPos+6], stackSubgrid[19*currStackPos+7], 0.5);
                    // Left
                    stackSubgrid[19*currStackPos+16] = sampleVecFieldLin_D(stackSubgrid[19*currStackPos+9], stackSubgrid[19*currStackPos+1], 0.5);
                    // Right
                    stackSubgrid[19*currStackPos+17] = sampleVecFieldLin_D(stackSubgrid[19*currStackPos+5], stackSubgrid[19*currStackPos+7], 0.5);
                    // Center
                    stackSubgrid[19*currStackPos+18] = sampleVecFieldLin_D(stackSubgrid[19*currStackPos+12], stackSubgrid[19*currStackPos+13], 0.5);
                }
                
                // Increment stack
                currStackPos++;
                cellSize = cellSize*0.5;
                    
                // Bisect and put cell on stack
                if(currSubCell[currStackPos-1] == 0) { // left/down/back
            
                    // Set cell corners
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    // Sample vector field at cell corners
                    stackV[currStackPos*8+0] = stackV[(currStackPos-1)*8+0];
                    stackV[currStackPos*8+1] = stackSubgrid[19*(currStackPos-1)+0];
                    stackV[currStackPos*8+2] = stackSubgrid[19*(currStackPos-1)+1];
                    stackV[currStackPos*8+3] = stackSubgrid[19*(currStackPos-1)+12];
                    stackV[currStackPos*8+4] = stackSubgrid[19*(currStackPos-1)+4];
                    stackV[currStackPos*8+5] = stackSubgrid[19*(currStackPos-1)+14];
                    stackV[currStackPos*8+6] = stackSubgrid[19*(currStackPos-1)+16];
                    stackV[currStackPos*8+7] = stackSubgrid[19*(currStackPos-1)+18];
                }
                else if(currSubCell[currStackPos-1] == 1) { // right/down/back
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y,
                                                                 stackCorners[(currStackPos-1)*8+1].z);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+0] = stackSubgrid[19*(currStackPos-1)+0];
                    stackV[currStackPos*8+1] = stackV[(currStackPos-1)*8+1];
                    stackV[currStackPos*8+2] = stackSubgrid[19*(currStackPos-1)+12];
                    stackV[currStackPos*8+3] = stackSubgrid[19*(currStackPos-1)+2];
                    stackV[currStackPos*8+4] = stackSubgrid[19*(currStackPos-1)+14];
                    stackV[currStackPos*8+5] = stackSubgrid[19*(currStackPos-1)+5];
                    stackV[currStackPos*8+6] = stackSubgrid[19*(currStackPos-1)+18];
                    stackV[currStackPos*8+7] = stackSubgrid[19*(currStackPos-1)+17];
                }
                else if(currSubCell[currStackPos-1] == 2) { // left/top/back
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+2].x,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+2].x,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);

                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+0] = stackSubgrid[19*(currStackPos-1)+1];
                    stackV[currStackPos*8+1] = stackSubgrid[19*(currStackPos-1)+12];
                    stackV[currStackPos*8+2] = stackV[(currStackPos-1)*8+2];
                    stackV[currStackPos*8+3] = stackSubgrid[19*(currStackPos-1)+3];
                    stackV[currStackPos*8+4] = stackSubgrid[19*(currStackPos-1)+16];
                    stackV[currStackPos*8+5] = stackSubgrid[19*(currStackPos-1)+18];
                    stackV[currStackPos*8+6] = stackSubgrid[19*(currStackPos-1)+6];
                    stackV[currStackPos*8+7] = stackSubgrid[19*(currStackPos-1)+15];
                }
                else if(currSubCell[currStackPos-1] == 3) { // right/top/back
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+3].x,
                                                                 stackCorners[(currStackPos-1)*8+3].y,
                                                                 stackCorners[(currStackPos-1)*8+3].z);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+3].x,
                                                                 stackCorners[(currStackPos-1)*8+3].y,
                                                                 stackCorners[(currStackPos-1)*8+3].z+cellSize);

                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+0] = stackSubgrid[19*(currStackPos-1)+12];
                    stackV[currStackPos*8+1] = stackSubgrid[19*(currStackPos-1)+2];
                    stackV[currStackPos*8+2] = stackSubgrid[19*(currStackPos-1)+3];
                    stackV[currStackPos*8+3] = stackV[(currStackPos-1)*8+3];
                    stackV[currStackPos*8+4] = stackSubgrid[19*(currStackPos-1)+18];
                    stackV[currStackPos*8+5] = stackSubgrid[19*(currStackPos-1)+17];
                    stackV[currStackPos*8+6] = stackSubgrid[19*(currStackPos-1)+15];
                    stackV[currStackPos*8+7] = stackSubgrid[19*(currStackPos-1)+7];
                }
                else if(currSubCell[currStackPos-1] == 4) { // left/bottom/front
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+4].x,
                                                                 stackCorners[(currStackPos-1)*8+4].y,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+4].x,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);

                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+0] = stackSubgrid[19*(currStackPos-1)+4];
                    stackV[currStackPos*8+1] = stackSubgrid[19*(currStackPos-1)+14];
                    stackV[currStackPos*8+2] = stackSubgrid[19*(currStackPos-1)+16];
                    stackV[currStackPos*8+3] = stackSubgrid[19*(currStackPos-1)+18];
                    stackV[currStackPos*8+4] = stackV[(currStackPos-1)*8+4];
                    stackV[currStackPos*8+5] = stackSubgrid[19*(currStackPos-1)+8];
                    stackV[currStackPos*8+6] = stackSubgrid[19*(currStackPos-1)+9];
                    stackV[currStackPos*8+7] = stackSubgrid[19*(currStackPos-1)+13];
                }
                else if(currSubCell[currStackPos-1] == 5) { // right/bottom/front
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+5].x,
                                                                 stackCorners[(currStackPos-1)*8+5].y,
                                                                 stackCorners[(currStackPos-1)*8+5].z);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+5].x,
                                                                 stackCorners[(currStackPos-1)*8+5].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+5].z);

                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+0] = stackSubgrid[19*(currStackPos-1)+14];
                    stackV[currStackPos*8+1] = stackSubgrid[19*(currStackPos-1)+5];
                    stackV[currStackPos*8+2] = stackSubgrid[19*(currStackPos-1)+18];
                    stackV[currStackPos*8+3] = stackSubgrid[19*(currStackPos-1)+17];
                    stackV[currStackPos*8+4] = stackSubgrid[19*(currStackPos-1)+8];
                    stackV[currStackPos*8+5] = stackV[(currStackPos-1)*8+5];
                    stackV[currStackPos*8+6] = stackSubgrid[19*(currStackPos-1)+13];
                    stackV[currStackPos*8+7] = stackSubgrid[19*(currStackPos-1)+10];
                }
                else if(currSubCell[currStackPos-1] == 6) { // left/top/front
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+2].x,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+4].x,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+6].x,
                                                                 stackCorners[(currStackPos-1)*8+6].y,
                                                                 stackCorners[(currStackPos-1)*8+6].z);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+6].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+6].y,
                                                                 stackCorners[(currStackPos-1)*8+6].z);

                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+0] = stackSubgrid[19*(currStackPos-1)+16];
                    stackV[currStackPos*8+1] = stackSubgrid[19*(currStackPos-1)+18];
                    stackV[currStackPos*8+2] = stackSubgrid[19*(currStackPos-1)+6];
                    stackV[currStackPos*8+3] = stackSubgrid[19*(currStackPos-1)+15];
                    stackV[currStackPos*8+4] = stackSubgrid[19*(currStackPos-1)+9];
                    stackV[currStackPos*8+5] = stackSubgrid[19*(currStackPos-1)+13];
                    stackV[currStackPos*8+6] = stackV[(currStackPos-1)*8+6];
                    stackV[currStackPos*8+7] = stackSubgrid[19*(currStackPos-1)+11];
                }
                else if(currSubCell[currStackPos-1] == 7) { // right/top/front
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+3].x,
                                                                 stackCorners[(currStackPos-1)*8+3].y,
                                                                 stackCorners[(currStackPos-1)*8+3].z+cellSize);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+5].x,
                                                                 stackCorners[(currStackPos-1)*8+5].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+5].z);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+6].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+6].y,
                                                                 stackCorners[(currStackPos-1)*8+6].z);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+7].x,
                                                                 stackCorners[(currStackPos-1)*8+7].y,
                                                                 stackCorners[(currStackPos-1)*8+7].z);
 
                    stackV[currStackPos*8+0] = stackSubgrid[19*(currStackPos-1)+18];
                    stackV[currStackPos*8+1] = stackSubgrid[19*(currStackPos-1)+17];
                    stackV[currStackPos*8+2] = stackSubgrid[19*(currStackPos-1)+15];
                    stackV[currStackPos*8+3] = stackSubgrid[19*(currStackPos-1)+7];
                    stackV[currStackPos*8+4] = stackSubgrid[19*(currStackPos-1)+13];
                    stackV[currStackPos*8+5] = stackSubgrid[19*(currStackPos-1)+10];
                    stackV[currStackPos*8+6] = stackSubgrid[19*(currStackPos-1)+11];
                    stackV[currStackPos*8+7] = stackV[(currStackPos-1)*8+7];
                }
            
                currSubCell[currStackPos] = 0;
            }
            else {
                // Put the center of the current (sub-)cell on the stack
                cellCoords_D[idx].x = stackCorners[8*currStackPos+0].x + cellSize*0.5;
                cellCoords_D[idx].y = stackCorners[8*currStackPos+0].y + cellSize*0.5;
                cellCoords_D[idx].z = stackCorners[8*currStackPos+0].z + cellSize*0.5;
                done = true;
            }
        }
        else {
            if(currStackPos > 0) {
                currStackPos--;
                cellSize = cellSize*2.0;
                currSubCell[currStackPos]++;
            }
            else {
                // Field is not vanishing in this cell
                done = true;
            }
        }
    }
}


/* calcCellCoords_D2 */
__global__
void calcCellCoords_D2(const float3 *vecField_D,     // dim.x*dim.y*dim.z
                      float3 *cellCoords_D) {

    // Get thread index
    uint nCells = (dim.x-1)*(dim.y-1)*(dim.z-1);
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nCells) return;

    // Get coordinates of the lower/left/back corner
    uint3 c;
    c.x = idx%(dim.x-1);
    c.y = (idx/(dim.x-1))%(dim.y-1);
    c.z = (idx/(dim.x-1))/(dim.y-1);


    // Init stack

    const uint maxStackSize = 20;
    uint currStackPos = 0;
    uint currSubCell[maxStackSize]; // 0 ... 7
    currSubCell[0] = 0;
    
    float cellSize = 1.0f;

    float3 stackCorners[maxStackSize*8];
    stackCorners[0] = make_float3(0.0, 0.0, 0.0);
    stackCorners[1] = make_float3(1.0, 0.0, 0.0);
    stackCorners[2] = make_float3(0.0, 1.0, 0.0);
    stackCorners[3] = make_float3(1.0, 1.0, 0.0);
    stackCorners[4] = make_float3(0.0, 0.0, 1.0);
    stackCorners[5] = make_float3(1.0, 0.0, 1.0);
    stackCorners[6] = make_float3(0.0, 1.0, 1.0);
    stackCorners[7] = make_float3(1.0, 1.0, 1.0);

    float3 stackV[8*maxStackSize]; // Vector field at corners of current (sub-)cell
    stackV[0] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+0))+c.x+0]);
    stackV[1] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+0))+c.x+1]);
    stackV[2] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+1))+c.x+0]);
    stackV[3] = normalize(vecField_D[dim.x*(dim.y*(c.z+0) + (c.y+1))+c.x+1]);
    stackV[4] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+0))+c.x+0]);
    stackV[5] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+0))+c.x+1]);
    stackV[6] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+1))+c.x+0]);
    stackV[7] = normalize(vecField_D[dim.x*(dim.y*(c.z+1) + (c.y+1))+c.x+1]);
    
    // Init cell coords
    cellCoords_D[idx].x = -1.0;
    cellCoords_D[idx].y = -1.0;
    cellCoords_D[idx].z = -1.0;
    
    bool done = false;
    while(!done) {
        if((isFieldVanishingInCell_D(&stackV[currStackPos*8]))&&(currSubCell[currStackPos] < 8)) {
            
            if(currStackPos < maxStackSize-1) {
                
                // Increment stack
                currStackPos++;
                cellSize = cellSize*0.5;
                    
                // Bisect and put cell on stack
                if(currSubCell[currStackPos-1] == 0) { // left/down/back
                    // Set cell corners
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    // Sample vector field at cell corners
                    stackV[currStackPos*8+0] = stackV[(currStackPos-1)*8+0];
                    stackV[currStackPos*8+1] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+0], stackV[(currStackPos-1)*8+1], 0.5);
                    stackV[currStackPos*8+2] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+0], stackV[(currStackPos-1)*8+2], 0.5);
                    stackV[currStackPos*8+3] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                            stackV[(currStackPos-1)*8+1], stackV[(currStackPos-1)*8+2],
                            stackV[(currStackPos-1)*8+3], 0.5, 0.5);
                    stackV[currStackPos*8+4] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+0], stackV[(currStackPos-1)*8+4], 0.5);
                    stackV[currStackPos*8+5] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                            stackV[(currStackPos-1)*8+1], stackV[(currStackPos-1)*8+4],
                            stackV[(currStackPos-1)*8+5], 0.5, 0.5);
                    stackV[currStackPos*8+6] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                            stackV[(currStackPos-1)*8+2], stackV[(currStackPos-1)*8+4],
                            stackV[(currStackPos-1)*8+6], 0.5, 0.5);
                    stackV[currStackPos*8+7] = sampleVecFieldTrilin_D(&stackV[(currStackPos-1)*8], 0.5, 0.5, 0.5);
                }
                else if(currSubCell[currStackPos-1] == 1) { // right/down/back
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y,
                                                                 stackCorners[(currStackPos-1)*8+1].z);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+0] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+0], stackV[(currStackPos-1)*8+1], 0.5);
                    stackV[currStackPos*8+4] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                            stackV[(currStackPos-1)*8+1], stackV[(currStackPos-1)*8+4],
                            stackV[(currStackPos-1)*8+5], 0.5, 0.5);
                    stackV[currStackPos*8+2] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                            stackV[(currStackPos-1)*8+1], stackV[(currStackPos-1)*8+2],
                            stackV[(currStackPos-1)*8+3], 0.5, 0.5);
                    stackV[currStackPos*8+6] = sampleVecFieldTrilin_D(&stackV[(currStackPos-1)*8], 0.5, 0.5, 0.5);
                    stackV[currStackPos*8+1] = stackV[(currStackPos-1)*8+1];
                    stackV[currStackPos*8+3] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+1], stackV[(currStackPos-1)*8+3], 0.5);
                    stackV[currStackPos*8+5] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+1], stackV[(currStackPos-1)*8+5], 0.5);
                    stackV[currStackPos*8+7] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+1],
                            stackV[(currStackPos-1)*8+3], stackV[(currStackPos-1)*8+5],
                            stackV[(currStackPos-1)*8+7], 0.5, 0.5);
                }
                else if(currSubCell[currStackPos-1] == 2) { // left/top/back
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+2].x,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+2].x,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);

                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+1] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                            stackV[(currStackPos-1)*8+1], stackV[(currStackPos-1)*8+2],
                            stackV[(currStackPos-1)*8+3], 0.5, 0.5);
                    stackV[currStackPos*8+5] = sampleVecFieldTrilin_D(&stackV[(currStackPos-1)*8], 0.5, 0.5, 0.5);
                    stackV[currStackPos*8+0] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+0], stackV[(currStackPos-1)*8+2], 0.5);
                    stackV[currStackPos*8+2] = stackV[(currStackPos-1)*8+2];
                    stackV[currStackPos*8+3] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+2], stackV[(currStackPos-1)*8+3], 0.5);
                    stackV[currStackPos*8+4] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                            stackV[(currStackPos-1)*8+2], stackV[(currStackPos-1)*8+4],
                            stackV[(currStackPos-1)*8+6], 0.5, 0.5);
                    stackV[currStackPos*8+6] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+2], stackV[(currStackPos-1)*8+6], 0.5);
                    stackV[currStackPos*8+7] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+2],
                            stackV[(currStackPos-1)*8+3], stackV[(currStackPos-1)*8+6],
                            stackV[(currStackPos-1)*8+7], 0.5, 0.5);
                }
                else if(currSubCell[currStackPos-1] == 3) { // right/top/back
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+3].x,
                                                                 stackCorners[(currStackPos-1)*8+3].y,
                                                                 stackCorners[(currStackPos-1)*8+3].z);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+3].x,
                                                                 stackCorners[(currStackPos-1)*8+3].y,
                                                                 stackCorners[(currStackPos-1)*8+3].z+cellSize);

                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+0] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                            stackV[(currStackPos-1)*8+1], stackV[(currStackPos-1)*8+2],
                            stackV[(currStackPos-1)*8+3], 0.5, 0.5);
                    stackV[currStackPos*8+4] = sampleVecFieldTrilin_D(&stackV[(currStackPos-1)*8], 0.5, 0.5, 0.5);
                    stackV[currStackPos*8+2] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+2], stackV[(currStackPos-1)*8+3], 0.5);
                    stackV[currStackPos*8+6] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+2],
                            stackV[(currStackPos-1)*8+3], stackV[(currStackPos-1)*8+6],
                            stackV[(currStackPos-1)*8+7], 0.5, 0.5);
                    stackV[currStackPos*8+1] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+1], stackV[(currStackPos-1)*8+3], 0.5);
                    stackV[currStackPos*8+3] = stackV[(currStackPos-1)*8+3];
                    stackV[currStackPos*8+5] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+1],
                            stackV[(currStackPos-1)*8+3], stackV[(currStackPos-1)*8+5],
                            stackV[(currStackPos-1)*8+7], 0.5, 0.5);
                    stackV[currStackPos*8+7] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+3], stackV[(currStackPos-1)*8+7], 0.5);
                }
                else if(currSubCell[currStackPos-1] == 4) { // left/bottom/front
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+4].x,
                                                                 stackCorners[(currStackPos-1)*8+4].y,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+4].x,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);

                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+3] = sampleVecFieldTrilin_D(&stackV[(currStackPos-1)*8], 0.5, 0.5, 0.5);
                    stackV[currStackPos*8+0] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+0], stackV[(currStackPos-1)*8+4], 0.5);
                    stackV[currStackPos*8+1] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                            stackV[(currStackPos-1)*8+1], stackV[(currStackPos-1)*8+4],
                            stackV[(currStackPos-1)*8+5], 0.5, 0.5);
                    stackV[currStackPos*8+2] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                            stackV[(currStackPos-1)*8+2], stackV[(currStackPos-1)*8+4],
                            stackV[(currStackPos-1)*8+6], 0.5, 0.5);
                    stackV[currStackPos*8+4] = stackV[(currStackPos-1)*8+4];
                    stackV[currStackPos*8+5] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+4], stackV[(currStackPos-1)*8+5], 0.5);
                    stackV[currStackPos*8+6] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+4], stackV[(currStackPos-1)*8+6], 0.5);
                    stackV[currStackPos*8+7] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+4],
                            stackV[(currStackPos-1)*8+5], stackV[(currStackPos-1)*8+6],
                            stackV[(currStackPos-1)*8+7], 0.5, 0.5);
                }
                else if(currSubCell[currStackPos-1] == 5) { // right/bottom/front
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+5].x,
                                                                 stackCorners[(currStackPos-1)*8+5].y,
                                                                 stackCorners[(currStackPos-1)*8+5].z);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+5].x,
                                                                 stackCorners[(currStackPos-1)*8+5].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+5].z);

                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+0] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                                                                     stackV[(currStackPos-1)*8+1], 
                                                                     stackV[(currStackPos-1)*8+4],
                                                                     stackV[(currStackPos-1)*8+5], 
                                                                     0.5, 0.5);
                    stackV[currStackPos*8+4] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+4], 
                                                                   stackV[(currStackPos-1)*8+5], 
                                                                   0.5);
                    stackV[currStackPos*8+2] = sampleVecFieldTrilin_D(&stackV[(currStackPos-1)*8], 
                                                                      0.5, 0.5, 0.5);
                    stackV[currStackPos*8+6] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+4],
                                                                     stackV[(currStackPos-1)*8+5], 
                                                                     stackV[(currStackPos-1)*8+6],
                                                                     stackV[(currStackPos-1)*8+7], 
                                                                     0.5, 0.5);
                    stackV[currStackPos*8+1] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+1],
                                                                   stackV[(currStackPos-1)*8+5], 
                                                                   0.5);
                    stackV[currStackPos*8+3] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+1],
                                                                     stackV[(currStackPos-1)*8+3], 
                                                                     stackV[(currStackPos-1)*8+5],
                                                                     stackV[(currStackPos-1)*8+7], 
                                                                     0.5, 0.5);
                    stackV[currStackPos*8+5] = stackV[(currStackPos-1)*8+5];
                    stackV[currStackPos*8+7] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+5],
                                                                   stackV[(currStackPos-1)*8+7], 
                                                                   0.5);
                }
                else if(currSubCell[currStackPos-1] == 6) { // left/top/front
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+2].x,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+4].x,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+6].x,
                                                                 stackCorners[(currStackPos-1)*8+6].y,
                                                                 stackCorners[(currStackPos-1)*8+6].z);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+6].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+6].y,
                                                                 stackCorners[(currStackPos-1)*8+6].z);

                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+1] = sampleVecFieldTrilin_D(&stackV[(currStackPos-1)*8], 0.5, 0.5, 0.5);
                    stackV[currStackPos*8+5] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+4],
                            stackV[(currStackPos-1)*8+5], stackV[(currStackPos-1)*8+6],
                            stackV[(currStackPos-1)*8+7], 0.5, 0.5);
                    stackV[currStackPos*8+2] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+2], stackV[(currStackPos-1)*8+6], 0.5);
                    stackV[currStackPos*8+0] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+0],
                            stackV[(currStackPos-1)*8+2], stackV[(currStackPos-1)*8+4],
                            stackV[(currStackPos-1)*8+6], 0.5, 0.5);
                    stackV[currStackPos*8+3] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+2],
                            stackV[(currStackPos-1)*8+3], stackV[(currStackPos-1)*8+6],
                            stackV[(currStackPos-1)*8+7], 0.5, 0.5);
                    stackV[currStackPos*8+4] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+4], stackV[(currStackPos-1)*8+6], 0.5);
                    stackV[currStackPos*8+6] = stackV[(currStackPos-1)*8+6];
                    stackV[currStackPos*8+7] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+6], stackV[(currStackPos-1)*8+7], 0.5);
                }
                else if(currSubCell[currStackPos-1] == 7) { // right/top/front
                    stackCorners[currStackPos*8+0] = make_float3(stackCorners[(currStackPos-1)*8+0].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+0].z+cellSize);
                    stackCorners[currStackPos*8+1] = make_float3(stackCorners[(currStackPos-1)*8+1].x,
                                                                 stackCorners[(currStackPos-1)*8+1].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+1].z+cellSize);
                    stackCorners[currStackPos*8+2] = make_float3(stackCorners[(currStackPos-1)*8+2].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+2].y,
                                                                 stackCorners[(currStackPos-1)*8+2].z+cellSize);
                    stackCorners[currStackPos*8+3] = make_float3(stackCorners[(currStackPos-1)*8+3].x,
                                                                 stackCorners[(currStackPos-1)*8+3].y,
                                                                 stackCorners[(currStackPos-1)*8+3].z+cellSize);
                    stackCorners[currStackPos*8+4] = make_float3(stackCorners[(currStackPos-1)*8+4].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+4].z);
                    stackCorners[currStackPos*8+5] = make_float3(stackCorners[(currStackPos-1)*8+5].x,
                                                                 stackCorners[(currStackPos-1)*8+5].y+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+5].z);
                    stackCorners[currStackPos*8+6] = make_float3(stackCorners[(currStackPos-1)*8+6].x+cellSize,
                                                                 stackCorners[(currStackPos-1)*8+6].y,
                                                                 stackCorners[(currStackPos-1)*8+6].z);
                    stackCorners[currStackPos*8+7] = make_float3(stackCorners[(currStackPos-1)*8+7].x,
                                                                 stackCorners[(currStackPos-1)*8+7].y,
                                                                 stackCorners[(currStackPos-1)*8+7].z);
                    // Sample vector field at cell corners (while reusing vals from the last subcell)
                    stackV[currStackPos*8+0] = sampleVecFieldTrilin_D(&stackV[(currStackPos-1)*8], 0.5, 0.5, 0.5);
                    stackV[currStackPos*8+4] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+4],
                            stackV[(currStackPos-1)*8+5], stackV[(currStackPos-1)*8+6],
                            stackV[(currStackPos-1)*8+7], 0.5, 0.5);
                    stackV[currStackPos*8+2] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+2],
                            stackV[(currStackPos-1)*8+3], stackV[(currStackPos-1)*8+6],
                            stackV[(currStackPos-1)*8+7], 0.5, 0.5);
                    stackV[currStackPos*8+6] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+6], stackV[(currStackPos-1)*8+7], 0.5);
                    stackV[currStackPos*8+1] = sampleVecFieldBilin_D(stackV[(currStackPos-1)*8+1],
                            stackV[(currStackPos-1)*8+3], stackV[(currStackPos-1)*8+5],
                            stackV[(currStackPos-1)*8+7], 0.5, 0.5);
                    stackV[currStackPos*8+3] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+3], stackV[(currStackPos-1)*8+7], 0.5);
                    stackV[currStackPos*8+5] = sampleVecFieldLin_D(stackV[(currStackPos-1)*8+5], stackV[(currStackPos-1)*8+7], 0.5);
                    stackV[currStackPos*8+7] = stackV[(currStackPos-1)*8+7];
                }
            
                currSubCell[currStackPos] = 0;
            }
            else {
                // Put the center of the current (sub-)cell on the stack
                cellCoords_D[idx].x = stackCorners[8*currStackPos+0].x + cellSize*0.5;
                cellCoords_D[idx].y = stackCorners[8*currStackPos+0].y + cellSize*0.5;
                cellCoords_D[idx].z = stackCorners[8*currStackPos+0].z + cellSize*0.5;
                done = true;
            }
        }
        else {
            if(currStackPos > 0) {
                currStackPos--;
                cellSize = cellSize*2.0;
                currSubCell[currStackPos]++;
            }
            else {
                // Field is not vanishing in this cell
                done = true;
            }
        }
    }
}


extern "C" {

/* SetGridParams */
cudaError_t SetGridParams(uint3 dim_h, float3 org_h,  float3 maxCoord_h, 
        float3 spacing_h) {
    checkCudaErrors(cudaMemcpyToSymbol(dim, &dim_h, sizeof(uint3)));
    checkCudaErrors(cudaMemcpyToSymbol(org, &org_h, sizeof(float3)));
    checkCudaErrors(cudaMemcpyToSymbol(maxCoord, &maxCoord_h, sizeof(float3)));
    checkCudaErrors(cudaMemcpyToSymbol(spacing, &spacing_h, sizeof(float3)));
    return cudaGetLastError();
}

/* SetStreamlineStepsize */
cudaError_t  SetStreamlineParams(float stepsize_h, uint maxSteps) {
    checkCudaErrors(cudaMemcpyToSymbol(streamlinesStep, &stepsize_h, sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(streamlinesStepCnt, &maxSteps, sizeof(uint)));
    return cudaGetLastError();
}

/* SetNumberOfPos */
cudaError_t SetNumberOfPos(uint nPos_h) {
    checkCudaErrors(cudaMemcpyToSymbol(nPos, &nPos_h, sizeof(uint)));
    return cudaGetLastError();
}

/* UpdatePosition */
cudaError_t UpdatePositionRK4(
        const float *vecField,
        uint3 dim,
        float *pos,
        uint nPos,
        uint maxIt,
        bool backward) {

    uint nThreadsPerBlock = min(512, nPos);
    uint nBlocks  = ceil((float)(nPos)/(float)(nThreadsPerBlock));

    float3 *vecField_D, *pos_D;

    // Allocate device memory
    checkCudaErrors(cudaMalloc((void **)&vecField_D, sizeof(float)*dim.x*dim.y*dim.z*3));
    checkCudaErrors(cudaMalloc((void **)&pos_D, sizeof(float)*nPos*3));

    // Copy vec field data to device memory
    checkCudaErrors(cudaMemcpy(vecField_D, vecField,
            sizeof(float)*dim.x*dim.y*dim.z*3,
            cudaMemcpyHostToDevice));
            
    // Copy positions to device memory
    checkCudaErrors(cudaMemcpy(pos_D, pos,
            sizeof(float)*nPos*3,
            cudaMemcpyHostToDevice));
            
    if(backward) {
        //printf("CUDA streamline integration (backward), max steps %u\n", maxIt); // DEBUG
        for(uint i = 0; i < maxIt; i++) {
            // Update position maxIt times
            UpdatePositionBackwardRK4_D <<< nBlocks, nThreadsPerBlock >>>
                 (vecField_D, 
                  pos_D);
            cudaThreadSynchronize();
        }
    }
    else {
        //printf("CUDA streamline integration, max steps %u\n", maxIt); // DEBUG
        for(uint i = 0; i < maxIt; i++) {
            // Update position maxIt times
            UpdatePositionRK4_D <<< nBlocks, nThreadsPerBlock >>>
                 (vecField_D, 
                  pos_D);
            cudaThreadSynchronize();
        }
    }
             
    // Copy updated positions back to host memory
    checkCudaErrors(cudaMemcpy(pos, pos_D,
            sizeof(float)*nPos*3,
            cudaMemcpyDeviceToHost));
            
    // Cleanup device memory
    checkCudaErrors(cudaFree(vecField_D));
    checkCudaErrors(cudaFree(pos_D));

    return cudaGetLastError();
}

/* SearchNullPoints */
cudaError_t SearchNullPoints(
        const float *vecField,
        uint3 dim,
        float3 org,
        float3 spacing,
        float *cellCoords,
        unsigned int maxStackSize) {

    uint n = (dim.x-1)*(dim.y-1)*(dim.z-1);
    uint nThreadsPerBlock = min(512, n);
    uint nBlocks  = ceil((float)(n)/(float)(nThreadsPerBlock));

    float3 *vecField_D, *cellCoords_D;

    // Allocate device memory
    checkCudaErrors(cudaMalloc((void **)&vecField_D, sizeof(float)*dim.x*dim.y*dim.z*3));
    checkCudaErrors(cudaMalloc((void **)&cellCoords_D, sizeof(float)*n*3));

    // Copy vec field data to device memory
    checkCudaErrors(cudaMemcpy(vecField_D, vecField,
            sizeof(float)*dim.x*dim.y*dim.z*3,
            cudaMemcpyHostToDevice));

    // Calculate cell coordinates of the critical points
    calcCellCoords_D2 <<< nBlocks, nThreadsPerBlock >>>
             (vecField_D, 
             cellCoords_D);
             
    // Copy cell coords back to host memory
    checkCudaErrors(cudaMemcpy(cellCoords, cellCoords_D,
            sizeof(float)*n*3,
            cudaMemcpyDeviceToHost));
            
    // Cleanup device memory
    checkCudaErrors(cudaFree(vecField_D));
    checkCudaErrors(cudaFree(cellCoords_D));

    return cudaGetLastError();
}

}



// Streamline integration //////////////////////////////////////////////////////


/**
 * Calculates the gradient field of a given scalar field.
 *
 * @param[in]  scalarField_D   The scalar field (device memory)
 * @param[out] gradientField_D The gradient field (device memory)
 */
__global__ void CalcGradient_D(float *scalarDield_D, float *gradientField_D) {

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get coordinates of the lower/left/back corner
//    uint3 c;
//    c.x = idx%(dim.x-1);
//    c.y = (idx/(dim.x-1))%(dim.y-1);
//    c.z = (idx/(dim.x-1))/(dim.y-1);

    uint3 c;
    c.x = idx%dim.x;
    c.y = (idx/dim.x)%dim.y;
    c.z = (idx/dim.x)/dim.y;

    // Omit border cells
    if (c.x == 0) {
        return;
    }
    if (c.y == 0) {
        return;
    }
    if (c.z == 0) {
        return;
    }
    if (c.x >= dim.x-1) {
        return;
    }
    if (c.y >= dim.y-1) {
        return;
    }
    if (c.z >= dim.z-1) {
        return;
    }

    float3 gradient;

    gradient.x = scalarDield_D[dim.x*(dim.y*c.z+c.y)+c.x+1];
    gradient.y = scalarDield_D[dim.x*(dim.y*c.z+c.y+1)+c.x];
    gradient.z = scalarDield_D[dim.x*(dim.y*(c.z+1)+c.y)+c.x];

    gradient.x -= scalarDield_D[dim.x*(dim.y*c.z+c.y)+c.x-1];
    gradient.y -= scalarDield_D[dim.x*(dim.y*c.z+c.y-1)+c.x];
    gradient.z -= scalarDield_D[dim.x*(dim.y*(c.z-1)+c.y)+c.x];

    gradient = normalize(gradient);

    gradientField_D[3*idx+0] = gradient.x;
    gradientField_D[3*idx+1] = gradient.y;
    gradientField_D[3*idx+2] = gradient.z;
}

extern "C"
cudaError_t CalcGradient(float *scalarDield_D, float *gradientField_D, uint volsize) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (volsize + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

    // Calculate gradient of the scalar field
    CalcGradient_D <<< grid, threadsPerBlock >>> (
            scalarDield_D,
            gradientField_D);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time 'CalcGradient_D':                            %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}

/** TODO */
__global__ void InitStartPos_D(float *vertexDataBuffer_D, float *streamlinePos_D,
        uint vertexDataBufferStride, uint vertexDataBufferOffsPos) {

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > nPos) {
        return;
    }

    streamlinePos_D[idx*streamlinesStepCnt*6+0] =
            vertexDataBuffer_D[idx*vertexDataBufferStride+vertexDataBufferOffsPos+0];
    streamlinePos_D[idx*streamlinesStepCnt*6+1] =
            vertexDataBuffer_D[idx*vertexDataBufferStride+vertexDataBufferOffsPos+1];
    streamlinePos_D[idx*streamlinesStepCnt*6+2] =
            vertexDataBuffer_D[idx*vertexDataBufferStride+vertexDataBufferOffsPos+2];

}


cudaError_t InitStartPos(float *vertexDataBuffer_D, float *streamlinePos_D,
        uint vertexDataBufferStride, uint vertexDataBufferOffsPos, uint vertexCnt) {

#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (vertexCnt + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

    // Calculate gradient of the scalar field
    InitStartPos_D <<< grid, threadsPerBlock >>> (
            vertexDataBuffer_D,
            streamlinePos_D,
            vertexDataBufferStride,
            vertexDataBufferOffsPos);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time 'InitStartPos_D':                            %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();

}

/** TODO */
__global__ void UpdateStreamlinePos_D(float *streamlinePos_D, float3 *gradientField_D, uint step) {

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > nPos) {
        return;
    }

    float3 currPos;
    currPos.x = streamlinePos_D[idx*streamlinesStepCnt*6+step*6+0];
    currPos.y = streamlinePos_D[idx*streamlinesStepCnt*6+step*6+1];
    currPos.z = streamlinePos_D[idx*streamlinesStepCnt*6+step*6+2];

    float3 x0, x1, x2, x3, v0, v1, v2, v3;
    v0 = make_float3(0.0, 0.0, 0.0);
    v1 = make_float3(0.0, 0.0, 0.0);
    v2 = make_float3(0.0, 0.0, 0.0);
    v3 = make_float3(0.0, 0.0, 0.0);

    x0 = currPos;
    v0 = normalize(sampleVecFieldAtTrilinNorm_D(x0, gradientField_D));
    v0 *= streamlinesStep;

    x1 = x0 + 0.5*v0;
    if(isValidGridPos_D(x1)) {
        v1 = normalize(sampleVecFieldAtTrilinNorm_D(x1, gradientField_D));
        v1 *= streamlinesStep;
    }

    x2 = x0 + 0.5f*v1;
    if(isValidGridPos_D(x2)) {
        v2 = normalize(sampleVecFieldAtTrilinNorm_D(x2, gradientField_D));
        v2 *= streamlinesStep;
    }

    x3 = x0 + v2;
    if(isValidGridPos_D(x3)) {
        v3 = normalize(sampleVecFieldAtTrilinNorm_D(x3, gradientField_D));
        v3 *= streamlinesStep;
    }

    x0 += (1.0f/6.0f)*(v0+2.0f*v1+2.0f*v2+v3);

    // Copy position to streamline position array if it is valid
    if(isValidGridPos_D(x0)) {
        streamlinePos_D[idx*streamlinesStepCnt*6+step*6+3] = x0.x;
        streamlinePos_D[idx*streamlinesStepCnt*6+step*6+4] = x0.y;
        streamlinePos_D[idx*streamlinesStepCnt*6+step*6+5] = x0.z;
    } else {
        streamlinePos_D[idx*streamlinesStepCnt*6+step*6+3] = streamlinePos_D[idx*streamlinesStepCnt*6+step*6+0];
        streamlinePos_D[idx*streamlinesStepCnt*6+step*6+4] = streamlinePos_D[idx*streamlinesStepCnt*6+step*6+1];
        streamlinePos_D[idx*streamlinesStepCnt*6+step*6+5] = streamlinePos_D[idx*streamlinesStepCnt*6+step*6+2];
    }

    // Copy new position to the next line segment
    streamlinePos_D[idx*streamlinesStepCnt*6+(step+1)*6+0] = streamlinePos_D[idx*streamlinesStepCnt*6+step*6+3];
    streamlinePos_D[idx*streamlinesStepCnt*6+(step+1)*6+1] = streamlinePos_D[idx*streamlinesStepCnt*6+step*6+4];
    streamlinePos_D[idx*streamlinesStepCnt*6+(step+1)*6+2] = streamlinePos_D[idx*streamlinesStepCnt*6+step*6+5];

}


cudaError_t UpdateStreamlinePos(float *streamlinePos_D,float *gradientField_D, uint vertexCnt, uint step) {
#ifdef USE_TIMER
    float dt_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
#endif

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (vertexCnt + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerGrid, 1, 1);

    // Calculate gradient of the scalar field
    UpdateStreamlinePos_D <<< grid, threadsPerBlock >>> (
            streamlinePos_D, (float3*)(gradientField_D), step);

#ifdef USE_TIMER
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&dt_ms, event1, event2);
    printf("CUDA time 'UpdateStreamlinePos_D':                     %.10f sec\n",
            dt_ms/1000.0f);
#endif

    return cudaGetLastError();
}

