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

#include "filter_cuda.cuh"
#include "helper_cuda.h"
#include "helper_math.h"

// Parameters in constant memory
__constant__ FilterParams fparams;


__device__ float getDistanceSq(float3 posA, float3 posB) {
    
    return (posA.x-posB.x)*(posA.x-posB.x) + (posA.y-posB.y)*(posA.y-posB.y) +
           (posA.z-posB.z)*(posA.z-posB.z); 
}


/*
 * hashValMultiGrid
 * 
 * Hash function resulting in a distribution of values with eight successive
 * hash values remaining inside the same cube.
 * 
 */
__device__
unsigned int hashValMultiGrid(int3 gridPos) {
    
    // 
    // (1) Grid size of coarse grid: 
    //      coarseGridSize  = fparams.gridSize.x / 2
    //
    // (2) Address of subcube in coarse grid: 
    //      subCubeAddr = 
    //      (int)(gridPos.y/2) * coarseGridSize 
    //      + (int)(gridPos.x/2) 
    //      + (int)(gridPos.z/2) * coarseGridSize  * coarseGridSize ;
    // 
    // (3) Hash value within one subcube: 
    //      h = ((gridPos.x)%2 + gridPos.y%2 * 2 + 2 * 2 * (gridPos.z%2))
    // 
    // (4) Actual hash value: 
    //      h + 8 * subCubeAddr
    
    unsigned int coarseGridSize = fparams.gridSize.x / 2;
    
    // TODO: gridPos.x % 2 = gridPos.x & (2 - 1) ?
    return (gridPos.x % 2) + (((gridPos.y % 2) + (gridPos.z % 2) * 2)
            + 2 * 2 * (gridPos.y/2 * coarseGridSize + gridPos.x/2 
                    + gridPos.z/2 * coarseGridSize) * coarseGridSize) * 2;
}


/*
 * calcFilterHashGridD
 */
__global__
void calcFilterHashGridD(unsigned int *gridHash,  
                         unsigned int *gridIndex,
                         float3       *atmPos) {
                             
    unsigned int idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    
    if(idx >= fparams.atmCntProt) 
        return;
    
    int3 gridPos = make_int3(floor((atmPos[idx].x - fparams.worldOrigin.x) / fparams.cellSize.x),
                             floor((atmPos[idx].y - fparams.worldOrigin.y) / fparams.cellSize.y),
                             floor((atmPos[idx].z - fparams.worldOrigin.z) / fparams.cellSize.z));

    // Wrap grid, assumes size is power of 2
    /*gridPos.x = gridPos.x & (fparams.gridSize.x - 1);  
    gridPos.y = gridPos.y & (fparams.gridSize.y - 1);
    gridPos.z = gridPos.z & (fparams.gridSize.z - 1);*/

    // Calculate hash value
    gridHash[idx]  = __umul24(__umul24(gridPos.z, fparams.gridSize.y), 
        fparams.gridSize.x) + __umul24(gridPos.y, fparams.gridSize.x) + gridPos.x;
    

    // Init index array 
    gridIndex[idx] = idx;
}



/*
 * reorderFilterDataD
 */
__global__
void reorderFilterDataD(unsigned int *cellStart,        
                        unsigned int *cellEnd,      
                        unsigned int *gridHash, 
                        unsigned int *gridIndex,
                        float3       *atmPos,
                        float3       *atmPosSorted) {
                        
    extern __shared__ unsigned int sharedHash[];    // blockSize + 1 elements
    
    unsigned int idx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    unsigned int hash;

    if(idx < fparams.atmCntProt) {
        hash = gridHash[idx];

        // Load hash data into shared memory so that we can look 
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if(idx > 0 && threadIdx.x == 0) {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridHash[idx - 1];
        }
    }

    __syncthreads();
    
    if(idx < fparams.atmCntProt) {
        
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell
        
        if(idx == 0 || hash != sharedHash[threadIdx.x]) {
            cellStart[hash] = idx;
            if(idx > 0)
                cellEnd[sharedHash[threadIdx.x]] = idx;
        }

        if(idx == fparams.atmCntProt - 1) {
            cellEnd[hash] = idx + 1;
        }

        // Now use the sorted index to reorder the pos data
        atmPosSorted[idx] = atmPos[gridIndex[idx]];
        
        // macro does either global read or texture fetch
        //float4 pos = FETCH( oldPos, sortedIndex);       
    }
}



/*
 * calcSolventVisibilityD
 */
__global__
void calcSolventVisibilityD(unsigned int *cellStart,
                            unsigned int *cellEnd,
                            float3       *atmPos,
                            float3       *atmPosProtSorted,
                            bool         *isSolventAtom,
                            int          *atomVisibility) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx >= fparams.atmCnt) {
        return;
    }
    
    // Non-solvent atoms are visible
    if(!isSolventAtom[idx]) {
        atomVisibility[idx] = 0;
        return;
    }
    
    // Note: startIndex/endIndex are referring to the position in the sorted array
    unsigned int startIdx, endIdx, hash, at, x, y, z;
    
    // Get position of the atom
    float3 p = atmPos[idx];

    int3 gridPos = make_int3(floor((p.x - fparams.worldOrigin.x) / fparams.cellSize.x),
                             floor((p.y - fparams.worldOrigin.y) / fparams.cellSize.y),
                             floor((p.z - fparams.worldOrigin.z) / fparams.cellSize.z));
                             
    int3 discRangeCeil = make_int3(
        (gridPos.x + fparams.discRange.x < fparams.gridSize.x) ? 
            (gridPos.x + fparams.discRange.x) : fparams.gridSize.x,
        (gridPos.y + fparams.discRange.y < fparams.gridSize.y) ? 
            (gridPos.y + fparams.discRange.y) : fparams.gridSize.y,
        (gridPos.z + fparams.discRange.z < fparams.gridSize.z) ? 
            (gridPos.z + fparams.discRange.z) : fparams.gridSize.z);
        
    int3 discRangeFloor = make_int3(
        (gridPos.x - fparams.discRange.x >= 0) ? (gridPos.x - fparams.discRange.x) : 0,
        (gridPos.y - fparams.discRange.y >= 0) ? (gridPos.y - fparams.discRange.y) : 0,
        (gridPos.z - fparams.discRange.z >= 0) ? (gridPos.z - fparams.discRange.z) : 0);
    
    // Examine neighbouring cells within the given range
    for(z = discRangeFloor.z; z < discRangeCeil.z; z++) {
        for(y = discRangeFloor.y; y < discRangeCeil.y; y++) {
            for(x = discRangeFloor.x; x < discRangeCeil.x; x++) {
                    
                hash = __umul24(__umul24(z, fparams.gridSize.y), fparams.gridSize.x) + 
                       __umul24(y, fparams.gridSize.x) + 
                       x;
                        
                startIdx = cellStart[hash];
                
                if(startIdx == 0xffffffff) {
                   continue; // Cell is empty - continue with next cell
                }
                else {

                    endIdx = cellEnd[hash];
            
                    // Iterate over all atoms in this cell
                    for(at = startIdx; at < endIdx; at++) {
                        if(getDistanceSq(atmPosProtSorted[at], p) <= fparams.solvRangeSq) {
                            
                            atomVisibility[idx] = 1;
                            return; 
                        }
                    }
                }
            }
        }
    }
}



/*
 * calcSolventVisibilityAltD
 */
__global__
void calcSolventVisibilityAltD(unsigned int *cellStart,
                               unsigned int *cellEnd,
                               float3       *atmPos,
                               float3       *atmPosProtSorted,
                               bool         *isSolventAtom,
                               int          *atomVisibility,
                               int3          *neighbourCellPos) {
                                
    extern __shared__ bool sharedVisibility[];

    unsigned int atmIdx = blockIdx.x;
    unsigned int neighbourCellIdx = threadIdx.x;
    
    if(!((atmIdx >= fparams.atmCnt) || (neighbourCellIdx >= fparams.numNeighbours))) {
        
        sharedVisibility[threadIdx.x] = false;
         
        if(!isSolventAtom[atmIdx]) { // Non-solvent atoms are visible
            sharedVisibility[threadIdx.x] = true;            
        }
        else {        
            // Get position of the atom
            float3 p = atmPos[atmIdx];
        
            int3 gridPos = make_int3(floor((p.x - fparams.worldOrigin.x) / fparams.cellSize.x),
                                     floor((p.y - fparams.worldOrigin.y) / fparams.cellSize.y),
                                     floor((p.z - fparams.worldOrigin.z) / fparams.cellSize.z));
                                     
            
            int3 neighbourPos = make_int3((gridPos.x + neighbourCellPos[neighbourCellIdx].x) & (fparams.gridSize.x - 1),  // wrap grid
                                         (gridPos.y + neighbourCellPos[neighbourCellIdx].y) & (fparams.gridSize.y - 1),
                                         (gridPos.z + neighbourCellPos[neighbourCellIdx].z) & (fparams.gridSize.z - 1));
            
                                    
            unsigned int hash = __umul24(__umul24(neighbourPos.z, fparams.gridSize.y), fparams.gridSize.x) + 
                                __umul24(neighbourPos.y, fparams.gridSize.x) + 
                                neighbourPos.x;
                    
            unsigned int startIdx = cellStart[hash];
            
            if(startIdx != 0xffffffff) {
        
                unsigned int endIdx = cellEnd[hash];

        
                // Iterate over all atoms in the neighbour cell
                // TODO
                for(unsigned int at = startIdx; at < endIdx; at++) {
                    if(getDistanceSq(atmPosProtSorted[at], p) <= fparams.solvRangeSq) {
                        sharedVisibility[threadIdx.x] = true;
                        break;
                    }
                }
            }
        }
    }
    else
        return;
    
    __syncthreads();
    
    // Merge visibility information
    if(threadIdx.x == 0) {
        atomVisibility[atmIdx] = 0;
        for(int i = 0; i < fparams.numNeighbours; i++) {
            if(sharedVisibility[i] == true) {
                atomVisibility[atmIdx] = 1;
                break;
            }
        }
    }
}




extern "C" {
 
    
    /*
     * setFilterParams
     */
    void setFilterParams(FilterParams *hostParams) {
        // Copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(fparams, hostParams, sizeof(FilterParams)));
    }


    /*
     * calcFilterHashGrid
     */
    void calcFilterHashGrid(unsigned int *gridHash,
                            unsigned int *gridIndex,
                            float        *atmPosProt,
                            unsigned int  atmCntProt) {
        
        // Compute grid size
        unsigned int numThreads = min(256, atmCntProt);
        unsigned int numBlocks  = (unsigned int)ceil((float)atmCntProt/(float)numThreads);
    
        // Execute the kernel
        calcFilterHashGridD <<< numBlocks, numThreads >>> (gridHash,
                                                           gridIndex,
                                                           (float3*) atmPosProt);
        
        getLastCudaError("calcFilterHashGridD");
    }


    /*
     * reorderFilterData
     */
    void reorderFilterData(unsigned int *cellStart,
                           unsigned int *cellEnd,
                           unsigned int *gridHash,
                           unsigned int *gridIndex,
                           float        *atmPosProt,
                           float        *atmPosProtSorted,
                           unsigned int  atmCntProt) {
  
        unsigned int numThreads = min(256, atmCntProt);
        unsigned int numBlocks  = (unsigned int)ceil((float)atmCntProt/(float)numThreads);
        
        // Compute memory size
        unsigned int memSize = sizeof(unsigned int)*(numThreads+1);
        
        // Execute kernel
        reorderFilterDataD <<< numBlocks, numThreads, memSize >>> (cellStart,
                                                                   cellEnd,
                                                                   gridHash,
                                                                   gridIndex,
                                                                   (float3*) atmPosProt,
                                                                   (float3*) atmPosProtSorted);
            
        getLastCudaError("reorderFilterDataD");
    }
                                       
    
    /*
     * calcSolventVisibilityAlt
     */
    void calcSolventVisibilityAlt(unsigned int *cellStart,
                                  unsigned int *cellEnd,
                                  float        *atmPos,
                                  float        *atmPosProtSorted,
                                  bool         *isSolventAtom,
                                  int          *atomVisibility,
                                  int          *neighbourCellPos,
                                  unsigned int  atmCnt,
                                  unsigned int  numNeighbours) {
                                   
        
        unsigned int numThreads = numNeighbours; // TODO this only works if number of neighbour cells is less than 512
        unsigned int memSize = numNeighbours*sizeof(bool);
        unsigned int numBlocks = atmCnt;
                                         
        // Execute kernel
        calcSolventVisibilityAltD <<< numBlocks, numThreads, memSize >>> (cellStart,
                                                          cellEnd,
                                                          (float3*) atmPos,
                                                          (float3*) atmPosProtSorted,
                                                          isSolventAtom,
                                                          atomVisibility,
                                                          (int3*) neighbourCellPos);
        
        getLastCudaError("calcSolventVisibilityAltD");
        
    }
    
    
    /*
     * calcSolventVisibility
     */
    void calcSolventVisibility(unsigned int *cellStart,
                               unsigned int *cellEnd,
                               float        *atmPos,
                               float        *atmPosProtSorted,
                               bool         *isSolventAtom,
                               int          *atomVisibility,
                               unsigned int  atmCnt) {
    
        // Compute grid size
        unsigned int numThreads = min(256, atmCnt);
        unsigned int numBlocks  = (unsigned int)ceil((float)atmCnt/(float)numThreads);                                     
                                         
        // Execute kernel
        calcSolventVisibilityD <<< numBlocks, numThreads >>> (cellStart,
                                                              cellEnd,
                                                              (float3*) atmPos,
                                                              (float3*) atmPosProtSorted,
                                                              isSolventAtom,
                                                              atomVisibility);
        
        getLastCudaError("calcSolventVisibilityD");                                                              
    }

                                 

} // extern "C"

