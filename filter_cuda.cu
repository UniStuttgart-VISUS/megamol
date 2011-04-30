#include <cutil_inline.h>
#include <vector_types.h>
#include <cutil_math.h>

#include "filter_cuda.cuh"

// Parameters in constant memory
__constant__ FilterParams fparams;


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
    //gridPos.x = gridPos.x & (fparams.gridSize.x - 1);  
    //gridPos.y = gridPos.y & (fparams.gridSize.y - 1);
    //gridPos.z = gridPos.z & (fparams.gridSize.z - 1);*/

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
        atomVisibility[idx] = 1;
        return;
    }
    
    unsigned int startIdx, endIdx, hash, j;
    int3 neighbourPos;
    int x,y,z;
    
    // Get position of the atom
    float3 p = atmPos[idx];

    int3 gridPos = make_int3(floor((p.x - fparams.worldOrigin.x) / fparams.cellSize.x),
                             floor((p.y - fparams.worldOrigin.y) / fparams.cellSize.y),
                             floor((p.z - fparams.worldOrigin.z) / fparams.cellSize.z));
    
    // Examine neighbouring cells within the given range
    for(z = -fparams.discRange.z; z <= fparams.discRange.z; z++) {
            
        neighbourPos.z = (gridPos.z + z);
        if((neighbourPos.z < 0) || (neighbourPos.z >= fparams.gridSize.z)) 
            continue;
        
        for(y = -fparams.discRange.y; y <= fparams.discRange.y; y++) {
            
            neighbourPos.y = (gridPos.y + y);
            if((neighbourPos.y < 0) || (neighbourPos.y >= fparams.gridSize.y)) 
                continue;
            
            for(x = -fparams.discRange.x; x <= fparams.discRange.x; x++) {
                
                neighbourPos.x = gridPos.x + x;
                if((neighbourPos.x < 0) || (neighbourPos.x >= fparams.gridSize.x)) 
                    continue;
                    
                hash = __umul24(__umul24(neighbourPos.z, fparams.gridSize.y), 
                           fparams.gridSize.x) + __umul24(neighbourPos.y, 
                           fparams.gridSize.x) + neighbourPos.x;
                           
                // Note: startIndex/endIndex are referring to the position in
                // the sorted array
                startIdx = cellStart[hash];
                
                if(startIdx == 0xffffffff) {
                   continue; // Cell is empty - continue with next cell
                }
                else {
                
                    // Note: startIndex/endIndex are referring to the position in
                    // the sorted array
                    endIdx = cellEnd[hash];
                    
                    // If cell contains non-solvent atoms and is within inner
                    // range the atom is visible
                    if((abs(neighbourPos.x) <= fparams.innerDiscRange) &&
                    (abs(neighbourPos.y) <= fparams.innerDiscRange) &&
                    (abs(neighbourPos.z) <= fparams.innerDiscRange)) {
                    
                        atomVisibility[idx] = 1;
                        return; 
                    }
                    else {
            
                        // Iterate over all atoms in this cell
                        for(j = startIdx; j < endIdx; j++) {
                            if(length(atmPosProtSorted[j] - p) <= fparams.solvRange) {
                                
                                atomVisibility[idx] = 1;
                                return; 
                            }
                        }
                    }
                }
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
        cutilSafeCall(cudaMemcpyToSymbol(fparams, hostParams, sizeof(FilterParams)));
    }


    /*
     * calcHashGrid
     */
    void calcFilterHashGrid(unsigned int *gridHash,
                            unsigned int *gridIndex,
                            float        *atmPosProt,
                            unsigned int  atmCntProt) {
        
        // Compute grid size
        unsigned int numThreads = min(256, atmCntProt);
        unsigned int numBlocks  = ceil((float)atmCntProt/(float)numThreads);
    
        // Execute the kernel
        calcFilterHashGridD <<< numBlocks, numThreads >>> (gridHash,
                                                           gridIndex,
                                                           (float3*) atmPosProt);
        
        cutilCheckMsg("calcFilterHashGridD");
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
  
        // Compute grid size
        unsigned int numThreads = min(256, atmCntProt);
        unsigned int numBlocks  = ceil((float)atmCntProt/(float)numThreads);
        
        // Compute memory size
        unsigned int memSize = sizeof(unsigned int)*(numThreads+1);
        
        // Execute kernel
        reorderFilterDataD <<< numBlocks, numThreads, memSize >>> (cellStart,
                                                                   cellEnd,
                                                                   gridHash,
                                                                   gridIndex,
                                                                   (float3*) atmPosProt,
                                                                   (float3*) atmPosProtSorted);
            
        cutilCheckMsg("reorderFilterDataD");
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
        unsigned int numBlocks  = ceil((float)atmCnt/(float)numThreads);                                     
                                         
        // Execute kernel
        calcSolventVisibilityD <<< numBlocks, numThreads >>> (cellStart,
                                                              cellEnd,
                                                              (float3*) atmPos,
                                                              (float3*) atmPosProtSorted,
                                                              isSolventAtom,
                                                              atomVisibility);
        
        cutilCheckMsg("calcSolventVisibilityD");                                                              
    }

} // extern "C"
