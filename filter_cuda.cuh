#ifndef FILTER_CUDA_CUH_INCLUDED
#define FILTER_CUDA_CUH_INCLUDED

#include <vector_types.h>


struct FilterParams {

    float3       cellSize;
    float3       worldOrigin;
    uint3        gridSize;
    unsigned int numCells;
    unsigned int atmCnt;
    unsigned int atmCntProt;
    float        solvRange;
    int3         discRange;
    int          innerDiscRange;
};


extern "C" {
    
    void setFilterParams(FilterParams *hostParams);
    
    
    void calcFilterHashGrid(unsigned int *gridHash,
                            unsigned int *gridIndex,
                            float        *atmPosProt,
                            unsigned int  atmCntProt);
    
                  
    void reorderFilterData(unsigned int *cellStart,
                           unsigned int *cellEnd,
                           unsigned int *gridHash,
                           unsigned int *gridIndex,
                           float        *atmPosProt,
                           float        *atmPosProtSorted,
                           unsigned int  atmCntProt);

    
    void calcSolventVisibility(unsigned int *cellStart,
                               unsigned int *cellEnd,
                               float        *atmPos,
                               float        *atmPosProtSorted,
                               bool         *isSolventAtom,
                               int          *atomVisibility,
                               unsigned int  atmCnt);

}

#endif // FILTER_CUDA_CUH_INCLUDED
