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
    float        solvRangeSq;
    int3         discRange;
    int3         discRangeWide;
    unsigned int numNeighbours;
    unsigned int innerCellRange;
    
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
                           
    
    void calcSolventVisibilityAlt(unsigned int *cellStart,
                                  unsigned int *cellEnd,
                                  float        *atmPos,
                                  float        *atmPosProtSorted,
                                  bool         *isSolventAtom,
                                  int          *atomVisibility,
                                  int          *neighbourCellPos,
                                  unsigned int  atmCnt,
                                  unsigned int  numNeighbours);  
                               
                       
    void calcSolventVisibility(unsigned int *cellStart,
                               unsigned int *cellEnd,
                               float        *atmPos,
                               float        *atmPosProtSorted,
                               bool         *isSolventAtom,
                               int          *atomVisibility,
                               unsigned int  atmCnt);                         
                       
                               
}

#endif // FILTER_CUDA_CUH_INCLUDED
