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
 
#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#define FETCH(t, i) tex1Dfetch(t##Tex, i)

#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct SimParams {

	uint3 gridSize;
    uint numCells;
    float3 worldOrigin;
    float3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

	float probeRadius;

    uint maxNumNeighbors;

    uint texSize;
};

// Reduced Surface parameters
struct RSParams {

    uint visibleAtomCount;
    uint maxNumProbeNeighbors;
    uint probeCount;

};

#endif
