//
// constantGridParams.cuh
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 18, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_CONSTANTGRIDPARAMS_CUH_INCLUDED
#define MMPROTEINCUDAPLUGIN_CONSTANTGRIDPARAMS_CUH_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

__constant__ __device__ uint3 gridSize_D;     // The size of the volume texture
__constant__ __device__ float3 gridOrg_D;     // The origin of the volume texture
__constant__ __device__ float3 gridDelta_D;   // The spacing of the volume texture

#endif // MMPROTEINCUDAPLUGIN_CONSTANTGRIDPARAMS_CUH_INCLUDED
