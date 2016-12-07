//
// PotentialCalculator.cuh
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 23, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_POTENTIALCALCULATOR_CUH_INCLUDED
#define MMPROTEINCUDAPLUGIN_POTENTIALCALCULATOR_CUH_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif // (defined(_MSC_VER) && (_MSC_VER > 1000))

#include "HostArr.h"
#include "CudaDevArr.h"

#if _WIN64

extern "C"
cudaError_t SolvePoissonEq(float gridSpacing, uint3 gridSize, float *charges,
        float *potential_D, float *potential);

extern "C"
cudaError_t DirectCoulombSummation(float *atomData, uint atomCount,
        float *potential, uint3 gridSize, float gridspacing);

#endif // _WIN64

#endif // MMPROTEINCUDAPLUGIN_POTENTIALCALCULATOR_CUH_INCLUDED
