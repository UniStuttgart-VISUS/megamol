//
// DiffusionSolver.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 16, 2013
// Author     : scharnkn
//

#include "stdafx.h"
#include "DiffusionSolver.h"

using namespace megamol;
using namespace megamol::protein;

#ifdef WITH_CUDA
/*
 * DiffusionSolver::CalcGVFCuda
 */
bool DiffusionSolver::CalcGVFCuda(double *v, size_t dim[3], unsigned int maxIt) {
    return true; // TODO
}
#endif // WITH_CUDA




