//
// VTKLegacyDataCallUnstructuredGrid.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 23, 2013
// Author     : scharnkn
//

#include "stdafx.h"
#include "VTKLegacyDataCallUnstructuredGrid.h"

using namespace megamol;
using namespace megamol::protein;

const unsigned int VTKLegacyDataCallUnstructuredGrid::CallForGetData = 0;
const unsigned int VTKLegacyDataCallUnstructuredGrid::CallForGetExtent = 1;


/*
 * VTKLegacyDataCallUnstructuredGrid::VTKLegacyDataCallUnstructuredGrid
 */
VTKLegacyDataCallUnstructuredGrid::VTKLegacyDataCallUnstructuredGrid(void)
    : core::AbstractGetData3DCall(), calltime(0.0) {}


/*
 * VTKLegacyDataCallUnstructuredGrid::~VTKLegacyDataCallUnstructuredGrid
 */
VTKLegacyDataCallUnstructuredGrid::~VTKLegacyDataCallUnstructuredGrid(void) {}
