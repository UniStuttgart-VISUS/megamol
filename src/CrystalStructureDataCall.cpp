/*
 * CrystalStructureDataCall.cpp
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#include "stdafx.h"
#include "CrystalStructureDataCall.h"

using namespace megamol;

/*
 * protein_cuda::CrystalStructureDataCallCallForGetData
 */
const unsigned int protein_cuda::CrystalStructureDataCall::CallForGetData = 0;


/*
 * protein_cuda::CrystalStructureDataCall:CallForGetExtent
 */
const unsigned int protein_cuda::CrystalStructureDataCall::CallForGetExtent = 1;


/*
 * CrystalStructureDataCall::CrystalStructureDataCall
 */
protein_cuda::CrystalStructureDataCall::CrystalStructureDataCall(void) :
        AbstractGetData3DCall(), atomCnt(0), dipoleCnt(0), conCnt(0),
        cellCnt(0), calltime(0.0f),
        atomPos(NULL), atomCon(NULL), cells(NULL), dipolePos(NULL),
        dipole(NULL) {
    // intentionally empty
}


/*
 * protein_cuda::CrystalStructureDataCall::~protein_cuda::CrystalStructureDataCall::
 */
protein_cuda::CrystalStructureDataCall::~CrystalStructureDataCall(void) {
}

