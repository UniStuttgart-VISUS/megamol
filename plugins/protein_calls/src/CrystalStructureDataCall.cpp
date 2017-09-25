/*
 * CrystalStructureDataCall.cpp
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id: CrystalStructureDataCall.cpp 1443 2015-07-08 12:18:12Z reina $
 */

#include "stdafx.h"
#include "protein_calls/CrystalStructureDataCall.h"

using namespace megamol;

/*
 * core::moldyn::CrystalStructureDataCallCallForGetData
 */
const unsigned int protein_calls::CrystalStructureDataCall::CallForGetData = 0;


/*
 * core::moldyn::CrystalStructureDataCall:CallForGetExtent
 */
const unsigned int protein_calls::CrystalStructureDataCall::CallForGetExtent = 1;


/*
 * CrystalStructureDataCall::CrystalStructureDataCall
 */
protein_calls::CrystalStructureDataCall::CrystalStructureDataCall(void) :
        AbstractGetData3DCall(), atomCnt(0), dipoleCnt(0), conCnt(0),
        cellCnt(0), calltime(0.0f),
        atomPos(NULL), atomCon(NULL), cells(NULL), dipolePos(NULL),
        dipole(NULL) {
    // intentionally empty
}


/*
 * core::moldyn::CrystalStructureDataCall::~core::moldyn::CrystalStructureDataCall::
 */
protein_calls::CrystalStructureDataCall::~CrystalStructureDataCall(void) {
}

