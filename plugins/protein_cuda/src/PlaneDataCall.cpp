/*
 *	PlaneDataCall.cpp
 *
 *	Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 *	All rights reserved
 */

#include "stdafx.h"

#include "PlaneDataCall.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_cuda;

const unsigned int PlaneDataCall::CallForGetData = 0;
const unsigned int PlaneDataCall::CallForGetExtent = 1;

/*
 *	PlaneDataCall::PlaneDataCall
 */
PlaneDataCall::PlaneDataCall(void) : Call(), 
	planeCnt(0), planeData(nullptr) {

}

/*
 *	PlaneDataCall::~PlaneDataCall
 */
PlaneDataCall::~PlaneDataCall(void) {

}

/*
 *	PlaneDataCall::DataHash
 */
SIZE_T PlaneDataCall::DataHash(void) {
	return this->dataHash;
}

/*
 *	PlaneDataCall::GetPlaneCnt
 */
unsigned int PlaneDataCall::GetPlaneCnt(void) {
	return this->planeCnt;
}

/*
 *	PlaneDataCall::GetPlaneData
 */
const vislib::math::Plane<float> * PlaneDataCall::GetPlaneData(void) {
	return this->planeData;
}

/*
 *	PlaneDataCall::SetDataHash
 */
void PlaneDataCall::SetDataHash(SIZE_T dataHash) {
	this->dataHash = dataHash;
}

/*
 *	PlaneDataCall::SetPlaneCnt
 */
void PlaneDataCall::SetPlaneCnt(unsigned int planeCnt) {
	this->planeCnt = planeCnt;
}

/*
 *	PlaneDataCall::SetPlaneData
 */
void PlaneDataCall::SetPlaneData(const vislib::math::Plane<float> * planeData) {
	this->planeData = planeData;
}