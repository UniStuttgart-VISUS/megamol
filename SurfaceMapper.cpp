//
// SurfaceMapper.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 16, 2013
// Author     : scharnkn
//

#include "stdafx.h"
#include "SurfaceMapper.h"
#include "VBODataCall.h"
#include "VTIDataCall.h"

using namespace megamol;
using namespace megamol::protein;


/*
 * SurfaceMapper::SurfaceMapper
 */
SurfaceMapper::SurfaceMapper(void) : core::Module() ,
        vtxInputSlot("vtxIn", "Input vertex data representing the source mesh"),
        vtxOutputSlot("vtxOut", "Output vertex data representing the surface mapping"),
        volInputSlot("volIn", "Input volume texture representing the target shape"),
        volOutputSlot("volOut", "Output volume texture containing the GVF") {

    /* Make data caller/callee slots available */

    // Vertex input
    this->vtxInputSlot.SetCompatibleCall<VBODataCallDescription>();
    this->MakeSlotAvailable(&this->vtxInputSlot);

    // Vertex output
    this->vtxOutputSlot.SetCallback(
            VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetData),
            &SurfaceMapper::getVtxData);
    this->vtxOutputSlot.SetCallback(
            VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetExtent),
            &SurfaceMapper::getVolExtent);
    this->MakeSlotAvailable(&this->vtxOutputSlot);

    // Volume input
    this->volInputSlot.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->volInputSlot);

    // Volume output
    this->volOutputSlot.SetCallback(
            VTIDataCall::ClassName(),
            VTIDataCall::FunctionName(VTIDataCall::CallForGetData),
            &SurfaceMapper::getVolData);
    this->volOutputSlot.SetCallback(
            VTIDataCall::ClassName(),
            VTIDataCall::FunctionName(VTIDataCall::CallForGetExtent),
            &SurfaceMapper::getVolExtent);
    this->MakeSlotAvailable(&this->volOutputSlot);
}


/*
 * SurfaceMapper::~SurfaceMapper
 */
SurfaceMapper::~SurfaceMapper(void) {
    this->Release();
}


/*
 * SurfaceMapper::create
 */
bool SurfaceMapper::create(void) {
    return true; // TODO
}


/*
 * SurfaceMapper::getVtxData
 */
bool SurfaceMapper::getVtxData(core::Call& call) {
    return true; // TODO
}


/*
 * SurfaceMapper::getVtxExtent
 */
bool SurfaceMapper::getVtxExtent(core::Call& call) {
    return true; // TODO
}


/*
 * SurfaceMapper::getVolData
 */
bool SurfaceMapper::getVolData(core::Call& call) {
    return true; // TODO
}


/*
 * SurfaceMapper::getVolExtent
 */
bool SurfaceMapper::getVolExtent(core::Call& call) {
    return true; // TODO
}


/*
 * SurfaceMapper::release
 */
void SurfaceMapper::release(void) {
    // TODO
}


