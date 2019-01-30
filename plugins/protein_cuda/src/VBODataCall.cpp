//
// VBODataCall.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 31, 2013
//     Author: scharnkn
//


#include "stdafx.h"
#include "VBODataCall.h"

using namespace megamol;
using namespace megamol::protein_cuda;


const unsigned int VBODataCall::CallForGetExtent = 0;
const unsigned int VBODataCall::CallForGetData = 1;


/*
 * VBODataCall::VBODataCall
 */
VBODataCall::VBODataCall(void) : Call(), vbo(0), tex(0), dataStride(-1),
        dataOffsPosition(-1), dataOffsNormal(-1), dataOffsTexCoord(-1),
        vboHandle(NULL) {

}


/*
 * VBODataCall::~VBODataCall
 */
VBODataCall::~VBODataCall(void) {

}
