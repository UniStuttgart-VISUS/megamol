/*
 * BlinnPhongRendererDeferred.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "BlinnPhongRendererDeferred.h"
#include "CallRender3D.h"
//#include "CallRenderDeferred3D.h"


using namespace megamol;
using namespace megamol::core;


/*
 * view::BlinnPhongRendererDeferred::BlinnPhongRendererDeferred
 */
view::BlinnPhongRendererDeferred::BlinnPhongRendererDeferred(void)
    : AbstractRendererDeferred3D() {
}


/*
 * view::BlinnPhongRendererDeferred::create
 */
bool view::BlinnPhongRendererDeferred::create(void) {
    return true;
}


/*
 * view::BlinnPhongRendererDeferred::release
 */
void view::BlinnPhongRendererDeferred::release(void) {
}


/*
 * view::BlinnPhongRendererDeferred::~BlinnPhongRendererDeferred
 */
view::BlinnPhongRendererDeferred::~BlinnPhongRendererDeferred(void) {
    this->Release();
}


/*
 * view::BlinnPhongRendererDeferred::GetCapabilities
 */
bool view::BlinnPhongRendererDeferred::GetCapabilities(Call& call) {

    CallRender3D *crIn = dynamic_cast<CallRender3D*>(&call);
    if(crIn == NULL) return false;

    CallRender3D *crOut = this->rendererSlot.CallAs<CallRender3D>();
    //CallRenderDeferred3D *crOut = this->rendererSlot.CallAs<CallRenderDeferred3D>();
    if(crOut == NULL) return false;

    // Call for getCapabilities
    if(!(*crOut)(2)) return false;

    // Set capabilities of for incoming render call
    *crIn = *crOut;

    return true;
}


/*
 * view::BlinnPhongRendererDeferred::GetExtents
 */
bool view::BlinnPhongRendererDeferred::GetExtents(Call& call) {

    CallRender3D *crIn = dynamic_cast<CallRender3D*>(&call);
    if(crIn == NULL) return false;

    CallRender3D *crOut = this->rendererSlot.CallAs<CallRender3D>();
    //CallRenderDeferred3D *crOut = this->rendererSlot.CallAs<CallRenderDeferred3D>();
    if(crOut == NULL) return false;

    // Call for getExtends
    if(!(*crOut)(1)) return false;

    // Set capabilities of for incoming render call
    *crIn = *crOut;

    return true;
}


/*
 * view::BlinnPhongRendererDeferred::Render
 */
bool view::BlinnPhongRendererDeferred::Render(Call& call) {

    CallRender3D *crIn = dynamic_cast<CallRender3D*>(&call);
    if(crIn == NULL) return false;

    CallRender3D *crOut = this->rendererSlot.CallAs<CallRender3D>();
    //CallRenderDeferred3D *crOut = this->rendererSlot.CallAs<CallRenderDeferred3D>();
    if(crOut == NULL) return false;

    // Get camera settings, outputbuffer etc
    // TODO
    *crOut = *crIn;

    // Call for render
    (*crOut)(0);

    return true;
}
