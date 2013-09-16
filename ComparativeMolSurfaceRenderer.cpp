//
// ComparativeMolSurfaceRenderer.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 16, 2013
// Author     : scharnkn
//

#include "stdafx.h"
#include "ComparativeMolSurfaceRenderer.h"
#include "VBODataCall.h"
#include "VTIDataCall.h"
#include "MolecularDataCall.h"
#include "view/AbstractCallRender3D.h"

#ifdef WITH_CUDA

using namespace megamol;
using namespace megamol::protein;


/*
 * ComparativeMolSurfaceRenderer::ComparativeMolSurfaceRenderer
 */
ComparativeMolSurfaceRenderer::ComparativeMolSurfaceRenderer(void) :
        Renderer3DModuleDS(),
        vtxInputSlot("vtxIn", "Incoming vertex data representing the mapped surface"),
        vtxOutputSlot("vtxOut", "Outgoing vertex data representing the source mesh"),
        volOutputSlot("volOut", "Initial external forces representing the target shape"),
        molDataSlot1("molIn1", "Input molecule #1"),
        molDataSlot2("molIn2", "Input molecule #2") {

    /* Make data caller/callee slots available */

    // Vertex input
    this->vtxInputSlot.SetCompatibleCall<VBODataCallDescription>();
    this->MakeSlotAvailable(&this->vtxInputSlot);

    // Vertex output
    this->vtxOutputSlot.SetCallback(
            VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetData),
            &ComparativeMolSurfaceRenderer::getVtxData);
    this->vtxOutputSlot.SetCallback(
            VBODataCall::ClassName(),
            VBODataCall::FunctionName(VBODataCall::CallForGetExtent),
            &ComparativeMolSurfaceRenderer::getVolExtent);
    this->MakeSlotAvailable(&this->vtxOutputSlot);

    // Volume output
    this->volOutputSlot.SetCallback(
            VTIDataCall::ClassName(),
            VTIDataCall::FunctionName(VTIDataCall::CallForGetData),
            &ComparativeMolSurfaceRenderer::getVolData);
    this->volOutputSlot.SetCallback(
            VTIDataCall::ClassName(),
            VTIDataCall::FunctionName(VTIDataCall::CallForGetExtent),
            &ComparativeMolSurfaceRenderer::getVolExtent);
    this->MakeSlotAvailable(&this->volOutputSlot);

    // Molecular data input #1
    this->molDataSlot1.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataSlot1);

    // Molecular data input #2
    this->molDataSlot2.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataSlot2);
}


/*
 * ComparativeMolSurfaceRenderer::~ComparativeMolSurfaceRenderer
 */
ComparativeMolSurfaceRenderer::~ComparativeMolSurfaceRenderer(void) {
    this->Release();
}


/*
 * ComparativeMolSurfaceRenderer::create
 */
bool ComparativeMolSurfaceRenderer::create(void) {
    return true; // TODO
}


/*
 * ComparativeMolSurfaceRenderer::GetCapabilities
 */
bool ComparativeMolSurfaceRenderer::GetCapabilities(core::Call& call) {
    core::view::AbstractCallRender3D *cr3d =
            dynamic_cast<core::view::AbstractCallRender3D *>(&call);

    if (cr3d == NULL) {
        return false;
    }

    cr3d->SetCapabilities(core::view::AbstractCallRender3D::CAP_RENDER |
                          core::view::AbstractCallRender3D::CAP_LIGHTING |
                          core::view::AbstractCallRender3D::CAP_ANIMATION);

    return true;
}


/*
 * ComparativeMolSurfaceRenderer::GetExtents
 */
bool ComparativeMolSurfaceRenderer::GetExtents(core::Call& call) {
    return true; // TODO
}


/*
 * ComparativeMolSurfaceRenderer::getVtxData
 */
bool ComparativeMolSurfaceRenderer::getVtxData(core::Call& call) {
    return true; // TODO
}

/*
 * ComparativeMolSurfaceRenderer::getVtxExtent
 */
bool ComparativeMolSurfaceRenderer::getVtxExtent(core::Call& call) {
    return true; // TODO
}


/*
 * ComparativeMolSurfaceRenderer::getVolData
 */
bool ComparativeMolSurfaceRenderer::getVolData(core::Call& call) {
    return true; // TODO
}


/*
 * ComparativeMolSurfaceRenderer::getVolExtent
 */
bool ComparativeMolSurfaceRenderer::getVolExtent(core::Call& call) {
    return true; // TODO
}


/*
 * ComparativeMolSurfaceRenderer::release
 */
void ComparativeMolSurfaceRenderer::release(void) {
    // TODO
}


/*
 *  ComparativeMolSurfaceRenderer::Render
 */
bool ComparativeMolSurfaceRenderer::Render(core::Call& call) {
    return true; // TODO
}

#endif // WITH_CUDA


