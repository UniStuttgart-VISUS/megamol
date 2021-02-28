/*
 * View3D.cpp
 *
 * Copyright (C) 2018, 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/View3D.h"
#include "mmcore/view/CallRenderView.h"


using namespace megamol::core;
using namespace megamol::core::view;

/*
 * View3D::View3D
 */
View3D::View3D(void)
    : view::AbstractView3D() {
    this->_rhsRenderSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->_rhsRenderSlot);

    this->MakeSlotAvailable(&this->_lhsRenderSlot);

}

/*
 * View3D::~View3D
 */
View3D::~View3D(void) {
    this->Release();
}

/*
 * View3D::Render
 */
void View3D::Render(double time, double instanceTime) {

    CallRender3D* cr3d = this->_rhsRenderSlot.CallAs<CallRender3D>();
    this->handleCameraMovement();

    if (cr3d == NULL) {
        return;
    }

    cr3d->SetFramebuffer(_framebuffer);

    AbstractView3D::beforeRender(time, instanceTime);

    cr3d->SetCamera(this->_camera);
    (*cr3d)(view::CallRender3D::FnRender);

    AbstractView3D::afterRender();

}

void megamol::core::view::View3D::ResetView() {
    AbstractView3D::ResetView(static_cast<float>(_framebuffer->width) / static_cast<float>(_framebuffer->height));
}

void megamol::core::view::View3D::Resize(unsigned int width, unsigned int height) {
    _framebuffer->width = width;
    _framebuffer->height = height;
    //TODO reallocate buffer?
}
