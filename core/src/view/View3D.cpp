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
void View3D::Render(const mmcRenderViewContext& context, Call* call) {

    CallRender3D* cr3d = this->_rhsRenderSlot.CallAs<CallRender3D>();
    this->handleCameraMovement();

    if (cr3d == NULL) {
        return;
    }

    if (call == nullptr) {
        _framebuffer->width = _camera.image_tile().width();
        _framebuffer->height = _camera.image_tile().height();
        cr3d->SetFramebuffer(_framebuffer);
    }
    else {
        auto cpu_call = dynamic_cast<view::CallRenderView*>(call);
        cr3d->SetFramebuffer(cpu_call->GetFramebuffer());
    }

    AbstractView3D::beforeRender(context);

    cr3d->SetCamera(this->_camera);
    (*cr3d)(view::CallRender3D::FnRender);

    AbstractView3D::afterRender(context);

}
