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
    this->rhsRenderSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rhsRenderSlot);

    this->MakeSlotAvailable(&this->lhsRenderSlot);

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
void View3D::Render(const mmcRenderViewContext& context) {

    CallRender3D* cr3d = this->rhsRenderSlot.CallAs<CallRender3D>();
    this->handleCameraMovement();

    if (cr3d == NULL) {
        return;
    }

    if (this->lhsCall == nullptr) {
        _framebuffer->width = cam.image_tile().width();
        _framebuffer->height = cam.image_tile().height();
        cr3d->SetFramebuffer(_framebuffer);
    }
    else {
        auto cpu_call = dynamic_cast<view::CallRenderView*>(this->lhsCall);
        cr3d->SetFramebuffer(cpu_call->GetFramebuffer());
    }

    AbstractView3D::beforeRender(context);

    cr3d->SetCameraState(this->cam);
    (*cr3d)(view::CallRender3D::FnRender);

    AbstractView3D::afterRender(context);

}
