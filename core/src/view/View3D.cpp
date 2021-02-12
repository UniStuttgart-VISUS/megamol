/*
 * View3D.cpp
 *
 * Copyright (C) 2018, 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/View3D.h"


using namespace megamol::core;
using namespace megamol::core::view;

/*
 * View3D::View3D
 */
View3D::View3D(void)
    : view::AbstractView3D() {
    this->rendererSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererSlot);

    this->MakeSlotAvailable(&this->renderSlot);

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

    CallRender3D* cr3d = this->rendererSlot.CallAs<CallRender3D>();
    this->handleCameraMovement();

    if (cr3d == NULL) {
        return;
    }
    cr3d->SetFramebuffer(_framebuffer);

    AbstractView3D::beforeRender(context);

    if (cr3d != nullptr) {
        cr3d->SetCameraState(this->cam);
        (*cr3d)(view::CallRender3D::FnRender);
    }

    AbstractView3D::afterRender(context);

}
