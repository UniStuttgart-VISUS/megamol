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
View3D::View3D(void) : view::AbstractView3D<CPUFramebuffer, cpu_fbo_resize, Camera3DController, Camera3DParameters>() {
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
ImageWrapper View3D::Render(double time, double instanceTime, bool present_fbo) {

    CallRender3D* cr3d = this->_rhsRenderSlot.CallAs<CallRender3D>();

    if (cr3d == NULL) {
        cr3d->SetFramebuffer(_fbo);
    
        AbstractView3D::beforeRender(time, instanceTime);
    
        cr3d->SetCamera(this->_camera);
        (*cr3d)(view::CallRender3D::FnRender);
    
        AbstractView3D::afterRender();
    }

    ImageWrapper::DataChannels channels =
        ImageWrapper::DataChannels::RGBA8; // vislib::graphics::gl::FramebufferObject seems to use RGBA8
    void* data_pointer = _fbo->colorBuffer.data();
    size_t fbo_width = _fbo->width;
    size_t fbo_height = _fbo->height;

    return frontend_resources::wrap_image<WrappedImageType::ByteArray>({fbo_width, fbo_height}, data_pointer, channels);
}

void megamol::core::view::View3D::ResetView() {
    AbstractView3D::ResetView(static_cast<float>(_fbo->width) / static_cast<float>(_fbo->height));
}
