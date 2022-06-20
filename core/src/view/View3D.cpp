/*
 * View3D.cpp
 *
 * Copyright (C) 2018, 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/View3D.h"

#include "GlobalValueStore.h"
#include "mmcore/view/CallRenderView.h"


using namespace megamol::core;
using namespace megamol::core::view;

/*
 * View3D::View3D
 */
View3D::View3D(void) {
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
ImageWrapper View3D::Render(double time, double instanceTime) {

    BaseView::beforeRender(time, instanceTime);

    CallRender3D* cr3d = this->_rhsRenderSlot.CallAs<CallRender3D>();

    if (cr3d != NULL) {
        cr3d->SetViewResolution({_fbo->getWidth(), _fbo->getHeight()});
        cr3d->SetFramebuffer(_fbo);
        cr3d->SetCamera(this->_camera);
        (*cr3d)(view::CallRender3D::FnRender);
    }

    BaseView::afterRender();

    return GetRenderingResult();
}

ImageWrapper megamol::core::view::View3D::GetRenderingResult() const {
    ImageWrapper::DataChannels channels =
        ImageWrapper::DataChannels::RGBA8; // vislib_gl::graphics::gl::FramebufferObject seems to use RGBA8
    size_t fbo_width = _fbo->width;
    size_t fbo_height = _fbo->height;

    return frontend_resources::wrap_image({fbo_width, fbo_height}, _fbo->colorBuffer, channels);
}

void megamol::core::view::View3D::Resize(unsigned int width, unsigned int height) {
    BaseView::Resize(width, height);

    _fbo->colorBuffer = std::vector<uint32_t>(width * height);
    _fbo->depthBuffer = std::vector<float>(width * height);
    _fbo->width = width;
    _fbo->height = height;
}

bool View3D::create() {

    _fbo = std::make_shared<CallRenderView::FBO_TYPE>();

    _fbo->depthBufferActive = false;
    _fbo->colorBuffer = std::vector<uint32_t>(1);
    _fbo->depthBuffer = std::vector<float>(1);
    _fbo->width = 1;
    _fbo->height = 1;
    _fbo->x = 0;
    _fbo->y = 0;

    const auto arcball_key = "arcball";

    // new frontend has global key-value resource
    auto maybe = this->frontend_resources.get<megamol::frontend_resources::GlobalValueStore>().maybe_get(arcball_key);
    if (maybe.has_value()) {
        this->_camera_controller.setArcballDefault(vislib::CharTraitsA::ParseBool(maybe.value().c_str()));
    }

    this->_firstImg = true;

    return true;
}
