/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "BaseAmortizedRenderer2D.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol;
using namespace megamol::infovis_gl;
using megamol::core::utility::log::Log;

BaseAmortizedRenderer2D::BaseAmortizedRenderer2D()
        : Renderer2D()
        , nextRendererSlot("nextRenderer", "connects to following Renderers, that will render in reduced resolution.")
        , enabledParam("Enabled", "Turn on switch") {
    this->nextRendererSlot.SetCompatibleCall<megamol::core_gl::view::CallRender2DGLDescription>();
    this->MakeSlotAvailable(&this->nextRendererSlot);

    this->enabledParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&enabledParam);
}

bool BaseAmortizedRenderer2D::create() {
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        Log::DefaultLog.WriteWarn("Ignore glError() from previous modules: %i", error);
    }

    auto const shaderOptions = msf::ShaderFactoryOptionsOpenGL(GetCoreInstance()->GetShaderPaths());
    createImpl(shaderOptions);

    return true;
}

void BaseAmortizedRenderer2D::release() {
    releaseImpl();
}

bool BaseAmortizedRenderer2D::GetExtents(core_gl::view::CallRender2DGL& call) {
    core_gl::view::CallRender2DGL* cr2d = this->nextRendererSlot.CallAs<core_gl::view::CallRender2DGL>();
    if (cr2d == nullptr) {
        return false;
    }

    cr2d->SetCamera(call.GetCamera());

    if (!(*cr2d)(core_gl::view::CallRender2DGL::FnGetExtents)) {
        return false;
    }

    call.SetTimeFramesCount(cr2d->TimeFramesCount());
    call.SetIsInSituTime(cr2d->IsInSituTime());

    call.AccessBoundingBoxes() = cr2d->GetBoundingBoxes();

    return true;
}

bool BaseAmortizedRenderer2D::Render(core_gl::view::CallRender2DGL& call) {
    core_gl::view::CallRender2DGL* cr2d = this->nextRendererSlot.CallAs<core_gl::view::CallRender2DGL>();

    if (cr2d == nullptr) {
        // Nothing to do really
        return true;
    }

    cr2d->SetTime(call.Time());
    cr2d->SetInstanceTime(call.InstanceTime());
    cr2d->SetLastFrameTime(call.LastFrameTime());

    cr2d->SetBackgroundColor(call.BackgroundColor());
    cr2d->AccessBoundingBoxes() = call.GetBoundingBoxes();
    cr2d->SetViewResolution(call.GetViewResolution());

    if (this->enabledParam.Param<core::param::BoolParam>()->Value()) {
        return renderImpl(call, *cr2d);
    } else {
        cr2d->SetFramebuffer(call.GetFramebuffer());
        cr2d->SetCamera(call.GetCamera());

        // send call to next renderer in line
        (*cr2d)(core::view::AbstractCallRender::FnRender);
    }
    return true;
}

bool BaseAmortizedRenderer2D::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    auto* cr = this->nextRendererSlot.CallAs<megamol::core_gl::view::CallRender2DGL>();
    if (cr) {
        megamol::core::view::InputEvent evt;
        evt.tag = megamol::core::view::InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cr->SetInputEvent(evt);
        return (*cr)(megamol::core_gl::view::CallRender2DGL::FnOnMouseButton);
    }
    return false;
}

bool BaseAmortizedRenderer2D::OnMouseMove(double x, double y) {
    auto* cr = this->nextRendererSlot.CallAs<megamol::core_gl::view::CallRender2DGL>();
    if (cr) {
        megamol::core::view::InputEvent evt;
        evt.tag = megamol::core::view::InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        cr->SetInputEvent(evt);
        return (*cr)(megamol::core_gl::view::CallRender2DGL::FnOnMouseMove);
    }
    return false;
}

bool BaseAmortizedRenderer2D::OnMouseScroll(double dx, double dy) {
    auto* cr = this->nextRendererSlot.CallAs<megamol::core_gl::view::CallRender2DGL>();
    if (cr) {
        megamol::core::view::InputEvent evt;
        evt.tag = megamol::core::view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;
        cr->SetInputEvent(evt);
        return (*cr)(megamol::core_gl::view::CallRender2DGL::FnOnMouseScroll);
    }
    return false;
}

bool BaseAmortizedRenderer2D::OnChar(unsigned int codePoint) {
    auto* cr = this->nextRendererSlot.CallAs<megamol::core_gl::view::CallRender2DGL>();
    if (cr) {
        megamol::core::view::InputEvent evt;
        evt.tag = megamol::core::view::InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;
        cr->SetInputEvent(evt);
        return (*cr)(megamol::core_gl::view::CallRender2DGL::FnOnChar);
    }
    return false;
}

bool BaseAmortizedRenderer2D::OnKey(
    megamol::core::view::Key key, megamol::core::view::KeyAction action, megamol::core::view::Modifiers mods) {
    auto* cr = this->nextRendererSlot.CallAs<megamol::core_gl::view::CallRender2DGL>();
    if (cr) {
        megamol::core::view::InputEvent evt;
        evt.tag = megamol::core::view::InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        cr->SetInputEvent(evt);
        return (*cr)(megamol::core_gl::view::CallRender2DGL::FnOnKey);
    }
    return false;
}
