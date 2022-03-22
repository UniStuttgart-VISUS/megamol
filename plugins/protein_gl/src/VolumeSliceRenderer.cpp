/*
 * VolumeSliceRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All Rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "VolumeSliceRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"
#include "mmcore_gl/view/Renderer2DModuleGL.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"
#include "vislib_gl/graphics/gl/ShaderSource.h"
#include <math.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_gl;
using namespace megamol::core::utility::log;


/*
 * VolumeSliceRenderer::VolumeSliceRenderer (CTOR)
 */
VolumeSliceRenderer::VolumeSliceRenderer(void)
        : core_gl::view::Renderer2DModuleGL()
        , volDataCallerSlot("getData", "Connects the volume slice rendering with data storage") {
    // volume data caller slot
    this->volDataCallerSlot.SetCompatibleCall<protein::VolumeSliceCallDescription>();
    this->MakeSlotAvailable(&this->volDataCallerSlot);
}

/*
 * VolumeSliceRenderer::~VolumeSliceRenderer (DTOR)
 */
VolumeSliceRenderer::~VolumeSliceRenderer(void) {
    this->Release();
}

/*
 * VolumeSliceRenderer::create
 */
bool VolumeSliceRenderer::create() {

    using namespace vislib_gl::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;

    // Load sphere shader
    auto ssf = std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());
    if (!ssf->MakeShaderSource("volume::std::textureSliceVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for volume slice rendering", this->ClassName());
        return false;
    }
    if (!ssf->MakeShaderSource("volume::std::textureSliceFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "%s: Unable to load fragment shader source for volume slice rendering", this->ClassName());
        return false;
    }
    try {
        if (!this->volumeSliceShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create volume slice rendering shader: %s\n",
            this->ClassName(), e.GetMsgA());
        return false;
    }

    return true;
}

/*
 * VolumeSliceRenderer::release
 */
void VolumeSliceRenderer::release() {}

bool VolumeSliceRenderer::GetExtents(core_gl::view::CallRender2DGL& call) {
    // get pointer to VolumeSliceCall
    protein::VolumeSliceCall* volume = this->volDataCallerSlot.CallAs<protein::VolumeSliceCall>();
    if (volume == NULL)
        return false;
    // execute the call
    if (!(*volume)(protein::VolumeSliceCall::CallForGetData))
        return false;
    // check clip plane normal vector against axes
    float lenX = (volume->getClipPlaneNormal().Cross(vislib::math::Vector<float, 3>(-1, 0, 0))).Length();
    float lenY = (volume->getClipPlaneNormal().Cross(vislib::math::Vector<float, 3>(0, -1, 0))).Length();
    float lenZ = (volume->getClipPlaneNormal().Cross(vislib::math::Vector<float, 3>(0, 0, -1))).Length();
    // check axes
    if (vislib::math::IsEqual(lenX, 0.0f)) {
        // positive x-axis
        call.AccessBoundingBoxes().SetBoundingBox(
            0.0f, 0.0f, volume->getBBoxDimensions().Y(), volume->getBBoxDimensions().Z());
    } else if (vislib::math::IsEqual(lenY, 0.0f)) {
        // positive y-axis
        call.AccessBoundingBoxes().SetBoundingBox(
            0.0f, 0.0f, volume->getBBoxDimensions().X(), volume->getBBoxDimensions().Z());
    } else if (vislib::math::IsEqual(lenZ, 0.0f)) {
        // positive z-axis
        call.AccessBoundingBoxes().SetBoundingBox(
            0.0f, 0.0f, volume->getBBoxDimensions().X(), volume->getBBoxDimensions().Y());
    } else {
        // default
        call.AccessBoundingBoxes().SetBoundingBox(0.0f, 0.0f, 1.0f, 1.0f);
    }

    return true;
}


/*
 * Callback for mouse events (move, press, and release)
 */
bool VolumeSliceRenderer::MouseEvent(float x, float y, view::MouseFlags flags) {
    // get pointer to VolumeSliceCall
    protein::VolumeSliceCall* volume = this->volDataCallerSlot.CallAs<protein::VolumeSliceCall>();
    if (volume == NULL)
        return false;
    // execute the call
    if (!(*volume)(protein::VolumeSliceCall::CallForGetData))
        return false;
    // check clip plane normal vector against axes
    float lenX = (volume->getClipPlaneNormal().Cross(vislib::math::Vector<float, 3>(-1, 0, 0))).Length();
    float lenY = (volume->getClipPlaneNormal().Cross(vislib::math::Vector<float, 3>(0, -1, 0))).Length();
    float lenZ = (volume->getClipPlaneNormal().Cross(vislib::math::Vector<float, 3>(0, 0, -1))).Length();
    // check axes
    if (vislib::math::IsEqual(lenX, 0.0f)) {
        // positive x-axis
        this->mousePos.Set(
            1.0f - volume->getTexRCoord(), x / volume->getBBoxDimensions().Y(), y / volume->getBBoxDimensions().Z());
    } else if (vislib::math::IsEqual(lenY, 0.0f)) {
        // positive y-axis
        this->mousePos.Set(
            x / volume->getBBoxDimensions().X(), 1.0f - volume->getTexRCoord(), y / volume->getBBoxDimensions().Z());
    } else if (vislib::math::IsEqual(lenZ, 0.0f)) {
        // positive z-axis
        this->mousePos.Set(
            x / volume->getBBoxDimensions().X(), y / volume->getBBoxDimensions().Y(), 1.0f - volume->getTexRCoord());
    } else {
        // default
        this->mousePos.Set(0, 0, 0);
    }
    // set the mouse position to the call
    volume->setMousePos(this->mousePos);

    bool consumeEvent = false;

    if (((flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED) != 0) && ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) == 0)) {
        // set the clicked mouse position
        volume->setClickPos(this->mousePos);
        consumeEvent = true;
    }

    return consumeEvent;
}

/*
 * VolumeSliceRenderer::Render
 */
bool VolumeSliceRenderer::Render(core_gl::view::CallRender2DGL& call) {

    // get pointer to VolumeSliceCall
    protein::VolumeSliceCall* volume = this->volDataCallerSlot.CallAs<protein::VolumeSliceCall>();
    if (volume == NULL)
        return false;
    // execute the call
    if (!(*volume)(protein::VolumeSliceCall::CallForGetData))
        return false;

    glEnable(GL_TEXTURE_3D);

    // enable slice shader
    this->volumeSliceShader.Enable();

    // set uniform variables
    glUniform1f(this->volumeSliceShader.ParameterLocation("isoValue"), volume->getIsovalue());
    glUniform1i(this->volumeSliceShader.ParameterLocation("volTex"), 0);

    // check if texture is available
    if (!volume->getVolumeTex())
        return false;
    // bind texture
    glBindTexture(GL_TEXTURE_3D, volume->getVolumeTex());
    // set color to white
    glColor3f(1, 1, 1);
    // start drawing a quad
    glBegin(GL_QUADS);
    float lenX = (volume->getClipPlaneNormal().Cross(vislib::math::Vector<float, 3>(-1, 0, 0))).Length();
    float dirX = volume->getClipPlaneNormal().Dot(vislib::math::Vector<float, 3>(-1, 0, 0));
    float lenY = (volume->getClipPlaneNormal().Cross(vislib::math::Vector<float, 3>(0, -1, 0))).Length();
    float dirY = volume->getClipPlaneNormal().Dot(vislib::math::Vector<float, 3>(0, -1, 0));
    float lenZ = (volume->getClipPlaneNormal().Cross(vislib::math::Vector<float, 3>(0, 0, -1))).Length();
    float dirZ = volume->getClipPlaneNormal().Dot(vislib::math::Vector<float, 3>(0, 0, -1));
    vislib::math::Vector<float, 3> box = volume->getBBoxDimensions();
    // check axis
    if (vislib::math::IsEqual(lenX, 0.0f) && (dirX > 0.0)) {
        // negativ x-axis
        glTexCoord3f(volume->getTexRCoord(), 0, 0);
        glVertex2f(0, 0);
        glTexCoord3f(volume->getTexRCoord(), 1, 0);
        glVertex2f(box.Y(), 0);
        glTexCoord3f(volume->getTexRCoord(), 1, 1);
        glVertex2f(box.Y(), box.Z());
        glTexCoord3f(volume->getTexRCoord(), 0, 1);
        glVertex2f(0, box.Z());
    } else if (vislib::math::IsEqual(lenX, 0.0f)) {
        // positive x-axis
        glTexCoord3f(1.0f - volume->getTexRCoord(), 0, 0);
        glVertex2f(0, 0);
        glTexCoord3f(1.0f - volume->getTexRCoord(), 1, 0);
        glVertex2f(box.Y(), 0);
        glTexCoord3f(1.0f - volume->getTexRCoord(), 1, 1);
        glVertex2f(box.Y(), box.Z());
        glTexCoord3f(1.0f - volume->getTexRCoord(), 0, 1);
        glVertex2f(0, box.Z());
    } else if (vislib::math::IsEqual(lenY, 0.0f) && (dirY > 0.0)) {
        // negativ y-axis
        glTexCoord3f(0, volume->getTexRCoord(), 0);
        glVertex2f(0, 0);
        glTexCoord3f(1, volume->getTexRCoord(), 0);
        glVertex2f(box.X(), 0);
        glTexCoord3f(1, volume->getTexRCoord(), 1);
        glVertex2f(box.X(), box.Z());
        glTexCoord3f(0, volume->getTexRCoord(), 1);
        glVertex2f(0, box.Z());
    } else if (vislib::math::IsEqual(lenY, 0.0f)) {
        // positive y-axis
        glTexCoord3f(0, 1.0f - volume->getTexRCoord(), 0);
        glVertex2f(0, 0);
        glTexCoord3f(1, 1.0f - volume->getTexRCoord(), 0);
        glVertex2f(box.X(), 0);
        glTexCoord3f(1, 1.0f - volume->getTexRCoord(), 1);
        glVertex2f(box.X(), box.Z());
        glTexCoord3f(0, 1.0f - volume->getTexRCoord(), 1);
        glVertex2f(0, box.Z());
    } else if (vislib::math::IsEqual(lenZ, 0.0f) && (dirZ > 0.0)) {
        // negativ z-axis
        glTexCoord3f(0, 0, volume->getTexRCoord());
        glVertex2f(0, 0);
        glTexCoord3f(1, 0, volume->getTexRCoord());
        glVertex2f(box.X(), 0);
        glTexCoord3f(1, 1, volume->getTexRCoord());
        glVertex2f(box.X(), box.Y());
        glTexCoord3f(0, 1, volume->getTexRCoord());
        glVertex2f(0, box.Y());
    } else if (vislib::math::IsEqual(lenZ, 0.0f)) {
        // positive z-axis
        glTexCoord3f(0, 0, 1.0f - volume->getTexRCoord());
        glVertex2f(0, 0);
        glTexCoord3f(1, 0, 1.0f - volume->getTexRCoord());
        glVertex2f(box.X(), 0);
        glTexCoord3f(1, 1, 1.0f - volume->getTexRCoord());
        glVertex2f(box.X(), box.Y());
        glTexCoord3f(0, 1, 1.0f - volume->getTexRCoord());
        glVertex2f(0, box.Y());
    }
    glEnd(); // GL_QUADS

    // disable slice shader
    this->volumeSliceShader.Disable();

    glDisable(GL_TEXTURE_3D);

    return true;
}
