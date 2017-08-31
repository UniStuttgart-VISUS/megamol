/*
 * AnaglyphStereoView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/special/AnaglyphStereoView.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/CallRenderView.h"
#include "vislib/sys/Log.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

using namespace megamol::core;


/*
 * view::special::AnaglyphStereoView::AnaglyphStereoView
 */
view::special::AnaglyphStereoView::AnaglyphStereoView(void)
        : AbstractStereoView(), leftBuffer(), rightBuffer(), shader(),
        leftColourSlot("lefteyecol", "The left eye colour"),
        rightColourSlot("righteyecol", "The right eye colour"),
        colourPresetsSlot("eyecolpreset", "Available colour presets") {

    this->leftColour[0] = 0.0f;
    this->leftColour[1] = 1.0f;
    this->leftColour[2] = 0.0f;
    this->leftColourSlot << new param::StringParam(utility::ColourParser::ToString(
        this->leftColour[0], this->leftColour[1], this->leftColour[2]));
    this->MakeSlotAvailable(&this->leftColourSlot);

    this->rightColour[0] = 1.0f;
    this->rightColour[1] = 0.0f;
    this->rightColour[2] = 1.0f;
    this->rightColourSlot << new param::StringParam(utility::ColourParser::ToString(
        this->rightColour[0], this->rightColour[1], this->rightColour[2]));
    this->MakeSlotAvailable(&this->rightColourSlot);

    param::EnumParam *presets = new param::EnumParam(0);
    presets->SetTypePair(0, "Custom");
    presets->SetTypePair(1, "Red-Green");
    presets->SetTypePair(2, "Red-Cyan");
    presets->SetTypePair(3, "Green-Magenta");
    this->colourPresetsSlot << presets;
    this->MakeSlotAvailable(&this->colourPresetsSlot);
}


/*
 * view::special::AnaglyphStereoView::~AnaglyphStereoView
 */
view::special::AnaglyphStereoView::~AnaglyphStereoView(void) {
    this->Release();
}


/*
 * view::special::AnaglyphStereoView::Resize
 */
void view::special::AnaglyphStereoView::Resize(unsigned int width, unsigned int height) {
    AbstractOverrideView::Resize(width, height);
    if ((width > 0) && (height > 0)) {
        this->leftBuffer.Release();
        this->leftBuffer.Create(width, height);
        this->rightBuffer.Release();
        this->rightBuffer.Create(width, height);
    }
}


/*
 * view::special::AnaglyphStereoView::Render
 */
void view::special::AnaglyphStereoView::Render(const mmcRenderViewContext& context) {
    CallRenderView *crv = this->getCallRenderView();
    if (crv == NULL) return;
    crv->SetTime(static_cast<float>(context.Time));
    crv->SetInstanceTime(context.InstanceTime);
    crv->SetGpuAffinity(context.GpuAffinity);

    if (this->colourPresetsSlot.IsDirty()) {
        this->colourPresetsSlot.ResetDirty();
        int preset = this->colourPresetsSlot.Param<param::EnumParam>()->Value();
        if (preset == 0) {
            this->leftColourSlot.ForceSetDirty();
            this->rightColourSlot.ForceSetDirty();
        } else {
            switch (preset) {
                case 1: // red-green
                    this->leftColour[0] = 1.0f;
                    this->leftColour[1] = 0.0f;
                    this->leftColour[2] = 0.0f;
                    this->rightColour[0] = 0.0f;
                    this->rightColour[1] = 1.0f;
                    this->rightColour[2] = 0.0f;
                    break;
                case 2: // red-cyan
                    this->leftColour[0] = 1.0f;
                    this->leftColour[1] = 0.0f;
                    this->leftColour[2] = 0.0f;
                    this->rightColour[0] = 0.0f;
                    this->rightColour[1] = 1.0f;
                    this->rightColour[2] = 1.0f;
                    break;
                case 3: // green-magenta
                    this->leftColour[0] = 0.0f;
                    this->leftColour[1] = 1.0f;
                    this->leftColour[2] = 0.0f;
                    this->rightColour[0] = 1.0f;
                    this->rightColour[1] = 0.0f;
                    this->rightColour[2] = 1.0f;
                    break;
                default:
                    preset = 0;
                    this->leftColourSlot.ForceSetDirty();
                    this->rightColourSlot.ForceSetDirty();
                    return;
            }
            if (preset != 0) {
                this->leftColourSlot.Param<param::StringParam>()->SetValue(
                    utility::ColourParser::ToString(
                    this->leftColour[0], this->leftColour[1], this->leftColour[2]), false);
                this->rightColourSlot.Param<param::StringParam>()->SetValue(
                    utility::ColourParser::ToString(
                    this->rightColour[0], this->rightColour[1], this->rightColour[2]), false);
            }
        }
    }
    if (this->leftColourSlot.IsDirty()) {
        this->leftColourSlot.ResetDirty();
        utility::ColourParser::FromString(this->leftColourSlot.Param<param::StringParam>()->Value(),
            this->leftColour[0], this->leftColour[1], this->leftColour[2]);
        this->colourPresetsSlot.Param<param::EnumParam>()->SetValue(0);
    }
    if (this->rightColourSlot.IsDirty()) {
        this->rightColourSlot.ResetDirty();
        utility::ColourParser::FromString(this->rightColourSlot.Param<param::StringParam>()->Value(),
            this->rightColour[0], this->rightColour[1], this->rightColour[2]);
        this->colourPresetsSlot.Param<param::EnumParam>()->SetValue(0);
    }

    vislib::graphics::CameraParameters::ProjectionType proj = this->getProjectionType();
    bool switchEyes = this->getSwitchEyes();

    UINT oldLevel = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(0);

    crv->SetProjection(proj,switchEyes
        ? vislib::graphics::CameraParameters::LEFT_EYE
        : vislib::graphics::CameraParameters::RIGHT_EYE);
    this->rightBuffer.Enable();
    crv->SetOutputBuffer(&this->rightBuffer);
    (*crv)(view::CallRenderView::CALL_RENDER);
    this->rightBuffer.Disable();

    crv->SetProjection(proj, switchEyes
        ? vislib::graphics::CameraParameters::RIGHT_EYE
        : vislib::graphics::CameraParameters::LEFT_EYE);
    this->leftBuffer.Enable();
    crv->SetOutputBuffer(&this->leftBuffer);
    (*crv)(view::CallRenderView::CALL_RENDER);
    this->leftBuffer.Disable();

    vislib::Trace::GetInstance().SetLevel(oldLevel);

    ::glDrawBuffer(GL_BACK);
    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();

    ::glDisable(GL_LIGHTING);
    ::glEnable(GL_TEXTURE_2D);
    ::glDisable(GL_BLEND);
    ::glDisable(GL_DEPTH_TEST);

    ::glActiveTextureARB(GL_TEXTURE0_ARB);
    this->leftBuffer.BindColourTexture();
    ::glActiveTextureARB(GL_TEXTURE1_ARB);
    this->rightBuffer.BindColourTexture();

    this->shader.Enable();
    this->shader.SetParameter("left", 0);
    this->shader.SetParameter("right", 1);
    this->shader.SetParameter("leftcol", this->leftColour[0],
        this->leftColour[1], this->leftColour[2]);
    this->shader.SetParameter("rightcol", this->rightColour[0],
        this->rightColour[1], this->rightColour[2]);
    ::glBegin(GL_QUADS);
    ::glColor3ub(255, 0, 0);
    ::glTexCoord2i(0, 0);
    ::glVertex2i(-1, -1);
    ::glTexCoord2i(1, 0);
    ::glVertex2i( 1, -1);
    ::glTexCoord2i(1, 1);
    ::glVertex2i( 1,  1);
    ::glTexCoord2i(0, 1);
    ::glVertex2i(-1,  1);
    ::glEnd();
    this->shader.Disable();

    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glActiveTextureARB(GL_TEXTURE0_ARB);
    ::glBindTexture(GL_TEXTURE_2D, 0);

}


/*
 * view::special::AnaglyphStereoView::create
 */
bool view::special::AnaglyphStereoView::create(void) {
    ASSERT(IsAvailable());

    if (!this->leftBuffer.Create(1, 1)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to initialize left FBO");
        return false;
    }
    if (!this->rightBuffer.Create(1, 1)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to initialize right FBO");
        return false;
    }

    try {
        if (!this->shader.Create(vislib::graphics::gl::GLSLShader::FTRANSFORM_VERTEX_SHADER_SRC,
"#version 120\n\
#extension GL_EXT_gpu_shader4 : enable\n\
\n\
uniform sampler2D left;\n\
uniform sampler2D right;\n\
uniform vec3 leftcol;\n\
uniform vec3 rightcol;\n\
\n\
void main() {\n\
    vec4 lcol = texelFetch2D(left, ivec2(gl_FragCoord.xy), 0);\n\
    float llum = (lcol.r + lcol.g + lcol.b) / 3.0;\n\
    vec4 rcol = texelFetch2D(right, ivec2(gl_FragCoord.xy), 0);\n\
    float rlum = (rcol.r + rcol.g + rcol.b) / 3.0;\n\
    gl_FragColor = vec4(leftcol * llum + rightcol * rlum, 1.0);\n\
}\n\
")) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile shader");
            return false;
        }
    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile shader: %s", ex.GetMsgA());
        return false;
    }

    return true;
}


/*
 * view::special::AnaglyphStereoView::release
 */
void view::special::AnaglyphStereoView::release(void) {
    this->leftBuffer.Release();
    this->rightBuffer.Release();
    this->shader.Release();
}
