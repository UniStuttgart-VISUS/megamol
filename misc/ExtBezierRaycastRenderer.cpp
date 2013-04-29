/*
 * ExtBezierRaycastRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "glh/glh_extensions.h"
#include "ExtBezierRaycastRenderer.h"
#include "misc/ExtBezierDataCall.h"
#include "CoreInstance.h"
#include "utility/ShaderSourceFactory.h"
#include "vislib/Exception.h"
#include "vislib/Log.h"
#include "vislib/ShaderSource.h"
//#include <cmath>

using namespace megamol::core;


/*
 * misc::ExtBezierRaycastRenderer::ExtBezierRaycastRenderer
 */
misc::ExtBezierRaycastRenderer::ExtBezierRaycastRenderer(void) : misc::AbstractBezierRaycastRenderer(),
        pbEllShader() {

    this->getDataSlot.SetCompatibleCall<ExtBezierDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * misc::ExtBezierRaycastRenderer::~ExtBezierRaycastRenderer
 */
misc::ExtBezierRaycastRenderer::~ExtBezierRaycastRenderer(void) {
    this->Release();
}


/*
 * misc::ExtBezierRaycastRenderer::create
 */
bool misc::ExtBezierRaycastRenderer::create(void) {
    using vislib::sys::Log;
    if (!AbstractBezierRaycastRenderer::create()) return false;

    vislib::graphics::gl::ShaderSource frag;
    vislib::graphics::gl::ShaderSource vert;

    try {
        frag.Clear();
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("extbezier::frag", frag)) {
            throw vislib::Exception("Unable to compile fragment shader", __FILE__, __LINE__);
        }
        vert.Clear();
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("extbezier::pointVert", vert)) {
            throw vislib::Exception("Unable to compile vertex shader", __FILE__, __LINE__);
        }

        if (!this->pbEllShader.Compile(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            throw vislib::Exception("Unable to compile shader", __FILE__, __LINE__);
        }

        this->pbEllShader.BindAttribute((this->pbEllShaderYAx1Pos = 2), "yax1");
        this->pbEllShader.BindAttribute((this->pbEllShaderPos2Pos = 3), "pos2");
        this->pbEllShader.BindAttribute((this->pbEllShaderYAx2Pos = 4), "yax2");
        this->pbEllShader.BindAttribute((this->pbEllShaderPos3Pos = 5), "pos3");
        this->pbEllShader.BindAttribute((this->pbEllShaderYAx3Pos = 6), "yax3");
        this->pbEllShader.BindAttribute((this->pbEllShaderPos4Pos = 7), "pos4");
        this->pbEllShader.BindAttribute((this->pbEllShaderYAx4Pos = 8), "yax4");
        this->pbEllShader.BindAttribute((this->pbEllShaderColours = 9), "cols");

        if (!this->pbEllShader.Link()) {
            throw vislib::Exception("Unable to link shader", __FILE__, __LINE__);
        }

    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteError("Unable to compile shader: %s\n", ex.GetMsgA());
        return false;
    } catch(...) {
        Log::DefaultLog.WriteError("Unable to compile shader: Unknown exception\n");
        return false;
    }

    this->shader = &this->pbEllShader;

    return true;
}


/*
 * misc::ExtBezierRaycastRenderer::release
 */
void misc::ExtBezierRaycastRenderer::release(void) {
    AbstractBezierRaycastRenderer::release();

    this->shader = NULL;
    this->pbEllShader.Release();

}


/*
 * misc::ExtBezierRaycastRenderer::render
 */
bool misc::ExtBezierRaycastRenderer::render(view::CallRender3D& call) {

    ExtBezierDataCall *ebdc = this->getDataSlot.CallAs<ExtBezierDataCall>();
    if ((ebdc == NULL) || (!(*ebdc)(0))) return false;

    ::glScalef(this->scaling, this->scaling, this->scaling);

    ::glDisable(GL_BLEND);
    ::glEnable(GL_DEPTH_TEST);
    ::glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    float viewportStuff[4];
    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
    ::glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    this->pbEllShader.Enable();
    this->pbEllShader.SetParameterArray4("viewAttr", 1, viewportStuff);

    ::glBegin(GL_POINTS);
    for (unsigned int i = 0; i < ebdc->CountElliptic(); i++) {
        const vislib::math::BezierCurve<ExtBezierDataCall::Point, 3>& curve = ebdc->EllipticCurves()[i];
        const ExtBezierDataCall::Point *cp;
        float c1, c2, c3;

        cp = &curve.ControlPoint(1);
        ::glVertexAttrib4fARB(this->pbEllShaderPos2Pos, cp->GetPosition().X(), cp->GetPosition().Y(), cp->GetPosition().Z(), cp->GetRadiusY());
        ::glVertexAttrib4fARB(this->pbEllShaderYAx2Pos, cp->GetY().X(), cp->GetY().Y(), cp->GetY().Z(), cp->GetRadiusZ());
        c1 = static_cast<float>(
            ((cp->GetColour().R() / 4) * 64
            + (cp->GetColour().G() / 4)) * 64
            + (cp->GetColour().B() / 4));

        cp = &curve.ControlPoint(2);
        ::glVertexAttrib4fARB(this->pbEllShaderPos3Pos, cp->GetPosition().X(), cp->GetPosition().Y(), cp->GetPosition().Z(), cp->GetRadiusY());
        ::glVertexAttrib4fARB(this->pbEllShaderYAx3Pos, cp->GetY().X(), cp->GetY().Y(), cp->GetY().Z(), cp->GetRadiusZ());
        c2 = static_cast<float>(
            ((cp->GetColour().R() / 4) * 64
            + (cp->GetColour().G() / 4)) * 64
            + (cp->GetColour().B() / 4));

        cp = &curve.ControlPoint(3);
        ::glVertexAttrib4fARB(this->pbEllShaderPos4Pos, cp->GetPosition().X(), cp->GetPosition().Y(), cp->GetPosition().Z(), cp->GetRadiusY());
        ::glVertexAttrib4fARB(this->pbEllShaderYAx4Pos, cp->GetY().X(), cp->GetY().Y(), cp->GetY().Z(), cp->GetRadiusZ());
        c3 = static_cast<float>(
            ((cp->GetColour().R() / 4) * 64
            + (cp->GetColour().G() / 4)) * 64
            + (cp->GetColour().B() / 4));

        cp = &curve.ControlPoint(0);
        ::glVertexAttrib4fARB(this->pbEllShaderYAx1Pos, cp->GetY().X(), cp->GetY().Y(), cp->GetY().Z(), cp->GetRadiusZ());
        ::glVertexAttrib4fARB(this->pbEllShaderColours,
            static_cast<float>(
            ((cp->GetColour().R() / 4) * 64
            + (cp->GetColour().G() / 4)) * 64
            + (cp->GetColour().B() / 4)),
            c1, c2, c3);
        ::glVertex4f(cp->GetPosition().X(), cp->GetPosition().Y(), cp->GetPosition().Z(), cp->GetRadiusY());

    }
    ::glEnd();

    this->pbEllShader.Disable();

    ebdc->Unlock();
    return true;
}
