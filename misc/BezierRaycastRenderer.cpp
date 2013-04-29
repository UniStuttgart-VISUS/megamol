/*
 * BezierRaycastRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
//#include "glh/glh_extensions.h"
#include "BezierRaycastRenderer.h"
#include "misc/BezierDataCall.h"
#include "CoreInstance.h"
#include "utility/ShaderSourceFactory.h"
#include "vislib/Exception.h"
#include "vislib/Log.h"
#include "vislib/ShaderSource.h"
//#include <cmath>

using namespace megamol::core;


/*
 * misc::BezierRaycastRenderer::BezierRaycastRenderer
 */
misc::BezierRaycastRenderer::BezierRaycastRenderer(void) : misc::AbstractBezierRaycastRenderer(),
        pbShader() {

    this->getDataSlot.SetCompatibleCall<BezierDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * misc::BezierRaycastRenderer::~BezierRaycastRenderer
 */
misc::BezierRaycastRenderer::~BezierRaycastRenderer(void) {
    this->Release();
}


/*
 * misc::BezierRaycastRenderer::create
 */
bool misc::BezierRaycastRenderer::create(void) {
    using vislib::sys::Log;
    if (!AbstractBezierRaycastRenderer::create()) return false;

    vislib::graphics::gl::ShaderSource frag;
    vislib::graphics::gl::ShaderSource vert;

    try {
        frag.Clear();
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("common::ConstColFragment", frag)) {
            throw vislib::Exception("Unable to compile fragment shader", __FILE__, __LINE__);
        }
        vert.Clear();
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("bezier::pointVert", vert)) {
            throw vislib::Exception("Unable to compile vertex shader", __FILE__, __LINE__);
        }

        if (!this->pbShader.Compile(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            throw vislib::Exception("Unable to compile shader", __FILE__, __LINE__);
        }

        this->pbShader.BindAttribute((this->pbShaderPos2Pos = 8), "pos2");
        this->pbShader.BindAttribute((this->pbShaderPos3Pos = 9), "pos3");
        this->pbShader.BindAttribute((this->pbShaderPos4Pos = 10), "pos4");
        this->pbShader.BindAttribute((this->pbShaderCol2Pos = 11), "col2");
        this->pbShader.BindAttribute((this->pbShaderCol3Pos = 12), "col3");
        this->pbShader.BindAttribute((this->pbShaderCol4Pos = 13), "col4");

        if (!this->pbShader.Link()) {
            throw vislib::Exception("Unable to link shader", __FILE__, __LINE__);
        }

    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteError("Unable to compile shader: %s\n", ex.GetMsgA());
        return false;
    } catch(...) {
        Log::DefaultLog.WriteError("Unable to compile shader: Unknown exception\n");
        return false;
    }

    this->shader = &this->pbShader;

    return true;
}


/*
 * misc::BezierRaycastRenderer::release
 */
void misc::BezierRaycastRenderer::release(void) {
    AbstractBezierRaycastRenderer::release();

    this->shader = NULL;
    this->pbShader.Release();

}


/*
 * misc::BezierRaycastRenderer::render
 */
bool misc::BezierRaycastRenderer::render(view::CallRender3D& call) {

    BezierDataCall *bdc = this->getDataSlot.CallAs<BezierDataCall>();
    if ((bdc == NULL) || (!(*bdc)(0))) return false;

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

    this->shader->Enable();
    this->shader->SetParameterArray4("viewAttr", 1, viewportStuff);

    // TODO: Implement

    ::glBegin(GL_POINTS); // not cool, but ok for now
    for (unsigned int t = 0; t < bdc->Count(); t++) {
        const vislib::math::BezierCurve<BezierDataCall::BezierPoint, 3>& curve = bdc->Curves()[t];
        const BezierDataCall::BezierPoint *cp;

        cp = &curve.ControlPoint(1);
        ::glVertexAttrib3fARB(this->pbShaderCol2Pos,
            static_cast<float>(cp->R()) / 255.0f,
            static_cast<float>(cp->G()) / 255.0f,
            static_cast<float>(cp->B()) / 255.0f);
        ::glVertexAttrib4fARB(this->pbShaderPos2Pos, cp->X(), cp->Y(), cp->Z(), cp->Radius());

        cp = &curve.ControlPoint(2);
        ::glVertexAttrib3fARB(this->pbShaderCol3Pos,
            static_cast<float>(cp->R()) / 255.0f,
            static_cast<float>(cp->G()) / 255.0f,
            static_cast<float>(cp->B()) / 255.0f);
        ::glVertexAttrib4fARB(this->pbShaderPos3Pos, cp->X(), cp->Y(), cp->Z(), cp->Radius());

        cp = &curve.ControlPoint(3);
        ::glVertexAttrib3fARB(this->pbShaderCol4Pos,
            static_cast<float>(cp->R()) / 255.0f,
            static_cast<float>(cp->G()) / 255.0f,
            static_cast<float>(cp->B()) / 255.0f);
        ::glVertexAttrib4fARB(this->pbShaderPos4Pos, cp->X(), cp->Y(), cp->Z(), cp->Radius());

        cp = &curve.ControlPoint(0);
        ::glColor3ub(cp->R(), cp->G(), cp->B());
        ::glVertex4f(cp->X(), cp->Y(), cp->Z(), cp->Radius());

    }
    ::glEnd();

    bdc->Unlock();

    this->shader->Disable();
    return true;
}
