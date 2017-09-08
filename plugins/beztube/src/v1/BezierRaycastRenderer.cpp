/*
 * BezierRaycastRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#define VLDEPRECATED
#define VISLIB_DEPRECATED_H_INCLUDED
//#include "glh/glh_extensions.h"
#include "v1/BezierRaycastRenderer.h"
#include "v1/BezierDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "vislib/Exception.h"
#include "vislib/sys/Log.h"
#include "vislib/graphics/gl/ShaderSource.h"
//#include <cmath>

using namespace megamol;
using namespace megamol::beztube;


/*
 * v1::BezierRaycastRenderer::BezierRaycastRenderer
 */
v1::BezierRaycastRenderer::BezierRaycastRenderer(void) : AbstractBezierRenderer(),
        pbShader() {

    this->getDataSlot.SetCompatibleCall<v1::BezierDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * v1::BezierRaycastRenderer::~BezierRaycastRenderer
 */
v1::BezierRaycastRenderer::~BezierRaycastRenderer(void) {
    this->Release();
}


/*
 * v1::BezierRaycastRenderer::create
 */
bool v1::BezierRaycastRenderer::create(void) {
    using vislib::sys::Log;
    if (!AbstractBezierRenderer::create()) return false;

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
 * v1::BezierRaycastRenderer::release
 */
void v1::BezierRaycastRenderer::release(void) {
    AbstractBezierRenderer::release();

    this->shader = NULL;
    this->pbShader.Release();

}


/*
 * v1::BezierRaycastRenderer::render
 */
bool v1::BezierRaycastRenderer::render(megamol::core::view::CallRender3D& call) {

    v1::BezierDataCall *bdc = this->getDataSlot.CallAs<v1::BezierDataCall>();
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

    // Implement missing

    ::glBegin(GL_POINTS); // not cool, but ok for now
    for (unsigned int t = 0; t < bdc->Count(); t++) {
        const vislib::math::BezierCurve<v1::BezierDataCall::BezierPoint, 3>& curve = bdc->Curves()[t];
        const v1::BezierDataCall::BezierPoint *cp;

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
