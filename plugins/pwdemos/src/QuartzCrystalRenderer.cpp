/*
 * QuartzCrystalRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "QuartzCrystalRenderer.h"
#include "QuartzCrystalDataCall.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace demos {

/*
 * CrystalRenderer::CrystalRenderer
 */
CrystalRenderer::CrystalRenderer(void) : core::view::Renderer3DModule(),
dataInSlot("datain", "slot to get the data"),
crystalIdx("idx", "The index of the selected crystal") {

    this->dataInSlot.SetCompatibleCall<core::factories::CallAutoDescription<CrystalDataCall> >();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->crystalIdx << new core::param::IntParam(0, 0);
    this->MakeSlotAvailable(&this->crystalIdx);

}


/*
 * CrystalRenderer::~CrystalRenderer
 */
CrystalRenderer::~CrystalRenderer(void) {
    this->Release();
}


/*
 * CrystalRenderer::GetExtents
 */
bool CrystalRenderer::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->AccessBoundingBoxes().Clear();
    cr->SetTimeFramesCount(1);

    return true;
}


/*
 * CrystalRenderer::Render
 */
bool CrystalRenderer::Render(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    unsigned int idx = static_cast<unsigned int>(this->crystalIdx.Param<core::param::IntParam>()->Value());

    CrystalDataCall *cdc = this->dataInSlot.CallAs<CrystalDataCall>();
    if ((cdc == NULL) || (!(*cdc)(0))) return false;

    if (cdc->GetCount() == 0) return false; // no data :-(

    idx = idx % cdc->GetCount();
    const CrystalDataCall::Crystal& c = cdc->GetCrystals()[idx];

    c.AssertMesh();

    float scaling = 1.0f / c.GetBoundingRadius();
    ::glScalef(scaling, scaling, scaling);
    scaling = 1.0f; //c.GetBaseRadius();

    ::glEnable(GL_NORMALIZE);
    ::glDisable(GL_BLEND);
    ::glEnable(GL_LIGHTING);
    ::glEnable(GL_DEPTH_TEST);
    ::glEnable(GL_COLOR_MATERIAL);
    ::glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

    ::glPointSize(8.0f);
    ::glLineWidth(4.0f);

    ::glDisable(GL_LIGHTING);
    ::glColor3ub(255, 128, 0);
    ::glBegin(GL_LINES);
    for (unsigned int i = 0; i < c.GetFaceCount(); i++) {
        vislib::math::Vector<float, 3> v = c.GetFace(i);
        v *= scaling;
        ::glVertex3fv(v.PeekComponents());
        v *= 1.1f;
        ::glVertex3fv(v.PeekComponents());
    }
    ::glEnd();
    ::glBegin(GL_POINTS);
    for (unsigned int i = 0; i < c.GetFaceCount(); i++) {
        vislib::math::Vector<float, 3> v = c.GetFace(i);
        v *= scaling;
        ::glVertex3fv(v.PeekComponents());
    }
    ::glEnd();

    ::glColor3ub(0, 255, 0);
    ::glBegin(GL_POINTS);
    const float *verts = c.GetMeshVertexData();
    for (unsigned int i = 0; i < c.GetFaceCount(); i++) {
        unsigned int cnt = c.GetMeshTriangleCounts()[i];
        for (unsigned int j = 0; j < cnt; j++) {
            ::glVertex3fv(verts + 3 * c.GetMeshTriangles()[i][j]);
        }
    }
    ::glEnd();

    ::glColor3ub(0, 0, 255);
    for (unsigned int i = 0; i < c.GetFaceCount(); i++) {
        ::glBegin(GL_LINE_LOOP);
        unsigned int cnt = c.GetMeshTriangleCounts()[i];
        for (unsigned int j = 0; j < cnt; j++) {
            ::glVertex3fv(verts + 3 * c.GetMeshTriangles()[i][j]);
        }
        ::glEnd();
    }


    ::glEnable(GL_LIGHTING);

    ::glColor3ub(192, 192, 192);
    //verts = c.GetMeshVertexData();
    for (unsigned int i = 0; i < c.GetFaceCount(); i++) {
        ::glNormal3fv(c.GetFace(i).PeekComponents());
        ::glBegin(GL_TRIANGLE_FAN);
        unsigned int cnt = c.GetMeshTriangleCounts()[i];
        for (unsigned int j = 0; j < cnt; j++) {
            ::glVertex3fv(verts + 3 * c.GetMeshTriangles()[i][j]);
        }
        ::glEnd();
    }


    return true;
}


/*
 * CrystalRenderer::create
 */
bool CrystalRenderer::create(void) {
    // intentionally empty
    return true;
}


/*
 * CrystalRenderer::release
 */
void CrystalRenderer::release(void) {
    // intentionally empty
}

} /* end namespace demos */
} /* end namespace megamol */