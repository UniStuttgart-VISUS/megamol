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
#include "mmcore/view/CallRender3D_2.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/sys/Log.h"

namespace megamol {
namespace demos {

/*
 * CrystalRenderer::CrystalRenderer
 */
CrystalRenderer::CrystalRenderer(void) : core::view::Renderer3DModule_2(),
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
bool CrystalRenderer::GetExtents(core::view::CallRender3D_2& call) {

    call.AccessBoundingBoxes().Clear();
    call.SetTimeFramesCount(1);

    return true;
}


/*
 * CrystalRenderer::Render
 */
bool CrystalRenderer::Render(core::view::CallRender3D_2& call) {

    unsigned int idx = static_cast<unsigned int>(this->crystalIdx.Param<core::param::IntParam>()->Value());

    CrystalDataCall *cdc = this->dataInSlot.CallAs<CrystalDataCall>();
    if ((cdc == NULL) || (!(*cdc)(0))) return false;

    if (cdc->GetCount() == 0) return false; // no data :-(

    idx = idx % cdc->GetCount();
    const CrystalDataCall::Crystal& c = cdc->GetCrystals()[idx];

    c.AssertMesh();

	core::view::Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;
    cam.calc_matrices(snapshot, viewTemp, projTemp, core::thecam::snapshot_content::all);
    glm::mat4 proj = projTemp;
    glm::mat4 view = viewTemp;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
    glLoadMatrixf(glm::value_ptr(proj));

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
    glLoadMatrixf(glm::value_ptr(view));

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
        ::glVertex3fv(v.PeekComponents());
        v *= 1.1f;
        ::glVertex3fv(v.PeekComponents());
    }
    ::glEnd();
    ::glBegin(GL_POINTS);
    for (unsigned int i = 0; i < c.GetFaceCount(); i++) {
        vislib::math::Vector<float, 3> v = c.GetFace(i);
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


	// determine position of point light
	this->GetLights();
    glm::vec4 lightPos = {0.0f, 0.0f, 0.0f, 1.0f};
    if (this->lightMap.size() != 1) {
		vislib::sys::Log::DefaultLog.WriteWarn("[BezierCPUMeshRenderer] Only one single point light source is supported by this renderer");
    }
    for (auto light : this->lightMap) {
        if (light.second.lightType != core::view::light::POINTLIGHT) {
        vislib::sys::Log::DefaultLog.WriteWarn("[BezierCPUMeshRenderer] Only single point light source is supported by this renderer");
        } else {
            auto lPos = light.second.pl_position;
            //light.second.lightColor;
            //light.second.lightIntensity;
            if (lPos.size() == 3) {
                lightPos[0] = lPos[0];
                lightPos[1] = lPos[1];
                lightPos[2] = lPos[2];
            }
            if (lPos.size() == 4) {
                lightPos[4] = lPos[4];
            }
            break;
        }
    }
    const float lp[4] = {-lightPos[0], -lightPos[1], -lightPos[1], 0.0f};
    const float zeros[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    const float ones[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, zeros);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, ones);
    glLightfv(GL_LIGHT0, GL_SPECULAR, ones);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, zeros);

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

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

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