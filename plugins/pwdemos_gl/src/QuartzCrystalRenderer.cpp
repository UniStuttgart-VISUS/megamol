/*
 * QuartzCrystalRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "QuartzCrystalRenderer.h"
#include "QuartzCrystalDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmstd/light/PointLight.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

#include <glm/ext.hpp>

namespace megamol::demos_gl {

/*
 * CrystalRenderer::CrystalRenderer
 */
CrystalRenderer::CrystalRenderer()
        : mmstd_gl::Renderer3DModuleGL()
        , dataInSlot("datain", "slot to get the data")
        , lightsSlot("lights", "Lights are retrieved over this slot.")
        , crystalIdx("idx", "The index of the selected crystal") {

    this->dataInSlot.SetCompatibleCall<core::factories::CallAutoDescription<CrystalDataCall>>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->lightsSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->lightsSlot);

    this->crystalIdx << new core::param::IntParam(0, 0);
    this->MakeSlotAvailable(&this->crystalIdx);
}


/*
 * CrystalRenderer::~CrystalRenderer
 */
CrystalRenderer::~CrystalRenderer() {
    this->Release();
}


/*
 * CrystalRenderer::GetExtents
 */
bool CrystalRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {

    call.AccessBoundingBoxes().Clear();
    call.SetTimeFramesCount(1);

    return true;
}


/*
 * CrystalRenderer::Render
 */
bool CrystalRenderer::Render(mmstd_gl::CallRender3DGL& call) {

    unsigned int idx = static_cast<unsigned int>(this->crystalIdx.Param<core::param::IntParam>()->Value());

    CrystalDataCall* cdc = this->dataInSlot.CallAs<CrystalDataCall>();
    if ((cdc == NULL) || (!(*cdc)(0)))
        return false;

    if (cdc->GetCount() == 0)
        return false; // no data :-(

    idx = idx % cdc->GetCount();
    const CrystalDataCall::Crystal& c = cdc->GetCrystals()[idx];

    c.AssertMesh();

    core::view::Camera cam = call.GetCamera();
    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();

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
    const float* verts = c.GetMeshVertexData();
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
    std::array<float, 3> lightPos = {0.0f, 0.0f, 0.0f};

    auto call_light = lightsSlot.CallAs<core::view::light::CallLight>();
    if (call_light != nullptr) {
        if (!(*call_light)(0)) {
            return false;
        }

        auto lights = call_light->getData();
        auto point_lights = lights.get<core::view::light::PointLightType>();

        if (point_lights.size() > 1) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[CartoonTessellationRenderer2000GT] Only one single 'Point Light' source is supported by this "
                "renderer");
        } else if (point_lights.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[CartoonTessellationRenderer2000GT] No 'Point Light' found");
        }

        for (auto const& light : point_lights) {
            // light.second.lightColor;
            // light.second.lightIntensity;
            lightPos[0] = light.position[0];
            lightPos[1] = light.position[1];
            lightPos[2] = light.position[2];
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
bool CrystalRenderer::create() {
    // intentionally empty
    return true;
}


/*
 * CrystalRenderer::release
 */
void CrystalRenderer::release() {
    // intentionally empty
}

} // namespace megamol::demos_gl
