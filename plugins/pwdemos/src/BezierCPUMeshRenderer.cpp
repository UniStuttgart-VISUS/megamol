/*
* BezierCPUMeshRenderer.cpp
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "BezierCPUMeshRenderer.h"
#include "mmcore/misc/BezierCurvesListDataCall.h"
#include "mmcore/param/IntParam.h"
#include <cmath>
#include "vislib/math/Vector.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/math/BezierCurve.h"
#include "vislib/math/Point.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/sys/Log.h"

namespace megamol {
namespace demos {

/*
 * BezierCPUMeshRenderer::BezierCPUMeshRenderer
 */
BezierCPUMeshRenderer::BezierCPUMeshRenderer(void) : AbstractBezierRenderer(),
        curveSectionsSlot("curveSections", "Linear sections approximating the curve"),
        profileSectionsSlot("profileSections", "Linear sections approximating the profile"),
        capSectionsSlot("capSections", "Linear sections approximating the cap spheres"),
        geo(0), dataHash(0) {

    this->getDataSlot.SetCompatibleCall<core::misc::BezierCurvesListDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->curveSectionsSlot << new core::param::IntParam(10, 1, 200);
    this->MakeSlotAvailable(&this->curveSectionsSlot);

    this->profileSectionsSlot << new core::param::IntParam(6, 3, 20);
    this->MakeSlotAvailable(&this->profileSectionsSlot);

    this->capSectionsSlot << new core::param::IntParam(1, 0, 1); // hazard: cone only, for now
    this->MakeSlotAvailable(&this->capSectionsSlot);

}


/*
 * BezierCPUMeshRenderer::~BezierCPUMeshRenderer
 */
BezierCPUMeshRenderer::~BezierCPUMeshRenderer(void) {
    this->Release();
}


/*
 * BezierCPUMeshRenderer::render
 */
bool BezierCPUMeshRenderer::render(megamol::core::view::CallRender3D_2& call) {
    using core::misc::BezierCurvesListDataCall;
    BezierCurvesListDataCall *data = this->getDataSlot.CallAs<BezierCurvesListDataCall>();
    if (data == nullptr) return false;
    data->SetFrameID(static_cast<unsigned int>(call.Time()));
    if (!(*data)(1)) return false;

    ::glDisable(GL_TEXTURE);
    ::glEnable(GL_DEPTH_TEST);
    ::glEnable(GL_COLOR_MATERIAL);
    ::glCullFace(GL_BACK);
    ::glFrontFace(GL_CCW);
    ::glDisable(GL_BLEND);
    ::glEnable(GL_LIGHTING);
    ::glEnable(GL_NORMALIZE);

    bool needUpdate = (data->DataHash() != this->dataHash) || (this->dataHash == 0);
    needUpdate |= this->curveSectionsSlot.IsDirty();
    needUpdate |= this->profileSectionsSlot.IsDirty();
    needUpdate |= this->capSectionsSlot.IsDirty();

    data->Unlock();

    if (needUpdate) {
        data->SetFrameID(static_cast<unsigned int>(call.Time()));
        if (!(*data)(0)) return false;
        this->dataHash = data->DataHash();
        this->curveSectionsSlot.ResetDirty();
        this->profileSectionsSlot.ResetDirty();
        this->capSectionsSlot.ResetDirty();

        int curSeg = this->curveSectionsSlot.Param<core::param::IntParam>()->Value();
        int proSeg = this->profileSectionsSlot.Param<core::param::IntParam>()->Value();
        int capSeg = this->capSectionsSlot.Param<core::param::IntParam>()->Value();

        ::glNewList(this->geo, GL_COMPILE);

        size_t cnt = data->Count();
        for (size_t i = 0; i < cnt; i++) {
            const BezierCurvesListDataCall::Curves& c = data->GetCurves()[i];
            const float globRad = c.GetGlobalRadius();
            const unsigned char* globCol = c.GetGlobalColour();

            size_t bpp = 0;
            bool with_col = false;
            bool with_rad = false;
            switch (c.GetDataLayout()) {
            case BezierCurvesListDataCall::DATALAYOUT_NONE :
                continue;
            case BezierCurvesListDataCall::DATALAYOUT_XYZ_F :
                bpp = 3 * 4;
                break;
            case BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B :
                bpp = 3 * 4 + 3 * 1;
                with_col = true;
                break;
            case BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B :
                bpp = 4 * 4 + 3 * 1;
                with_col = true;
                with_rad = true;
                break;
            case BezierCurvesListDataCall::DATALAYOUT_XYZR_F:
                bpp = 4 * 4;
                with_rad = true;
                break;
            }

            size_t p_cnt = c.GetIndexCount();

            for (size_t j = 0; j < p_cnt; j += 4) {
                size_t jp = c.GetIndex()[j + 0] * bpp;

                const float *p1 = static_cast<const float*>(c.GetDataAt(jp));
                const float *r1 = static_cast<const float*>(with_rad ? c.GetDataAt(jp + 12) : &globRad);
                const unsigned char *c1 = static_cast<const unsigned char*>(with_col ? c.GetDataAt(jp + (with_rad ? 16 : 12)) : globCol);

                jp = c.GetIndex()[j + 1] * bpp;

                const float *p2 = static_cast<const float*>(c.GetDataAt(jp));
                const float *r2 = static_cast<const float*>(with_rad ? c.GetDataAt(jp + 12) : &globRad);
                const unsigned char *c2 = static_cast<const unsigned char*>(with_col ? c.GetDataAt(jp + (with_rad ? 16 : 12)) : globCol);

                jp = c.GetIndex()[j + 2] * bpp;

                const float *p3 = static_cast<const float*>(c.GetDataAt(jp));
                const float *r3 = static_cast<const float*>(with_rad ? c.GetDataAt(jp + 12) : &globRad);
                const unsigned char *c3 = static_cast<const unsigned char*>(with_col ? c.GetDataAt(jp + (with_rad ? 16 : 12)) : globCol);

                jp = c.GetIndex()[j + 3] * bpp;

                const float *p4 = static_cast<const float*>(c.GetDataAt(jp));
                const float *r4 = static_cast<const float*>(with_rad ? c.GetDataAt(jp + 12) : &globRad);
                const unsigned char *c4 = static_cast<const unsigned char*>(with_col ? c.GetDataAt(jp + (with_rad ? 16 : 12)) : globCol);

                this->drawTube(p1, r1, c1, p2, r2, c2, p3, r3, c3, p4, r4, c4, with_rad, with_col, curSeg, proSeg, capSeg);
            }

        }

        ::glEndList();

        data->Unlock();

    }

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

    glCallList(this->geo);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);

    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);
    ::glDisable(GL_POINT_SMOOTH);

    return true;
}


/*
 * BezierCPUMeshRenderer::create
 */
bool BezierCPUMeshRenderer::create(void) {
    if (!AbstractBezierRenderer::create()) return false;

    this->geo = ::glGenLists(1);

    return true;
}


/*
 * BezierCPUMeshRenderer::release
 */
void BezierCPUMeshRenderer::release(void) {
    ::glDeleteLists(this->geo, 1);
    this->geo = 0;
}


/*
 * BezierCPUMeshRenderer::drawTube
 */
void BezierCPUMeshRenderer::drawTube(
        float const *p1, float const *r1, unsigned char const *c1,
        float const *p2, float const *r2, unsigned char const *c2,
        float const *p3, float const *r3, unsigned char const *c3,
        float const *p4, float const *r4, unsigned char const *c4,
        bool hasRad, bool hasCol, int curSeg, int proSeg, int capSeg) {

    typedef vislib::math::Vector<float, 3> vec3;
    typedef vislib::math::Point<float, 3> pt3;
    typedef vislib::math::Point<float, 1> pt1;
    typedef vislib::math::Vector<float, 2> vec2;
    typedef vislib::math::ShallowVector<float, 3> svec3;
    typedef vislib::math::ShallowPoint<float, 3> spt3;

    vec2 *ringfac = new vec2[proSeg];
    for (int i = 0; i < proSeg; ++i) {
        ringfac[i].Set(
            static_cast<float>(::cos(M_PI * 2.0 * i / proSeg)),
            static_cast<float>(::sin(M_PI * 2.0 * i / proSeg)));
    }

    vislib::math::BezierCurve<pt3, 3> pos;
    pos[0] = spt3(const_cast<float*>(p1));
    pos[1] = spt3(const_cast<float*>(p2));
    pos[2] = spt3(const_cast<float*>(p3));
    pos[3] = spt3(const_cast<float*>(p4));

    vislib::math::BezierCurve<pt1, 3> rad;
    rad[0][0] = *r1;
    rad[1][0] = *r2;
    rad[2][0] = *r3;
    rad[3][0] = *r4;

    vislib::math::BezierCurve<pt3, 3> col;
    col[0].Set(static_cast<float>(c1[0]) / 255.f, static_cast<float>(c1[1]) / 255.f, static_cast<float>(c1[2]) / 255.f);
    col[1].Set(static_cast<float>(c2[0]) / 255.f, static_cast<float>(c2[1]) / 255.f, static_cast<float>(c2[2]) / 255.f);
    col[2].Set(static_cast<float>(c3[0]) / 255.f, static_cast<float>(c3[1]) / 255.f, static_cast<float>(c3[2]) / 255.f);
    col[3].Set(static_cast<float>(c4[0]) / 255.f, static_cast<float>(c4[1]) / 255.f, static_cast<float>(c4[2]) / 255.f);

    pt1 rad1;
    pt3 pos1;
    vec3 dir1;
    rad.CalcPoint(rad1, 0.f);
    pos.CalcPoint(pos1, 0.f);
    pos.CalcTangent(dir1, 0.f);
    dir1.Normalise();
    vec3 x1(1.f, 0.f, 0.f);
    if (dir1.IsParallel(x1)) x1.Set(0.f, 1.f, 0.f);
    vec3 y1 = x1.Cross(dir1);
    y1.Normalise();
    x1 = dir1.Cross(y1);
    pt3 col1 = col[0];

    pt1 rad2;
    pt3 pos2, col2;
    vec3 dir2, x2, y2;

    if (capSeg > 0) {
        ::glColor3fv(col1.PeekCoordinates());
        ::glBegin(GL_TRIANGLE_FAN);

        ::glNormal3fv((-dir1).PeekComponents());
        ::glVertex3fv((pos1 - dir1 * rad1[0]).PeekCoordinates());

        for (int i = 0; i <= proSeg; i++) {
            ::glColor3fv(col1.PeekCoordinates());
            vec3 n = x1 * ringfac[i % proSeg].X() + y1 * ringfac[i % proSeg].Y();
            ::glNormal3fv(n.PeekComponents());
            ::glVertex3fv((pos1 + n * rad1[0]).PeekCoordinates());
        }

        ::glEnd();
    }

    for (int j = 0; j < curSeg; ++j) {
        float alpha = static_cast<float>(j + 1) / static_cast<float>(curSeg);

        rad.CalcPoint(rad2, alpha);
        pos.CalcPoint(pos2, alpha);
        pos.CalcTangent(dir2, alpha);
        dir2.Normalise();
        if (x1.IsParallel(dir2)) {
            x2 = dir2.Cross(y1);
            x2.Normalise();
            y2 = x2.Cross(dir2);
        } else {
            y2 = x1.Cross(dir2);
            y2.Normalise();
            x2 = dir2.Cross(y2);
        }
        col.CalcPoint(col2, alpha);

        ::glBegin(GL_TRIANGLE_STRIP);
        for (int i = 0; i <= proSeg; i++) {
            ::glColor3fv(col1.PeekCoordinates());
            vec3 n = x1 * ringfac[i % proSeg].X() + y1 * ringfac[i % proSeg].Y();
            ::glNormal3fv(n.PeekComponents());
            ::glVertex3fv((pos1 + n * rad1[0]).PeekCoordinates());

            ::glColor3fv(col2.PeekCoordinates());
            n = x2 * ringfac[i % proSeg].X() + y2 * ringfac[i % proSeg].Y();
            ::glNormal3fv(n.PeekComponents());
            ::glVertex3fv((pos2 + n * rad2[0]).PeekCoordinates());
        }
        ::glEnd();

        rad1 = rad2;
        pos1 = pos2;
        col1 = col2;
        dir1 = dir2;
        x1 = x2;
        y1 = y2;

    }

    if (capSeg > 0) {
        ::glColor3fv(col1.PeekCoordinates());
        ::glBegin(GL_TRIANGLE_FAN);

        ::glNormal3fv(dir1.PeekComponents());
        ::glVertex3fv((pos1 + dir1 * rad1[0]).PeekCoordinates());

        for (int i =proSeg; i >= 0; --i) {
            ::glColor3fv(col1.PeekCoordinates());
            vec3 n = x1 * ringfac[i % proSeg].X() + y1 * ringfac[i % proSeg].Y();
            ::glNormal3fv(n.PeekComponents());
            ::glVertex3fv((pos1 + n * rad1[0]).PeekCoordinates());
        }

        ::glEnd();
    }

}

} /* end namespace demos */
} /* end namespace megamol */
