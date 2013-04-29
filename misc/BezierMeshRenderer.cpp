/*
 * BezierMeshRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "glh/glh_extensions.h"
#include "BezierMeshRenderer.h"
#include "BezierDataCall.h"
#include "param/EnumParam.h"
#include "param/IntParam.h"
#include "view/CallRender3D.h"
#include <cmath>

using namespace megamol::core;


/*
 * misc::BezierMeshRenderer::BezierMeshRenderer
 */
misc::BezierMeshRenderer::BezierMeshRenderer(void) : Renderer3DModule(),
        getDataSlot("getdata", "Connects to the data source"),
        curveSectionsSlot("curveSections", "The number of linear sections along the curve"),
        profileSectionsSlot("profileSections", "The number of section along the profile"),
        capTypeSlot("capType", "Controlls the type of the curve caps"),
        objs(0), objsHash(0) {

    this->getDataSlot.SetCompatibleCall<misc::BezierDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->curveSectionsSlot << new param::IntParam(25, 1);
    this->MakeSlotAvailable(&this->curveSectionsSlot);

    this->profileSectionsSlot << new param::IntParam(8, 3);
    this->MakeSlotAvailable(&this->profileSectionsSlot);

    param::EnumParam *ep = new param::EnumParam(2);
    ep->SetTypePair(0, "None");
    ep->SetTypePair(1, "Disk");
    ep->SetTypePair(2, "Hemisphere");
    this->capTypeSlot << ep;
    this->MakeSlotAvailable(&this->capTypeSlot);

}


/*
 * misc::BezierMeshRenderer::~BezierMeshRenderer
 */
misc::BezierMeshRenderer::~BezierMeshRenderer(void) {
    this->Release();
}


/*
 * misc::BezierMeshRenderer::create
 */
bool misc::BezierMeshRenderer::create(void) {

    this->objs = ::glGenLists(1);
    this->curveSectionsSlot.ForceSetDirty();

    return true;
}


/*
 * misc::BezierMeshRenderer::GetCapabilities
 */
bool misc::BezierMeshRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        | view::CallRender3D::CAP_LIGHTING
        );

    return true;
}


/*
 * misc::BezierMeshRenderer::GetExtents
 */
bool misc::BezierMeshRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    BezierDataCall *bdc = this->getDataSlot.CallAs<misc::BezierDataCall>();
    if ((bdc != NULL) && ((*bdc)(1))) {
        cr->SetTimeFramesCount(bdc->FrameCount());
        cr->AccessBoundingBoxes() = bdc->AccessBoundingBoxes();
        if (cr->AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {

            float sizing = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
            if (sizing > 0.0000001) {
                sizing = 10.0f / sizing;
            } else {
                sizing = 1.0f;
            }
            cr->AccessBoundingBoxes().MakeScaledWorld(sizing);

        }

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();

    }

    return true;
}


/*
 * misc::BezierMeshRenderer::release
 */
void misc::BezierMeshRenderer::release(void) {
    ::glDeleteLists(this->objs, 1);
    this->objs = 0;
}


/*
 * misc::BezierMeshRenderer::Render
 */
bool misc::BezierMeshRenderer::Render(Call& call) {
    typedef misc::BezierDataCall::BezierPoint Pt;
    typedef vislib::math::Vector<float, 2> Vec2;
    typedef vislib::math::Vector<float, 3> Vec3;

    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    BezierDataCall *bdc = this->getDataSlot.CallAs<misc::BezierDataCall>();
    float scaling = 1.0f;
    if (bdc != NULL) {
        bdc->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*bdc)(1)) return false;

        // calculate scaling
        scaling = bdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }

        bdc->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*bdc)(0)) return false;

    } else {
        return false;
    }

    ::glScalef(scaling, scaling, scaling);

    ::glDisable(GL_TEXTURE);
    ::glEnable(GL_DEPTH_TEST);
    ::glEnable(GL_COLOR_MATERIAL);
    ::glCullFace(GL_BACK);
    ::glFrontFace(GL_CCW);
    ::glDisable(GL_BLEND);
    ::glEnable(GL_LIGHTING);
    ::glEnable(GL_NORMALIZE);

    if ((this->objsHash != bdc->DataHash())
            || this->curveSectionsSlot.IsDirty()
            || this->profileSectionsSlot.IsDirty()
            || this->capTypeSlot.IsDirty()) {
        this->objsHash = bdc->DataHash();
        this->curveSectionsSlot.ResetDirty();
        this->profileSectionsSlot.ResetDirty();
        this->capTypeSlot.ResetDirty();

        int curveSegs = this->curveSectionsSlot.Param<param::IntParam>()->Value();
        float curveStep = 1.0f / static_cast<float>(curveSegs);
        int profileSegs = this->profileSectionsSlot.Param<param::IntParam>()->Value();
        unsigned int profileCnt = profileSegs + 1;
        int capType = this->capTypeSlot.Param<param::EnumParam>()->Value();

        Vec2 *profile = new Vec2[profileCnt];
        for (unsigned int i = 0; i < profileCnt; i++) {
            float a = 2.0f * static_cast<float>(M_PI) * static_cast<float>(i)
                / static_cast<float>(profileSegs);
            profile[i].Set(cos(a), sin(a));
        }

        unsigned int capProfileCnt = (profileSegs / 2) + (profileSegs % 2);
        Vec2 *capProfile = new Vec2[capProfileCnt];
        for (unsigned int i = 0; i < capProfileCnt; i++) {
            float a = 0.5f * static_cast<float>(M_PI) * static_cast<float>(i)
                / static_cast<float>(capProfileCnt - 1);
            capProfile[i].Set(cos(a), sin(a));
        }

        ::glNewList(this->objs, GL_COMPILE_AND_EXECUTE);

        for (unsigned int i = 0; i < bdc->Count(); i++) {
            const vislib::math::BezierCurve<Pt, 3> &curve = bdc->Curves()[i];

            Pt p1 = curve.ControlPoint(0); // point
            Vec3 t1;
            curve.CalcTangent(t1, 0.0f);
            if (t1.IsNull()) {
                t1.Set(0.0f, 0.0f, 1.0f);
            } else {
                t1.Normalise(); // tangent
            }
            Vec3 vp1(1.0f, 0.0f, 0.0f); // primary
            if (t1.IsParallel(vp1)) vp1.Set(0.0f, 1.0f, 0.0f);
            Vec3 vs1 = t1.Cross(vp1); // secondary
            vs1.Normalise();
            vp1 = vs1.Cross(t1);
            vp1.Normalise();

            Pt p2; // seconds
            Vec3 t2, vp2, vs2;

            // Start Cap
            if (capType == 1) {
                ::glColor3ub(p1.R(), p1.G(), p1.B());
                ::glBegin(GL_TRIANGLE_FAN);
                ::glNormal3f(-t1.X(), -t1.Y(), -t1.Z());
                for (unsigned int i = profileCnt - 1; i > 0; i--) {
                    ::glVertex3f(
                        p1.X() + p1.Radius() * (profile[i].X() * vp1.X() + profile[i].Y() * vs1.X()),
                        p1.Y() + p1.Radius() * (profile[i].X() * vp1.Y() + profile[i].Y() * vs1.Y()),
                        p1.Z() + p1.Radius() * (profile[i].X() * vp1.Z() + profile[i].Y() * vs1.Z()));
                }
                ::glEnd();
            } else if (capType == 2) {
                ::glColor3ub(p1.R(), p1.G(), p1.B());
                for (unsigned int j = 1; j < capProfileCnt; j++) {
                    ::glBegin(GL_QUAD_STRIP);
                    for (unsigned int i = 0; i < profileCnt; i++) {
                        Vec3 v3 = vp1 * profile[i].X() + vs1 * profile[i].Y();

                        Vec3 v4 = v3 * capProfile[j - 1].X() - t1 * capProfile[j - 1].Y();
                        ::glNormal3fv(v4.PeekComponents());
                        ::glVertex3f(p1.X() + p1.Radius() * v4.X(), p1.Y() + p1.Radius() * v4.Y(), p1.Z() + p1.Radius() * v4.Z());

                        v4 = v3 * capProfile[j].X() - t1 * capProfile[j].Y();
                        ::glNormal3fv(v4.PeekComponents());
                        ::glVertex3f(p1.X() + p1.Radius() * v4.X(), p1.Y() + p1.Radius() * v4.Y(), p1.Z() + p1.Radius() * v4.Z());
                    }
                    ::glEnd();
                }
            }

            // curve
            // *1 holding values for t=0
            for (float t = curveStep; t < 1.0f + 0.5f * curveStep; t += curveStep) {
                if (t > 1.0f - 0.5f * curveStep) t = 1.0f;
                p2 = p1;
                t2 = t1;
                vp2 = vp1;
                vs2 = vs1;

                curve.CalcPoint(p1, t);
                curve.CalcTangent(t1, t);
                if (t1.IsNull()) {
                    t1.Set(0.0f, 0.0f, 1.0f);
                } else {
                    t1.Normalise(); // tangent
                }
                if (t1.IsParallel(vp1)) {
                    vp1 = vs1.Cross(t1);
                    vp1.Normalise();
                    vs1 = t1.Cross(vp1);
                    vs1.Normalise();
                } else {
                    vs1 = t1.Cross(vp1);
                    vs1.Normalise();
                    vp1 = vs1.Cross(t1);
                    vp1.Normalise();
                }

                ::glBegin(GL_QUAD_STRIP);
                for (unsigned int i = 0; i < profileCnt; i++) {

                    ::glColor3ub(p1.R(), p1.G(), p1.B());
                    ::glNormal3f(
                        profile[i].X() * vp1.X() + profile[i].Y() * vs1.X(),
                        profile[i].X() * vp1.Y() + profile[i].Y() * vs1.Y(),
                        profile[i].X() * vp1.Z() + profile[i].Y() * vs1.Z());
                    ::glVertex3f(
                        p1.X() + p1.Radius() * (profile[i].X() * vp1.X() + profile[i].Y() * vs1.X()),
                        p1.Y() + p1.Radius() * (profile[i].X() * vp1.Y() + profile[i].Y() * vs1.Y()),
                        p1.Z() + p1.Radius() * (profile[i].X() * vp1.Z() + profile[i].Y() * vs1.Z()));

                    ::glColor3ub(p2.R(), p2.G(), p2.B());
                    ::glNormal3f(
                        profile[i].X() * vp2.X() + profile[i].Y() * vs2.X(),
                        profile[i].X() * vp2.Y() + profile[i].Y() * vs2.Y(),
                        profile[i].X() * vp2.Z() + profile[i].Y() * vs2.Z());
                    ::glVertex3f(
                        p2.X() + p2.Radius() * (profile[i].X() * vp2.X() + profile[i].Y() * vs2.X()),
                        p2.Y() + p2.Radius() * (profile[i].X() * vp2.Y() + profile[i].Y() * vs2.Y()),
                        p2.Z() + p2.Radius() * (profile[i].X() * vp2.Z() + profile[i].Y() * vs2.Z()));

                }
                ::glEnd();

            }

            // End Cap
            if (capType == 1) {
                ::glColor3ub(p1.R(), p1.G(), p1.B());
                ::glBegin(GL_TRIANGLE_FAN);
                ::glNormal3f(t1.X(), t1.Y(), t1.Z());
                for (unsigned int i = 1; i < profileCnt; i++) {
                    ::glVertex3f(
                        p1.X() + p1.Radius() * (profile[i].X() * vp1.X() + profile[i].Y() * vs1.X()),
                        p1.Y() + p1.Radius() * (profile[i].X() * vp1.Y() + profile[i].Y() * vs1.Y()),
                        p1.Z() + p1.Radius() * (profile[i].X() * vp1.Z() + profile[i].Y() * vs1.Z()));
                }
                ::glEnd();
            } else if (capType == 2) {
                ::glColor3ub(p1.R(), p1.G(), p1.B());
                for (unsigned int j = 1; j < capProfileCnt; j++) {
                    ::glBegin(GL_QUAD_STRIP);
                    for (unsigned int i = 0; i < profileCnt; i++) {
                        Vec3 v3 = vp1 * profile[i].X() + vs1 * profile[i].Y();

                        Vec3 v4 = v3 * capProfile[j].X() + t1 * capProfile[j].Y();
                        ::glNormal3fv(v4.PeekComponents());
                        ::glVertex3f(p1.X() + p1.Radius() * v4.X(), p1.Y() + p1.Radius() * v4.Y(), p1.Z() + p1.Radius() * v4.Z());

                        v4 = v3 * capProfile[j - 1].X() + t1 * capProfile[j - 1].Y();
                        ::glNormal3fv(v4.PeekComponents());
                        ::glVertex3f(p1.X() + p1.Radius() * v4.X(), p1.Y() + p1.Radius() * v4.Y(), p1.Z() + p1.Radius() * v4.Z());
                    }
                    ::glEnd();
                }
            }

        }

        ::glEndList();
        delete[] profile;
        delete[] capProfile;

    } else {
        ::glCallList(this->objs);

    }

    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);
    ::glDisable(GL_POINT_SMOOTH);

    return true;
}
