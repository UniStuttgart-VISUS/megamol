/*
 * ExtBezierMeshRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "misc/ExtBezierMeshRenderer.h"
#include "param/IntParam.h"
#include "view/CallRender3D.h"
#include "vislib/Array.h"
#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include <GL/gl.h>
#include <cmath>

using namespace megamol::core;


/*
 * misc::ExtBezierMeshRenderer::ExtBezierMeshRenderer
 */
misc::ExtBezierMeshRenderer::ExtBezierMeshRenderer(void) : Renderer3DModule(),
        getDataSlot("getdata", "Connects to the data source"),
        curveSectionsSlot("curveSections", "The number of linear sections along the curve"),
        profileSectionsSlot("profileSections", "The number of section along the profile"),
        objs(0), objsHash(0) {

    this->getDataSlot.SetCompatibleCall<misc::ExtBezierDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->curveSectionsSlot << new param::IntParam(25, 1);
    this->MakeSlotAvailable(&this->curveSectionsSlot);

    this->profileSectionsSlot << new param::IntParam(8, 3);
    this->MakeSlotAvailable(&this->profileSectionsSlot);

}


/*
 * misc::ExtBezierMeshRenderer::~ExtBezierMeshRenderer
 */
misc::ExtBezierMeshRenderer::~ExtBezierMeshRenderer(void) {
    this->Release();
}


/*
 * misc::ExtBezierMeshRenderer::create
 */
bool misc::ExtBezierMeshRenderer::create(void) {

    this->objs = ::glGenLists(1);
    this->curveSectionsSlot.ForceSetDirty();

    return true;
}


/*
 * misc::ExtBezierMeshRenderer::GetCapabilities
 */
bool misc::ExtBezierMeshRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        | view::CallRender3D::CAP_LIGHTING
        );

    return true;
}


/*
 * misc::ExtBezierMeshRenderer::GetExtents
 */
bool misc::ExtBezierMeshRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    ExtBezierDataCall *ebdc = this->getDataSlot.CallAs<misc::ExtBezierDataCall>();
    if ((ebdc != NULL) && ((*ebdc)(1))) {
        cr->SetTimeFramesCount(ebdc->FrameCount());
        cr->AccessBoundingBoxes() = ebdc->AccessBoundingBoxes();
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
 * misc::ExtBezierMeshRenderer::release
 */
void misc::ExtBezierMeshRenderer::release(void) {
    ::glDeleteLists(this->objs, 1);
    this->objs = 0;
}


/*
 * misc::ExtBezierMeshRenderer::Render
 */
bool misc::ExtBezierMeshRenderer::Render(Call& call) {
    typedef misc::ExtBezierDataCall::Point Pt;
    typedef vislib::math::Vector<float, 2> Vec2;
    typedef vislib::math::Vector<float, 3> Vec3;

    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    ExtBezierDataCall *ebdc = this->getDataSlot.CallAs<misc::ExtBezierDataCall>();
    float scaling = 1.0f;
    if (ebdc != NULL) {
        ebdc->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*ebdc)(1)) return false;

        // calculate scaling
        scaling = ebdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }

        ebdc->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*ebdc)(0)) return false;

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
    ::glEnable(GL_CULL_FACE);

    if ((this->objsHash != ebdc->DataHash())
            || this->curveSectionsSlot.IsDirty()
            || this->profileSectionsSlot.IsDirty()) {
        this->objsHash = ebdc->DataHash();
        this->curveSectionsSlot.ResetDirty();
        this->profileSectionsSlot.ResetDirty();

        ::glNewList(this->objs, GL_COMPILE_AND_EXECUTE);

        unsigned int profileSegs = this->profileSectionsSlot.Param<param::IntParam>()->Value();
        if (profileSegs < 3) profileSegs = 3;
        unsigned int lengthSegs = this->curveSectionsSlot.Param<param::IntParam>()->Value();
        if (lengthSegs < 1) lengthSegs = 1;

        this->drawEllipCurves(ebdc->EllipticCurves(), ebdc->CountElliptic(), profileSegs, lengthSegs);
        this->drawRectCurves(ebdc->RectangularCurves(), ebdc->CountRectangular(), lengthSegs);

        ::glEndList();

    } else {
        ::glCallList(this->objs);

    }

    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);
    ::glDisable(GL_POINT_SMOOTH);

    return true;
}


/*
 * misc::ExtBezierMeshRenderer::calcBase
 */
void misc::ExtBezierMeshRenderer::calcBase(const vislib::math::BezierCurve<
        ExtBezierDataCall::Point, 3>& curve, float t,
        vislib::math::Vector<float, 3>& x, vislib::math::Vector<float, 3>& y,
        vislib::math::Vector<float, 3>& z) {

    vislib::math::Vector<float, 3> ox(x);
    curve.CalcTangent(x, t);
    if (x.IsNull()) {
        if (ox.IsNull()) {
            x.Set(1.0f, 0.0f, 0.0f);
        } else {
            x = ox;
        }
    }
    x.Normalise();

    if (x.IsParallel(y)) { // should never happen, but I already have the code, so ...
        y = z.Cross(x);
        y.Normalise();
        z = x.Cross(y);
        z.Normalise(); // should not be required ...
    } else {
        z = x.Cross(y);
        z.Normalise();
        y = z.Cross(x);
        y.Normalise(); // should not be required ...
    }

    ASSERT(x.IsNormalised());
    ASSERT(y.IsNormalised());
    ASSERT(z.IsNormalised());
    ASSERT(vislib::math::IsEqual(x.Dot(y), 0.0f));
    ASSERT(vislib::math::IsEqual(x.Dot(z), 0.0f));
    ASSERT(vislib::math::IsEqual(y.Dot(z), 0.0f));
}


/*
 * misc::ExtBezierMeshRenderer::drawCurves
 */
void misc::ExtBezierMeshRenderer::drawCurves(
        const vislib::math::BezierCurve<misc::ExtBezierDataCall::Point, 3> *curves,
        SIZE_T cnt, const vislib::math::Point<float, 2> *profile, SIZE_T profCnt,
        unsigned int lengthSections, bool profSmooth) {
    vislib::math::Point<float, 3> *p1 = new vislib::math::Point<float, 3>[profCnt]; // profile
    vislib::math::Point<float, 3> *p2 = new vislib::math::Point<float, 3>[profCnt];
    vislib::math::Vector<float, 3> *n1 = new vislib::math::Vector<float, 3>[profCnt]; // normal
    vislib::math::Vector<float, 3> *n2 = new vislib::math::Vector<float, 3>[profCnt];
    vislib::graphics::ColourRGBAu8 c1, c2; // colours
    misc::ExtBezierDataCall::Point p;
    vislib::math::Vector<float, 3> x(1.0f, 0.0f, 0.0f), y(0.0f, 1.0f, 0.0f), z(0.0f, 0.0f, 1.0f);
    ASSERT(lengthSections >= 1);

    for (SIZE_T i = 0; i < cnt; i++) {

        // Start Cap
        curves[i].CalcPoint(p, 0.0f);
        y = p.GetY();
        this->calcBase(curves[i], 0.0f, x, y, z);
        c1 = p.GetColour();

        ::glColor3ubv(c1.PeekComponentes());
        ::glNormal3f(-x.X(), -x.Y(), -x.Z());
        ::glBegin(GL_TRIANGLE_FAN);
        for (int j = static_cast<int>(profCnt - 1); j >= 0; j--) {
            p1[j] = p.GetPosition()
                + y * (profile[j].X() * p.GetRadiusY())
                + z * (profile[j].Y() * p.GetRadiusZ());
            ::glVertex3fv(p1[j].PeekCoordinates());
        }
        // faked as usual (normal fixed in the yz-plane
        if (profSmooth) {
            for (SIZE_T j = 1; j <= profCnt; j++) {
                n1[j % profCnt] = (p1[(j + 1) % profCnt] - p1[j - 1]).Cross(x);
                n1[j % profCnt].Normalise();
            }
        } else {
            for (SIZE_T j = 0; j < profCnt; j++) {
                n1[j] = (p1[(j + 1) % profCnt] - p1[j]).Cross(x);
                n1[j].Normalise();
            }
        }
        ::glEnd();


        // Curve
        for (SIZE_T k = 1; k < lengthSections; k++) {
            float t = static_cast<float>(k) / static_cast<float>(lengthSections - 1);

            curves[i].CalcPoint(p, t);
            y = p.GetY();
            this->calcBase(curves[i], t, x, y, z);
            c2 = p.GetColour();

            for (SIZE_T j = 0; j < profCnt; j++) {
                p2[j] = p.GetPosition()
                    + y * (profile[j].X() * p.GetRadiusY())
                    + z * (profile[j].Y() * p.GetRadiusZ());
            }
            if (profSmooth) {
                for (SIZE_T j = 1; j <= profCnt; j++) {
                    n2[j % profCnt] = (p2[(j + 1) % profCnt] - p2[j - 1]).Cross(x);
                    n2[j % profCnt].Normalise();
                }
            } else {
                for (SIZE_T j = 0; j < profCnt; j++) {
                    n2[j] = (p2[(j + 1) % profCnt] - p2[j]).Cross(x);
                    n2[j].Normalise();
                }
            }

            if (profSmooth) {
                ::glBegin(GL_QUAD_STRIP);
                for (SIZE_T j = 0; j < profCnt; j++) {
                    ::glNormal3fv(n2[j].PeekComponents());
                    ::glColor3ubv(c2.PeekComponentes());
                    ::glVertex3fv(p2[j].PeekCoordinates());
                    ::glNormal3fv(n1[j].PeekComponents());
                    ::glColor3ubv(c1.PeekComponentes());
                    ::glVertex3fv(p1[j].PeekCoordinates());
                }
                ::glNormal3fv(n2[0].PeekComponents());
                ::glColor3ubv(c2.PeekComponentes());
                ::glVertex3fv(p2[0].PeekCoordinates());
                ::glNormal3fv(n1[0].PeekComponents());
                ::glColor3ubv(c1.PeekComponentes());
                ::glVertex3fv(p1[0].PeekCoordinates());
                ::glEnd();

            } else {
                ::glBegin(GL_QUADS);
                for (SIZE_T j = 0; j < profCnt; j++) {
                    ::glColor3ubv(c2.PeekComponentes());
                    ::glNormal3fv(n2[j].PeekComponents());
                    ::glVertex3fv(p2[(j + 1) % profCnt].PeekCoordinates());
                    ::glVertex3fv(p2[j].PeekCoordinates());
                    ::glColor3ubv(c1.PeekComponentes());
                    ::glNormal3fv(n1[j].PeekComponents());
                    ::glVertex3fv(p1[j].PeekCoordinates());
                    ::glVertex3fv(p1[(j + 1) % profCnt].PeekCoordinates());
                }
                ::glEnd();

            }

            for (SIZE_T j = 0; j < profCnt; j++) {
                p1[j] = p2[j];
                n1[j] = n2[j];
            }
            c1 = c2;

        }

        // End Cap
        ::glColor3ubv(c1.PeekComponentes());
        ::glNormal3fv(x.PeekComponents());
        ::glBegin(GL_TRIANGLE_FAN);
        for (SIZE_T j = 0; j < profCnt; j++) {
            ::glVertex3fv(p1[j].PeekCoordinates());
        }
        ::glEnd();

    }

    delete[] n2;
    delete[] n1;
    delete[] p2;
    delete[] p1;
}


/*
 * misc::ExtBezierMeshRenderer::drawEllipCurves
 */
void misc::ExtBezierMeshRenderer::drawEllipCurves(
        const vislib::math::BezierCurve<misc::ExtBezierDataCall::Point, 3> *curves,
        SIZE_T cnt, unsigned int profileSections, unsigned int lengthSections) {
    static vislib::Array<vislib::math::Point<float, 2> > profile;
    if (profile.Count() != profileSections) {
        profile.SetCount(profileSections);
        for (SIZE_T i = 0; i < static_cast<SIZE_T>(profileSections); i++) {
            double ang = static_cast<double>(i) * 2.0 * M_PI
                / static_cast<double>(profileSections);
            profile[i].Set(static_cast<float>(cos(ang)), static_cast<float>(sin(ang)));
        }
    }
    this->drawCurves(curves, cnt, profile.PeekElements(), profile.Count(), lengthSections + 1, true);
}


/*
 * misc::ExtBezierMeshRenderer::drawRectCurves
 */
void misc::ExtBezierMeshRenderer::drawRectCurves(
        const vislib::math::BezierCurve<misc::ExtBezierDataCall::Point, 3> *curves,
        SIZE_T cnt, unsigned int lengthSections) {
    static vislib::math::Point<float, 2> profile[4] = {
        vislib::math::Point<float, 2>(-1.0f, -1.0f),
        vislib::math::Point<float, 2>( 1.0f, -1.0f),
        vislib::math::Point<float, 2>( 1.0f, 1.0f),
        vislib::math::Point<float, 2>(-1.0f, 1.0f) };
    this->drawCurves(curves, cnt, profile, 4, lengthSections + 1, false);
}



//int curveSegs = this->curveSectionsSlot.Param<param::IntParam>()->Value();
//float curveStep = 1.0f / static_cast<float>(curveSegs);
//int profileSegs = this->profileSectionsSlot.Param<param::IntParam>()->Value();
//unsigned int profileCnt = profileSegs + 1;
//int capType = this->capTypeSlot.Param<param::EnumParam>()->Value();

//Vec2 *profile = new Vec2[profileCnt];
//for (unsigned int i = 0; i < profileCnt; i++) {
//    float a = 2.0f * static_cast<float>(M_PI) * static_cast<float>(i)
//        / static_cast<float>(profileSegs);
//    profile[i].Set(cos(a), sin(a));
//}

//unsigned int capProfileCnt = (profileSegs / 2) + (profileSegs % 2);
//Vec2 *capProfile = new Vec2[capProfileCnt];
//for (unsigned int i = 0; i < capProfileCnt; i++) {
//    float a = 0.5f * static_cast<float>(M_PI) * static_cast<float>(i)
//        / static_cast<float>(capProfileCnt - 1);
//    capProfile[i].Set(cos(a), sin(a));
//}

//for (unsigned int i = 0; i < bdc->Count(); i++) {
//    const vislib::math::BezierCurve<Pt, 3> &curve = bdc->Curves()[i];

//    Pt p1 = curve.ControlPoint(0); // point
//    Vec3 t1;
//    curve.CalcTangent(t1, 0.0f);
//    if (t1.IsNull()) {
//        t1.Set(0.0f, 0.0f, 1.0f);
//    } else {
//        t1.Normalise(); // tangent
//    }
//    Vec3 vp1(1.0f, 0.0f, 0.0f); // primary
//    if (t1.IsParallel(vp1)) vp1.Set(0.0f, 1.0f, 0.0f);
//    Vec3 vs1 = t1.Cross(vp1); // secondary
//    vs1.Normalise();
//    vp1 = vs1.Cross(t1);
//    vp1.Normalise();

//    Pt p2; // seconds
//    Vec3 t2, vp2, vs2;

//    // Start Cap
//    if (capType == 1) {
//        ::glColor3ub(p1.R(), p1.G(), p1.B());
//        ::glBegin(GL_TRIANGLE_FAN);
//        ::glNormal3f(-t1.X(), -t1.Y(), -t1.Z());
//        for (unsigned int i = profileCnt - 1; i > 0; i--) {
//            ::glVertex3f(
//                p1.X() + p1.Radius() * (profile[i].X() * vp1.X() + profile[i].Y() * vs1.X()),
//                p1.Y() + p1.Radius() * (profile[i].X() * vp1.Y() + profile[i].Y() * vs1.Y()),
//                p1.Z() + p1.Radius() * (profile[i].X() * vp1.Z() + profile[i].Y() * vs1.Z()));
//        }
//        ::glEnd();
//    } else if (capType == 2) {
//        ::glColor3ub(p1.R(), p1.G(), p1.B());
//        for (unsigned int j = 1; j < capProfileCnt; j++) {
//            ::glBegin(GL_QUAD_STRIP);
//            for (unsigned int i = 0; i < profileCnt; i++) {
//                Vec3 v3 = vp1 * profile[i].X() + vs1 * profile[i].Y();

//                Vec3 v4 = v3 * capProfile[j - 1].X() - t1 * capProfile[j - 1].Y();
//                ::glNormal3fv(v4.PeekComponents());
//                ::glVertex3f(p1.X() + p1.Radius() * v4.X(), p1.Y() + p1.Radius() * v4.Y(), p1.Z() + p1.Radius() * v4.Z());

//                v4 = v3 * capProfile[j].X() - t1 * capProfile[j].Y();
//                ::glNormal3fv(v4.PeekComponents());
//                ::glVertex3f(p1.X() + p1.Radius() * v4.X(), p1.Y() + p1.Radius() * v4.Y(), p1.Z() + p1.Radius() * v4.Z());
//            }
//            ::glEnd();
//        }
//    }

//    // curve
//    // *1 holding values for t=0
//    for (float t = curveStep; t < 1.0f + 0.5f * curveStep; t += curveStep) {
//        if (t > 1.0f - 0.5f * curveStep) t = 1.0f;
//        p2 = p1;
//        t2 = t1;
//        vp2 = vp1;
//        vs2 = vs1;

//        curve.CalcPoint(p1, t);
//        curve.CalcTangent(t1, t);
//        if (t1.IsNull()) {
//            t1.Set(0.0f, 0.0f, 1.0f);
//        } else {
//            t1.Normalise(); // tangent
//        }
//        if (t1.IsParallel(vp1)) {
//            vp1 = vs1.Cross(t1);
//            vp1.Normalise();
//            vs1 = t1.Cross(vp1);
//            vs1.Normalise();
//        } else {
//            vs1 = t1.Cross(vp1);
//            vs1.Normalise();
//            vp1 = vs1.Cross(t1);
//            vp1.Normalise();
//        }

//        ::glBegin(GL_QUAD_STRIP);
//        for (unsigned int i = 0; i < profileCnt; i++) {

//            ::glColor3ub(p1.R(), p1.G(), p1.B());
//            ::glNormal3f(
//                profile[i].X() * vp1.X() + profile[i].Y() * vs1.X(),
//                profile[i].X() * vp1.Y() + profile[i].Y() * vs1.Y(),
//                profile[i].X() * vp1.Z() + profile[i].Y() * vs1.Z());
//            ::glVertex3f(
//                p1.X() + p1.Radius() * (profile[i].X() * vp1.X() + profile[i].Y() * vs1.X()),
//                p1.Y() + p1.Radius() * (profile[i].X() * vp1.Y() + profile[i].Y() * vs1.Y()),
//                p1.Z() + p1.Radius() * (profile[i].X() * vp1.Z() + profile[i].Y() * vs1.Z()));

//            ::glColor3ub(p2.R(), p2.G(), p2.B());
//            ::glNormal3f(
//                profile[i].X() * vp2.X() + profile[i].Y() * vs2.X(),
//                profile[i].X() * vp2.Y() + profile[i].Y() * vs2.Y(),
//                profile[i].X() * vp2.Z() + profile[i].Y() * vs2.Z());
//            ::glVertex3f(
//                p2.X() + p2.Radius() * (profile[i].X() * vp2.X() + profile[i].Y() * vs2.X()),
//                p2.Y() + p2.Radius() * (profile[i].X() * vp2.Y() + profile[i].Y() * vs2.Y()),
//                p2.Z() + p2.Radius() * (profile[i].X() * vp2.Z() + profile[i].Y() * vs2.Z()));

//        }
//        ::glEnd();

//    }

//    // End Cap
//    if (capType == 1) {
//        ::glColor3ub(p1.R(), p1.G(), p1.B());
//        ::glBegin(GL_TRIANGLE_FAN);
//        ::glNormal3f(t1.X(), t1.Y(), t1.Z());
//        for (unsigned int i = 1; i < profileCnt; i++) {
//            ::glVertex3f(
//                p1.X() + p1.Radius() * (profile[i].X() * vp1.X() + profile[i].Y() * vs1.X()),
//                p1.Y() + p1.Radius() * (profile[i].X() * vp1.Y() + profile[i].Y() * vs1.Y()),
//                p1.Z() + p1.Radius() * (profile[i].X() * vp1.Z() + profile[i].Y() * vs1.Z()));
//        }
//        ::glEnd();
//    } else if (capType == 2) {
//        ::glColor3ub(p1.R(), p1.G(), p1.B());
//        for (unsigned int j = 1; j < capProfileCnt; j++) {
//            ::glBegin(GL_QUAD_STRIP);
//            for (unsigned int i = 0; i < profileCnt; i++) {
//                Vec3 v3 = vp1 * profile[i].X() + vs1 * profile[i].Y();

//                Vec3 v4 = v3 * capProfile[j].X() + t1 * capProfile[j].Y();
//                ::glNormal3fv(v4.PeekComponents());
//                ::glVertex3f(p1.X() + p1.Radius() * v4.X(), p1.Y() + p1.Radius() * v4.Y(), p1.Z() + p1.Radius() * v4.Z());

//                v4 = v3 * capProfile[j - 1].X() + t1 * capProfile[j - 1].Y();
//                ::glNormal3fv(v4.PeekComponents());
//                ::glVertex3f(p1.X() + p1.Radius() * v4.X(), p1.Y() + p1.Radius() * v4.Y(), p1.Z() + p1.Radius() * v4.Z());
//            }
//            ::glEnd();
//        }
//    }

//}

//delete[] profile;
//delete[] capProfile;
