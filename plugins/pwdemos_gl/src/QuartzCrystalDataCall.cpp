/*
 * QuartzCrystalDataCall.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "QuartzCrystalDataCall.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/Trace.h"
#include "vislib/math/Plane.h"
#include "vislib/math/Point.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/math/Vector.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/memutils.h"
#include <cmath>


namespace megamol {
namespace demos_gl {

/*
 * CrystalDataCall::Crystal::Crystal
 */
CrystalDataCall::Crystal::Crystal(void)
        : baseRad(1.0f)
        , boundRad(1.0f)
        , faces()
        , vertices(NULL)
        , triangleCnt(NULL)
        , triangles(NULL) {
    // intentionally empty
}


/*
 * CrystalDataCall::Crystal::~Crystal
 */
CrystalDataCall::Crystal::~Crystal(void) {
    this->ClearMesh();
}


/*
 * CrystalDataCall::Crystal::AssertMesh
 */
void CrystalDataCall::Crystal::AssertMesh(void) const {
    const_cast<CrystalDataCall::Crystal*>(this)->CalculateMesh(true, false);
}


/*
 * CrystalDataCall::Crystal::AddFace
 */
void CrystalDataCall::Crystal::AddFace(const vislib::math::Vector<float, 3>& vec) {
    this->ClearMesh();
    this->faces.Add(vec);
}


/*
 * CrystalDataCall::Crystal::CalculateMesh
 */
void CrystalDataCall::Crystal::CalculateMesh(bool setBoundingRad, bool force) {
    using vislib::Pair;
    using vislib::math::Plane;
    using vislib::math::Point;
    using vislib::math::ShallowPoint;
    using vislib::math::Vector;

    const double ARGL_EPSILON1 = 0.002;
    const double ARGL_EPSILON2 = 0.004;
    const double ARGL_EPSILON3 = 0.004;

    if (this->vertices != NULL) {
        if (force)
            this->ClearMesh();
        else
            return;
    }

    SIZE_T facesCnt = this->faces.Count();

    Plane<double>* planes = new Plane<double>[facesCnt];
    for (SIZE_T i = 0; i < facesCnt; i++) {
        const Vector<float, 3>& v = this->faces[i];
        planes[i].Set(Point<float, 3>(v.PeekComponents()), v);
    }
    vislib::Array<Point<double, 3>> verts;
    Point<double, 3> pt;
    bool validPt;
    double len = 0.0;
    for (SIZE_T i = 0; i < facesCnt; i++) {
        for (SIZE_T j = i + 1; j < facesCnt; j++) {
            for (SIZE_T k = j + 1; k < facesCnt; k++) {
                if (!planes[i].CalcIntersectionPoint(planes[j], planes[k], pt))
                    continue; // no point.
                validPt = true;

                for (SIZE_T l = 0; l < facesCnt; l++) {
                    double dist = planes[l].Distance(pt);
                    if (dist > ARGL_EPSILON1) { // too far off in negative half space to be acceptable
                        validPt = false;
                        break;
                    }
                }
                if (validPt) {
                    // Use ARGL_EPSILON to avoid numeric problems. Argl!
                    //if (!verts.Contains(pt)) verts.Add(pt);
                    bool contains = false;
                    for (SIZE_T l = 0; l < verts.Count(); l++) {
                        if (vislib::math::Abs(verts[l].Distance(pt)) < ARGL_EPSILON2) {
                            contains = true;
                            break;
                        }
                    }
                    if (!contains) {
                        verts.Add(pt);
                    }
                    double l = pt.Distance(Point<double, 3>());
                    if (l > len) {
                        len = l;
                    }
                }
            }
        }
    }
    if (setBoundingRad) {
        this->boundRad = static_cast<float>(len); //:-/
    }

    this->vertices = new float[3 * verts.Count()];
    for (unsigned int i = 0; i < verts.Count(); i++) {
        this->vertices[i * 3 + 0] = static_cast<float>(verts[i].X());
        this->vertices[i * 3 + 1] = static_cast<float>(verts[i].Y());
        this->vertices[i * 3 + 2] = static_cast<float>(verts[i].Z());
    }
    unsigned int* idz = new unsigned int[verts.Count()];
    this->triangles = new unsigned int*[facesCnt];
    this->triangleCnt = new unsigned int[facesCnt];
    float* angs = new float[verts.Count()]; // angles of points on plane

    for (SIZE_T i = 0; i < facesCnt; i++) {
        unsigned int idzc = 0;
        Vector<float, 3> n(this->faces[i]);
        n.Normalise();
        Point<float, 3> mid(n.PeekComponents());
        Vector<float, 3> x, y;
        for (unsigned int j = 0; j < verts.Count(); j++) {
            // point to check verts[j];
            double dist = planes[i].Distance(verts[j]);
            if (vislib::math::Abs(dist) < ARGL_EPSILON3) {
                Vector<float, 2> p2d;                   // point on plane
                Vector<float, 3> diff = verts[j] - mid; // diffvector
                float off = n.Dot(diff);
                diff -= n * off; // now diff really is in plane! (as good as it gets)
                if (idzc == 0) {
                    idz[0] = j;
                    x = diff;
                    p2d.Set(x.Normalise(), 0.0f);
                    ASSERT(vislib::math::IsEqual(n.Dot(x), 0.0f));
                    y = n.Cross(x);
                    ASSERT(y.IsNormalised());

                } else {
                    idz[idzc] = j;
                    p2d.Set(x.Dot(diff), y.Dot(diff));
                }
                angs[idzc] = -::atan2(p2d.X(), p2d.Y());
                idzc++;
            }
        }

#if defined(DEBUG) || defined(_DEBUG)
        //printf("Face %d: (%u vertices)\n", static_cast<int>(i), idzc);
        //for (unsigned int j = 0; j < idzc; j++) {
        //    printf("  %u: (%f, %f, %f) ~ %f\n", idz[j],
        //        this->vertices[idz[j] * 3 + 0],
        //        this->vertices[idz[j] * 3 + 1],
        //        this->vertices[idz[j] * 3 + 2],
        //        angs[j]);
        //}
#endif

        this->triangles[i] = new unsigned int[idzc];
        if (idzc > 0) {
            //VLTRACE(VISLIB_TRCELVL_INFO, "Building face fan with %u vertices\n", idzc);
            float minAng = 2.0f * static_cast<float>(M_PI);
            for (unsigned int j = 0; j < idzc; j++) {
                if (angs[j] < minAng) {
                    minAng = angs[j];
                    this->triangles[i][0] = idz[j];
                }
            }
            //VLTRACE(VISLIB_TRCELVL_INFO, "  Starting with pt %u @ %f\n", this->triangles[i][0], minAng);

            for (unsigned int j = 1; j < idzc; j++) {
                float next = 2.0f * static_cast<float>(M_PI);
                unsigned int nextPos = idzc + 1;
                for (unsigned int k = 0; k < idzc; k++) {
                    if (angs[k] <= minAng)
                        continue;
                    if (angs[k] < next) {
                        next = angs[k];
                        this->triangles[i][j] = idz[k];
                        nextPos = k;
                    }
                }
                if (nextPos >= idzc) {
                    idzc = j;
                    //VLTRACE(VISLIB_TRCELVL_INFO, "  early fan termination\n");
                    break;
                }
                //VLTRACE(VISLIB_TRCELVL_INFO, "  next pt %u @ %f\n", idz[nextPos], next);
                minAng = next;
            }
        }
        this->triangleCnt[i] = idzc;
    }

    delete[] idz;
    delete[] angs;

    delete[] planes;
}


/*
 * CrystalDataCall::Crystal::Clear
 */
void CrystalDataCall::Crystal::Clear(void) {
    this->ClearMesh();
    this->faces.Clear();
    this->baseRad = 1.0f;
    this->boundRad = 1.0f;
}


/*
 * CrystalDataCall::Crystal::ClearMesh
 */
void CrystalDataCall::Crystal::ClearMesh(void) {
    ARY_SAFE_DELETE(this->triangleCnt);
    ARY_SAFE_DELETE(this->vertices);
    if (this->triangles != NULL) {
        for (SIZE_T i = 0; i < this->faces.Count(); i++) {
            delete[] this->triangles[i];
        }
        ARY_SAFE_DELETE(this->triangles);
    }
}


/*
 * CrystalDataCall::Crystal::operator==
 */
bool CrystalDataCall::Crystal::operator==(const CrystalDataCall::Crystal& rhs) const {
    return vislib::math::IsEqual(this->baseRad, rhs.baseRad) && vislib::math::IsEqual(this->boundRad, rhs.boundRad) &&
           (this->faces == rhs.faces); // deep compare
}


/*
 * CrystalDataCall::CallForGetData
 */
const unsigned int CrystalDataCall::CallForGetData = 0;


/*
 * CrystalDataCall::CallForGetExtent
 */
const unsigned int CrystalDataCall::CallForGetExtent = 1;


/*
 * CrystalDataCall::CrystalDataCall
 */
CrystalDataCall::CrystalDataCall(void) : core::AbstractGetData3DCall(), count(0), crystals(NULL) {}


/*
 * CrystalDataCall::~CrystalDataCall
 */
CrystalDataCall::~CrystalDataCall(void) {
    this->count = 0;
    this->crystals = NULL; // DO NOT DELETE
}
} // namespace demos_gl
} /* end namespace megamol */
