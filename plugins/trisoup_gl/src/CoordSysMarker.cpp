/*
 * CoordSysMarker.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "CoordSysMarker.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/assert.h"
#include "vislib/math/Point.h"
#include "vislib/math/Vector.h"

using namespace megamol;
using namespace megamol::trisoup_gl;


/*
 * CoordSysMarker::CoordSysMarker
 */
CoordSysMarker::CoordSysMarker() : AbstractTriMeshDataSource() {
    // intentionally empty
}


/*
 * CoordSysMarker::~CoordSysMarker
 */
CoordSysMarker::~CoordSysMarker() {
    this->Release();
}


/*
 * CoordSysMarker::assertData
 */
void CoordSysMarker::assertData() {
    typedef vislib::math::Point<float, 3> Point;
    typedef vislib::math::Vector<float, 3> Vector;

    // materials
    this->mats.SetCount(4);
    this->mats[0].MakeDefault();

    this->mats[1].MakeDefault();
    this->mats[1].Dye(1.0f, 0.0f, 0.0f);

    this->mats[2].MakeDefault();
    this->mats[2].Dye(0.0f, 1.0f, 0.0f);

    this->mats[3].MakeDefault();
    this->mats[3].Dye(0.0f, 0.0f, 1.0f);

    // objects
    const unsigned int cntSeg = 24;
    float sin[cntSeg + 1];
    float cos[cntSeg + 1];
    for (unsigned int i = 0; i < cntSeg; i++) {
        sin[i] = static_cast<float>(
            ::sin(vislib::math::PI_DOUBLE * 2.0 * static_cast<double>(i) / static_cast<double>(cntSeg)));
        cos[i] = static_cast<float>(
            ::cos(vislib::math::PI_DOUBLE * 2.0 * static_cast<double>(i) / static_cast<double>(cntSeg)));
    }
    sin[cntSeg] = sin[0];
    cos[cntSeg] = cos[0];

    // triangle data of arrow geometry
    vislib::Array<Point> v;
    vislib::Array<Vector> n;

    for (unsigned int i = 0; i < cntSeg; i++) {

        // cone
        v.Add(Point(0.0f, sin[i] * 0.1f, cos[i] * 0.1f));
        n.Add(Vector(0.0f, sin[i], cos[i]));
        v.Add(Point(0.5f, sin[i] * 0.1f, cos[i] * 0.1f));
        n.Add(Vector(0.0f, sin[i], cos[i]));
        v.Add(Point(0.0f, sin[i + 1] * 0.1f, cos[i + 1] * 0.1f));
        n.Add(Vector(0.0f, sin[i + 1], cos[i + 1]));

        v.Add(Point(0.5f, sin[i] * 0.1f, cos[i] * 0.1f));
        n.Add(Vector(0.0f, sin[i], cos[i]));
        v.Add(Point(0.5f, sin[i + 1] * 0.1f, cos[i + 1] * 0.1f));
        n.Add(Vector(0.0f, sin[i + 1], cos[i + 1]));
        v.Add(Point(0.0f, sin[i + 1] * 0.1f, cos[i + 1] * 0.1f));
        n.Add(Vector(0.0f, sin[i + 1], cos[i + 1]));

        // head
        v.Add(Point(0.5f, sin[i] * 0.3f, cos[i] * 0.3f));
        n.Add(Vector(0.0f, sin[i], cos[i]));
        v.Add(Point(0.9f, sin[i] * 0.06f, cos[i] * 0.06f));
        n.Add(Vector(0.5f, sin[i], cos[i]));
        v.Add(Point(0.5f, sin[i + 1] * 0.3f, cos[i + 1] * 0.3f));
        n.Add(Vector(0.0f, sin[i + 1], cos[i + 1]));

        v.Add(Point(0.9f, sin[i] * 0.06f, cos[i] * 0.06f));
        n.Add(Vector(0.5f, sin[i], cos[i]));
        v.Add(Point(0.9f, sin[i + 1] * 0.06f, cos[i + 1] * 0.06f));
        n.Add(Vector(0.5f, sin[i + 1], cos[i + 1]));
        v.Add(Point(0.5f, sin[i + 1] * 0.3f, cos[i + 1] * 0.3f));
        n.Add(Vector(0.0f, sin[i + 1], cos[i + 1]));

        // tip
        v.Add(Point(0.9f, sin[i] * 0.06f, cos[i] * 0.06f));
        n.Add(Vector(0.5f, sin[i], cos[i]));
        v.Add(Point(1.0f, 0.0f, 0.0f));
        n.Add(Vector(1.0f, 0.0f, 0.0f));
        v.Add(Point(0.9f, sin[i + 1] * 0.06f, cos[i + 1] * 0.06f));
        n.Add(Vector(0.5f, sin[i + 1], cos[i + 1]));
    }

    // caps
    for (unsigned int i = 1; i < cntSeg; i++) {
        v.Add(Point(0.5f, sin[0] * 0.3f, cos[0] * 0.3f));
        n.Add(Vector(-1.0f, 0.0f, 0.0f));
        v.Add(Point(0.5f, sin[i] * 0.3f, cos[i] * 0.3f));
        n.Add(Vector(-1.0f, 0.0f, 0.0f));
        v.Add(Point(0.5f, sin[i + 1] * 0.3f, cos[i + 1] * 0.3f));
        n.Add(Vector(-1.0f, 0.0f, 0.0f));
    }

    const float tm[6][3][3] = {{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{0.0f, 1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}},
        {{-1.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}};

    ASSERT(v.Count() == n.Count());
    this->objs.SetCount(6);
    for (SIZE_T i = 0; i < 6; i++) {
        this->objs[i].SetMaterial((i < 3) ? &this->mats[i + 1] : &this->mats[0]);

        float* vDat = new float[v.Count() * 3];
        float* nDat = new float[v.Count() * 3];
        for (SIZE_T j = 0; j < v.Count(); j++) {
            vDat[j * 3 + 0] = tm[i][0][0] * v[j].X() + tm[i][1][0] * v[j].Y() + tm[i][2][0] * v[j].Z();
            vDat[j * 3 + 1] = tm[i][0][1] * v[j].X() + tm[i][1][1] * v[j].Y() + tm[i][2][1] * v[j].Z();
            vDat[j * 3 + 2] = tm[i][0][2] * v[j].X() + tm[i][1][2] * v[j].Y() + tm[i][2][2] * v[j].Z();
            nDat[j * 3 + 0] = tm[i][0][0] * n[j].X() + tm[i][1][0] * n[j].Y() + tm[i][2][0] * n[j].Z();
            nDat[j * 3 + 1] = tm[i][0][1] * n[j].X() + tm[i][1][1] * n[j].Y() + tm[i][2][1] * n[j].Z();
            nDat[j * 3 + 2] = tm[i][0][2] * n[j].X() + tm[i][1][2] * n[j].Y() + tm[i][2][2] * n[j].Z();
        }

        this->objs[i].SetVertexData(static_cast<unsigned int>(v.Count()), vDat, nDat, NULL, NULL, true);
    }

    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->datahash = 1;
}
