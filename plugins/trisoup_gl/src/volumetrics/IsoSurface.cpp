/*
 * IsoSurface.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "volumetrics/IsoSurface.h"
#include "geometry_calls/VolumetricDataCall.h"
#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "trisoup/volumetrics/MarchingCubeTables.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/math/Plane.h"
#include "vislib/math/Vector.h"
#include <cfloat>
#include <climits>
#include <cmath>

using namespace megamol;
using namespace megamol::trisoup_gl;
using namespace megamol::trisoup_gl::volumetrics;


/*
 * IsoSurface::tets
 */
const unsigned int IsoSurface::tets[6][4] = {
    {0, 2, 3, 7}, {0, 2, 6, 7}, {0, 4, 6, 7}, {0, 6, 1, 2}, {0, 6, 1, 4}, {5, 6, 1, 4}};


/*
 * IsoSurface::IsoSurface
 */
IsoSurface::IsoSurface(void)
        : inDataSlot("inData", "The slot for requesting input data")
        , outDataSlot("outData", "Gets the data")
        , attributeSlot("attr", "The attribute to show")
        , isoValueSlot("isoval", "The iso value")
        , dataHash(0)
        , frameIdx(0)
        , index()
        , vertex()
        , normal()
        , mesh() {

    this->inDataSlot.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->outDataSlot.SetCallback("CallTriMeshData", "GetData", &IsoSurface::outDataCallback);
    this->outDataSlot.SetCallback("CallTriMeshData", "GetExtent", &IsoSurface::outExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->attributeSlot << new core::param::StringParam("0");
    this->MakeSlotAvailable(&this->attributeSlot);

    this->isoValueSlot << new core::param::FloatParam(0.5f);
    this->MakeSlotAvailable(&this->isoValueSlot);
}


/*
 * IsoSurface::~IsoSurface
 */
IsoSurface::~IsoSurface(void) {
    this->Release();
}


/*
 * IsoSurface::create
 */
bool IsoSurface::create(void) {
    // intentionally empty
    return true;
}


/*
 * IsoSurface::release
 */
void IsoSurface::release(void) {
    this->index.EnforceSize(0);
    this->vertex.EnforceSize(0);
    this->normal.EnforceSize(0);
}


/*
 * IsoSurface::outDataCallback
 */
bool IsoSurface::outDataCallback(core::Call& caller) {
    megamol::geocalls_gl::CallTriMeshDataGL* tmd = dynamic_cast<megamol::geocalls_gl::CallTriMeshDataGL*>(&caller);
    if (tmd == NULL)
        return false;

    geocalls::VolumetricDataCall* cvd = this->inDataSlot.CallAs<geocalls::VolumetricDataCall>();
    if (cvd != NULL) {

        bool recalc = false;

        if (this->isoValueSlot.IsDirty()) {
            this->isoValueSlot.ResetDirty();
            recalc = true;
        }

        if (this->attributeSlot.IsDirty()) {
            this->attributeSlot.ResetDirty();
            recalc = true;
        }

        cvd->SetFrameID(tmd->FrameID(), tmd->IsFrameForced());
        if (!(*cvd)(0)) {
            recalc = false;
        } else {
            if ((this->dataHash != cvd->DataHash()) || (this->frameIdx != cvd->FrameID())) {
                recalc = true;
            }
        }

        unsigned int attrIdx = UINT_MAX;
        if (recalc) {
            if (cvd->GetScalarType() != geocalls::VolumetricDataCall::ScalarType::FLOATING_POINT) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("Only float volumes are supported ATM");
                recalc = false;
            }
        }

        if (recalc) {
            float isoVal = this->isoValueSlot.Param<core::param::FloatParam>()->Value();

            this->index.EnforceSize(0);
            this->vertex.EnforceSize(0);
#ifdef WITH_COLOUR_DATA
            this->colour.EnforceSize(0);
#endif /* WITH_COLOUR_DATA */
            this->normal.EnforceSize(0);

            vislib::RawStorageWriter i(this->index);
            vislib::RawStorageWriter v(this->vertex);
#ifdef WITH_COLOUR_DATA
            vislib::RawStorageWriter c(this->colour);
#endif /* WITH_COLOUR_DATA */
            vislib::RawStorageWriter n(this->normal);
            i.SetIncrement(1024 * 1024);
            v.SetIncrement(1024 * 1024);
#ifdef WITH_COLOUR_DATA
            c.SetIncrement(1024 * 1024);
#endif /* WITH_COLOUR_DATA */
            n.SetIncrement(1024 * 1024);

            // Rebuild mesh data
            this->buildMesh(i, v,
#ifdef WITH_COLOUR_DATA
                c,
#endif /* WITH_COLOUR_DATA */
                n, isoVal, static_cast<float*>(cvd->GetData()), cvd->GetResolution(0), cvd->GetResolution(1),
                cvd->GetResolution(2));

            this->index.EnforceSize(i.End(), true);
            this->vertex.EnforceSize(v.End(), true);
#ifdef WITH_COLOUR_DATA
            this->colour.EnforceSize(c.End(), true);
#endif /* WITH_COLOUR_DATA */
            this->normal.EnforceSize(n.End(), true);

            this->mesh.SetMaterial(NULL);
            this->mesh.SetVertexData(static_cast<unsigned int>(this->vertex.GetSize() / (3 * sizeof(float))),
                this->vertex.As<float>(), this->normal.As<float>(),
#ifdef WITH_COLOUR_DATA
                this->colour.As<float>(),
#else  /* WITH_COLOUR_DATA */
                NULL,
#endif /* WITH_COLOUR_DATA */
                NULL, false);
            this->mesh.SetTriangleData(static_cast<unsigned int>(this->index.GetSize() / (3 * sizeof(unsigned int))),
                this->index.As<unsigned int>(), false);

            this->dataHash = cvd->DataHash();
            this->frameIdx = cvd->FrameID();
        }
    }

    tmd->SetDataHash(this->dataHash);
    tmd->SetFrameID(this->frameIdx);
    tmd->SetObjects(1, &this->mesh);
    tmd->SetUnlocker(NULL);

    return true;
}


/*
 * IsoSurface::outExtentCallback
 */
bool IsoSurface::outExtentCallback(megamol::core::Call& caller) {
    megamol::geocalls_gl::CallTriMeshDataGL* tmd = dynamic_cast<megamol::geocalls_gl::CallTriMeshDataGL*>(&caller);
    if (tmd == NULL)
        return false;

    tmd->AccessBoundingBoxes().Clear();
    geocalls::VolumetricDataCall* cvd = this->inDataSlot.CallAs<geocalls::VolumetricDataCall>();
    cvd->SetFrameID(tmd->FrameID(), tmd->IsFrameForced());
    if ((cvd == NULL) || (!(*cvd)(geocalls::VolumetricDataCall::IDX_GET_METADATA))) {
        // no input data
        tmd->SetDataHash(0);
        tmd->SetFrameCount(1);

    } else {
        // input data in cvd
        tmd->SetDataHash(cvd->DataHash());
        tmd->SetExtent(cvd->FrameCount(), cvd->AccessBoundingBoxes());
        this->osbb = cvd->AccessBoundingBoxes().ObjectSpaceBBox();
    }
    tmd->SetUnlocker(NULL);

    return true;
}


/*
 * IsoSurface::buildMesh
 */
void IsoSurface::buildMesh(vislib::RawStorageWriter& i, vislib::RawStorageWriter& v,
#ifdef WITH_COLOUR_DATA
    vislib::RawStorageWriter& c,
#endif /* WITH_COLOUR_DATA */
    vislib::RawStorageWriter& n, float val, const float* vol, unsigned int sx, unsigned int sy, unsigned int sz) {

    // DEBUG: though all voxel
    float cubeValues[8];
    vislib::math::Point<float, 3> pts[8];
    const float cellSizeX = this->osbb.Width() / static_cast<float>(sx);
    const float cellSizeY = this->osbb.Height() / static_cast<float>(sy);
    const float cellSizeZ = this->osbb.Depth() / static_cast<float>(sz);

    for (unsigned int z = 0; z < sz - 1; z++) {
        vislib::math::Point<float, 3> p(0.0f, 0.0f, (static_cast<float>(z) + 0.5f) / static_cast<float>(sz));
        p[2] = p[2] * this->osbb.Depth() + this->osbb.Back();

        for (unsigned int y = 0; y < sy - 1; y++) {
            p[1] = (static_cast<float>(y) + 0.5f) / static_cast<float>(sy);
            p[1] = p[1] * this->osbb.Height() + this->osbb.Bottom();

            for (unsigned int x = 0; x < sx - 1; x++) {
                p[0] = (static_cast<float>(x) + 0.5f) / static_cast<float>(sx);
                p[0] = p[0] * this->osbb.Width() + this->osbb.Left();

                bool bigger = false;
                bool smaller = false;
                for (unsigned int j = 0; j < 8; j++) {
                    cubeValues[j] =
                        vol[(x + trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[j][0]) +
                            sx * ((y + trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[j][1]) +
                                     sy * (z + trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[j][2]))];
                    bigger = bigger || (cubeValues[j] >= val);
                    smaller = smaller || (cubeValues[j] < val);
                }
                if (!bigger || !smaller)
                    continue;

                for (unsigned int tetIdx = 0; tetIdx < 6; tetIdx++) {
                    unsigned int triIdx = 0;
                    if (cubeValues[tets[tetIdx][0]] < val)
                        triIdx |= 1;
                    if (cubeValues[tets[tetIdx][1]] < val)
                        triIdx |= 2;
                    if (cubeValues[tets[tetIdx][2]] < val)
                        triIdx |= 4;
                    if (cubeValues[tets[tetIdx][3]] < val)
                        triIdx |= 8;

#if 0
                    for (unsigned int j = 0; j < 4; j++) {
                        pts[j].Set(
                            p.X() + static_cast<float>(MarchingCubeTables::a2fVertexOffset[tets[tetIdx][j]][0]) * cellSizeX,
                            p.Y() + static_cast<float>(MarchingCubeTables::a2fVertexOffset[tets[tetIdx][j]][1]) * cellSizeY,
                            p.Z() + static_cast<float>(MarchingCubeTables::a2fVertexOffset[tets[tetIdx][j]][2]) * cellSizeZ);
                    }

                    this->makeTet(triIdx, pts,
                        cubeValues[tets[tetIdx][0]],
                        cubeValues[tets[tetIdx][1]],
                        cubeValues[tets[tetIdx][2]],
                        cubeValues[tets[tetIdx][3]],
                        val, i, v, n);
#else
                    for (unsigned int j = 0; j < 8; j++) {
                        pts[j].Set(p.X() + static_cast<float>(
                                               trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[j][0]) *
                                               cellSizeX,
                            p.Y() +
                                static_cast<float>(trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[j][1]) *
                                    cellSizeY,
                            p.Z() +
                                static_cast<float>(trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[j][2]) *
                                    cellSizeZ);
                    }
                    this->makeTet(triIdx, tetIdx, pts, cubeValues, val, i, v, n);
#endif
                }
            }
        }
    }

#ifdef WITH_COLOUR_DATA
    float* vd = this->vertex.As<float>();
    SIZE_T vc = v.End() / (3 * sizeof(float));
    for (SIZE_T j = 0; j < vc; j++, vd += 3) {
        float r, g, b;

        float x = (vd[0] - this->osbb.Left()) / this->osbb.Width();
        float y = (vd[1] - this->osbb.Bottom()) / this->osbb.Height();
        float z = (vd[2] - this->osbb.Back()) / this->osbb.Depth();
        x *= static_cast<float>(sx);
        y *= static_cast<float>(sy);
        z *= static_cast<float>(sz);
        x -= 0.5f;
        y -= 0.5f;
        z -= 0.5f;
        int ix = static_cast<int>(x);
        int iy = static_cast<int>(y);
        int iz = static_cast<int>(z);
        x -= static_cast<float>(ix);
        y -= static_cast<float>(iy);
        z -= static_cast<float>(iz);

        float vv[8];
        for (int ox = 0; ox < 2; ox++) {
            int px = ix + ox;
            if (px < 0)
                px = 0;
            if (px >= static_cast<int>(sx))
                px = static_cast<int>(sx) - 1;
            for (int oy = 0; oy < 2; oy++) {
                int py = iy + oy;
                if (py < 0)
                    py = 0;
                if (py >= static_cast<int>(sy))
                    py = static_cast<int>(sy) - 1;
                for (int oz = 0; oz < 2; oz++) {
                    int pz = iz + oz;
                    if (pz < 0)
                        pz = 0;
                    if (pz >= static_cast<int>(sz))
                        pz = static_cast<int>(sz) - 1;

                    vv[ox + 2 * (oy + 2 * oz)] = vol[px + sx * (py + sy * pz)];
                }
            }
        }

        vv[0] = (1.0f - x) * vv[0] + x * vv[1];
        vv[2] = (1.0f - x) * vv[2] + x * vv[3];
        vv[4] = (1.0f - x) * vv[4] + x * vv[5];
        vv[6] = (1.0f - x) * vv[6] + x * vv[7];

        vv[0] = (1.0f - y) * vv[0] + y * vv[2];
        vv[4] = (1.0f - y) * vv[4] + y * vv[6];

        vv[0] = (1.0f - z) * vv[0] + z * vv[4];

        if (vv[0] < val) {
            r = 1.0f;
            g = 0.0f;
            b = 0.0f;
            vv[0] = val - vv[0];
        } else {
            r = 0.0f;
            g = 0.0f;
            b = 1.0f;
            vv[0] -= val;
        }
        vv[0] *= 1000.0f;
        ASSERT(vv[0] >= 0.0f);
        if (vv[0] > 1.0f)
            vv[0] = 1.0f;
        vv[1] = 1.0f - vv[0];
        r = vv[1] + vv[0] * r;
        g = vv[1] + vv[0] * g;
        b = vv[1] + vv[0] * b;

        c.Write(r);
        c.Write(g);
        c.Write(b);
    }
#endif /* WITH_COLOUR_DATA */
}


/*
 * IsoSurface::getOffset
 */
float IsoSurface::getOffset(float fValue1, float fValue2, float fValueDesired) {
    float fDelta = fValue2 - fValue1;
    ASSERT(fDelta != 0.0f);
    float res = (fValueDesired - fValue1) / fDelta;
    ASSERT((res <= 1.0f) && (res >= 0.0f));
    return res;
}


float getValue(float* cv, unsigned int idx0, unsigned int idx1, float a) {
    float b = 1.0f - a;
    float x = b * trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[idx0][0] +
              a * trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[idx1][0];
    float y = b * trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[idx0][1] +
              a * trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[idx1][1];
    float z = b * trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[idx0][2] +
              a * trisoup::volumetrics::MarchingCubeTables::a2fVertexOffset[idx1][2];
    float vv[4];

    vv[0] = (1.0f - x) * cv[0] + x * cv[1];
    vv[1] = (1.0f - x) * cv[3] + x * cv[2];
    vv[2] = (1.0f - x) * cv[4] + x * cv[5];
    vv[3] = (1.0f - x) * cv[7] + x * cv[6];

    vv[0] = (1.0f - y) * vv[0] + y * vv[1];
    vv[2] = (1.0f - y) * vv[2] + y * vv[3];

    return (1.0f - z) * vv[0] + z * vv[2];
}


/*
 * IsoSurface::interpolate
 */
vislib::math::Point<float, 3> IsoSurface::interpolate(
    vislib::math::Point<float, 3>* pts, float* cv, float val, unsigned int idx0, unsigned int idx1) {
    float a0 = 0.0f;
    float v0 = getValue(cv, idx0, idx1, a0);
    if (vislib::math::IsEqual(v0, val))
        return pts[idx0];
    float a1 = 1.0f;
    float v1 = getValue(cv, idx0, idx1, a1);
    if (vislib::math::IsEqual(v1, val))
        return pts[idx1];
    float a = getOffset(cv[idx0], cv[idx1], val);
    float v = getValue(cv, idx0, idx1, a);
    unsigned int maxStep = 100;
    bool flip = cv[idx0] > cv[idx1];

    while ((maxStep > 0) && !vislib::math::IsEqual(v, val)) {
        ASSERT(((v0 <= val) && (val <= v1)) || ((v1 <= val) && (val <= v0)));

        if ((!flip && (v > val)) || (flip && (v < val))) {
            a1 = a;
            v1 = v;
        } else {
            a0 = a;
            v0 = v;
        }
        a = a0 + getOffset(v0, v1, val) * (a1 - a0);
        v = getValue(cv, idx0, idx1, a);

        maxStep--;
    }

    return pts[idx0].Interpolate(pts[idx1], a);
}


/*
 * IsoSurface::makeTet
 */
void IsoSurface::makeTet(unsigned int triIdx, vislib::math::Point<float, 3>* pts, float v0, float v1, float v2,
    float v3, float val, vislib::RawStorageWriter& idxWrtr, vislib::RawStorageWriter& vrtWrtr,
    vislib::RawStorageWriter& nrlWrtr) {
    vislib::math::Point<float, 3> tri[3];
    vislib::math::Point<float, 3> tri2[3];
    vislib::math::Point<float, 3>& p0 = pts[0];
    vislib::math::Point<float, 3>& p1 = pts[1];
    vislib::math::Point<float, 3>& p2 = pts[2];
    vislib::math::Point<float, 3>& p3 = pts[3];
    vislib::math::Vector<float, 3> norm = (p2 - p1).Cross(p3 - p1);
    norm.Normalise();
    vislib::math::Plane<float> pln(p1, norm);
    bool flip = (pln.Halfspace(p0) == vislib::math::HALFSPACE_POSITIVE);

    unsigned int triOffset = 0;
    switch (triIdx) {
    case 0x00:
    case 0x0F:
        break;
    case 0x01:
        flip = !flip;
    case 0x0E:
        tri[0] = p0.Interpolate(p1, getOffset(v0, v1, val));
        tri[flip ? 2 : 1] = p0.Interpolate(p2, getOffset(v0, v2, val));
        tri[flip ? 1 : 2] = p0.Interpolate(p3, getOffset(v0, v3, val));
        triOffset++;
        break;
    case 0x02:
        flip = !flip;
    case 0x0D:
        tri[0] = p1.Interpolate(p0, getOffset(v1, v0, val));
        tri[flip ? 2 : 1] = p1.Interpolate(p3, getOffset(v1, v3, val));
        tri[flip ? 1 : 2] = p1.Interpolate(p2, getOffset(v1, v2, val));
        triOffset++;
        break;
    case 0x0C:
        flip = !flip;
    case 0x03:
        // tetrahedron 1: around p1: p1->p2, p0->p2, p1->p3
        // tetrahedron 2: around p0: p0->p3, p1->p3, p0->p2
        // tetrahedron 3: around p1: p0, p1->p3, p0->p2
        tri[0] = p0.Interpolate(p3, getOffset(v0, v3, val));
        tri[flip ? 2 : 1] = p0.Interpolate(p2, getOffset(v0, v2, val));
        tri[flip ? 1 : 2] = p1.Interpolate(p3, getOffset(v1, v3, val));
        triOffset++;
        tri2[0] = tri[flip ? 1 : 2];
        tri2[flip ? 1 : 2] = p1.Interpolate(p2, getOffset(v1, v2, val));
        tri2[flip ? 2 : 1] = tri[flip ? 2 : 1];
        triOffset++;
        break;
    case 0x04:
        flip = !flip;
    case 0x0B:
        tri[0] = p2.Interpolate(p0, getOffset(v2, v0, val));
        tri[flip ? 2 : 1] = p2.Interpolate(p1, getOffset(v2, v1, val));
        tri[flip ? 1 : 2] = p2.Interpolate(p3, getOffset(v2, v3, val));
        triOffset++;
        break;
    case 0x05:
        flip = !flip;
    case 0x0A:
        // WARNING: per analogy = 3, subst 1 with 2
        // tetrahedron 1: around p2: p1->p2, p0->p1, p2->p3
        // tetrahedron 2: around p0: p0->p3, p2->p3, p0->p1
        // tetrahedron 3: around p2: p0, p2->p3, p0->p1
        tri[0] = p0.Interpolate(p1, getOffset(v0, v1, val));
        tri[flip ? 2 : 1] = p2.Interpolate(p3, getOffset(v2, v3, val));
        tri[flip ? 1 : 2] = p0.Interpolate(p3, getOffset(v0, v3, val));
        triOffset++;
        tri2[0] = tri[0];
        tri2[flip ? 2 : 1] = p1.Interpolate(p2, getOffset(v1, v2, val));
        tri2[flip ? 1 : 2] = tri[flip ? 2 : 1];
        triOffset++;
        break;
    case 0x06:
        flip = !flip;
    case 0x09:
        // WARNING: per analogy = 3, subst 0 with 2
        // tetrahedron 1: around p1: p1->p0, p0->p2, p1->p3
        // tetrahedron 2: around p2: p2->p3, p1->p3, p0->p2
        // tetrahedron 3: around p1: p2, p1->p3, p0->p2
        tri[0] = p0.Interpolate(p1, getOffset(v0, v1, val));
        tri[flip ? 2 : 1] = p1.Interpolate(p3, getOffset(v1, v3, val));
        tri[flip ? 1 : 2] = p2.Interpolate(p3, getOffset(v2, v3, val));
        triOffset++;
        tri2[0] = tri[0];
        tri2[flip ? 1 : 2] = p0.Interpolate(p2, getOffset(v0, v2, val));
        tri2[flip ? 2 : 1] = tri[flip ? 1 : 2];
        triOffset++;
        break;
    case 0x08:
        flip = !flip;
    case 0x07:
        tri[0] = p3.Interpolate(p0, getOffset(v3, v0, val));
        tri[flip ? 2 : 1] = p3.Interpolate(p2, getOffset(v3, v2, val));
        tri[flip ? 1 : 2] = p3.Interpolate(p1, getOffset(v3, v1, val));
        triOffset++;
        break;
    }

    if (triOffset == 0)
        return;
    // norm?
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            vrtWrtr.Write(tri[i][j]);
        }
    }
    norm = (tri[1] - tri[0]).Cross(tri[2] - tri[0]);
    norm.Normalise();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            nrlWrtr.Write(norm[j]);
        }
    }
    if (triOffset == 1)
        return;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            vrtWrtr.Write(tri2[i][j]);
        }
    }
    norm = (tri2[1] - tri2[0]).Cross(tri2[2] - tri2[0]);
    norm.Normalise();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            nrlWrtr.Write(norm[j]);
        }
    }
}


/*
 * IsoSurface::makeTet
 */
void IsoSurface::makeTet(unsigned int triIdx, unsigned int tetIdx, vislib::math::Point<float, 3>* pts, float* cv,
    float val, vislib::RawStorageWriter& idxWrtr, vislib::RawStorageWriter& vrtWrtr,
    vislib::RawStorageWriter& nrlWrtr) {
    vislib::math::Point<float, 3> tri[3];
    vislib::math::Point<float, 3> tri2[3];
    vislib::math::Point<float, 3>& p0 = pts[tets[tetIdx][0]];
    vislib::math::Point<float, 3>& p1 = pts[tets[tetIdx][1]];
    vislib::math::Point<float, 3>& p2 = pts[tets[tetIdx][2]];
    vislib::math::Point<float, 3>& p3 = pts[tets[tetIdx][3]];
    vislib::math::Vector<float, 3> norm = (p2 - p1).Cross(p3 - p1);
    norm.Normalise();
    vislib::math::Plane<float> pln(p1, norm);
    bool flip = (pln.Halfspace(p0) == vislib::math::HALFSPACE_POSITIVE);

    unsigned int triOffset = 0;
    switch (triIdx) {
    case 0x00:
    case 0x0F:
        break;
    case 0x01:
        flip = !flip;
    case 0x0E:
        tri[0] = interpolate(pts, cv, val, tets[tetIdx][0], tets[tetIdx][1]);
        tri[flip ? 2 : 1] = interpolate(pts, cv, val, tets[tetIdx][0], tets[tetIdx][2]);
        tri[flip ? 1 : 2] = interpolate(pts, cv, val, tets[tetIdx][0], tets[tetIdx][3]);
        triOffset++;
        break;
    case 0x02:
        flip = !flip;
    case 0x0D:
        tri[0] = interpolate(pts, cv, val, tets[tetIdx][1], tets[tetIdx][0]);
        tri[flip ? 2 : 1] = interpolate(pts, cv, val, tets[tetIdx][1], tets[tetIdx][3]);
        tri[flip ? 1 : 2] = interpolate(pts, cv, val, tets[tetIdx][1], tets[tetIdx][2]);
        triOffset++;
        break;
    case 0x0C:
        flip = !flip;
    case 0x03:
        // tetrahedron 1: around p1: p1->p2, p0->p2, p1->p3
        // tetrahedron 2: around p0: p0->p3, p1->p3, p0->p2
        // tetrahedron 3: around p1: p0, p1->p3, p0->p2
        tri[0] = interpolate(pts, cv, val, tets[tetIdx][0], tets[tetIdx][3]);
        tri[flip ? 2 : 1] = interpolate(pts, cv, val, tets[tetIdx][0], tets[tetIdx][2]);
        tri[flip ? 1 : 2] = interpolate(pts, cv, val, tets[tetIdx][1], tets[tetIdx][3]);
        triOffset++;
        tri2[0] = tri[flip ? 1 : 2];
        tri2[flip ? 1 : 2] = interpolate(pts, cv, val, tets[tetIdx][1], tets[tetIdx][2]);
        tri2[flip ? 2 : 1] = tri[flip ? 2 : 1];
        triOffset++;
        break;
    case 0x04:
        flip = !flip;
    case 0x0B:
        tri[0] = interpolate(pts, cv, val, tets[tetIdx][2], tets[tetIdx][0]);
        tri[flip ? 2 : 1] = interpolate(pts, cv, val, tets[tetIdx][2], tets[tetIdx][1]);
        tri[flip ? 1 : 2] = interpolate(pts, cv, val, tets[tetIdx][2], tets[tetIdx][3]);
        triOffset++;
        break;
    case 0x05:
        flip = !flip;
    case 0x0A:
        // WARNING: per analogy = 3, subst 1 with 2
        // tetrahedron 1: around p2: p1->p2, p0->p1, p2->p3
        // tetrahedron 2: around p0: p0->p3, p2->p3, p0->p1
        // tetrahedron 3: around p2: p0, p2->p3, p0->p1
        tri[0] = interpolate(pts, cv, val, tets[tetIdx][0], tets[tetIdx][1]);
        tri[flip ? 2 : 1] = interpolate(pts, cv, val, tets[tetIdx][2], tets[tetIdx][3]);
        tri[flip ? 1 : 2] = interpolate(pts, cv, val, tets[tetIdx][0], tets[tetIdx][3]);
        triOffset++;
        tri2[0] = tri[0];
        tri2[flip ? 2 : 1] = interpolate(pts, cv, val, tets[tetIdx][1], tets[tetIdx][2]);
        tri2[flip ? 1 : 2] = tri[flip ? 2 : 1];
        triOffset++;
        break;
    case 0x06:
        flip = !flip;
    case 0x09:
        // WARNING: per analogy = 3, subst 0 with 2
        // tetrahedron 1: around p1: p1->p0, p0->p2, p1->p3
        // tetrahedron 2: around p2: p2->p3, p1->p3, p0->p2
        // tetrahedron 3: around p1: p2, p1->p3, p0->p2
        tri[0] = interpolate(pts, cv, val, tets[tetIdx][0], tets[tetIdx][1]);
        tri[flip ? 2 : 1] = interpolate(pts, cv, val, tets[tetIdx][1], tets[tetIdx][3]);
        tri[flip ? 1 : 2] = interpolate(pts, cv, val, tets[tetIdx][2], tets[tetIdx][3]);
        triOffset++;
        tri2[0] = tri[0];
        tri2[flip ? 1 : 2] = interpolate(pts, cv, val, tets[tetIdx][0], tets[tetIdx][2]);
        tri2[flip ? 2 : 1] = tri[flip ? 1 : 2];
        triOffset++;
        break;
    case 0x08:
        flip = !flip;
    case 0x07:
        tri[0] = interpolate(pts, cv, val, tets[tetIdx][3], tets[tetIdx][0]);
        tri[flip ? 2 : 1] = interpolate(pts, cv, val, tets[tetIdx][3], tets[tetIdx][2]);
        tri[flip ? 1 : 2] = interpolate(pts, cv, val, tets[tetIdx][3], tets[tetIdx][1]);
        triOffset++;
        break;
    }

    if (triOffset == 0)
        return;
    // norm?
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            vrtWrtr.Write(tri[i][j]);
        }
    }
    norm = (tri[1] - tri[0]).Cross(tri[2] - tri[0]);
    norm.Normalise();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            nrlWrtr.Write(norm[j]);
        }
    }
    if (triOffset == 1)
        return;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            vrtWrtr.Write(tri2[i][j]);
        }
    }
    norm = (tri2[1] - tri2[0]).Cross(tri2[2] - tri2[0]);
    norm.Normalise();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            nrlWrtr.Write(norm[j]);
        }
    }
}
