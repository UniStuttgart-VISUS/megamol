/*
 * IsoSurface.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "volumetrics/IsoSurface.h"

#include <cfloat>
#include <climits>
#include <cmath>

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/math/Plane.h"
#include "vislib/math/Vector.h"
#include "volumetrics/MarchingCubeTables.h"

using namespace megamol;
using namespace megamol::trisoup;
using namespace megamol::trisoup::volumetrics;


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
        , isoValueSlot("isoval", "The iso value")
        , dataHash(0)
        , frameIdx(0)
        , index()
        , vertex()
        , normal()
        , mesh() {

    this->inDataSlot.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->outDataSlot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &IsoSurface::outDataCallback);
    this->outDataSlot.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &IsoSurface::outExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

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
    auto tmd = dynamic_cast<mesh::CallMesh*>(&caller);
    if (tmd == NULL)
        return false;

    auto cvd = this->inDataSlot.CallAs<core::misc::VolumetricDataCall>();
    if (cvd != NULL) {

        bool recalc = false;

        if (this->isoValueSlot.IsDirty()) {
            this->isoValueSlot.ResetDirty();
            recalc = true;
        }

        auto mesh_meta = tmd->getMetaData();
        cvd->SetFrameID(mesh_meta.m_frame_ID);
        if (!(*cvd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS) ||
            !(*cvd)(core::misc::VolumetricDataCall::IDX_GET_METADATA) ||
            !(*cvd)(core::misc::VolumetricDataCall::IDX_GET_DATA)) {
            recalc = false;
        } else {
            if ((this->dataHash != cvd->DataHash()) || (this->frameIdx != cvd->FrameID())) {
                recalc = true;
            }
        }

        /*unsigned int attrIdx = UINT_MAX;
        if (recalc) {
            vislib::StringA attrName(this->attributeSlot.Param<core::param::StringParam>()->Value());
            attrIdx = cvd->FindAttribute(attrName);
            if (attrIdx == UINT_MAX) {
                try {
                    attrIdx = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(attrName));
                } catch (...) { attrIdx = UINT_MAX; }
            }
            if (attrIdx >= cvd->AttributeCount()) {
                recalc = false;
            } else if (cvd->Attribute(attrIdx).Type() != core::CallVolumeData::TYPE_FLOAT) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("Only float volumes are supported ATM");
                recalc = false;
            }
        }*/

        if (recalc) {
            float isoVal = this->isoValueSlot.Param<core::param::FloatParam>()->Value();

            auto const metadata = cvd->GetMetadata();

            if (metadata->ScalarType != core::misc::FLOATING_POINT) {
                core::utility::log::Log::DefaultLog.WriteError("[IsoSurface] Only floating point data allowed");
                return false;
            }

            auto const data = reinterpret_cast<float const*>(cvd->GetData());

            auto const x_res = metadata->Resolution[0];
            auto const y_res = metadata->Resolution[1];
            auto const z_res = metadata->Resolution[2];

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
                n, isoVal, data, x_res, y_res, z_res);

            this->index.EnforceSize(i.End(), true);
            this->vertex.EnforceSize(v.End(), true);
#ifdef WITH_COLOUR_DATA
            this->colour.EnforceSize(c.End(), true);
#endif /* WITH_COLOUR_DATA */
            this->normal.EnforceSize(n.End(), true);

            mesh = std::make_shared<mesh::MeshDataAccessCollection>();
            mesh::MeshDataAccessCollection::IndexData index_data;
            index_data.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
            index_data.byte_size = index.GetSize();
            index_data.data = index.AsAt<uint8_t>(0);

            std::vector<mesh::MeshDataAccessCollection::VertexAttribute> attributes;
            attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{vertex.AsAt<uint8_t>(0),
                vertex.GetSize(), 3, mesh::MeshDataAccessCollection::ValueType::FLOAT, 3 * sizeof(float), 0,
                mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION});
            attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{normal.AsAt<uint8_t>(0),
                normal.GetSize(), 3, mesh::MeshDataAccessCollection::ValueType::FLOAT, 3 * sizeof(float), 0,
                mesh::MeshDataAccessCollection::AttributeSemanticType::NORMAL});

            mesh->addMesh("isosurface", attributes, index_data);

            this->dataHash = cvd->DataHash();
            this->frameIdx = cvd->FrameID();
        }
    }

    tmd->setData(mesh, dataHash);
    auto mesh_meta = tmd->getMetaData();
    mesh_meta.m_frame_ID = frameIdx;
    tmd->setMetaData(mesh_meta);

    return true;
}


/*
 * IsoSurface::outExtentCallback
 */
bool IsoSurface::outExtentCallback(megamol::core::Call& caller) {
    auto tmd = dynamic_cast<mesh::CallMesh*>(&caller);
    if (tmd == NULL)
        return false;

    auto mesh_meta = tmd->getMetaData();
    auto cvd = this->inDataSlot.CallAs<core::misc::VolumetricDataCall>();
    cvd->SetFrameID(mesh_meta.m_frame_ID);
    if ((cvd == NULL) || (!(*cvd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS))) {
        // no input data
        mesh_meta.m_frame_cnt = 1;
        tmd->setMetaData(mesh_meta);

    } else {
        // input data in cvd
        mesh_meta.m_frame_cnt = cvd->FrameCount();
        mesh_meta.m_bboxs.SetBoundingBox(cvd->AccessBoundingBoxes().ObjectSpaceBBox());
        mesh_meta.m_bboxs.SetClipBox(cvd->AccessBoundingBoxes().ClipBox());
        tmd->setMetaData(mesh_meta);
        this->osbb = cvd->AccessBoundingBoxes().ObjectSpaceBBox();
    }

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

    unsigned int baseIdx = 0;

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
                    cubeValues[j] = vol[(x + MarchingCubeTables::a2fVertexOffset[j][0]) +
                                        sx * ((y + MarchingCubeTables::a2fVertexOffset[j][1]) +
                                                 sy * (z + MarchingCubeTables::a2fVertexOffset[j][2]))];
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
                        pts[j].Set(p.X() + static_cast<float>(MarchingCubeTables::a2fVertexOffset[j][0]) * cellSizeX,
                            p.Y() + static_cast<float>(MarchingCubeTables::a2fVertexOffset[j][1]) * cellSizeY,
                            p.Z() + static_cast<float>(MarchingCubeTables::a2fVertexOffset[j][2]) * cellSizeZ);
                    }
                    this->makeTet(triIdx, tetIdx, pts, cubeValues, val, i, v, n, baseIdx);
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
    float x = b * MarchingCubeTables::a2fVertexOffset[idx0][0] + a * MarchingCubeTables::a2fVertexOffset[idx1][0];
    float y = b * MarchingCubeTables::a2fVertexOffset[idx0][1] + a * MarchingCubeTables::a2fVertexOffset[idx1][1];
    float z = b * MarchingCubeTables::a2fVertexOffset[idx0][2] + a * MarchingCubeTables::a2fVertexOffset[idx1][2];
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
    vislib::RawStorageWriter& nrlWrtr, unsigned int& baseIdx) {
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
        idxWrtr.Write(baseIdx++);
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
        idxWrtr.Write(baseIdx++);
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
    float val, vislib::RawStorageWriter& idxWrtr, vislib::RawStorageWriter& vrtWrtr, vislib::RawStorageWriter& nrlWrtr,
    unsigned int& baseIdx) {
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
        idxWrtr.Write(baseIdx++);
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
        idxWrtr.Write(baseIdx++);
    }
    norm = (tri2[1] - tri2[0]).Cross(tri2[2] - tri2[0]);
    norm.Normalise();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            nrlWrtr.Write(norm[j]);
        }
    }
}
