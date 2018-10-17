/*
 * SombreroMeshRenderer.cpp
 * Copyright (C) 2006-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "SombreroMeshRenderer.h"
#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

#include "infovis/FlagCall.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/Vector.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/KeyCode.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/MemmappedFile.h"

#include <iterator>
#include <set>
#include <tuple>

using namespace megamol;
using namespace megamol::sombreros;

using namespace megamol::core;


/*
 * SombreroMeshRenderer::SombreroMeshRenderer
 */
SombreroMeshRenderer::SombreroMeshRenderer(void)
    : Renderer3DModule()
    , getDataSlot("getData", "The slot to fetch the tri-mesh data")
    , getVolDataSlot("getVolData", "The slot to fetch the volume data (experimental)")
    , getFlagDataSlot("getFlagData", "The slot to fetch the data from the flag storage")
    , showVertices("showVertices", "Flag whether to show the verices of the object")
    , lighting("lighting", "Flag whether or not use lighting for the surface")
    , surFrontStyle("frontstyle", "The rendering style for the front surface")
    , surBackStyle("backstyle", "The rendering style for the back surface")
    , windRule("windingrule", "The triangle edge winding rule")
    , colorSlot("color", "The triangle color (if not colors are read from file)")
    , brushColorSlot("brushColor", "The color for the brushing")
    , doScaleSlot("doScale", "Do Scaling of model data")
    , showSweatBandSlot("showSweatband", "Activates the display of the sweatband line") {

    this->getDataSlot.SetCompatibleCall<megamol::geocalls::CallTriMeshDataDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getFlagDataSlot.SetCompatibleCall<megamol::infovis::FlagCallDescription>();
    this->MakeSlotAvailable(&this->getFlagDataSlot);

    this->showVertices.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showVertices);

    this->lighting.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->lighting);

    param::EnumParam* ep = new param::EnumParam(0);
    ep->SetTypePair(0, "Filled");
    ep->SetTypePair(1, "Wireframe");
    ep->SetTypePair(2, "Points");
    ep->SetTypePair(3, "None");
    this->surFrontStyle << ep;
    this->MakeSlotAvailable(&this->surFrontStyle);

    ep = new param::EnumParam(3);
    ep->SetTypePair(0, "Filled");
    ep->SetTypePair(1, "Wireframe");
    ep->SetTypePair(2, "Points");
    ep->SetTypePair(3, "None");
    this->surBackStyle << ep;
    this->MakeSlotAvailable(&this->surBackStyle);

    ep = new param::EnumParam(0);
    ep->SetTypePair(0, "Counter-Clock Wise");
    ep->SetTypePair(1, "Clock Wise");
    this->windRule << ep;
    this->MakeSlotAvailable(&this->windRule);

    this->colorSlot.SetParameter(new param::StringParam("white"));
    this->MakeSlotAvailable(&this->colorSlot);

    this->brushColorSlot.SetParameter(new param::StringParam("red"));
    this->MakeSlotAvailable(&this->brushColorSlot);

    this->doScaleSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->doScaleSlot);

    this->showSweatBandSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showSweatBandSlot);

    this->lastTime = 0.0f;
}


/*
 * SombreroMeshRenderer::~SombreroMeshRenderer
 */
SombreroMeshRenderer::~SombreroMeshRenderer(void) { this->Release(); }


/*
 * SombreroMeshRenderer::create
 */
bool SombreroMeshRenderer::create(void) {
    // intentionally empty
    return true;
}


/*
 * SombreroMeshRenderer::GetCapabilities
 */
bool SombreroMeshRenderer::GetCapabilities(Call& call) {
    view::CallRender3D* cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER | view::CallRender3D::CAP_LIGHTING | view::CallRender3D::CAP_ANIMATION);

    return true;
}


/*
 * TriSoupRenderer::GetExtents
 */
bool SombreroMeshRenderer::GetExtents(Call& call) {
    view::CallRender3D* cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;
    megamol::geocalls::CallTriMeshData* ctmd = this->getDataSlot.CallAs<megamol::geocalls::CallTriMeshData>();
    if (ctmd == NULL) return false;
    ctmd->SetFrameID(static_cast<int>(cr->Time()));
    this->lastTime = cr->Time();
    if (!(*ctmd)(1)) return false;

    cr->SetTimeFramesCount(ctmd->FrameCount());
    cr->AccessBoundingBoxes().Clear();
    cr->AccessBoundingBoxes() = ctmd->AccessBoundingBoxes();
    if (this->doScaleSlot.Param<param::BoolParam>()->Value()) {
        float scale = ctmd->AccessBoundingBoxes().ClipBox().LongestEdge();
        if (scale > 0.0f) scale = 2.0f / scale;
        cr->AccessBoundingBoxes().MakeScaledWorld(scale);
    } else {
        cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);
    }

    return true;
}


/*
 * SombreroMeshRenderer::release
 */
void SombreroMeshRenderer::release(void) {
    // intentionally empty
}

/*
 * SombreroMeshRenderer::MouseEvent
 */
bool SombreroMeshRenderer::MouseEvent(float x, float y, megamol::core::view::MouseFlags flags) {
    bool consume = false;

    // vislib::sys::Log::DefaultLog.WriteInfo("%s %f %f", this->Name(), x, y);

    auto flagsc = this->getFlagDataSlot.CallAs<infovis::FlagCall>();
    if (flagsc == nullptr) {
        return false;
    }

    auto pixelDir = getPixelDirection(x, y);
    auto camPos = this->lastCamState.camPos;

    if ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) || (flags & view::MOUSEFLAG_BUTTON_RIGHT_DOWN)) {

        std::tuple<float, unsigned int, unsigned int, unsigned int> mark = std::make_tuple(FLT_MAX, 0, 0, 0);
        bool found = false;
        for (auto& tri : this->triangles) {
            vislib::math::Vector<float, 3> p1 = this->vertexPositions[tri.GetX()];
            vislib::math::Vector<float, 3> p2 = this->vertexPositions[tri.GetY()];
            vislib::math::Vector<float, 3> p3 = this->vertexPositions[tri.GetZ()];
            float dist = 0.0f;
            bool isect = this->rayTriIntersect(camPos, pixelDir, p1, p2, p3, dist);
            found |= isect;
            if (isect && dist < std::get<0>(mark)) {
                mark = std::make_tuple(
                    dist, this->indexAttrib[tri.GetX()], this->indexAttrib[tri.GetY()], this->indexAttrib[tri.GetZ()]);
            }
        }

        if (found) {
            vislib::sys::Log::DefaultLog.WriteWarn("Hit");
        }

        (*flagsc)(infovis::FlagCall::CallForGetFlags);
        if (flagsc->has_data()) {
            const auto& fl = flagsc->GetFlags();
            this->flagSet.insert(fl.begin(), fl.end());
        }

        if (flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) {
            if (found) {
                this->flagSet.insert(std::get<1>(mark));
                this->flagSet.insert(std::get<2>(mark));
                this->flagSet.insert(std::get<3>(mark));
            }
        } else {
            if (found) {
                this->flagSet.erase(std::get<1>(mark));
                this->flagSet.erase(std::get<2>(mark));
                this->flagSet.erase(std::get<3>(mark));
            }
        }
        std::shared_ptr<infovis::FlagStorage::FlagVectorType> v;
        v = std::make_shared<infovis::FlagStorage::FlagVectorType>();
        v->assign(this->vertexPositions.size(), infovis::FlagStorage::ENABLED);
        for (const auto& smem : this->flagSet) {
            v->operator[](smem) = infovis::FlagStorage::SELECTED;
        }

        flagsc->SetFlags(v);
        (*flagsc)(infovis::FlagCall::CallForSetFlags);

        consume = true;
    }

    return consume;
}

/*
 * SombreroMeshRenderer::rayTriIntersect
 */
bool SombreroMeshRenderer::rayTriIntersect(const vislib::math::Vector<float, 3>& pos,
    const vislib::math::Vector<float, 3>& dir, const vislib::math::Vector<float, 3>& p1,
    const vislib::math::Vector<float, 3>& p2, const vislib::math::Vector<float, 3>& p3, float& intersectDist) {

    const auto& e_1 = p2 - p1;
    const auto& e_2 = p3 - p1;

    auto& n = e_1.Cross(e_2);
    n.Normalise();
    const auto& q = dir.Cross(e_2);
    const float a = e_1.Dot(q);

    // back facing or parallel?
    if ((n.Dot(dir) >= 0.0f) || std::abs(a) <= 0.000001f) {
        return false;
    }

    const auto& s = (pos - p1) / a;
    const auto& r = s.Cross(e_1);

    vislib::math::Vector<float, 3> b;
    b[0] = s.Dot(q);
    b[1] = r.Dot(dir);
    b[2] = 1.0f - b[0] - b[1];

    if ((b[0] < 0.0f) || (b[1] < 0.0f) || (b[2] < 0.0f)) {
        return false;
    }
    intersectDist = e_2.Dot(r);

    return (intersectDist >= 0.0f);
}

/*
 * SombreroMeshRenderer::getPixelDirection
 */
vislib::math::Vector<float, 3> SombreroMeshRenderer::getPixelDirection(float x, float y) {
    vislib::math::Vector<float, 3> result(0.0f, 0.0f, 0.0f);
    if (this->lastCamState.camDir.Length() < 0.5f) return result;
    result = this->lastCamState.camDir;

    // TODO get direction correct

    float u = (x / static_cast<float>(this->lastCamState.width));
    float v = (y / static_cast<float>(this->lastCamState.height));
    float zNear = this->lastCamState.zNear;
    float zFar = this->lastCamState.zFar;
    float fovx = this->lastCamState.fovx;
    float fovy = this->lastCamState.fovy;

    auto& camUp = this->lastCamState.camUp;
    auto& camRight = this->lastCamState.camRight;
    auto& camDir = this->lastCamState.camDir;
    auto& camPos = this->lastCamState.camPos;

    auto oL = (tan(fovx * 0.5f) * zNear) * (-camRight) + (tan(fovy * 0.5f) * zNear) * camUp + camDir * zNear + camPos;
    auto uL =
        (tan(fovx * 0.5f) * zNear) * (-camRight) + (tan(fovy * 0.5f) * zNear) * (-camUp) + camDir * zNear + camPos;
    auto oR = (tan(fovx * 0.5f) * zNear) * camRight + (tan(fovy * 0.5f) * zNear) * camUp + camDir * zNear + camPos;
    auto uR = (tan(fovx * 0.5f) * zNear) * camRight + (tan(fovy * 0.5f) * zNear) * (-camUp) + camDir * zNear + camPos;

    auto targetL = (1.0f - v) * uL + v * oL;
    auto targetR = (1.0f - v) * uR + v * oR;

    auto target = (1.0f - u) * targetL + u * targetR;

    result = target - camPos;
    result.Normalise();

    return result;
}

/*
 * SombreroMeshRenderer::overrideColors
 */
void SombreroMeshRenderer::overrideColors(const vislib::math::Vector<float, 3>& color) {
    if (this->flagSet.empty()) return;
    for (size_t i = 0; i < this->indexAttrib.size(); i++) {
        if (this->flagSet.count(this->indexAttrib[i]) > 0) {
            this->newColors[3 * i + 0] = color[0];
            this->newColors[3 * i + 1] = color[1];
            this->newColors[3 * i + 2] = color[2];
        }
    }
}

/*
 * SombreroMeshRenderer::Render
 */
bool SombreroMeshRenderer::Render(Call& call) {
    view::CallRender3D* cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;
    megamol::geocalls::CallTriMeshData* ctmd = this->getDataSlot.CallAs<megamol::geocalls::CallTriMeshData>();
    if (ctmd == NULL) return false;

    ctmd->SetFrameID(static_cast<int>(cr->Time()));
    if (!(*ctmd)(1)) return false;
    if (this->doScaleSlot.Param<param::BoolParam>()->Value()) {
        float scale = ctmd->AccessBoundingBoxes().ClipBox().LongestEdge();
        if (scale > 0.0f) scale = 2.0f / scale;
        // float mat[16] = {
        //    scale, 0.0f, 0.0f, 0.0f,
        //    0.0f, scale, 0.0f, 0.0f,
        //    0.0f, 0.0f, scale, 0.0f,
        //    0.0f, 0.0f, 0.0f, 1.0f
        //};
        //::glMultMatrixf(mat);
        glPushMatrix();
        ::glScalef(scale, scale, scale);
    }

    this->lastTime = cr->Time();
    ctmd->SetFrameID(static_cast<int>(cr->Time()));
    if (!(*ctmd)(0)) return false;

    bool datadirty = false;
    if (this->lastDataHash != ctmd->DataHash()) {
        datadirty = true;
        this->lastDataHash = ctmd->DataHash();
    }

    // auto bb = ctmd->AccessBoundingBoxes().ObjectSpaceBBox();
    // printf("min: %f %f %f ; max: %f %f %f\n", bb.Left(), bb.Bottom(), bb.Back(), bb.Right(), bb.Top(), bb.Front());

    bool normals = false;
    bool colors = false;
    bool textures = false;
    ::glEnable(GL_DEPTH_TEST);
    bool doLighting = this->lighting.Param<param::BoolParam>()->Value();
    if (doLighting) {
        ::glEnable(GL_LIGHTING);
    } else {
        ::glDisable(GL_LIGHTING);
    }
    ::glDisable(GL_BLEND);
    ::glEnableClientState(GL_VERTEX_ARRAY);
    ::glDisableClientState(GL_NORMAL_ARRAY);
    ::glDisableClientState(GL_COLOR_ARRAY);
    ::glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    ::glEnable(GL_COLOR_MATERIAL);
    ::glEnable(GL_TEXTURE_2D);
    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    ::glEnable(GL_NORMALIZE);

    // GLsizei** oldbind = new GLsizei*[16];
    // for (int i = 0; i < 16; i++) {
    //    oldbind[i] = new GLsizei();
    //}
    ////glGetVertexAttribPointerv(0, GL_VERTEX_ARRAY_POINTER, (GLvoid**)oldbind);
    // glGetPointerv(GL_VERTEX_ARRAY_POINTER, (GLvoid**)oldbind);
    glVertexAttribPointer(0, 0, GL_FLOAT, GL_FALSE, 0, nullptr);
    // glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &oldbind);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    // glBindVertexArray(0);


    GLint cfm;
    ::glGetIntegerv(GL_CULL_FACE_MODE, &cfm);
    GLint pm[2];
    ::glGetIntegerv(GL_POLYGON_MODE, pm);
    GLint twr;
    ::glGetIntegerv(GL_FRONT_FACE, &twr);

    if (this->windRule.Param<param::EnumParam>()->Value() == 0) {
        ::glFrontFace(GL_CCW);
    } else {
        ::glFrontFace(GL_CW);
    }

    int fpm, bpm, cf = 0;

    switch (this->surFrontStyle.Param<param::EnumParam>()->Value()) {
    default:
        fpm = GL_FILL;
        break;
    case 1:
        fpm = GL_LINE;
        break;
    case 2:
        fpm = GL_POINT;
        break;
    case 3:
        fpm = GL_FILL;
        cf = GL_FRONT;
        break;
    }
    switch (this->surBackStyle.Param<param::EnumParam>()->Value()) {
    default:
        bpm = GL_FILL;
        break;
    case 1:
        bpm = GL_LINE;
        break;
    case 2:
        bpm = GL_POINT;
        break;
    case 3:
        bpm = GL_FILL;
        cf = (cf == 0) ? GL_BACK : GL_FRONT_AND_BACK;
        break;
    }
    ::glPolygonMode(GL_FRONT, fpm);
    ::glPolygonMode(GL_BACK, bpm);
    if (cf == 0) {
        ::glDisable(GL_CULL_FACE);
    } else {
        ::glEnable(GL_CULL_FACE);
        ::glCullFace(cf);
    }

    //::glColor3f(1.0f, 1.0f, 1.0f);
    float r, g, b;
    this->colorSlot.ResetDirty();
    utility::ColourParser::FromString(this->colorSlot.Param<param::StringParam>()->Value(), r, g, b);
    ::glColor3f(r, g, b);

    utility::ColourParser::FromString(this->brushColorSlot.Param<param::StringParam>()->Value(), r, g, b);
    vislib::math::Vector<float, 3> brushCol(r, g, b);

    // construct the lines for the sweatband
    std::vector<std::pair<uint, uint>> lines;
    if (ctmd->Count() >= 2 && this->showSweatBandSlot.Param<param::BoolParam>()->Value()) {
        std::set<uint> firstset, secondset;
        const megamol::geocalls::CallTriMeshData::Mesh& obj0 = ctmd->Objects()[0];
        const megamol::geocalls::CallTriMeshData::Mesh& obj1 = ctmd->Objects()[1];

        switch (obj0.GetTriDataType()) {
        case megamol::geocalls::CallTriMeshData::Mesh::DT_BYTE: {
            auto ptrb = obj0.GetTriIndexPointerByte();
            if (ptrb == nullptr) break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                firstset.insert(static_cast<uint>(ptrb[j * 3 + 0]));
                firstset.insert(static_cast<uint>(ptrb[j * 3 + 1]));
                firstset.insert(static_cast<uint>(ptrb[j * 3 + 2]));
            }
        } break;
        case megamol::geocalls::CallTriMeshData::Mesh::DT_UINT16: {
            auto ptr16 = obj0.GetTriIndexPointerUInt16();
            if (ptr16 == nullptr) break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                firstset.insert(static_cast<uint>(ptr16[j * 3 + 0]));
                firstset.insert(static_cast<uint>(ptr16[j * 3 + 1]));
                firstset.insert(static_cast<uint>(ptr16[j * 3 + 2]));
            }
        } break;
        case megamol::geocalls::CallTriMeshData::Mesh::DT_UINT32: {
            auto ptr32 = obj0.GetTriIndexPointerUInt32();
            if (ptr32 == nullptr) break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                firstset.insert(static_cast<uint>(ptr32[j * 3 + 0]));
                firstset.insert(static_cast<uint>(ptr32[j * 3 + 1]));
                firstset.insert(static_cast<uint>(ptr32[j * 3 + 2]));
            }
        } break;
        default:
            break;
        }

        switch (obj1.GetTriDataType()) {
        case megamol::geocalls::CallTriMeshData::Mesh::DT_BYTE: {
            auto ptrb = obj1.GetTriIndexPointerByte();
            if (ptrb == nullptr) break;
            for (size_t j = 0; j < obj1.GetTriCount(); j++) {
                secondset.insert(static_cast<uint>(ptrb[j * 3 + 0]));
                secondset.insert(static_cast<uint>(ptrb[j * 3 + 1]));
                secondset.insert(static_cast<uint>(ptrb[j * 3 + 2]));
            }
        } break;
        case megamol::geocalls::CallTriMeshData::Mesh::DT_UINT16: {
            auto ptr16 = obj1.GetTriIndexPointerUInt16();
            if (ptr16 == nullptr) break;
            for (size_t j = 0; j < obj1.GetTriCount(); j++) {
                secondset.insert(static_cast<uint>(ptr16[j * 3 + 0]));
                secondset.insert(static_cast<uint>(ptr16[j * 3 + 1]));
                secondset.insert(static_cast<uint>(ptr16[j * 3 + 2]));
            }
        } break;
        case megamol::geocalls::CallTriMeshData::Mesh::DT_UINT32: {
            auto ptr32 = obj1.GetTriIndexPointerUInt32();
            if (ptr32 == nullptr) break;
            for (size_t j = 0; j < obj1.GetTriCount(); j++) {
                secondset.insert(static_cast<uint>(ptr32[j * 3 + 0]));
                secondset.insert(static_cast<uint>(ptr32[j * 3 + 1]));
                secondset.insert(static_cast<uint>(ptr32[j * 3 + 2]));
            }
        } break;
        default:
            break;
        }

        std::vector<uint> resvec;
        std::set_intersection(
            firstset.begin(), firstset.end(), secondset.begin(), secondset.end(), std::back_inserter(resvec));
        std::set<uint> resset(resvec.begin(), resvec.end());

        switch (obj0.GetTriDataType()) {
        case megamol::geocalls::CallTriMeshData::Mesh::DT_BYTE: {
            auto ptrb = obj0.GetTriIndexPointerByte();
            if (ptrb == nullptr) break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                uint v0 = static_cast<uint>(ptrb[j * 3 + 0]);
                uint v1 = static_cast<uint>(ptrb[j * 3 + 1]);
                uint v2 = static_cast<uint>(ptrb[j * 3 + 2]);
                std::pair<uint, uint> resline;
                if (resset.count(v0) > 0 && resset.count(v1) > 0) {
                    if (v0 > v1) std::swap(v0, v1);
                    resline = std::make_pair(v0, v1);
                    lines.push_back(resline);
                }
                if (resset.count(v1) > 0 && resset.count(v2) > 0) {
                    if (v1 > v2) std::swap(v1, v2);
                    resline = std::make_pair(v1, v2);
                    lines.push_back(resline);
                }
                if (resset.count(v0) > 0 && resset.count(v2) > 0) {
                    if (v0 > v2) std::swap(v0, v2);
                    resline = std::make_pair(v0, v2);
                    lines.push_back(resline);
                }
            }
        } break;
        case megamol::geocalls::CallTriMeshData::Mesh::DT_UINT16: {
            auto ptr16 = obj0.GetTriIndexPointerUInt16();
            if (ptr16 == nullptr) break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                uint v0 = static_cast<uint>(ptr16[j * 3 + 0]);
                uint v1 = static_cast<uint>(ptr16[j * 3 + 1]);
                uint v2 = static_cast<uint>(ptr16[j * 3 + 2]);
                std::pair<uint, uint> resline;
                if (resset.count(v0) > 0 && resset.count(v1) > 0) {
                    if (v0 > v1) std::swap(v0, v1);
                    resline = std::make_pair(v0, v1);
                    lines.push_back(resline);
                }
                if (resset.count(v1) > 0 && resset.count(v2) > 0) {
                    if (v1 > v2) std::swap(v1, v2);
                    resline = std::make_pair(v1, v2);
                    lines.push_back(resline);
                }
                if (resset.count(v0) > 0 && resset.count(v2) > 0) {
                    if (v0 > v2) std::swap(v0, v2);
                    resline = std::make_pair(v0, v2);
                    lines.push_back(resline);
                }
            }
        } break;
        case megamol::geocalls::CallTriMeshData::Mesh::DT_UINT32: {
            auto ptr32 = obj0.GetTriIndexPointerUInt32();
            if (ptr32 == nullptr) break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                uint v0 = static_cast<uint>(ptr32[j * 3 + 0]);
                uint v1 = static_cast<uint>(ptr32[j * 3 + 1]);
                uint v2 = static_cast<uint>(ptr32[j * 3 + 2]);
                std::pair<uint, uint> resline;
                if (resset.count(v0) > 0 && resset.count(v1) > 0) {
                    if (v0 > v1) std::swap(v0, v1);
                    resline = std::make_pair(v0, v1);
                    lines.push_back(resline);
                }
                if (resset.count(v1) > 0 && resset.count(v2) > 0) {
                    if (v1 > v2) std::swap(v1, v2);
                    resline = std::make_pair(v1, v2);
                    lines.push_back(resline);
                }
                if (resset.count(v0) > 0 && resset.count(v2) > 0) {
                    if (v0 > v2) std::swap(v0, v2);
                    resline = std::make_pair(v0, v2);
                    lines.push_back(resline);
                }
            }
        } break;
        default:
            break;
        }
    }

    std::vector<uint> linevec;
    for (auto& p : lines) {
        linevec.push_back(p.first);
        linevec.push_back(p.second);
    }

    for (unsigned int i = 0; i < ctmd->Count(); i++) {
        const megamol::geocalls::CallTriMeshData::Mesh& obj = ctmd->Objects()[i];

        switch (obj.GetVertexDataType()) {
        case megamol::geocalls::CallTriMeshData::Mesh::DT_FLOAT:
            ::glVertexPointer(3, GL_FLOAT, 0, obj.GetVertexPointerFloat());
            if (datadirty && i == 0) {
                this->vertexPositions.resize(obj.GetVertexCount());
                for (size_t j = 0; j < obj.GetVertexCount(); j++) {
                    this->vertexPositions[j] = vislib::math::Vector<float, 3>(&obj.GetVertexPointerFloat()[3 * j]);
                }
            }
            break;
        case megamol::geocalls::CallTriMeshData::Mesh::DT_DOUBLE:
            ::glVertexPointer(3, GL_DOUBLE, 0, obj.GetVertexPointerDouble());
            if (datadirty && i == 0) {
                this->vertexPositions.resize(obj.GetVertexCount());
                for (size_t j = 0; j < obj.GetVertexCount(); j++) {
                    this->vertexPositions[j] =
                        vislib::math::Vector<float, 3>(static_cast<float>(obj.GetVertexPointerDouble()[3 * j + 0]),
                            static_cast<float>(obj.GetVertexPointerDouble()[3 * j + 1]),
                            static_cast<float>(obj.GetVertexPointerDouble()[3 * j + 2]));
                }
            }
            break;
        default:
            continue;
        }

        if (obj.HasNormalPointer() != NULL) {
            if (!normals) {
                ::glEnableClientState(GL_NORMAL_ARRAY);
                normals = true;
            }
            switch (obj.GetNormalDataType()) {
            case megamol::geocalls::CallTriMeshData::Mesh::DT_FLOAT:
                ::glNormalPointer(GL_FLOAT, 0, obj.GetNormalPointerFloat());
                break;
            case megamol::geocalls::CallTriMeshData::Mesh::DT_DOUBLE:
                ::glNormalPointer(GL_DOUBLE, 0, obj.GetNormalPointerDouble());
                break;
            default:
                continue;
            }
        } else if (normals) {
            ::glDisableClientState(GL_NORMAL_ARRAY);
            normals = false;
        }

        if (obj.HasColourPointer() != NULL) {
            if (!colors) {
                ::glEnableClientState(GL_COLOR_ARRAY);
                colors = true;
            }
            switch (obj.GetColourDataType()) {
            case megamol::geocalls::CallTriMeshData::Mesh::DT_BYTE:
                if (i == 0 && !this->flagSet.empty()) {
                    this->newColors.resize(this->vertexPositions.size() * 3);
                    for (size_t j = 0; j < this->newColors.size(); j++) {
                        this->newColors[j] = static_cast<float>(obj.GetColourPointerByte()[j]) / 255.0f;
                    }
                }
                overrideColors(brushCol);
                if (flagSet.empty()) {
                    ::glColorPointer(3, GL_UNSIGNED_BYTE, 0, obj.GetColourPointerByte());
                } else {
                    ::glColorPointer(3, GL_FLOAT, 0, newColors.data());
                }
                break;
            case megamol::geocalls::CallTriMeshData::Mesh::DT_FLOAT:
                if (i == 0 && !this->flagSet.empty()) {
                    this->newColors.resize(this->vertexPositions.size() * 3);
                    for (size_t j = 0; j < this->newColors.size(); j++) {
                        this->newColors[j] = obj.GetColourPointerFloat()[j];
                    }
                }
                overrideColors(brushCol);
                if (flagSet.empty()) {
                    ::glColorPointer(3, GL_FLOAT, 0, obj.GetColourPointerFloat());
                } else {
                    ::glColorPointer(3, GL_FLOAT, 0, newColors.data());
                }
                break;
            case megamol::geocalls::CallTriMeshData::Mesh::DT_DOUBLE:
                if (i == 0 && !this->flagSet.empty()) {
                    this->newColors.resize(this->vertexPositions.size() * 3);
                    for (size_t j = 0; j < this->newColors.size(); j++) {
                        this->newColors[j] = static_cast<float>(obj.GetColourPointerDouble()[j]);
                    }
                }
                overrideColors(brushCol);
                if (flagSet.empty()) {
                    ::glColorPointer(3, GL_DOUBLE, 0, obj.GetColourPointerDouble());
                } else {
                    ::glColorPointer(3, GL_FLOAT, 0, newColors.data());
                }

                break;
            default:
                continue;
            }
        } else if (colors) {
            ::glDisableClientState(GL_COLOR_ARRAY);
            colors = false;
        }

        if (obj.HasTextureCoordinatePointer() != NULL) {
            if (!textures) {
                ::glEnableClientState(GL_TEXTURE_COORD_ARRAY);
                textures = true;
            }
            switch (obj.GetTextureCoordinateDataType()) {
            case megamol::geocalls::CallTriMeshData::Mesh::DT_FLOAT:
                ::glTexCoordPointer(2, GL_FLOAT, 0, obj.GetTextureCoordinatePointerFloat());
                break;
            case megamol::geocalls::CallTriMeshData::Mesh::DT_DOUBLE:
                ::glTexCoordPointer(2, GL_DOUBLE, 0, obj.GetTextureCoordinatePointerDouble());
                break;
            default:
                continue;
            }
        } else if (textures) {
            ::glDisableClientState(GL_TEXTURE_COORD_ARRAY);
            textures = false;
        }

        if (obj.GetMaterial() != NULL) {
            const megamol::geocalls::CallTriMeshData::Material& mat = *obj.GetMaterial();

            if (doLighting) {
                ::glDisable(GL_COLOR_MATERIAL);
                GLfloat mat_ambient[4] = {mat.GetKa()[0], mat.GetKa()[1], mat.GetKa()[2], 1.0f};
                GLfloat mat_diffuse[4] = {mat.GetKd()[0], mat.GetKd()[1], mat.GetKd()[2], 1.0f};
                GLfloat mat_specular[4] = {mat.GetKs()[0], mat.GetKs()[1], mat.GetKs()[2], 1.0f};
                GLfloat mat_emission[4] = {mat.GetKe()[0], mat.GetKe()[1], mat.GetKe()[2], 1.0f};
                GLfloat mat_shininess[1] = {mat.GetNs()};
                ::glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
                ::glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
                ::glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
                ::glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission);
                ::glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
            } else {
                ::glColor3f(mat.GetKd()[0], mat.GetKd()[1], mat.GetKd()[2]);
            }

            GLuint mapid = mat.GetMapID();
            if (mapid > 0) {
                //::glActiveTexture(GL_TEXTURE0);
                ::glEnable(GL_COLOR_MATERIAL);
                ::glBindTexture(GL_TEXTURE_2D, mapid);
                ::glEnable(GL_TEXTURE_2D);
                ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            }
        } else {
            GLfloat mat_ambient[4] = {0.2f, 0.2f, 0.2f, 1.0f};
            GLfloat mat_diffuse[4] = {0.8f, 0.8f, 0.8f, 1.0f};
            GLfloat mat_specular[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            GLfloat mat_emission[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            GLfloat mat_shininess[1] = {0.0f};
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
            ::glBindTexture(GL_TEXTURE_2D, 0);
            ::glEnable(GL_COLOR_MATERIAL);
        }

        if (obj.HasTriIndexPointer() != NULL) {
            switch (obj.GetTriDataType()) {
            case megamol::geocalls::CallTriMeshData::Mesh::DT_BYTE:
                ::glDrawElements(GL_TRIANGLES, obj.GetTriCount() * 3, GL_UNSIGNED_BYTE, obj.GetTriIndexPointerByte());
                if (datadirty && i == 0) {
                    this->triangles.resize(obj.GetTriCount());
                    for (size_t j = 0; j < obj.GetTriCount(); j++) {
                        this->triangles[j] = vislib::math::Vector<unsigned int, 3>(
                            static_cast<unsigned int>(obj.GetTriIndexPointerByte()[3 * j + 0]),
                            static_cast<unsigned int>(obj.GetTriIndexPointerByte()[3 * j + 1]),
                            static_cast<unsigned int>(obj.GetTriIndexPointerByte()[3 * j + 2]));
                    }
                }
                break;
            case megamol::geocalls::CallTriMeshData::Mesh::DT_UINT16:
                ::glDrawElements(
                    GL_TRIANGLES, obj.GetTriCount() * 3, GL_UNSIGNED_SHORT, obj.GetTriIndexPointerUInt16());
                if (datadirty && i == 0) {
                    this->triangles.resize(obj.GetTriCount());
                    for (size_t j = 0; j < obj.GetTriCount(); j++) {
                        this->triangles[j] = vislib::math::Vector<unsigned int, 3>(
                            static_cast<unsigned int>(obj.GetTriIndexPointerUInt16()[3 * j + 0]),
                            static_cast<unsigned int>(obj.GetTriIndexPointerUInt16()[3 * j + 1]),
                            static_cast<unsigned int>(obj.GetTriIndexPointerUInt16()[3 * j + 2]));
                    }
                }
                break;
            case megamol::geocalls::CallTriMeshData::Mesh::DT_UINT32:
                ::glDrawElements(GL_TRIANGLES, obj.GetTriCount() * 3, GL_UNSIGNED_INT, obj.GetTriIndexPointerUInt32());
                if (datadirty && i == 0) {
                    this->triangles.resize(obj.GetTriCount());
                    for (size_t j = 0; j < obj.GetTriCount(); j++) {
                        this->triangles[j] = vislib::math::Vector<unsigned int, 3>(
                            static_cast<unsigned int>(obj.GetTriIndexPointerUInt32()[3 * j + 0]),
                            static_cast<unsigned int>(obj.GetTriIndexPointerUInt32()[3 * j + 1]),
                            static_cast<unsigned int>(obj.GetTriIndexPointerUInt32()[3 * j + 2]));
                    }
                }
                break;
            default:
                continue;
            }
        } else {
            ::glDrawArrays(GL_TRIANGLES, 0, obj.GetVertexCount());
        }

        if (!doLighting) {
            ::glColor3f(r, g, b);
        }

        if (datadirty && i == 0 && obj.GetVertexAttribCount() > 0 &&
            obj.GetVertexAttribDataType(0) == megamol::geocalls::CallTriMeshData::Mesh::DT_UINT32) {
            auto dt = obj.GetVertexAttribDataType(0);
            this->indexAttrib.resize(obj.GetVertexCount());
            auto ptr = obj.GetVertexAttribPointerUInt32(0);
            for (size_t j = 0; j < obj.GetVertexCount(); j++) {
                this->indexAttrib[j] = ptr[j];
            }
        }
    }

    if (normals) ::glDisableClientState(GL_NORMAL_ARRAY);
    if (colors) ::glDisableClientState(GL_COLOR_ARRAY);
    if (textures) ::glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    {
        GLfloat mat_ambient[4] = {0.2f, 0.2f, 0.2f, 1.0f};
        GLfloat mat_diffuse[4] = {0.8f, 0.8f, 0.8f, 1.0f};
        GLfloat mat_specular[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        GLfloat mat_emission[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        GLfloat mat_shininess[1] = {0.0f};
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
        ::glBindTexture(GL_TEXTURE_2D, 0);
    }

    ::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (this->showVertices.Param<param::BoolParam>()->Value()) {
        //::glEnable(GL_POINT_SIZE);
        ::glPointSize(3.0f);
        ::glDisable(GL_LIGHTING);

        ::glColor3f(1.0f, 0.0f, 0.0f);
        for (unsigned int i = 0; i < ctmd->Count(); i++) {
            switch (ctmd->Objects()[i].GetVertexDataType()) {
            case megamol::geocalls::CallTriMeshData::Mesh::DT_FLOAT:
                ::glVertexPointer(3, GL_FLOAT, 0, ctmd->Objects()[i].GetVertexPointerFloat());
                break;
            case megamol::geocalls::CallTriMeshData::Mesh::DT_DOUBLE:
                ::glVertexPointer(3, GL_DOUBLE, 0, ctmd->Objects()[i].GetVertexPointerDouble());
                break;
            default:
                continue;
            }
            ::glDrawArrays(GL_POINTS, 0, ctmd->Objects()[i].GetVertexCount());
        }

        //::glEnable(GL_POINT_SIZE);
        ::glPointSize(1.0f);
    }

    if (this->showSweatBandSlot.Param<param::BoolParam>()->Value() && lines.size() > 0) {
        ::glLineWidth(3.0f);
        ::glDisable(GL_LIGHTING);
        ::glDisable(GL_DEPTH_TEST);

        ::glColor3f(1.0f, 0.0f, 0.0f);
        switch (ctmd->Objects()[0].GetVertexDataType()) {
        case megamol::geocalls::CallTriMeshData::Mesh::DT_FLOAT:
            ::glVertexPointer(3, GL_FLOAT, 0, ctmd->Objects()[0].GetVertexPointerFloat());
            break;
        case megamol::geocalls::CallTriMeshData::Mesh::DT_DOUBLE:
            ::glVertexPointer(3, GL_DOUBLE, 0, ctmd->Objects()[0].GetVertexPointerDouble());
            break;
        default:
            break;
        }
        ::glDrawElements(GL_LINES, static_cast<GLsizei>(linevec.size()), GL_UNSIGNED_INT, &linevec[0]);
        ::glEnable(GL_DEPTH_TEST);
    }

    // copy index attributes


    // query the current state of the camera (needed for picking)
    GLfloat m[16];
    GLfloat m_proj[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, m);
    glGetFloatv(GL_PROJECTION_MATRIX, m_proj);
    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> modelMatrix(&m[0]);
    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> projectionMatrix(&m_proj[0]);
    // TODO invert?

    auto cam = cr->GetCameraParameters();
    auto viewport = cr->GetViewport().GetSize();

    // the camera position from the matrix seems to be wrong
    /*this->lastCamState.camPos =
        vislib::math::Vector<float, 3>(modelMatrix.GetAt(0, 3), modelMatrix.GetAt(1, 3), modelMatrix.GetAt(2, 3));*/
    this->lastCamState.camPos = cam->Position();
    this->lastCamState.camDir =
        vislib::math::Vector<float, 3>(modelMatrix.GetAt(0, 2), modelMatrix.GetAt(1, 2), -modelMatrix.GetAt(2, 2));
    this->lastCamState.camUp =
        vislib::math::Vector<float, 3>(modelMatrix.GetAt(0, 1), modelMatrix.GetAt(1, 1), modelMatrix.GetAt(2, 1));
    this->lastCamState.camRight =
        vislib::math::Vector<float, 3>(modelMatrix.GetAt(0, 0), modelMatrix.GetAt(1, 0), modelMatrix.GetAt(2, 0));
    this->lastCamState.camDir.Normalise();
    this->lastCamState.camUp.Normalise();
    this->lastCamState.camRight.Normalise();
    this->lastCamState.zNear = cam->NearClip();
    this->lastCamState.zFar = cam->FarClip();
    this->lastCamState.fovy = (float)(cam->ApertureAngle() * M_PI / 180.0f);
    this->lastCamState.aspect = (float)viewport.GetWidth() / (float)viewport.GetHeight();
    if (viewport.GetHeight() == 0) {
        this->lastCamState.aspect = 0.0f;
    }
    this->lastCamState.fovx = 2.0f * atan(tan(this->lastCamState.fovy / 2.0f) * this->lastCamState.aspect);
    this->lastCamState.width = viewport.GetWidth();
    this->lastCamState.height = viewport.GetHeight();

    vislib::sys::Log::DefaultLog.WriteInfo(
        "dir: %f %f %f", lastCamState.camDir.GetX(), lastCamState.camDir.GetY(), lastCamState.camDir.GetZ());
    /*vislib::sys::Log::DefaultLog.WriteInfo(
        "up: %f %f %f", lastCamState.camUp.GetX(), lastCamState.camUp.GetY(), lastCamState.camUp.GetZ());*/
    /*vislib::sys::Log::DefaultLog.WriteInfo(
        "pos: %f %f %f", lastCamState.camPos.GetX(), lastCamState.camPos.GetY(), lastCamState.camPos.GetZ());*/
    /*vislib::sys::Log::DefaultLog.WriteInfo(
        "right: %f %f %f", lastCamState.camRight.GetX(), lastCamState.camRight.GetY(), lastCamState.camRight.GetZ());*/

    // vislib::sys::Log::DefaultLog.WriteInfo("-------------------------");

    ::glCullFace(cfm);
    ::glFrontFace(twr);
    ::glPolygonMode(GL_FRONT, pm[0]);
    ::glPolygonMode(GL_BACK, pm[1]);

    ::glEnable(GL_CULL_FACE);
    ::glDisableClientState(GL_VERTEX_ARRAY);
    //::glDisable(GL_POINT_SIZE);
    ::glEnable(GL_BLEND);

#if (defined(_MSC_VER) && (_MSC_VER > 1000))
    ::GetLastError();
#endif
    ::glCullFace(cfm);
    ::glFrontFace(twr);
    ::glDisableClientState(GL_VERTEX_ARRAY);

    return true;
}
