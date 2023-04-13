/*
 * SombreroMeshRenderer.cpp
 * Copyright (C) 2006-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */


#include "SombreroMeshRenderer.h"
#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

//#include "mmcore/FlagCall.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/Vector.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/KeyCode.h"
#include "vislib/sys/MemmappedFile.h"

#include <array>
#include <climits>
#include <iterator>
#include <set>
#include <tuple>

using namespace megamol;
using namespace megamol::protein_gl;

using namespace megamol::core;


/*
 * SombreroMeshRenderer::SombreroMeshRenderer
 */
SombreroMeshRenderer::SombreroMeshRenderer()
        : mmstd_gl::Renderer3DModuleGL()
        , getDataSlot("getData", "The slot to fetch the tri-mesh data")
        , getVolDataSlot("getVolData", "The slot to fetch the volume data (experimental)")
        , getFlagDataSlot("getFlagData", "The slot to fetch the data from the flag storage")
        , showVertices("showVertices", "Flag whether to show the verices of the object")
        , lighting("lighting", "Flag whether or not use lighting for the surface")
        , surFrontStyle("frontstyle", "The rendering style for the front surface")
        , surBackStyle("backstyle", "The rendering style for the back surface")
        , windRule("windingrule", "The triangle edge winding rule")
        , colorSlot("colors::color", "The triangle color (if not colors are read from file)")
        , brushColorSlot("colors::brushColor", "The color for the brushing")
        , innerColorSlot("colors::innerColor", "The color of the inner radius line")
        , outerColorSlot("colors::outerColor", "The color of the outer radius line")
        , borderColorSlot("colors::sweatbandColor", "The color of the sweatband line")
        , fontColorSlot("colors::fontColor", "The color of the font")
        , showRadiiSlot("showRadii", "Enable the textual annotation of the radii")
        , showSweatBandSlot("showSweatband", "Activates the display of the sweatband line")
        , theFont(megamol::core::utility::SDFFont::PRESET_ROBOTO_SANS) {

    this->getDataSlot.SetCompatibleCall<megamol::geocalls_gl::CallTriMeshDataGLDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

#if 0
    this->getFlagDataSlot.SetCompatibleCall<megamol::core::FlagCallDescription>();
    this->MakeSlotAvailable(&this->getFlagDataSlot);
#endif

    this->showVertices.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showVertices);

    this->showRadiiSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showRadiiSlot);

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

    this->borderColorSlot.SetParameter(new param::StringParam("red"));
    this->MakeSlotAvailable(&this->borderColorSlot);

    this->fontColorSlot.SetParameter(new param::StringParam("black"));
    this->MakeSlotAvailable(&this->fontColorSlot);

    this->innerColorSlot.SetParameter(new param::StringParam("red"));
    this->MakeSlotAvailable(&this->innerColorSlot);

    this->outerColorSlot.SetParameter(new param::StringParam("red"));
    this->MakeSlotAvailable(&this->outerColorSlot);

    this->showSweatBandSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showSweatBandSlot);

    this->lastTime = 0.0f;
    this->flagVersion = 0;
}


/*
 * SombreroMeshRenderer::~SombreroMeshRenderer
 */
SombreroMeshRenderer::~SombreroMeshRenderer() {
    this->Release();
}


/*
 * SombreroMeshRenderer::create
 */
bool SombreroMeshRenderer::create() {
    // intentionally empty
    return true;
}


/*
 * TriSoupRenderer::GetExtents
 */
bool SombreroMeshRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    mmstd_gl::CallRender3DGL* cr = dynamic_cast<mmstd_gl::CallRender3DGL*>(&call);
    if (cr == NULL)
        return false;
    megamol::geocalls_gl::CallTriMeshDataGL* ctmd = this->getDataSlot.CallAs<megamol::geocalls_gl::CallTriMeshDataGL>();
    if (ctmd == NULL)
        return false;
    ctmd->SetFrameID(static_cast<int>(cr->Time()));
    this->lastTime = cr->Time();
    if (!(*ctmd)(1))
        return false;

    cr->SetTimeFramesCount(ctmd->FrameCount());
    cr->AccessBoundingBoxes().Clear();
    cr->AccessBoundingBoxes() = ctmd->AccessBoundingBoxes();

    return true;
}


/*
 * SombreroMeshRenderer::release
 */
void SombreroMeshRenderer::release() {
    // intentionally empty
}

#if 0
/*
 * SombreroMeshRenderer::MouseEvent
 */
bool SombreroMeshRenderer::MouseEvent(float x, float y, megamol::core::view::MouseFlags flags) {
    bool consume = false;

    // megamol::core::utility::log::Log::DefaultLog.WriteInfo("%s %f %f", this->Name(), x, y);

    auto flagsc = this->getFlagDataSlot.CallAs<core::FlagCall>();
    if (flagsc == nullptr) {
        return false;
    }

    auto pixelDir = getPixelDirection(x, y);
    auto camPos = this->lastCamState.camPos;

    if ((flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) || (flags & view::MOUSEFLAG_BUTTON_RIGHT_DOWN)) {
        std::tuple<float, unsigned int, unsigned int, unsigned int> mark = std::make_tuple(FLT_MAX, 0, 0, 0);
        bool found = false;
        for (size_t o = 0; o < this->triangles.size(); o++) {
            for (auto& tri : this->triangles[o]) {
                vislib::math::Vector<float, 3> p1 = this->vertexPositions[o][tri.GetX()];
                vislib::math::Vector<float, 3> p2 = this->vertexPositions[o][tri.GetY()];
                vislib::math::Vector<float, 3> p3 = this->vertexPositions[o][tri.GetZ()];
                float dist = 0.0f;
                bool isect = this->rayTriIntersect(camPos, pixelDir, p1, p2, p3, dist);
                found |= isect;
                if (isect && dist < std::get<0>(mark)) {
                    mark = std::make_tuple(dist, this->indexAttrib[o][tri.GetX()], this->indexAttrib[o][tri.GetY()],
                        this->indexAttrib[o][tri.GetZ()]);
                }
            }
        }

        // this stuff kinda rapes the flag storage because we use arbitrary flags.
        // but hey, it works ;-)
        (*flagsc)(core::FlagCall::CallMapFlags);
        if (this->flagVersion != flagsc->GetVersion()) {
            this->flagVersion = flagsc->GetVersion();
            const auto& fl = flagsc->GetFlags();
            this->flagSet.insert(fl->cbegin(), fl->cend());
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
        std::shared_ptr<core::FlagStorage::FlagVectorType> v;
        v = std::make_shared<core::FlagStorage::FlagVectorType>();
        v->resize(this->flagSet.size());
        v->assign(this->flagSet.begin(), this->flagSet.end());

        this->flagVersion++;
        flagsc->SetFlags(v, this->flagVersion);
        (*flagsc)(core::FlagCall::CallUnmapFlags);

        consume = true;
    }

    return consume;
}
#endif

/*
 * SombreroMeshRenderer::rayTriIntersect
 */
bool SombreroMeshRenderer::rayTriIntersect(const vislib::math::Vector<float, 3>& pos,
    const vislib::math::Vector<float, 3>& dir, const vislib::math::Vector<float, 3>& p1,
    const vislib::math::Vector<float, 3>& p2, const vislib::math::Vector<float, 3>& p3, float& intersectDist) {

    const vislib::math::Vector<float, 3> e_1 = p2 - p1;
    const vislib::math::Vector<float, 3> e_2 = p3 - p1;

    vislib::math::Vector<float, 3> n = e_1.Cross(e_2);
    n.Normalise();
    const auto& q = dir.Cross(e_2);
    const float a = e_1.Dot(q);

    // parallel?
    if (std::abs(a) <= 0.000001f) {
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
    if (this->lastCamState.camDir.Length() < 0.5f)
        return result;
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

    auto oL = static_cast<float>(tan(fovx * 0.5f) * zNear) * (-camRight) +
              static_cast<float>(tan(fovy * 0.5f) * zNear) * camUp + camDir * zNear + camPos;
    auto uL = static_cast<float>(tan(fovx * 0.5f) * zNear) * (-camRight) +
              static_cast<float>(tan(fovy * 0.5f) * zNear) * (-camUp) + camDir * zNear + camPos;
    auto oR = static_cast<float>(tan(fovx * 0.5f) * zNear) * camRight +
              static_cast<float>(tan(fovy * 0.5f) * zNear) * camUp + camDir * zNear + camPos;
    auto uR = static_cast<float>(tan(fovx * 0.5f) * zNear) * camRight +
              static_cast<float>(tan(fovy * 0.5f) * zNear) * (-camUp) + camDir * zNear + camPos;

    auto targetL = v * uL + (1.0f - v) * oL;
    auto targetR = v * uR + (1.0f - v) * oR;

    auto target = (1.0f - u) * targetL + u * targetR;

    result = target - camPos;
    result.Normalise();

    return result;
}

/*
 * SombreroMeshRenderer::overrideColors
 */
void SombreroMeshRenderer::overrideColors(const int meshIdx, const vislib::math::Vector<float, 3>& color) {
    if (this->flagSet.empty())
        return;
    for (size_t i = 0; i < this->indexAttrib[meshIdx].size(); i++) {
        if (this->flagSet.count(this->indexAttrib[meshIdx][i]) > 0) {
            this->newColors[meshIdx][3 * i + 0] = color[0];
            this->newColors[meshIdx][3 * i + 1] = color[1];
            this->newColors[meshIdx][3 * i + 2] = color[2];
        }
    }
}

/*
 * SombreroMeshRenderer::Render
 */
bool SombreroMeshRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    mmstd_gl::CallRender3DGL* cr = dynamic_cast<mmstd_gl::CallRender3DGL*>(&call);
    if (cr == NULL)
        return false;
    megamol::geocalls_gl::CallTriMeshDataGL* ctmd = this->getDataSlot.CallAs<megamol::geocalls_gl::CallTriMeshDataGL>();
    if (ctmd == NULL)
        return false;

    ctmd->SetFrameID(static_cast<int>(cr->Time()));
    if (!(*ctmd)(1))
        return false;

    this->lastTime = cr->Time();
    ctmd->SetFrameID(static_cast<int>(cr->Time()));
    if (!(*ctmd)(0))
        return false;

    bool datadirty = false;
    if (this->lastDataHash != ctmd->DataHash()) {
        datadirty = true;
        this->lastDataHash = ctmd->DataHash();
    }

#if 0
    auto flagsc = this->getFlagDataSlot.CallAs<core::FlagCall>();
    if (flagsc != nullptr) {
        (*flagsc)(core::FlagCall::CallMapFlags);
        if (flagVersion != flagsc->GetVersion()) {
            this->flagVersion = flagsc->GetVersion();
            const auto& fl = flagsc->GetFlags();
            this->flagSet.clear();
            this->flagSet.insert(fl->begin(), fl->end());
        }
        (*flagsc)(core::FlagCall::CallUnmapFlags);
    }
#endif

    auto bb = ctmd->AccessBoundingBoxes().ObjectSpaceBBox();
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
    utility::ColourParser::FromString(this->colorSlot.Param<param::StringParam>()->Value().c_str(), r, g, b);
    ::glColor3f(r, g, b);

    utility::ColourParser::FromString(this->brushColorSlot.Param<param::StringParam>()->Value().c_str(), r, g, b);
    vislib::math::Vector<float, 3> brushCol(r, g, b);

    // construct the lines for the sweatband
    std::vector<std::pair<uint32_t, uint32_t>> lines;
    if (ctmd->Count() >= 2) {
        std::set<uint32_t> firstset, secondset;
        const megamol::geocalls_gl::CallTriMeshDataGL::Mesh& obj0 = ctmd->Objects()[0];
        const megamol::geocalls_gl::CallTriMeshDataGL::Mesh& obj1 = ctmd->Objects()[1];

        switch (obj0.GetTriDataType()) {
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE: {
            auto ptrb = obj0.GetTriIndexPointerByte();
            if (ptrb == nullptr)
                break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                firstset.insert(static_cast<uint32_t>(ptrb[j * 3 + 0]));
                firstset.insert(static_cast<uint32_t>(ptrb[j * 3 + 1]));
                firstset.insert(static_cast<uint32_t>(ptrb[j * 3 + 2]));
            }
        } break;
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT16: {
            auto ptr16 = obj0.GetTriIndexPointerUInt16();
            if (ptr16 == nullptr)
                break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                firstset.insert(static_cast<uint32_t>(ptr16[j * 3 + 0]));
                firstset.insert(static_cast<uint32_t>(ptr16[j * 3 + 1]));
                firstset.insert(static_cast<uint32_t>(ptr16[j * 3 + 2]));
            }
        } break;
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32: {
            auto ptr32 = obj0.GetTriIndexPointerUInt32();
            if (ptr32 == nullptr)
                break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                firstset.insert(static_cast<uint32_t>(ptr32[j * 3 + 0]));
                firstset.insert(static_cast<uint32_t>(ptr32[j * 3 + 1]));
                firstset.insert(static_cast<uint32_t>(ptr32[j * 3 + 2]));
            }
        } break;
        default:
            break;
        }

        switch (obj1.GetTriDataType()) {
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE: {
            auto ptrb = obj1.GetTriIndexPointerByte();
            if (ptrb == nullptr)
                break;
            for (size_t j = 0; j < obj1.GetTriCount(); j++) {
                secondset.insert(static_cast<uint32_t>(ptrb[j * 3 + 0]));
                secondset.insert(static_cast<uint32_t>(ptrb[j * 3 + 1]));
                secondset.insert(static_cast<uint32_t>(ptrb[j * 3 + 2]));
            }
        } break;
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT16: {
            auto ptr16 = obj1.GetTriIndexPointerUInt16();
            if (ptr16 == nullptr)
                break;
            for (size_t j = 0; j < obj1.GetTriCount(); j++) {
                secondset.insert(static_cast<uint32_t>(ptr16[j * 3 + 0]));
                secondset.insert(static_cast<uint32_t>(ptr16[j * 3 + 1]));
                secondset.insert(static_cast<uint32_t>(ptr16[j * 3 + 2]));
            }
        } break;
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32: {
            auto ptr32 = obj1.GetTriIndexPointerUInt32();
            if (ptr32 == nullptr)
                break;
            for (size_t j = 0; j < obj1.GetTriCount(); j++) {
                secondset.insert(static_cast<uint32_t>(ptr32[j * 3 + 0]));
                secondset.insert(static_cast<uint32_t>(ptr32[j * 3 + 1]));
                secondset.insert(static_cast<uint32_t>(ptr32[j * 3 + 2]));
            }
        } break;
        default:
            break;
        }

        std::vector<uint32_t> resvec;
        std::set_intersection(
            firstset.begin(), firstset.end(), secondset.begin(), secondset.end(), std::back_inserter(resvec));
        std::set<uint32_t> resset(resvec.begin(), resvec.end());

        switch (obj0.GetTriDataType()) {
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE: {
            auto ptrb = obj0.GetTriIndexPointerByte();
            if (ptrb == nullptr)
                break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                uint32_t v0 = static_cast<uint32_t>(ptrb[j * 3 + 0]);
                uint32_t v1 = static_cast<uint32_t>(ptrb[j * 3 + 1]);
                uint32_t v2 = static_cast<uint32_t>(ptrb[j * 3 + 2]);
                std::pair<uint32_t, uint32_t> resline;
                if (resset.count(v0) > 0 && resset.count(v1) > 0) {
                    if (v0 > v1)
                        std::swap(v0, v1);
                    resline = std::make_pair(v0, v1);
                    lines.push_back(resline);
                }
                if (resset.count(v1) > 0 && resset.count(v2) > 0) {
                    if (v1 > v2)
                        std::swap(v1, v2);
                    resline = std::make_pair(v1, v2);
                    lines.push_back(resline);
                }
                if (resset.count(v0) > 0 && resset.count(v2) > 0) {
                    if (v0 > v2)
                        std::swap(v0, v2);
                    resline = std::make_pair(v0, v2);
                    lines.push_back(resline);
                }
            }
        } break;
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT16: {
            auto ptr16 = obj0.GetTriIndexPointerUInt16();
            if (ptr16 == nullptr)
                break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                uint32_t v0 = static_cast<uint32_t>(ptr16[j * 3 + 0]);
                uint32_t v1 = static_cast<uint32_t>(ptr16[j * 3 + 1]);
                uint32_t v2 = static_cast<uint32_t>(ptr16[j * 3 + 2]);
                std::pair<uint32_t, uint32_t> resline;
                if (resset.count(v0) > 0 && resset.count(v1) > 0) {
                    if (v0 > v1)
                        std::swap(v0, v1);
                    resline = std::make_pair(v0, v1);
                    lines.push_back(resline);
                }
                if (resset.count(v1) > 0 && resset.count(v2) > 0) {
                    if (v1 > v2)
                        std::swap(v1, v2);
                    resline = std::make_pair(v1, v2);
                    lines.push_back(resline);
                }
                if (resset.count(v0) > 0 && resset.count(v2) > 0) {
                    if (v0 > v2)
                        std::swap(v0, v2);
                    resline = std::make_pair(v0, v2);
                    lines.push_back(resline);
                }
            }
        } break;
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32: {
            auto ptr32 = obj0.GetTriIndexPointerUInt32();
            if (ptr32 == nullptr)
                break;
            for (size_t j = 0; j < obj0.GetTriCount(); j++) {
                uint32_t v0 = static_cast<uint32_t>(ptr32[j * 3 + 0]);
                uint32_t v1 = static_cast<uint32_t>(ptr32[j * 3 + 1]);
                uint32_t v2 = static_cast<uint32_t>(ptr32[j * 3 + 2]);
                std::pair<uint32_t, uint32_t> resline;
                if (resset.count(v0) > 0 && resset.count(v1) > 0) {
                    if (v0 > v1)
                        std::swap(v0, v1);
                    resline = std::make_pair(v0, v1);
                    lines.push_back(resline);
                }
                if (resset.count(v1) > 0 && resset.count(v2) > 0) {
                    if (v1 > v2)
                        std::swap(v1, v2);
                    resline = std::make_pair(v1, v2);
                    lines.push_back(resline);
                }
                if (resset.count(v0) > 0 && resset.count(v2) > 0) {
                    if (v0 > v2)
                        std::swap(v0, v2);
                    resline = std::make_pair(v0, v2);
                    lines.push_back(resline);
                }
            }
        } break;
        default:
            break;
        }
    }

    std::vector<uint32_t> linevec;
    for (auto& p : lines) {
        linevec.push_back(p.first);
        linevec.push_back(p.second);
    }

    if (this->vertexPositions.size() != ctmd->Count()) {
        this->vertexPositions.resize(ctmd->Count());
        this->triangles.resize(ctmd->Count());
        this->indexAttrib.resize(ctmd->Count());
        this->newColors.resize(ctmd->Count());
    }

    for (unsigned int i = 0; i < ctmd->Count(); i++) {
        const megamol::geocalls_gl::CallTriMeshDataGL::Mesh& obj = ctmd->Objects()[i];

        switch (obj.GetVertexDataType()) {
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
            ::glVertexPointer(3, GL_FLOAT, 0, obj.GetVertexPointerFloat());
            if (datadirty) {
                this->vertexPositions[i].resize(obj.GetVertexCount());
                for (size_t j = 0; j < obj.GetVertexCount(); j++) {
                    this->vertexPositions[i][j] = vislib::math::Vector<float, 3>(&obj.GetVertexPointerFloat()[3 * j]);
                }
            }
            break;
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
            ::glVertexPointer(3, GL_DOUBLE, 0, obj.GetVertexPointerDouble());
            if (datadirty) {
                this->vertexPositions[i].resize(obj.GetVertexCount());
                for (size_t j = 0; j < obj.GetVertexCount(); j++) {
                    this->vertexPositions[i][j] =
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
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                ::glNormalPointer(GL_FLOAT, 0, obj.GetNormalPointerFloat());
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
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
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE:
                if (!this->flagSet.empty()) {
                    this->newColors[i].resize(this->vertexPositions[i].size() * 3);
                    for (size_t j = 0; j < this->newColors[i].size(); j++) {
                        this->newColors[i][j] = static_cast<float>(obj.GetColourPointerByte()[j]) / 255.0f;
                    }
                }
                overrideColors(i, brushCol);
                if (flagSet.empty()) {
                    ::glColorPointer(3, GL_UNSIGNED_BYTE, 0, obj.GetColourPointerByte());
                } else {
                    ::glColorPointer(3, GL_FLOAT, 0, newColors[i].data());
                }
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                if (!this->flagSet.empty()) {
                    this->newColors[i].resize(this->vertexPositions[i].size() * 3);
                    for (size_t j = 0; j < this->newColors[i].size(); j++) {
                        this->newColors[i][j] = obj.GetColourPointerFloat()[j];
                    }
                }
                overrideColors(i, brushCol);
                if (flagSet.empty()) {
                    ::glColorPointer(3, GL_FLOAT, 0, obj.GetColourPointerFloat());
                } else {
                    ::glColorPointer(3, GL_FLOAT, 0, newColors[i].data());
                }
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
                if (!this->flagSet.empty()) {
                    this->newColors[i].resize(this->vertexPositions[i].size() * 3);
                    for (size_t j = 0; j < this->newColors[i].size(); j++) {
                        this->newColors[i][j] = static_cast<float>(obj.GetColourPointerDouble()[j]);
                    }
                }
                overrideColors(i, brushCol);
                if (flagSet.empty()) {
                    ::glColorPointer(3, GL_DOUBLE, 0, obj.GetColourPointerDouble());
                } else {
                    ::glColorPointer(3, GL_FLOAT, 0, newColors[i].data());
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
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                ::glTexCoordPointer(2, GL_FLOAT, 0, obj.GetTextureCoordinatePointerFloat());
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
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
            const megamol::geocalls_gl::CallTriMeshDataGL::Material& mat = *obj.GetMaterial();

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
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE:
                ::glDrawElements(GL_TRIANGLES, obj.GetTriCount() * 3, GL_UNSIGNED_BYTE, obj.GetTriIndexPointerByte());
                if (datadirty) {
                    this->triangles[i].resize(obj.GetTriCount());
                    for (size_t j = 0; j < obj.GetTriCount(); j++) {
                        this->triangles[i][j] = vislib::math::Vector<unsigned int, 3>(
                            static_cast<unsigned int>(obj.GetTriIndexPointerByte()[3 * j + 0]),
                            static_cast<unsigned int>(obj.GetTriIndexPointerByte()[3 * j + 1]),
                            static_cast<unsigned int>(obj.GetTriIndexPointerByte()[3 * j + 2]));
                    }
                }
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT16:
                ::glDrawElements(
                    GL_TRIANGLES, obj.GetTriCount() * 3, GL_UNSIGNED_SHORT, obj.GetTriIndexPointerUInt16());
                if (datadirty) {
                    this->triangles[i].resize(obj.GetTriCount());
                    for (size_t j = 0; j < obj.GetTriCount(); j++) {
                        this->triangles[i][j] = vislib::math::Vector<unsigned int, 3>(
                            static_cast<unsigned int>(obj.GetTriIndexPointerUInt16()[3 * j + 0]),
                            static_cast<unsigned int>(obj.GetTriIndexPointerUInt16()[3 * j + 1]),
                            static_cast<unsigned int>(obj.GetTriIndexPointerUInt16()[3 * j + 2]));
                    }
                }
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32:
                ::glDrawElements(GL_TRIANGLES, obj.GetTriCount() * 3, GL_UNSIGNED_INT, obj.GetTriIndexPointerUInt32());
                if (datadirty) {
                    this->triangles[i].resize(obj.GetTriCount());
                    for (size_t j = 0; j < obj.GetTriCount(); j++) {
                        this->triangles[i][j] = vislib::math::Vector<unsigned int, 3>(
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

        if (datadirty && obj.GetVertexAttribCount() > 0 &&
            obj.GetVertexAttribDataType(0) == megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32) {
            auto dt = obj.GetVertexAttribDataType(0);
            this->indexAttrib[i].resize(obj.GetVertexCount());
            auto ptr = obj.GetVertexAttribPointerUInt32(0);
            for (size_t j = 0; j < obj.GetVertexCount(); j++) {
                this->indexAttrib[i][j] = ptr[j];
            }
        }
    }

    if (normals)
        ::glDisableClientState(GL_NORMAL_ARRAY);
    if (colors)
        ::glDisableClientState(GL_COLOR_ARRAY);
    if (textures)
        ::glDisableClientState(GL_TEXTURE_COORD_ARRAY);

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
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                ::glVertexPointer(3, GL_FLOAT, 0, ctmd->Objects()[i].GetVertexPointerFloat());
                break;
            case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
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

    utility::ColourParser::FromString(this->innerColorSlot.Param<param::StringParam>()->Value().c_str(), r, g, b);
    vislib::math::Vector<float, 3> innerColor(r, g, b);

    utility::ColourParser::FromString(this->outerColorSlot.Param<param::StringParam>()->Value().c_str(), r, g, b);
    vislib::math::Vector<float, 3> outerColor(r, g, b);

    utility::ColourParser::FromString(this->fontColorSlot.Param<param::StringParam>()->Value().c_str(), r, g, b);
    vislib::math::Vector<float, 3> fontColor(r, g, b);

    utility::ColourParser::FromString(this->borderColorSlot.Param<param::StringParam>()->Value().c_str(), r, g, b);
    vislib::math::Vector<float, 3> borderColor(r, g, b);

    if (this->showSweatBandSlot.Param<param::BoolParam>()->Value() && lines.size() > 0) {
        ::glLineWidth(3.0f);
        ::glDisable(GL_LIGHTING);
        ::glDisable(GL_DEPTH_TEST);

        ::glColor3f(borderColor.GetX(), borderColor.GetY(), borderColor.GetZ());
        switch (ctmd->Objects()[0].GetVertexDataType()) {
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
            ::glVertexPointer(3, GL_FLOAT, 0, ctmd->Objects()[0].GetVertexPointerFloat());
            break;
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE:
            ::glVertexPointer(3, GL_DOUBLE, 0, ctmd->Objects()[0].GetVertexPointerDouble());
            break;
        default:
            break;
        }
        ::glDrawElements(GL_LINES, static_cast<GLsizei>(linevec.size()), GL_UNSIGNED_INT, &linevec[0]);
        ::glEnable(GL_DEPTH_TEST);
    }

    auto bbCenter = bb.CalcCenter();

    // query the current state of the camera (needed for picking)
    view::Camera cam = cr->GetCamera();
    glm::mat4 projMat = cam.getProjectionMatrix();
    glm::mat4 modelViewMat = cam.getViewMatrix();
    auto modelMatrixInv = modelViewMat;
    modelMatrixInv = glm::inverse(modelMatrixInv);
    std::array<int, 2> viewport = {cr->GetFramebuffer()->getWidth(), cr->GetFramebuffer()->getHeight()};

    if (this->showRadiiSlot.Param<param::BoolParam>()->Value()) {
        // find closest line vertex location
        vislib::math::Point<float, 3> closest;
        float closestDist = FLT_MAX;
        switch (ctmd->Objects()[0].GetVertexDataType()) {
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT: {
            auto ptr = ctmd->Objects()[0].GetVertexPointerFloat();
            for (size_t i = 0; i < linevec.size(); i++) {
                vislib::math::Point<float, 3> pp(&ptr[3 * linevec[i] + 0]);
                if (pp.GetX() > bbCenter.GetX() && std::abs(pp.GetY() - bbCenter.GetY()) < closestDist) {
                    closest = pp;
                    closestDist = std::abs(pp.GetY() - bbCenter.GetY());
                }
            }
        } break;
        case megamol::geocalls_gl::CallTriMeshDataGL::Mesh::DT_DOUBLE: {
            auto ptr = ctmd->Objects()[0].GetVertexPointerDouble();
            break;
        }
        default:
            break;
        }

        ::glLineWidth(3.0f);
        ::glDisable(GL_LIGHTING);
        ::glDisable(GL_DEPTH_TEST);
        ::glColor3f(1.0f, 0.0f, 0.0f);

        glBegin(GL_LINES);
        glColor3f(innerColor.GetX(), innerColor.GetY(), innerColor.GetZ());
        glVertex3f(bbCenter.GetX(), bbCenter.GetY(), bbCenter.GetZ());
        glVertex3f(closest.GetX(), closest.GetY(), closest.GetZ());
        glColor3f(outerColor.GetX(), outerColor.GetY(), outerColor.GetZ());
        glVertex3f(closest.GetX(), closest.GetY(), closest.GetZ());
        glVertex3f(bb.Right(), closest.GetY(), closest.GetZ());
        glEnd();

        if (this->theFont.Initialise(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>())) {
            float distleft = std::abs(bbCenter.GetX() - closest.GetX());
            float distright = std::abs(closest.GetX() - bb.Right());
            vislib::StringA textleft = (trunc(distleft, 2) + " Å").c_str();
            vislib::StringA textright = (trunc(distright, 2) + " Å").c_str();

            float sizeleft = 5.0f;
            float sizeright = 5.0f;

            while (theFont.LineWidth(sizeleft, textleft) > distleft && sizeleft > 1.0f) {
                sizeleft -= 0.5f;
            }
            while (theFont.LineWidth(sizeright, textright) > distright && sizeright > 1.0f) {
                sizeright -= 0.5f;
            }

            sizeleft = std::min(sizeleft, sizeright);
            sizeright = sizeleft;

            float leftwidth = theFont.LineWidth(sizeleft, textleft);
            float rightwidth = theFont.LineWidth(sizeright, textright);
            float leftheight = theFont.LineHeight(sizeleft);
            float rightheight = theFont.LineHeight(sizeright);

            float remainleft = distleft - leftwidth;
            float remainright = distright - rightwidth;

            float fontCol[4] = {fontColor.GetX(), fontColor.GetY(), fontColor.GetZ(), 1.0f};
            this->theFont.DrawString(projMat, modelViewMat, fontCol, bbCenter.GetX() + (remainleft / 2.0f),
                bbCenter.GetY() + 1.0 * leftheight, bbCenter.GetZ(), leftwidth, leftheight, sizeleft, false, textleft,
                megamol::core::utility::SDFFont::ALIGN_LEFT_BOTTOM);
            this->theFont.DrawString(projMat, modelViewMat, fontCol, closest.GetX() + (remainright / 2.0f),
                closest.GetY() + 1.0 * rightheight, closest.GetZ(), rightwidth, rightheight, sizeright, false,
                textright, megamol::core::utility::SDFFont::ALIGN_LEFT_BOTTOM);
        }

        ::glEnable(GL_DEPTH_TEST);
    }

    // the camera position from the matrix seems to be wrong
    auto cam_pose = cam.get<megamol::core::view::Camera::Pose>();
    this->lastCamState.camPos =
        vislib::math::Vector<float, 3>(cam_pose.position.x, cam_pose.position.y, cam_pose.position.x);
    this->lastCamState.camDir =
        vislib::math::Vector<float, 3>(cam_pose.direction.x, cam_pose.direction.y, cam_pose.direction.x);
    this->lastCamState.camUp = vislib::math::Vector<float, 3>(cam_pose.up.x, cam_pose.up.y, cam_pose.up.x);
    this->lastCamState.camRight = vislib::math::Vector<float, 3>(cam_pose.right.x, cam_pose.right.y, cam_pose.right.x);
    this->lastCamState.camDir.Normalise();
    this->lastCamState.camUp.Normalise();
    this->lastCamState.camRight.Normalise();

    try {
        auto cam_intrinsics = cam.get<megamol::core::view::Camera::PerspectiveParameters>();
        this->lastCamState.zNear = cam_intrinsics.near_plane;
        this->lastCamState.zFar = cam_intrinsics.far_plane;
        this->lastCamState.fovy = cam_intrinsics.fovy;
        this->lastCamState.aspect = cam_intrinsics.aspect;
        if (std::get<1>(viewport) == 0) {
            this->lastCamState.aspect = 0.0f;
        }
        this->lastCamState.fovx = 2.0f * atan(tan(this->lastCamState.fovy / 2.0f) * this->lastCamState.aspect);
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "SombreroMeshRenderer - Error when getting camera intrinsics");
    }

    this->lastCamState.width = std::get<0>(viewport);
    this->lastCamState.height = std::get<1>(viewport);

    ::glCullFace(cfm);
    ::glFrontFace(twr);
    ::glPolygonMode(GL_FRONT, pm[0]);
    ::glPolygonMode(GL_BACK, pm[1]);

    ::glEnable(GL_CULL_FACE);
    ::glDisableClientState(GL_VERTEX_ARRAY);
    ::glEnable(GL_BLEND);

#if (defined(_MSC_VER) && (_MSC_VER > 1000))
    ::GetLastError();
#endif
    ::glCullFace(cfm);
    ::glFrontFace(twr);
    ::glDisableClientState(GL_VERTEX_ARRAY);

    return true;
}
