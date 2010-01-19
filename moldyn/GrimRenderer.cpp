/*
 * GrimRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GrimRenderer.h"
#include "ParticleGridDataCall.h"
#include "CoreInstance.h"
#include "param/IntParam.h"
#include "view/CallClipPlane.h"
#include "view/CallGetTransferFunction.h"
#include "view/CallRender3D.h"
#include <GL/gl.h>
#include "vislib/Array.h"
#include "vislib/assert.h"
#include "vislib/Camera.h"
#include "vislib/mathfunctions.h"
#include "vislib/mathtypes.h"
#include "vislib/Pair.h"
#include "vislib/Plane.h"
#include "vislib/Point.h"
#include "vislib/Trace.h"
#include "vislib/Vector.h"

using namespace megamol::core;

/****************************************************************************/


/*
 * moldyn::GrimRenderer::CellInfo::CellInfo
 */
moldyn::GrimRenderer::CellInfo::CellInfo(void) {
    ::glGenOcclusionQueriesNV(1, &this->oQuery);
}


/*
 * moldyn::GrimRenderer::CellInfo::~CellInfo
 */
moldyn::GrimRenderer::CellInfo::~CellInfo(void) {
    ::glDeleteOcclusionQueriesNV(1, &this->oQuery);
}

/****************************************************************************/


/*
 * moldyn::GrimRenderer::GrimRenderer
 */
moldyn::GrimRenderer::GrimRenderer(void) : Renderer3DModule(),
        sphereShader(), initDepthShader(), initDepthMapShader(),
        depthMipShader(), pointShader(), initDepthPointShader(), fbo(),
        getDataSlot("getdata", "Connects to the data source"),
        getTFSlot("gettransferfunction", "Connects to the transfer function module"),
        greyTF(0), cellDists(), cellInfos(0) {

    this->getDataSlot.SetCompatibleCall<moldyn::ParticleGridDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

}


/*
 * moldyn::GrimRenderer::~GrimRenderer
 */
moldyn::GrimRenderer::~GrimRenderer(void) {
    this->Release();
}


/*
 * moldyn::GrimRenderer::create
 */
bool moldyn::GrimRenderer::create(void) {
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()
            || !vislib::graphics::gl::FramebufferObject::InitialiseExtensions()
            || (glh_init_extensions("GL_NV_occlusion_query GL_ARB_multitexture") == GL_FALSE)) {
        return false;
    }

    vislib::graphics::gl::ShaderSource vert, frag;

    const char *shaderName = "sphere";
    try {

        shaderName = "sphereShader";
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::theOtherSphereVertex", vert)) { return false; }
        if (!instance()->ShaderSourceFactory().MakeShaderSource("simplesphere::fragment", frag)) { return false; }
        //printf("\nVertex Shader:\n%s\n\nFragment Shader:\n%s\n",
        //    vert.WholeCode().PeekBuffer(),
        //    frag.WholeCode().PeekBuffer());
        if (!this->sphereShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "initDepthShader";
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::init::vertex", vert)) { return false; }
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::init::fragment", frag)) { return false; }
        if (!this->initDepthShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "initDepthMapShader";
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::depthmap::vert", vert)) { return false; }
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::depthmap::initfrag", frag)) { return false; }
        if (!this->initDepthMapShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "depthMipShader";
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::depthmap::vert", vert)) { return false; }
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::depthmap::mipfrag", frag)) { return false; }
        if (!this->depthMipShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "pointShader";
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::point::vert", vert)) { return false; }
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::point::frag", frag)) { return false; }
        if (!this->pointShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

        shaderName = "initDepthPointShader";
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::point::simplevert", vert)) { return false; }
        if (!instance()->ShaderSourceFactory().MakeShaderSource("mipdepth::point::simplefrag", frag)) { return false; }
        if (!this->initDepthPointShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile %s: Unknown error\n", shaderName);
            return false;
        }

    } catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s shader (@%s): %s\n", shaderName,
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()) ,ce.GetMsgA());
        return false;
    } catch(vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s shader: %s\n", shaderName, e.GetMsgA());
        return false;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile %s shader: Unknown exception\n", shaderName);
        return false;
    }

    glEnable(GL_TEXTURE_1D);
    glGenTextures(1, &this->greyTF);
    unsigned char tex[6] = {
        0, 0, 0,  255, 255, 255
    };
    glBindTexture(GL_TEXTURE_1D, this->greyTF);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);

    glDisable(GL_TEXTURE_1D);

    this->fbo.Create(1, 1); // year, right.
    this->depthmap[0].Create(1, 1);
    this->depthmap[1].Create(1, 1);

    return true;
}


/*
 * moldyn::GrimRenderer::GetCapabilities
 */
bool moldyn::GrimRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        | view::CallRender3D::CAP_LIGHTING
        //| view::CallRender3D::CAP_ANIMATION
        );

    return true;
}


/*
 * moldyn::GrimRenderer::GetExtents
 */
bool moldyn::GrimRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    ParticleGridDataCall *pgdc = this->getDataSlot.CallAs<ParticleGridDataCall>();
    if (pgdc == NULL) return false;
    if (!(*pgdc)(1)) return false;

    cr->SetTimeFramesCount(pgdc->FrameCount());
    cr->AccessBoundingBoxes() = pgdc->AccessBoundingBoxes();

    float scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    if (scaling > 0.0000001) {
        scaling = 10.0f / scaling;
    } else {
        scaling = 1.0f;
    }
    cr->AccessBoundingBoxes().MakeScaledWorld(scaling);

    return true;
}


/*
 * moldyn::GrimRenderer::release
 */
void moldyn::GrimRenderer::release(void) {
    this->sphereShader.Release();
    this->initDepthMapShader.Release();
    this->initDepthShader.Release();
    this->depthMipShader.Release();
    this->pointShader.Release();
    this->fbo.Release();
    this->depthmap[0].Release();
    this->depthmap[1].Release();
    ::glDeleteTextures(1, &this->greyTF);
    this->cellDists.Clear();
    this->cellInfos.Clear();
}


/*
 * moldyn::GrimRenderer::Render
 */
bool moldyn::GrimRenderer::Render(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    ParticleGridDataCall *pgdc = this->getDataSlot.CallAs<ParticleGridDataCall>();
    if (pgdc == NULL) return false;

    // ask for extend to calculate the data scaling
    if (!(*pgdc)(1)) return false;
    float scaling = pgdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    if (scaling > 0.0000001) {
        scaling = 10.0f / scaling;
    } else {
        scaling = 1.0f;
    }

    // fetch real data
    if (!(*pgdc)(0)) return false;
    unsigned int cellcnt = pgdc->CellsCount();
    unsigned int typecnt = pgdc->TypesCount();

    // update fbo size, if required
    GLint viewport[4];
    ::glGetIntegerv(GL_VIEWPORT, viewport);
    if ((this->fbo.GetWidth() != static_cast<UINT>(viewport[2]))
            || (this->fbo.GetHeight() != static_cast<UINT>(viewport[3]))) {
        this->fbo.Release();
        this->fbo.Create(viewport[2], viewport[3],
                GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, // colour buffer
                vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE,
                GL_DEPTH_COMPONENT24); // depth buffer

        unsigned int dmw = vislib::math::NextPowerOfTwo(viewport[2]);
        unsigned int dmh = vislib::math::NextPowerOfTwo(viewport[3]);
        dmh += dmh / 2;
        if ((this->depthmap[0].GetWidth() != dmw) || (this->depthmap[0].GetHeight() != dmh)) {
            for (int i = 0; i < 2; i++) {
                this->depthmap[i].Release();
                this->depthmap[i].Create(dmw, dmh, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT,
                    vislib::graphics::gl::FramebufferObject::ATTACHMENT_DISABLED);
            }
        }
    }

    if (this->cellDists.Count() != cellcnt) {
        this->cellDists.SetCount(cellcnt);
        this->cellInfos.SetCount(cellcnt);
        for (unsigned int i = 0; i < cellcnt; i++) {
            this->cellDists[i].First() = i;
            this->cellInfos[i].wasvisible = true; // TODO: refine with Reina-Approach
            this->cellInfos[i].maxrad = 0.0f;
            for (unsigned int j = 0; j < typecnt; j++) {
                this->cellInfos[i].maxrad = vislib::math::Max(this->cellInfos[i].maxrad,
                    pgdc->Cells()[i].AccessParticleLists()[j].GetMaxRadius() * scaling);
            }
        }
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(5.0f);

    float viewDist = 
        0.5f * cr->GetCameraParameters()->VirtualViewSize().Height() /
        tanf(cr->GetCameraParameters()->HalfApertureAngle());

    // depth-sort of cells
    vislib::Array<vislib::Pair<unsigned int, float> > &dists = this->cellDists;
    vislib::Array<CellInfo> &infos = this->cellInfos;
    // The usage of these references is required in order to get performance !!! WTF !!!
    for (unsigned int i = 0; i < cellcnt; i++) {
        unsigned int idx = dists[i].First();
        const ParticleGridDataCall::GridCell& cell = pgdc->Cells()[idx];
        CellInfo& info = infos[idx];
        const vislib::math::Cuboid<float> &bbox = cell.GetBoundingBox();

        vislib::math::Point<float, 3> cellPos(
            (bbox.Left() + bbox.Right()) * 0.5f * scaling,
            (bbox.Bottom() + bbox.Top()) * 0.5f * scaling,
            (bbox.Back() + bbox.Front()) * 0.5f * scaling);

        vislib::math::Vector<float, 3> cellDistV = cellPos - cr->GetCameraParameters()->Position();
        float cellDist = cr->GetCameraParameters()->Front().Dot(cellDistV);

        dists[i].Second() = cellDist;

        // calculate view size of the max sphere
        float sphereImgRad = info.maxrad * viewDist / cellDist;
        info.dots = (sphereImgRad < 0.75f);

        info.isvisible = true;
        // TODO: Test against the viewing frustum

    }
    dists.Sort(&GrimRenderer::depthSort);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    // z-buffer-filling
#if defined(DEBUG) || defined(_DEBUG)
    UINT oldlevel = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_NONE);
#endif
    this->fbo.Enable();
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ::glScalef(scaling, scaling, scaling);

    // initialize depth buffer
    this->initDepthPointShader.Enable();
    ::glPointSize(1.0f);
    for (int i = cellcnt - 1; i >= 0; i--) { // front to back
        unsigned int idx = dists[i].First();
        const ParticleGridDataCall::GridCell *cell = &pgdc->Cells()[idx];
        CellInfo &info = infos[idx];
        if (!info.wasvisible) continue;
        // only draw cells which were visible last frame
        if (!info.dots) continue;

        for (unsigned int j = 0; j < typecnt; j++) {
            const ParticleGridDataCall::Particles &parts = cell->AccessParticleLists()[j];
            const ParticleGridDataCall::ParticleType &ptype = pgdc->Types()[j];

            // radius and position
            switch (ptype.GetVertexDataType()) {
                case MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glVertexPointer(3, GL_FLOAT,
                        vislib::math::Max(16U, parts.GetVertexDataStride()),
                        parts.GetVertexData());
                    break;
                default:
                    continue;
            }

            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

            glDisableClientState(GL_VERTEX_ARRAY);
        }
    }
    this->initDepthPointShader.Disable();

    ::glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    float viewportStuff[4] = {
        cr->GetCameraParameters()->TileRect().Left(),
        cr->GetCameraParameters()->TileRect().Bottom(),
        cr->GetCameraParameters()->TileRect().Width(),
        cr->GetCameraParameters()->TileRect().Height()};
    float defaultPointSize = vislib::math::Max(viewportStuff[2], viewportStuff[3]);
    ::glPointSize(defaultPointSize);
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    unsigned int cial = glGetAttribLocationARB(this->sphereShader, "colIdx");
    unsigned int cial2 = glGetAttribLocationARB(this->pointShader, "colIdx");

    this->initDepthShader.Enable();

    glUniform4fvARB(this->initDepthShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->initDepthShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fvARB(this->initDepthShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fvARB(this->initDepthShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
    // no clipping plane for now
    glColor4ub(192, 192, 192, 255);
    glDisableClientState(GL_COLOR_ARRAY);

    for (int i = cellcnt - 1; i >= 0; i--) { // front to back
        unsigned int idx = dists[i].First();
        const ParticleGridDataCall::GridCell *cell = &pgdc->Cells()[idx];
        CellInfo &info = infos[idx];
        if (!info.wasvisible) continue;
        // only draw cells which were visible last frame
        if (info.dots) continue;

        //glColor4ub(192, 192, 192, 255);
        float a = static_cast<float>(i) / static_cast<float>(cellcnt - 1);
        ASSERT((a >= 0.0) && (a <= 1.0f));
        glColor3f(1.0f - a, 0.0f, a);
        if (info.dots) {
            glColor3ub(255, 0, 0);
        }

        for (unsigned int j = 0; j < typecnt; j++) {
            const ParticleGridDataCall::Particles &parts = cell->AccessParticleLists()[j];
            const ParticleGridDataCall::ParticleType &ptype = pgdc->Types()[j];

            // radius and position
            switch (ptype.GetVertexDataType()) {
                case MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4fARB(this->initDepthShader.ParameterLocation("inConsts1"), ptype.GetGlobalRadius(), 0.0f, 0.0f, 0.0f);
                    glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4fARB(this->initDepthShader.ParameterLocation("inConsts1"), -1.0f, 0.0f, 0.0f, 0.0f);
                    glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                default:
                    continue;
            }

            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

            glDisableClientState(GL_VERTEX_ARRAY);

        }

        //glBegin(GL_LINES);
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Front());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Front());
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Front());
        //glVertex3f(cell->GetBoundingBox().Right(), cell->GetBoundingBox().Top(), cell->GetBoundingBox().Back());
        //glVertex3f(cell->GetBoundingBox().Left(), cell->GetBoundingBox().Bottom(), cell->GetBoundingBox().Front());
        //glEnd();

    }

    this->initDepthShader.Disable();

    // occlusion queries ftw
    ::glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    ::glDepthMask(GL_FALSE);
    ::glDisable(GL_CULL_FACE);

    // also disable texturing and any fancy shading features
    for (int i = cellcnt - 1; i >= 0; i--) { // front to back
        const ParticleGridDataCall::GridCell& cell = pgdc->Cells()[i];
        CellInfo& info = infos[i];
        const vislib::math::Cuboid<float>& bbox = cell.GetBoundingBox();
        if (!info.isvisible) continue; // frustum culling

        ::glBeginOcclusionQueryNV(info.oQuery);

        // render bounding box for cell idx
        ::glBegin(GL_QUADS);

        ::glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Back());
        ::glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Back());
        ::glVertex3f(bbox.Right(), bbox.Top(), bbox.Back());
        ::glVertex3f(bbox.Left(), bbox.Top(), bbox.Back());

        ::glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Front());
        ::glVertex3f(bbox.Left(), bbox.Top(), bbox.Front());
        ::glVertex3f(bbox.Right(), bbox.Top(), bbox.Front());
        ::glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Front());

        ::glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Back());
        ::glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Back());
        ::glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Front());
        ::glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Front());

        ::glVertex3f(bbox.Left(), bbox.Top(), bbox.Front());
        ::glVertex3f(bbox.Left(), bbox.Top(), bbox.Back());
        ::glVertex3f(bbox.Right(), bbox.Top(), bbox.Back());
        ::glVertex3f(bbox.Right(), bbox.Top(), bbox.Front());

        ::glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Back());
        ::glVertex3f(bbox.Left(), bbox.Top(), bbox.Back());
        ::glVertex3f(bbox.Left(), bbox.Top(), bbox.Front());
        ::glVertex3f(bbox.Left(), bbox.Bottom(), bbox.Front());

        ::glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Front());
        ::glVertex3f(bbox.Right(), bbox.Bottom(), bbox.Back());
        ::glVertex3f(bbox.Right(), bbox.Top(), bbox.Back());
        ::glVertex3f(bbox.Right(), bbox.Top(), bbox.Front());

        ::glEnd();

        ::glEndOcclusionQueryNV();
    }

    ::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    ::glDepthMask(GL_TRUE);
    ::glEnable(GL_CULL_FACE);
    // reenable other state

    this->fbo.Disable();
    // END Depth buffer initialized

    // create depth mipmap
    this->depthmap[0].Enable();

    //::glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
    ::glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT);

    ::glEnable(GL_TEXTURE_2D);
    ::glEnable(GL_DEPTH_TEST);
    ::glDisable(GL_LIGHTING);
    ::glDisable(GL_DEPTH_TEST);
    ::glActiveTextureARB(GL_TEXTURE0_ARB);
    this->fbo.BindDepthTexture();

    ::glMatrixMode(GL_PROJECTION);
    ::glPushMatrix();
    ::glLoadIdentity();
    ::glMatrixMode(GL_MODELVIEW);
    ::glPushMatrix();
    ::glLoadIdentity();

    this->initDepthMapShader.Enable();
    this->initDepthMapShader.SetParameter("datex", 0);

    ::glBegin(GL_QUADS);
    float xf = float(this->fbo.GetWidth()) / float(this->depthmap[0].GetWidth());
    float yf = float(this->fbo.GetHeight()) / float(this->depthmap[0].GetHeight());
    ::glVertex2f(-1.0f, -1.0f);
    ::glVertex2f(-1.0f + 2.0f * xf, -1.0f);
    ::glVertex2f(-1.0f + 2.0f * xf, -1.0f + 2.0f * yf);
    ::glVertex2f(-1.0f, -1.0f + 2.0f * yf);
    ::glEnd();

    this->initDepthMapShader.Disable();

    int maxLevel = 0;
    int lw = this->depthmap[0].GetWidth() / 2;
    int ly = this->depthmap[0].GetHeight() * 2 / 3;
    int lh = ly / 2;
    int ls = vislib::math::Min(lh, lw);

    this->depthMipShader.Enable();
    this->depthMipShader.SetParameter("datex", 0);
    this->depthMipShader.SetParameter("src", 0, 0);
    this->depthMipShader.SetParameter("dst", 0, ly);

    maxLevel = 1; // we created one! hui!
    ::glBegin(GL_QUADS);
    ::glVertex2f(-1.0f + 2.0f * 0.0f,
        -1.0f + 2.0f * float(ly) / float(this->depthmap[0].GetHeight()));
    ::glVertex2f(-1.0f + 2.0f * float(this->fbo.GetWidth() / 2) / float(this->depthmap[0].GetWidth()),
        -1.0f + 2.0f * float(ly) / float(this->depthmap[0].GetHeight()));
    ::glVertex2f(-1.0f + 2.0f * float(this->fbo.GetWidth() / 2) / float(this->depthmap[0].GetWidth()),
        -1.0f + 2.0f * float(ly + this->fbo.GetHeight() / 2) / float(this->depthmap[0].GetHeight()));
    ::glVertex2f(-1.0f + 2.0f * 0.0f,
        -1.0f + 2.0f * float(ly + this->fbo.GetHeight() / 2) / float(this->depthmap[0].GetHeight()));
    ::glEnd();

    this->depthmap[0].Disable();

    int lx = lw;
    while (ls > 1) {
        this->depthmap[maxLevel % 2].Enable();
        this->depthmap[1 - (maxLevel % 2)].BindColourTexture();

        this->depthMipShader.SetParameter("src", lx - lw, ly);
        this->depthMipShader.SetParameter("dst", lx, ly);

        lw /= 2;
        lh /= 2;
        ls /= 2;

        float x1, x2, y1, y2;

        x1 = float(lx) / float(this->depthmap[0].GetWidth());
        x2 = float(lx + lw) / float(this->depthmap[0].GetWidth());
        y1 = float(ly) / float(this->depthmap[0].GetHeight());
        y2 = float(ly + lh) / float(this->depthmap[0].GetHeight());

        ::glBegin(GL_QUADS);
        ::glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y1);
        ::glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y1);
        ::glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y2);
        ::glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y2);
        ::glEnd();

        this->depthmap[maxLevel % 2].Disable();
        ::glBindTexture(GL_TEXTURE_2D, 0);

        lx += lw;
        maxLevel++;
    }

    this->depthMipShader.Disable();

    this->depthmap[0].Enable();
    this->initDepthMapShader.Enable();
    this->initDepthMapShader.SetParameter("datex", 0);
    this->depthmap[1].BindColourTexture();

    lw = this->depthmap[0].GetWidth() / 2;
    ly = this->depthmap[0].GetHeight() * 2 / 3;
    lh = ly / 2;
    ls = vislib::math::Min(lh, lw);
    lx = lw;
    while (ls > 1) {

        lw /= 2;
        lh /= 2;
        ls /= 2;

        float x1, x2, y1, y2;

        x1 = float(lx) / float(this->depthmap[0].GetWidth());
        x2 = float(lx + lw) / float(this->depthmap[0].GetWidth());
        y1 = float(ly) / float(this->depthmap[0].GetHeight());
        y2 = float(ly + lh) / float(this->depthmap[0].GetHeight());

        ::glBegin(GL_QUADS);
        ::glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y1);
        ::glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y1);
        ::glVertex2f(-1.0f + 2.0f * x2, -1.0f + 2.0f * y2);
        ::glVertex2f(-1.0f + 2.0f * x1, -1.0f + 2.0f * y2);
        ::glEnd();

        lx += lw;

        // and skip one
        lw /= 2;
        lh /= 2;
        ls /= 2;
        lx += lw;
    }

    this->initDepthMapShader.Disable();
    this->depthmap[0].Disable();

    ::glMatrixMode(GL_PROJECTION);
    ::glPopMatrix();
    ::glMatrixMode(GL_MODELVIEW);
    ::glPopMatrix();

    ::glBindTexture(GL_TEXTURE_2D, 0);
    // END generation of depth-max mipmap

#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(oldlevel);
#endif

    unsigned int visCnt = 0;
    // draw visible data (dots)
    ::glEnable(GL_DEPTH_TEST);
    ::glPointSize(1.0f);
    this->pointShader.Enable();
    for (int i = cellcnt - 1; i >= 0; i--) { // front to back
        const ParticleGridDataCall::GridCell& cell = pgdc->Cells()[i];
        CellInfo& info = infos[i];
        unsigned int pixelCount;
        if (!info.isvisible) continue; // frustum culling
        if (!info.dots) continue;

        ::glGetOcclusionQueryuivNV(info.oQuery, GL_PIXEL_COUNT_NV, &pixelCount);
        info.isvisible = (pixelCount > 0);
        //printf("PixelCount of cell %u is %u\n", idx, pixelCount);
        if (!info.isvisible) continue; // occlusion culling
        visCnt++;


        for (unsigned int j = 0; j < typecnt; j++) {
            const ParticleGridDataCall::Particles &parts = cell.AccessParticleLists()[j];
            const ParticleGridDataCall::ParticleType &ptype = pgdc->Types()[j];
            float minC = 0.0f, maxC = 0.0f;
            unsigned int colTabSize = 0;

            // colour
            switch (ptype.GetColourDataType()) {
                case MultiParticleDataCall::Particles::COLDATA_NONE:
                    glColor3ubv(ptype.GetGlobalColour());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(3, GL_UNSIGNED_BYTE,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(4, GL_UNSIGNED_BYTE,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(3, GL_FLOAT,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(4, GL_FLOAT,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                    glEnableVertexAttribArrayARB(cial2);
                    glVertexAttribPointerARB(cial2, 1, GL_FLOAT, GL_FALSE,
                        parts.GetColourDataStride(), parts.GetColourData());

                    glEnable(GL_TEXTURE_1D);

                    view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
                    if ((cgtf != NULL) && ((*cgtf)())) {
                        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                        colTabSize = cgtf->TextureSize();
                    } else {
                        glBindTexture(GL_TEXTURE_1D, this->greyTF);
                        colTabSize = 2;
                    }

                    glUniform1iARB(this->pointShader.ParameterLocation("colTab"), 0);
                    minC = ptype.GetMinColourIndexValue();
                    maxC = ptype.GetMaxColourIndexValue();
                    glColor3ub(127, 127, 127);
                } break;
                default:
                    glColor3ub(127, 127, 127);
                    break;
            }

            // radius and position
            switch (ptype.GetVertexDataType()) {
                case MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4fARB(this->pointShader.ParameterLocation("inConsts1"),
                        ptype.GetGlobalRadius(), minC, maxC, float(colTabSize));
                    glVertexPointer(3, GL_FLOAT,
                        parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4fARB(this->pointShader.ParameterLocation("inConsts1"),
                        -1.0f, minC, maxC, float(colTabSize));
                    glVertexPointer(3, GL_FLOAT,
                        vislib::math::Max(16U, parts.GetVertexDataStride()),
                        parts.GetVertexData());
                    break;
                default:
                    continue;
            }

            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableVertexAttribArrayARB(cial2);
            glDisable(GL_TEXTURE_1D);
        }

    }
    this->pointShader.Disable();

    // draw visible data (spheres)
    this->sphereShader.Enable();

    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
    // no clipping plane for now
    glUniform4fARB(this->sphereShader.ParameterLocation("clipDat"), 0.0f, 0.0f, 0.0f, 0.0f);
    glUniform3fARB(this->sphereShader.ParameterLocation("clipCol"), 0.0f, 0.0f, 0.0f);
    this->sphereShader.SetParameter("depthTexParams", this->depthmap[0].GetWidth(),
        this->depthmap[0].GetHeight() * 2 / 3, maxLevel);

    ::glEnable(GL_TEXTURE_2D);
    ::glActiveTextureARB(GL_TEXTURE2_ARB);
    this->depthmap[0].BindColourTexture();
    this->sphereShader.SetParameter("depthTex", 2);
    ::glActiveTextureARB(GL_TEXTURE0_ARB);
    ::glPointSize(defaultPointSize);

    for (int i = cellcnt - 1; i >= 0; i--) { // front to back
        unsigned int idx = dists[i].First();
        const ParticleGridDataCall::GridCell& cell = pgdc->Cells()[idx];
        CellInfo& info = infos[idx];

        unsigned int pixelCount;
        if (!info.isvisible) continue; // frustum culling
        if (info.dots) continue;

        ::glGetOcclusionQueryuivNV(info.oQuery, GL_PIXEL_COUNT_NV, &pixelCount);
        info.isvisible = (pixelCount > 0);
        //printf("PixelCount of cell %u is %u\n", idx, pixelCount);
        if (!info.isvisible) continue; // occlusion culling
        visCnt++;

        for (unsigned int j = 0; j < typecnt; j++) {
            const ParticleGridDataCall::Particles &parts = cell.AccessParticleLists()[j];
            const ParticleGridDataCall::ParticleType &ptype = pgdc->Types()[j];
            float minC = 0.0f, maxC = 0.0f;
            unsigned int colTabSize = 0;

            // colour
            switch (ptype.GetColourDataType()) {
                case MultiParticleDataCall::Particles::COLDATA_NONE:
                    glColor3ubv(ptype.GetGlobalColour());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(3, GL_UNSIGNED_BYTE,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(4, GL_UNSIGNED_BYTE,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(3, GL_FLOAT,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(4, GL_FLOAT,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                    glEnableVertexAttribArrayARB(cial);
                    glVertexAttribPointerARB(cial, 1, GL_FLOAT, GL_FALSE,
                        parts.GetColourDataStride(), parts.GetColourData());

                    glEnable(GL_TEXTURE_1D);

                    view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
                    if ((cgtf != NULL) && ((*cgtf)())) {
                        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                        colTabSize = cgtf->TextureSize();
                    } else {
                        glBindTexture(GL_TEXTURE_1D, this->greyTF);
                        colTabSize = 2;
                    }

                    glUniform1iARB(this->sphereShader.ParameterLocation("colTab"), 0);
                    minC = ptype.GetMinColourIndexValue();
                    maxC = ptype.GetMaxColourIndexValue();
                    glColor3ub(127, 127, 127);
                } break;
                default:
                    glColor3ub(127, 127, 127);
                    break;
            }

            // radius and position
            switch (ptype.GetVertexDataType()) {
                case MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"),
                        ptype.GetGlobalRadius(), minC, maxC, float(colTabSize));
                    glVertexPointer(3, GL_FLOAT,
                        parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"),
                        -1.0f, minC, maxC, float(colTabSize));
                    glVertexPointer(4, GL_FLOAT,
                        parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                default:
                    continue;
            }

            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableVertexAttribArrayARB(cial);
            glDisable(GL_TEXTURE_1D);
        }

    }
    //printf("%f%% cells visible\n", float(visCnt * 100) / float(cellcnt));
    this->sphereShader.Disable();
    ::glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    ::glDisable(GL_TEXTURE_2D);

    //// DEBUG OUTPUT OF FBO
    //::glEnable(GL_TEXTURE_2D);
    //::glDisable(GL_LIGHTING);
    //::glDisable(GL_DEPTH_TEST);
    ////this->fbo.BindDepthTexture();
    ////this->fbo.BindColourTexture();
    //this->depthmap[0].BindColourTexture();
    //::glMatrixMode(GL_PROJECTION);
    //::glPushMatrix();
    //::glLoadIdentity();
    //::glMatrixMode(GL_MODELVIEW);
    //::glPushMatrix();
    //::glLoadIdentity();
    //::glColor3ub(255, 255, 255);
    //::glBegin(GL_QUADS);
    //::glTexCoord2f(0.0f, 0.0f);
    //::glVertex2i(-1, -1);
    //::glTexCoord2f(1.0f, 0.0f);
    //::glVertex2i(1, -1);
    //::glTexCoord2f(1.0f, 1.0f);
    //::glVertex2i(1, 1);
    //::glTexCoord2f(0.0f, 1.0f);
    //::glVertex2i(-1, 1);
    //::glEnd();
    //::glMatrixMode(GL_PROJECTION);
    //::glPopMatrix();
    //::glMatrixMode(GL_MODELVIEW);
    //::glPopMatrix();
    //::glBindTexture(GL_TEXTURE_2D, 0);

    // done!
    pgdc->Unlock();

    for (int i = cellcnt - 1; i >= 0; i--) {
        CellInfo& info = infos[i];
        info.wasvisible = info.isvisible;
    }

    return true;
}


/*
 * moldyn::GrimRenderer::depthSort
 */
int moldyn::GrimRenderer::depthSort(const vislib::Pair<unsigned int, float>& lhs,
           const vislib::Pair<unsigned int, float>& rhs) {
    float d = rhs.Second() - lhs.Second();
    if (d > vislib::math::FLOAT_EPSILON) return 1;
    if (d < -vislib::math::FLOAT_EPSILON) return -1;
    return 0;
}
