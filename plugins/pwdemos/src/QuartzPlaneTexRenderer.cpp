/*
 * QuartzPlaneTexRenderer.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "QuartzPlaneTexRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "vislib/assert.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/graphicsfunctions.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/math/Vector.h"

namespace megamol {
namespace demos {

/*
 * QuartzPlaneTexRenderer::QuartzPlaneTexRenderer
 */
QuartzPlaneTexRenderer::QuartzPlaneTexRenderer(void) : core::view::Renderer2DModule(),
AbstractTexQuartzRenderer(),
useClipColSlot("useClipCol", "Use clipping plane or grain colour for grains"),
cryShader() {

    this->useClipColSlot << new core::param::BoolParam(false);

    this->MakeSlotAvailable(&this->dataInSlot);
    this->MakeSlotAvailable(&this->typesInSlot);
    this->MakeSlotAvailable(&this->clipPlaneSlot);
    this->MakeSlotAvailable(&this->grainColSlot);
    this->MakeSlotAvailable(&this->showClipPlanePolySlot);
    this->MakeSlotAvailable(&this->useClipColSlot);
    this->MakeSlotAvailable(&this->correctPBCSlot);

}


/*
 * QuartzPlaneTexRenderer::~QuartzPlaneTexRenderer
 */
QuartzPlaneTexRenderer::~QuartzPlaneTexRenderer(void) {
    this->Release();
}


/*
 * QuartzPlaneTexRenderer::create
 */
bool QuartzPlaneTexRenderer::create(void) {
    using vislib::graphics::gl::GLSLShader;
    using megamol::core::utility::log::Log;
    using vislib::graphics::gl::ShaderSource;

    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Failed to initialise OpenGL GLSL Shader");
        return false;
    }
    if (!ogl_IsVersionGEQ(2, 0) || !isExtAvailable("GL_ARB_multitexture")) {
        Log::DefaultLog.WriteError("GL2.0 not present");
        return false;
    }

    ShaderSource vert, frag;
    try {
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quartz::ray::plane::tex::vert", vert)) {
            throw vislib::Exception("Generic vertex shader build failure", __FILE__, __LINE__);
        }
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quartz::ray::plane::tex::frag", frag)) {
            throw vislib::Exception("Generic fragment shader build failure", __FILE__, __LINE__);
        }
        if (!this->cryShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            throw vislib::Exception("Generic shader create failure", __FILE__, __LINE__);
        }
    }
    catch (vislib::Exception ex) {
        Log::DefaultLog.WriteError("Unable to compile shader: %s", ex.GetMsgA());
        this->release(); // Because I know that 'release' ONLY releases all the shaders
        return false;
    }
    catch (...) {
        Log::DefaultLog.WriteError("Unable to compile shader: Unexpected Exception");
        this->release(); // Because I know that 'release' ONLY releases all the shaders
        return false;
    }

    return true;
}


/*
 * QuartzPlaneTexRenderer::GetExtents
 */
bool QuartzPlaneTexRenderer::GetExtents(core::view::CallRender2DGL& call) {
    ParticleGridDataCall *pgdc = this->getParticleData();
    core::view::CallClipPlane *ccp = this->getClipPlaneData();
    if ((pgdc != NULL) && (ccp != NULL)) {
        if ((*pgdc)(ParticleGridDataCall::CallForGetExtent)) {
            if ((*ccp)()) {
                vislib::math::Vector<float, 3> cx, cy, p;
                float minX, minY, maxX, maxY, x, y;
                const vislib::math::Cuboid<float>& bbox =
                    pgdc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()
                    ? pgdc->AccessBoundingBoxes().ObjectSpaceBBox()
                    : pgdc->AccessBoundingBoxes().ClipBox();
                ccp->CalcPlaneSystem(cx, cy);

                p = bbox.GetLeftBottomBack();
                x = cx.Dot(p);
                y = cy.Dot(p);
                minX = maxX = x;
                minY = maxY = y;

                p = bbox.GetLeftBottomFront();
                x = cx.Dot(p);
                y = cy.Dot(p);
                if (minX > x) minX = x;
                if (maxX < x) maxX = x;
                if (minY > y) minY = y;
                if (maxY < y) maxY = y;

                p = bbox.GetLeftTopBack();
                x = cx.Dot(p);
                y = cy.Dot(p);
                if (minX > x) minX = x;
                if (maxX < x) maxX = x;
                if (minY > y) minY = y;
                if (maxY < y) maxY = y;

                p = bbox.GetLeftTopFront();
                x = cx.Dot(p);
                y = cy.Dot(p);
                if (minX > x) minX = x;
                if (maxX < x) maxX = x;
                if (minY > y) minY = y;
                if (maxY < y) maxY = y;

                p = bbox.GetRightBottomBack();
                x = cx.Dot(p);
                y = cy.Dot(p);
                if (minX > x) minX = x;
                if (maxX < x) maxX = x;
                if (minY > y) minY = y;
                if (maxY < y) maxY = y;

                p = bbox.GetRightBottomFront();
                x = cx.Dot(p);
                y = cy.Dot(p);
                if (minX > x) minX = x;
                if (maxX < x) maxX = x;
                if (minY > y) minY = y;
                if (maxY < y) maxY = y;

                p = bbox.GetRightTopBack();
                x = cx.Dot(p);
                y = cy.Dot(p);
                if (minX > x) minX = x;
                if (maxX < x) maxX = x;
                if (minY > y) minY = y;
                if (maxY < y) maxY = y;

                p = bbox.GetRightTopFront();
                x = cx.Dot(p);
                y = cy.Dot(p);
                if (minX > x) minX = x;
                if (maxX < x) maxX = x;
                if (minY > y) minY = y;
                if (maxY < y) maxY = y;

                call.AccessBoundingBoxes().SetBoundingBox(minX, minY, 0, maxX, maxY, 0);

                return true;
            }
            pgdc->Unlock();
        }
    }

    call.AccessBoundingBoxes().SetBoundingBox(-1.0f, -1.0f, 0, 1.0f, 1.0f, 0);
    return false;
}


/*
 * QuartzPlaneTexRenderer::release
 */
void QuartzPlaneTexRenderer::release(void) {
    AbstractTexQuartzRenderer::releaseTypeTexture();
    this->cryShader.Release();
}


/*
 * QuartzPlaneTexRenderer::Render
 */
bool QuartzPlaneTexRenderer::Render(core::view::CallRender2DGL& call) {
    ParticleGridDataCall *pgdc = this->getParticleData();
    CrystalDataCall *tdc = this->getCrystaliteData();
    core::view::CallClipPlane *ccp = this->getClipPlaneData();
    if ((pgdc == NULL) || (tdc == NULL) || (ccp == NULL)) {
        if (pgdc != NULL) pgdc->Unlock();
        if (tdc != NULL) tdc->Unlock();
        return false;
    }
    this->assertGrainColour();
    this->assertTypeTexture(*tdc);

    ::glDisable(GL_BLEND); // for now, may use in-shader super-sampling later on
    ::glDisable(GL_DEPTH_TEST);
    ::glDisable(GL_CULL_FACE);
    ::glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    ::glDisable(GL_LIGHTING);

    vislib::math::Vector<float, 3> cx, cy, cz;
    ccp->CalcPlaneSystem(cx, cy, cz);

    float viewportStuff[4];
    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    float shaderPointSize = vislib::math::Max(viewportStuff[2], viewportStuff[3]);
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    ::glPointSize(shaderPointSize);

    ::glEnableClientState(GL_VERTEX_ARRAY); // xyzr
    ::glEnableClientState(GL_TEXTURE_COORD_ARRAY); // quart

    if (this->useClipColSlot.Param<core::param::BoolParam>()->Value()) {
        ::glColor3ubv(ccp->GetColour());
    }
    else {
        ::glColor3fv(this->grainCol);
    }

    float planeZ = ccp->GetPlane().Distance(vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f));
    vislib::math::Cuboid<float> bbox(pgdc->GetBoundingBoxes().ObjectSpaceBBox());
    vislib::math::Point<float, 3> bboxmin(
        vislib::math::Min(bbox.Left(), bbox.Right()),
        vislib::math::Min(bbox.Bottom(), bbox.Top()),
        vislib::math::Min(bbox.Back(), bbox.Front()));
    vislib::math::Point<float, 3> bboxmax(
        vislib::math::Max(bbox.Left(), bbox.Right()),
        vislib::math::Max(bbox.Bottom(), bbox.Top()),
        vislib::math::Max(bbox.Back(), bbox.Front()));
    bool fixPBC = this->correctPBCSlot.Param<core::param::BoolParam>()->Value();
    if (!fixPBC) {
        bboxmin.Set(0.0f, 0.0f, 0.0f);
        bboxmax.Set(0.0f, 0.0f, 0.0f);
    }

    this->cryShader.Enable();
    ::glEnable(GL_LIGHTING);
    ::glPointSize(shaderPointSize);
    this->cryShader.SetParameterArray4("viewAttr", 1, viewportStuff);
    this->cryShader.SetParameterArray3("camX", 1, cx.PeekComponents());
    this->cryShader.SetParameterArray3("camY", 1, cy.PeekComponents());
    this->cryShader.SetParameterArray3("camZ", 1, cz.PeekComponents());
    this->cryShader.SetParameterArray3("bboxmin", 1, bboxmin.PeekCoordinates());
    this->cryShader.SetParameterArray3("bboxmax", 1, bboxmax.PeekCoordinates());
    this->cryShader.SetParameter("planeZ", planeZ);

    ::glActiveTexture(GL_TEXTURE0);
    ::glBindTexture(GL_TEXTURE_2D, this->typeTexture);
    this->cryShader.SetParameter("typeData", 0);

    for (int cellX = (fixPBC ? -1 : 0); cellX < static_cast<int>(pgdc->SizeX() + (fixPBC ? 1 : 0)); cellX++) {
        int ccx = cellX;
        float xoff = 0.0f;
        if (ccx < 0) {
            ccx = pgdc->SizeX() - 1;
            xoff -= bbox.Width();
        }
        if (ccx >= static_cast<int>(pgdc->SizeX())) {
            ccx = 0;
            xoff += bbox.Width();
        }

        for (int cellY = (fixPBC ? -1 : 0); cellY < static_cast<int>(pgdc->SizeY() + (fixPBC ? 1 : 0)); cellY++) {
            int ccy = cellY;
            float yoff = 0.0f;
            if (ccy < 0) {
                ccy = pgdc->SizeY() - 1;
                yoff -= bbox.Height();
            }
            if (ccy >= static_cast<int>(pgdc->SizeY())) {
                ccy = 0;
                yoff += bbox.Height();
            }

            for (int cellZ = (fixPBC ? -1 : 0); cellZ < static_cast<int>(pgdc->SizeZ() + (fixPBC ? 1 : 0)); cellZ++) {
                int ccz = cellZ;
                float zoff = 0.0f;
                if (ccz < 0) {
                    ccz = pgdc->SizeZ() - 1;
                    zoff -= bbox.Depth();
                }
                if (ccz >= static_cast<int>(pgdc->SizeZ())) {
                    ccz = 0;
                    zoff += bbox.Depth();
                }
                GL_VERIFY(this->cryShader.SetParameter("posoffset", xoff, yoff, zoff));

                unsigned int cellIdx = static_cast<unsigned int>(ccx
                    + pgdc->SizeX() * (ccy + pgdc->SizeY() * ccz));

                const ParticleGridDataCall::Cell& cell = pgdc->Cells()[cellIdx];

                bool hasPos = false, hasNeg = false;
                vislib::math::Cuboid<float> ccbox = cell.ClipBox();
                ccbox.Move(xoff, yoff, zoff);
                if (ccp->GetPlane().Halfspace(ccbox.GetRightTopFront()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetRightTopBack()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetRightBottomFront()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetRightBottomBack()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetLeftTopFront()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetLeftTopBack()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetLeftBottomFront()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetLeftBottomBack()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (!hasPos || !hasNeg) continue; // not visible cell

                for (unsigned int listIdx = 0; listIdx < cell.Count(); listIdx++) {
                    const ParticleGridDataCall::List& list = cell.Lists()[listIdx];
                    //if (list.Type() != 0) continue; // DEBUG!

                    this->cryShader.SetParameter("typeInfo",
                        static_cast<int>(list.Type()),
                        static_cast<int>(tdc->GetCrystals()[list.Type()].GetFaceCount()));
                    this->cryShader.SetParameter("outerRad",
                        tdc->GetCrystals()[list.Type()].GetBoundingRadius());

                    ::glVertexPointer(4, GL_FLOAT, 8 * sizeof(float), list.Data());
                    ::glTexCoordPointer(4, GL_FLOAT, 8 * sizeof(float), list.Data() + 4);
                    ::glDrawArrays(GL_POINTS, 0, list.Count());
                }
            }
        }
    }

    this->cryShader.Disable();
    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glDisableClientState(GL_VERTEX_ARRAY); // xyzr
    ::glDisableClientState(GL_TEXTURE_COORD_ARRAY); // quart

    if (this->showClipPlanePolySlot.Param<core::param::BoolParam>()->Value()) {
        ::glColor3ubv(ccp->GetColour());
        // cut plane with bbox and show outline
        ::glDisable(GL_LIGHTING);
        ::glEnable(GL_BLEND);
        ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        ::glEnable(GL_LINE_SMOOTH);
        vislib::math::Plane<float> px(1.0f, 0.0f, 0.0f, -bbox.Right());
        vislib::math::Plane<float> nx(-1.0f, 0.0f, 0.0f, bbox.Left());
        vislib::math::Plane<float> py(0.0f, 1.0f, 0.0f, -bbox.Top());
        vislib::math::Plane<float> ny(0.0f, -1.0f, 0.0f, bbox.Bottom());
        vislib::math::Plane<float> pz(0.0f, 0.0f, 1.0f, -bbox.Front());
        vislib::math::Plane<float> nz(0.0f, 0.0f, -1.0f, bbox.Back());
        const vislib::math::Plane<float>& cp(ccp->GetPlane());
        vislib::math::Point<float, 3> p;
        vislib::Array<vislib::math::Point<float, 3> > poly;
        bbox.Grow(bbox.LongestEdge() * 0.001f);

        if (px.CalcIntersectionPoint(py, cp, p) && bbox.Contains(p)) poly.Add(p);
        if (px.CalcIntersectionPoint(pz, cp, p) && bbox.Contains(p)) poly.Add(p);
        if (px.CalcIntersectionPoint(ny, cp, p) && bbox.Contains(p)) poly.Add(p);
        if (px.CalcIntersectionPoint(nz, cp, p) && bbox.Contains(p)) poly.Add(p);
        if (nx.CalcIntersectionPoint(py, cp, p) && bbox.Contains(p)) poly.Add(p);
        if (nx.CalcIntersectionPoint(pz, cp, p) && bbox.Contains(p)) poly.Add(p);
        if (nx.CalcIntersectionPoint(ny, cp, p) && bbox.Contains(p)) poly.Add(p);
        if (nx.CalcIntersectionPoint(nz, cp, p) && bbox.Contains(p)) poly.Add(p);
        if (py.CalcIntersectionPoint(pz, cp, p) && bbox.Contains(p)) poly.Add(p);
        if (py.CalcIntersectionPoint(nz, cp, p) && bbox.Contains(p)) poly.Add(p);
        if (ny.CalcIntersectionPoint(pz, cp, p) && bbox.Contains(p)) poly.Add(p);
        if (ny.CalcIntersectionPoint(nz, cp, p) && bbox.Contains(p)) poly.Add(p);

        if (poly.Count() > 0) {
            vislib::graphics::FlatPolygonSort(poly);

            ::glLineWidth(2.5f);
            ::glBegin(GL_LINE_LOOP);
            for (SIZE_T i = 0; i < poly.Count(); i++) {
                vislib::math::Vector<float, 3> v(poly[i].PeekCoordinates());
                ::glVertex2f(v.Dot(cx), v.Dot(cy));
            }
            ::glEnd();
        }
    }

    tdc->Unlock();
    pgdc->Unlock();

    return true;
}

} /* end namespace demos */
} /* end namespace megamol */
