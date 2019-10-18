/*
 * QuartzRenderer.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "QuartzRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/view/CallClipPlane.h"
#include "vislib/sys/Log.h"
#include "vislib/memutils.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/glfunctions.h"
#include "vislib/graphics/graphicsfunctions.h"
#include <cfloat>

namespace megamol {
namespace demos {

/*
 * QuartzRenderer::QuartzRenderer
 */
QuartzRenderer::QuartzRenderer(void)
    : core::view::Renderer3DModule_2()
    , AbstractMultiShaderQuartzRenderer()
    , showClipAxesSlot("showClipAxes", "Shows/Hides the axes (x and y) of the clipping plane") {

    this->showClipAxesSlot << new core::param::BoolParam(true);

    this->MakeSlotAvailable(&this->dataInSlot);
    this->MakeSlotAvailable(&this->typesInSlot);
    this->MakeSlotAvailable(&this->clipPlaneSlot);
    this->MakeSlotAvailable(&this->grainColSlot);
    this->MakeSlotAvailable(&this->showClipPlanePolySlot);
    this->MakeSlotAvailable(&this->showClipAxesSlot);
    this->MakeSlotAvailable(&this->correctPBCSlot);
}


/*
 * QuartzRenderer::~QuartzRenderer
 */
QuartzRenderer::~QuartzRenderer(void) {
    this->Release();
    ASSERT(this->shaders == NULL);
}


/*
 * QuartzRenderer::GetExtents
 */
bool QuartzRenderer::GetExtents(core::view::CallRender3D_2& call) {
    ParticleGridDataCall *pgdc = this->dataInSlot.CallAs<ParticleGridDataCall>();
    if ((pgdc != NULL) && ((*pgdc)(ParticleGridDataCall::CallForGetExtent))) {
        call.AccessBoundingBoxes().SetBoundingBox(pgdc->AccessBoundingBoxes().ObjectSpaceBBox());
        call.AccessBoundingBoxes().SetClipBox(pgdc->AccessBoundingBoxes().ObjectSpaceClipBox());
        pgdc->Unlock();

    } else {
        call.AccessBoundingBoxes().Clear();
    }

    call.SetTimeFramesCount(1); // I really don't want to support time-dependent data

    return true;
}


/*
 * QuartzRenderer::Render
 */
bool QuartzRenderer::Render(core::view::CallRender3D_2& call) {
    ParticleGridDataCall *pgdc = this->getParticleData();
    if (pgdc == NULL) return false;
    CrystalDataCall *tdc = this->getCrystaliteData();
    if (tdc== NULL) return false;
    this->assertGrainColour();
    core::view::CallClipPlane *ccp = this->getClipPlaneData();

    unsigned int shaderInitCnt = 3; // only initialize n shader(s) per frame

	// camera setup
	core::view::Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;

	// Generate complete snapshot and calculate matrices
    cam.calc_matrices(snapshot, viewTemp, projTemp, core::thecam::snapshot_content::all);
    glm::vec4 viewport = glm::vec4(0, 0, cam.resolution_gate().width(), cam.resolution_gate().height());
    if (viewport.z < 1.0f) viewport.z = 1.0f;
    if (viewport.w < 1.0f) viewport.w = 1.0f;
    viewport = glm::vec4(0, 0, 2.f / viewport.z, 2.f / viewport.w);
    float shaderPointSize = 1.f; vislib::math::Max(viewport.z, viewport.w);

    glm::vec4 camView = snapshot.view_vector;
    glm::vec4 camRight = snapshot.right_vector;
    glm::vec4 camUp = snapshot.up_vector;
	glm::vec4 camPos = snapshot.position;

	glm::mat4 view = viewTemp;
    glm::mat4 proj = projTemp;
    glm::mat4 MVinv = glm::inverse(view);
    glm::mat4 MVP = proj * view;
    glm::mat4 MVPinv = glm::inverse(MVP);
    glm::mat4 MVPtransp = glm::transpose(MVP);

	// color setup
    glm::vec4 grainColor = glm::vec4(this->grainCol[0], this->grainCol[1], this->grainCol[2], 1.f);
    glm::vec4 ambient = glm::vec4(0.2f, 0.2f, 0.2f, 1.f);
    glm::vec4 diffuse = glm::vec4(1.f);
    glm::vec4 specular = glm::vec4(1.f);

	// light setup
	this->GetLights();
    
    float lightIntensity = 1.f;
    glm::vec4 lightPos = {0.f, 0.f, 0.f, 1.f};
    glm::vec4 lightCol = {0.f, 0.f, 0.f, 1.f};
    
	// TODO use e.g. SSBOs to store lights and edit shader accordingly to support multiple lights
    if (this->lightMap.size() < 1) { // #lights added in configurator
        vislib::sys::Log::DefaultLog.WriteWarn("No lights available in lightmap");
    } else {
		for (auto light : this->lightMap) {
		   auto lc = this->lightMap.begin()->second.lightColor;
		   lightCol = glm::vec4(lc[0], lc[1], lc[2], lc[3]);
		   lightIntensity = this->lightMap.begin()->second.lightIntensity;

		   auto lightPosTemp = this->lightMap.begin()->second.pl_position;
		   if (lightPosTemp.size() == 3) {
		       lightPos[0] = lightPosTemp[0];
		       lightPos[1] = lightPosTemp[1];
		       lightPos[2] = lightPosTemp[2];
		   }
		   if (lightPosTemp.size() == 4) {
               lightPos[0] = lightPosTemp[0];
               lightPos[1] = lightPosTemp[1];
               lightPos[2] = lightPosTemp[2];
		       lightPos[3] = lightPosTemp[3];
		   }
		}
	}
    

    ::glDisable(GL_BLEND);
    ::glEnable(GL_DEPTH_TEST);
    ::glEnable(GL_CULL_FACE);
    ::glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    ::glBindBuffer(GL_ARRAY_BUFFER, 0); // TODO IMO necessary, but Karsten might have fixed it
    ::glEnableClientState(GL_VERTEX_ARRAY); // xyzr
    ::glEnableClientState(GL_TEXTURE_COORD_ARRAY); // quart

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

                unsigned int cellIdx = static_cast<unsigned int>(ccx
                    + pgdc->SizeX() * (ccy + pgdc->SizeY() * ccz));

                const ParticleGridDataCall::Cell& cell = pgdc->Cells()[cellIdx];

                if (ccp != NULL) {
                    bool hasPos = false;
                    vislib::math::Cuboid<float> ccbox = cell.ClipBox();
                    ccbox.Move(xoff, yoff, zoff);
                    if (ccp->GetPlane().Halfspace(ccbox.GetRightTopFront()) == 
                            vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                        hasPos = true;
                    } else if (ccp->GetPlane().Halfspace(ccbox.GetRightTopBack()) == 
                            vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                        hasPos = true;
                    } else if (ccp->GetPlane().Halfspace(ccbox.GetRightBottomFront()) == 
                            vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                        hasPos = true;
                    } else if (ccp->GetPlane().Halfspace(ccbox.GetRightBottomBack()) == 
                            vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                        hasPos = true;
                    } else if (ccp->GetPlane().Halfspace(ccbox.GetLeftBottomBack()) == 
                            vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                        hasPos = true;
                    } else if (ccp->GetPlane().Halfspace(ccbox.GetLeftBottomFront()) == 
                            vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                        hasPos = true;
                    } else if (ccp->GetPlane().Halfspace(ccbox.GetLeftTopBack()) == 
                            vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                        hasPos = true;
                    } else if (ccp->GetPlane().Halfspace(ccbox.GetLeftTopFront()) == 
                            vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                        hasPos = true;
                    }
                    if (!hasPos) continue;
                }

                //::glColor3ub(255, 127, 0);
                //vislib::graphics::gl::DrawCuboidLines(cell.ClipBox());
                //::glColor3fv(this->grainCol);

                for (unsigned int l = 0; l < cell.Count(); l++) {
                    const ParticleGridDataCall::List& list = cell.Lists()[l];
                    //if (list.Type() != 0) continue; // TODO: DEBUG! Remove me!

                    vislib::graphics::gl::GLSLShader *shader = this->shaders[list.Type() % this->cntShaders];
                    if ((shader == NULL) && (shaderInitCnt > 0)) {
                        unsigned int t = list.Type() % this->cntShaders;
                        try {
                            shader = this->shaders[t]
                                = this->makeShader(tdc->GetCrystals()[t]);
                        } catch(...) {
                        }
                        shaderInitCnt--;
                    }

                    if (shader != NULL) {
                        shader->Enable();
                        ::glPointSize(shaderPointSize);
                        ::glUniform4fv(shader->ParameterLocation("viewAttr"), 1,
                            glm::value_ptr(viewport));
                        ::glUniform3fv(shader->ParameterLocation("camIn"), 1,
                            glm::value_ptr(glm::vec3(camView.x, camView.y, camView.z)));
                        ::glUniform3fv(shader->ParameterLocation("camRight"), 1, 
							glm::value_ptr(glm::vec3(camRight.x, camRight.y, camRight.z)));
                        ::glUniform3fv(shader->ParameterLocation("camUp"), 1, 
							glm::value_ptr(glm::vec3(camUp.x, camUp.y, camUp.z)));
                        ::glUniform4fv(shader->ParameterLocation("camPosInit"), 1,glm::value_ptr(camPos));
                        ::glUniform4fv(shader->ParameterLocation("lightPos"), 1, glm::value_ptr(lightPos));
                        ::glUniform4fv(shader->ParameterLocation("color"), 1, glm::value_ptr(grainColor));
                        ::glUniform4fv(shader->ParameterLocation("ambientCol"), 1, glm::value_ptr(ambient));
                        ::glUniform4fv(shader->ParameterLocation("diffuseCol"), 1, glm::value_ptr(diffuse));
                        ::glUniform4fv(shader->ParameterLocation("specularCol"), 1, glm::value_ptr(specular));
                        ::glUniform1f(shader->ParameterLocation("lightIntensity"), lightIntensity);
                        ::glUniformMatrix4fv(shader->ParameterLocation("ModelViewMatrixInverse"), 1, 
							GL_FALSE, glm::value_ptr(MVinv));
                        ::glUniformMatrix4fv(shader->ParameterLocation("ModelViewMatrixInverseTranspose"), 1, 
							GL_FALSE, glm::value_ptr(glm::transpose(MVinv)));
                        ::glUniformMatrix4fv(shader->ParameterLocation("ModelViewProjectionMatrix"), 1, 
							GL_FALSE, glm::value_ptr(MVP));
                        ::glUniformMatrix4fv(shader->ParameterLocation("ModelViewProjectionMatrixInverse"), 1, 
							GL_FALSE, glm::value_ptr(MVPinv));
                        ::glUniformMatrix4fv(shader->ParameterLocation("ModelViewProjectionMatrixTranspose"), 1, 
							GL_FALSE, glm::value_ptr(MVPtransp));
                        if (ccp != NULL) {
                            shader->SetParameter("clipcol",
                                static_cast<float>(ccp->GetColour()[0]) / 255.0f,
                                static_cast<float>(ccp->GetColour()[1]) / 255.0f,
                                static_cast<float>(ccp->GetColour()[2]) / 255.0f);
                        }
                        shader->SetParameterArray3("bboxmin", 1, bboxmin.PeekCoordinates());
                        shader->SetParameterArray3("bboxmax", 1, bboxmax.PeekCoordinates());
                        shader->SetParameter("posoffset", xoff, yoff, zoff);

                    } else {
                        ::glPointSize(4.0f);
                        this->errShader.Enable();
                        shader = &this->errShader;
                    }
                    if (ccp != NULL) {
                        shader->SetParameter("clipplane", ccp->GetPlane().A(),
                            ccp->GetPlane().B(), ccp->GetPlane().C(), ccp->GetPlane().D());
                    } else {
                        shader->SetParameter("clipplane", 0.0f, 0.0f, 0.0f, 0.0f);
                    }

                    ::glVertexPointer(4, GL_FLOAT, 8 * sizeof(float), list.Data());
                    ::glTexCoordPointer(4, GL_FLOAT, 8 * sizeof(float), list.Data() + 4);
                    ::glDrawArrays(GL_POINTS, 0, list.Count());

                    shader->Disable();
                }

            }
        }
    }

    ::glDisableClientState(GL_VERTEX_ARRAY); // xyzr
    ::glDisableClientState(GL_TEXTURE_COORD_ARRAY); // quart

    if ((ccp != NULL) && (
            (this->showClipPlanePolySlot.Param<core::param::BoolParam>()->Value())
            || (this->showClipAxesSlot.Param<core::param::BoolParam>()->Value()))) {
        ::glColor3ubv(ccp->GetColour());
        // cut plane with bbox and show outline
        ::glEnable(GL_BLEND);
        ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        ::glEnable(GL_LINE_SMOOTH);
        vislib::math::Cuboid<float> bbox(pgdc->GetBoundingBoxes().ObjectSpaceBBox());
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
            if (this->showClipPlanePolySlot.Param<core::param::BoolParam>()->Value()) {
                ::glBegin(GL_LINE_LOOP);
                for (SIZE_T i = 0; i < poly.Count(); i++) {
                    ::glVertex3fv(poly[i].PeekCoordinates());
                }
                ::glEnd();
            }

            if (this->showClipAxesSlot.Param<core::param::BoolParam>()->Value()) {
                p = poly[0];
                for (SIZE_T i = 1; i < poly.Count(); i++) p.Set(p.X() + poly[i].X(), p.Y() + poly[i].Y(), p.Z() + poly[i].Z());
                p.Set(p.X() / static_cast<float>(poly.Count()),
                    p.Y() / static_cast<float>(poly.Count()),
                    p.Z() / static_cast<float>(poly.Count()));
                float l = FLT_MAX;
                for (SIZE_T i = 0; i < poly.Count(); i++) {
                    float d = (p - poly[i]).Length();
                    if (d < l) l = d;
                }

                vislib::math::Vector<float, 3> cx, cy;
                ccp->CalcPlaneSystem(cx, cy);

                cx *= l * 0.2f;
                cy *= l * 0.2f;

                ::glBegin(GL_LINES);
                ::glColor3ub(255, 0, 0);
                ::glVertex3f(p.X(), p.Y(), p.Z());
                ::glVertex3f(p.X() + cx.X(), p.Y() + cx.Y(), p.Z() + cx.Z());
                ::glColor3ub(0, 255, 0);
                ::glVertex3f(p.X(), p.Y(), p.Z());
                ::glVertex3f(p.X() + cy.X(), p.Y() + cy.Y(), p.Z() + cy.Z());
                ::glEnd();

            }
        }

    }

    if (tdc) tdc->Unlock();
    pgdc->Unlock();

    return true;
}


/*
 * QuartzRenderer::create
 */
bool QuartzRenderer::create(void) {
    using vislib::graphics::gl::GLSLShader;
    using vislib::sys::Log;
    using vislib::graphics::gl::ShaderSource;

    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        vislib::sys::Log::DefaultLog.WriteError("Failed to initialise OpenGL GLSL Shader");
        return false;
    }

    ShaderSource vert, frag;
    try {
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quartz::simple::errVert", vert)) {
            throw vislib::Exception("Generic vertex shader build failure", __FILE__, __LINE__);
        }
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quartz::simple::frag", frag)) {
            throw vislib::Exception("Generic fragment shader build failure", __FILE__, __LINE__);
        }
        if (!this->errShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            throw vislib::Exception("Generic shader create failure", __FILE__, __LINE__);
        }
    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteError("Unable to compile shader: %s", ex.GetMsgA());
        this->release(); // Because I know that 'release' ONLY releases all the shaders
        return false;
    } catch(...) {
        Log::DefaultLog.WriteError("Unable to compile shader: Unexpected Exception");
        this->release(); // Because I know that 'release' ONLY releases all the shaders
        return false;
    }

    return true;
}


/*
 * QuartzRenderer::release
 */
void QuartzRenderer::release(void) {
    this->releaseShaders();
    this->errShader.Release();
}


/*
 * QuartzRenderer::makeShader
 */
vislib::graphics::gl::GLSLShader* QuartzRenderer::makeShader(const CrystalDataCall::Crystal& c) {
    using vislib::graphics::gl::GLSLShader;
    using vislib::sys::Log;
    using vislib::graphics::gl::ShaderSource;

    GLSLShader *s = new GLSLShader();

    c.AssertMesh();
    const float *v = c.GetMeshVertexData();

    ShaderSource vert, frag;

    try {

        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quartz::ray::vertclipped", vert)) {
            throw vislib::Exception("Generic vertex shader build failure", __FILE__, __LINE__);
        }
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quartz::ray::fragclipped", frag)) {
            throw vislib::Exception("Generic fragment shader build failure", __FILE__, __LINE__);
        }

        vislib::StringA str, line;

        str.Format("#define OUTERRAD %f\n", c.GetBoundingRadius());
        vert.Replace(1 /* HAZARD */, new ShaderSource::StringSnippet(str));

        str = "vec2 ll;\n"\
            "vec2 lams = vec2(-1000000.0, 1000000.0);\n";
        for (unsigned int j = 0; j < c.GetFaceCount(); j++) {
            vislib::math::Vector<float, 3> n(c.GetFace(j));
            n.Normalise();
            line.Format("faceNormal = vec3(%f, %f, %f);\n", n.X(), n.Y(), n.Z());
            str += line;
            line.Format("ll = planeCast(faceNormal, rad * %f, ray);\n", c.GetFace(j).Length());
            line.Append("if (ll.y < 0.0) {\n"\
                "  // hit from front\n"\
                "  if (lams.x < ll.x) {\n"\
                "    lams.x = ll.x;\n"\
                "    normal = faceNormal;\n"\
                "  }\n"\
                "}\n"\
                "if (ll.y > 0.0) {\n"\
                "  // hit from behind\n"\
                "  if (lams.y > ll.x) {\n"\
                "    lams.y = ll.x;\n"\
                "  }\n"\
                "}\n");
            str += line;
        }
        str.Append("if (lams.y < lams.x) discard;\n");
        str.Append("lambda = lams.x;\n");
        frag.Replace(5 /* HAZARD */, new ShaderSource::StringSnippet(str));

        if (!s->Compile(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            throw vislib::Exception("Generic compilation failure", __FILE__, __LINE__);
        }
        if (!s->Link()) {
            throw vislib::Exception("Generic linking failure", __FILE__, __LINE__);
        }

    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteError("Unable to compile shader: %s", ex.GetMsgA());
        delete s;
        return NULL;
    } catch(...) {
        Log::DefaultLog.WriteError("Unable to compile shader: Unexpected Exception");
        delete s;
        return NULL;
    }

    return s;
}

} /* end namespace demos */
} /* end namespace megamol */