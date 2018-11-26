//
// SphereRendererMouse.cpp
//
// Copyright (C) 2012 by University of Stuttgart (VISUS).
// All rights reserved.
//

#include "stdafx.h"

#define _USE_MATH_DEFINES
#include "SphereRendererMouse.h"

#include <math.h>

#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>

#include "mmcore/CoreInstance.h"
#include "protein_calls/MolecularDataCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/AbstractCallRender3D.h"

#include "vislib/math/Matrix.h"
#include "vislib/math/Vector.h"

using namespace megamol;
using namespace megamol::protein_calls;

/*
 * protein::special::SphereRendererMouse::SphereRendererMouse
 */
protein::SphereRendererMouse::SphereRendererMouse() : Renderer3DModuleMouse(),
        molDataCallerSlot("getData", "Connects the molecule rendering with molecule data storage."),
        sphereRadSclParam("sphereRadScl", "Scale factor for the sphere radius."),
        atomColParam("atomColor", "The color of unselected atoms." ),
        atomColHoverParam("atomColorHover", "The color for atoms which are hovered." ),
        atomColSelParam("atomColorSel", "The color for currently selected atoms." ),
        useGeomShaderParam("useGeomShader", "Use geometry shader for glyph ray casting." ),
        showSelRectParam("showSelRect", "Show rectangle when selecting." ),
        atomCnt(0),
        mouseX(-1), mouseY(-1),
        startSelect(-1, -1), endSelect(-1, -1),
        drag(false), resetSelection(true) {

    // Data caller slot
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->molDataCallerSlot);

    // Param slot for sphere radius
    this->sphereRadSclParam << new core::param::FloatParam(1.0f, 0.0);
    this->MakeSlotAvailable(&this->sphereRadSclParam);

    // General atom color
    this->atomColParam.SetParameter(new core::param::StringParam("#6095c6"));
    this->MakeSlotAvailable(&this->atomColParam);

    // Hover atom color
    this->atomColHoverParam.SetParameter(new core::param::StringParam("#FFFFFF"));
    this->MakeSlotAvailable( &this->atomColHoverParam);

    // Atom color for selected atoms
    this->atomColSelParam.SetParameter(new core::param::StringParam("#21c949"));
    this->MakeSlotAvailable(&this->atomColSelParam);

    // Param slot for sphere radius
    this->useGeomShader = false;
    this->useGeomShaderParam << new core::param::BoolParam(this->useGeomShader);
    this->MakeSlotAvailable(&this->useGeomShaderParam);

    // Param slot for selection rectangle
    this->showSelRect = true;
    this->showSelRectParam << new core::param::BoolParam(this->showSelRect);
    this->MakeSlotAvailable(&this->showSelRectParam);
}


/*
 * protein::SphereRendererMouse::~SphereRendererMouse
 */
protein::SphereRendererMouse::~SphereRendererMouse() {
    this->Release();
}


/*
 * protein::SphereRendererMouse::create
 */
bool protein::SphereRendererMouse::create(void) {
    if (!areExtsAvailable( "GL_EXT_gpu_shader4 GL_EXT_geometry_shader4 GL_EXT_bindable_uniform")
        || !ogl_IsVersionGEQ(2,0)
        || !areExtsAvailable( "GL_ARB_vertex_shader GL_ARB_vertex_program GL_ARB_shader_objects")
        || !vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc, geomSrc, fragSrc;

    // Load sphere shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    try {
        if (!this->sphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch(vislib::Exception &e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load alternative sphere shader (uses geometry shader)
    //if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereVertexGeom", vertSrc)) {
    //    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
    //    return false;
    //}
    //if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereGeom", geomSrc)) {
    //    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load geometry shader source for sphere shader");
    //    return false;
    //}
    //if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereFragmentGeom", fragSrc)) {
    //    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for sphere shader");
    //    return false;
    //}
    //try {
    //    this->sphereShaderGeo.Compile( vertSrc.Code(), vertSrc.Count(), geomSrc.Code(),
    //        geomSrc.Count(), fragSrc.Code(), fragSrc.Count());
    //    this->sphereShaderGeo.Link();
    //}
    //catch(vislib::Exception &e) {
    //    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create sphere geometry shader: %s\n", e.GetMsgA());
    //    return false;
    //}

    return true;
}


/*
 * protein::SphereRendererMouse::GetExtents
 */
bool protein::SphereRendererMouse::GetExtents(core::Call& call) {

    core::view::AbstractCallRender3D *cr3d = dynamic_cast<core::view::AbstractCallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL ) return false;
    if (!(*mol)(MolecularDataCall::CallForGetExtent)) return false;

    float scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    cr3d->AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
    cr3d->SetTimeFramesCount(mol->FrameCount());

    this->atomCnt = mol->AtomCount();

    return true;
}


/*
 * protein::SphereRendererMouse::Render
 */
bool protein::SphereRendererMouse::Render(core::Call& call) {
    using namespace vislib::math;

    // Update parameters
    if(this->useGeomShaderParam.IsDirty() ) {
        this->useGeomShader = this->useGeomShaderParam.Param<core::param::BoolParam>()->Value();
    }
    if(this->showSelRectParam.IsDirty() ) {
        this->showSelRect = this->showSelRectParam.Param<core::param::BoolParam>()->Value();
    }

    // cast the call to Render3D
    core::view::AbstractCallRender3D *cr3d =
            dynamic_cast<core::view::AbstractCallRender3D *>(&call);

    if(cr3d == NULL) return false;

    // Get camera information of render call
    this->cameraInfo = cr3d->GetCameraParameters();

    // Get calltime of render call
    float callTime = cr3d->Time();

    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL) return false;

    int cnt;

    // Set call time in data call
    mol->SetCalltime(callTime);

    // Set frame ID in data call and call data
    mol->SetFrameID(static_cast<int>( callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) return false;

    // Check if atom count is zero
    if( mol->AtomCount() == 0 ) return true;

    // get positions of the first frame
    float *pos0 = new float[mol->AtomCount() * 3];
    memcpy(pos0, mol->AtomPositions(), mol->AtomCount()*3*sizeof(float));

    // set next frame ID and get positions of the second frame
    if((static_cast<int>(callTime)+1) < int(mol->FrameCount()))
        mol->SetFrameID(static_cast<int>(callTime)+1);
    else
        mol->SetFrameID(static_cast<int>( callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) {
        delete[] pos0;
        return false;
    }
    float *pos1 = new float[mol->AtomCount()*3];
    memcpy(pos1, mol->AtomPositions(), mol->AtomCount()*3*sizeof(float));

    // interpolate atom positions between frames
    float *posInter = new float[mol->AtomCount()*4];
    float inter = callTime - static_cast<float>(static_cast<int>(callTime));
    float threshold = vislib::math::Min(mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
            vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
                    mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
    for(cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
        if(std::sqrt( std::pow( pos0[3*cnt+0] - pos1[3*cnt+0], 2) +
                std::pow( pos0[3*cnt+1] - pos1[3*cnt+1], 2) +
                std::pow( pos0[3*cnt+2] - pos1[3*cnt+2], 2) ) < threshold ) {
            posInter[4*cnt+0] = (1.0f - inter) * pos0[3*cnt+0] + inter * pos1[3*cnt+0];
            posInter[4*cnt+1] = (1.0f - inter) * pos0[3*cnt+1] + inter * pos1[3*cnt+1];
            posInter[4*cnt+2] = (1.0f - inter) * pos0[3*cnt+2] + inter * pos1[3*cnt+2];
        } else if(inter < 0.5f) {
            posInter[4*cnt+0] = pos0[3*cnt+0];
            posInter[4*cnt+1] = pos0[3*cnt+1];
            posInter[4*cnt+2] = pos0[3*cnt+2];
        } else {
            posInter[4*cnt+0] = pos1[3*cnt+0];
            posInter[4*cnt+1] = pos1[3*cnt+1];
            posInter[4*cnt+2] = pos1[3*cnt+2];
        }
        posInter[4*cnt+3] = this->sphereRadSclParam.Param<core::param::FloatParam>()->Value();
    }

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    // compute scale factor and scale world
    float scale;
    if(!vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef(scale, scale, scale);

    float viewportStuff[4] = {
            this->cameraInfo->TileRect().Left(),
            this->cameraInfo->TileRect().Bottom(),
            this->cameraInfo->TileRect().Width(),
            this->cameraInfo->TileRect().Height()};
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);


    if(this->resetSelection) {
        // Reset selections
        this->atomSelect.SetCount(mol->AtomCount());
#pragma omp parallel for
        for(int at = 0; at < static_cast<int>(mol->AtomCount()); at++) {
            this->atomSelect[at] = false;
        }
        this->resetSelection = false;
    }

    // Get GL_MODELVIEW matrix
    GLfloat modelMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelMatrix_column);
    Matrix<GLfloat, 4, COLUMN_MAJOR> modelMatrix(&modelMatrix_column[0]);

    // Get GL_PROJECTION matrix
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    Matrix<GLfloat, 4, COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);

    // Compute modelviewprojection matrix
    Matrix<GLfloat, 4, ROW_MAJOR> modelProjMatrix = projMatrix*modelMatrix;

    // Get light position
    GLfloat lightPos[4];
    glGetLightfv(GL_LIGHT0, GL_POSITION, lightPos);


    // Apply positional filter to all atoms if dragging is enabled
    if(this->drag) {

        // Get GL_VIEWPORT
        int viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);

        // Compute NDC coordinates of the rectangle
        float rectNDC[4];
        rectNDC[0] = static_cast<float>(this->startSelect.X())/static_cast<float>(viewport[2]);
        rectNDC[0] -= 0.5f;
        rectNDC[0] *= 2.0f;

        rectNDC[1] = static_cast<float>(this->endSelect.X())/static_cast<float>(viewport[2]);
        rectNDC[1] -= 0.5f;
        rectNDC[1] *= 2.0f;

        rectNDC[2] = 1.0f - static_cast<float>(this->startSelect.Y())/static_cast<float>(viewport[3]);
        rectNDC[2] -= 0.5f;
        rectNDC[2] *= 2.0f;

        rectNDC[3] = 1.0f - static_cast<float>(this->endSelect.Y())/static_cast<float>(viewport[3]);
        rectNDC[3] -= 0.5f;
        rectNDC[3] *= 2.0f;

        // Sort coordinates
        float swap;
        if(rectNDC[0] > rectNDC[1]) {
            swap = rectNDC[1];
            rectNDC[1] = rectNDC[0];
            rectNDC[0] = swap;
        }
        if(rectNDC[2] > rectNDC[3]) {
            swap = rectNDC[2];
            rectNDC[2] = rectNDC[3];
            rectNDC[3] = swap;
        }
        /*printf("RECT (%i %i) (%i %i) | (%f %f) (%f %f)\n", this->startSelect.X(), this->startSelect.Y(),
                this->endSelect.X(), this->endSelect.Y(), rectNDC[0], rectNDC[2],
                rectNDC[1], rectNDC[3]);*/


#pragma omp parallel for
        for(int at = 0; at < static_cast<int>(mol->AtomCount()); at++) {

            this->atomSelect[at] = false;

            // Get atom (object space) position
            Vector<float, 4> posOS;
            posOS.SetX(mol->AtomPositions()[at*3+0]);
            posOS.SetY(mol->AtomPositions()[at*3+1]);
            posOS.SetZ(mol->AtomPositions()[at*3+2]);
            posOS.SetW(1.0f);

            // Compute eye space position
            Vector<float, 4> posES;
            posES = modelProjMatrix*posOS;

            // Compute normalized device coordinates
            Vector<float, 3> posNDC;
            posNDC.SetX(posES.X()/posES.W());
            posNDC.SetY(posES.Y()/posES.W());
            posNDC.SetZ(posES.Z()/posES.W());

            //printf("#%i PosNDC (%f %f %f)\n", at, posNDC.X(), posNDC.Y(), posNDC.Z());

            if(posNDC.X() < rectNDC[0]) continue;
            if(posNDC.X() > rectNDC[1]) continue;
            if(posNDC.Y() < rectNDC[2]) continue;
            if(posNDC.Y() > rectNDC[3]) continue;

            this->atomSelect[at] = true;
        }

    }

    //unsigned int atomSelCnt = 0;

    // Set color
    this->atomColor.SetCount(mol->AtomCount()*3);
#pragma omp parallel for
    for(int at = 0; at < static_cast<int>(mol->AtomCount()); at++) {
        if(this->atomSelect[at] == false) { // Atom is not selected
            float r, g, b;
            core::utility::ColourParser::FromString(
                    this->atomColParam.Param<core::param::StringParam>()->Value(), r, g, b);
            this->atomColor[3*at+0] = r;
            this->atomColor[3*at+1] = g;
            this->atomColor[3*at+2] = b;
        }
        else { // Atom is selected
            float r, g, b;
            core::utility::ColourParser::FromString(
                    this->atomColSelParam.Param<core::param::StringParam>()->Value(), r, g, b);
            this->atomColor[3*at+0] = r;
            this->atomColor[3*at+1] = g;
            this->atomColor[3*at+2] = b;
        }
    }

    if(this->useGeomShader) { // Use geometry shader
        // Enable sphere shader
        this->sphereShaderGeo.Enable();

        // Set shader variables
        glUniform4fvARB(this->sphereShaderGeo.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fvARB(this->sphereShaderGeo.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
        glUniform3fvARB(this->sphereShaderGeo.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
        glUniform3fvARB(this->sphereShaderGeo.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());

        glUniformMatrix4fvARB(this->sphereShaderGeo.ParameterLocation("modelview"), 1, false, modelMatrix_column);
        glUniformMatrix4fvARB(this->sphereShaderGeo.ParameterLocation("proj"), 1, false, projMatrix_column);
        glUniform4fvARB(this->sphereShaderGeo.ParameterLocation("lightPos"), 1, lightPos);

        // Vertex attributes
        GLint vertexPos = glGetAttribLocation(this->sphereShaderGeo, "vertex");
        GLint vertexColor = glGetAttribLocation(this->sphereShaderGeo, "color");

        // Enable arrays for attributes
        glEnableVertexAttribArray(vertexPos);
        glEnableVertexAttribArray(vertexColor);

        // Set attribute pointers
        glVertexAttribPointer(vertexPos, 4, GL_FLOAT, GL_FALSE, 0, posInter);
        glVertexAttribPointer(vertexColor, 3, GL_FLOAT, GL_FALSE, 0, this->atomColor.PeekElements());

        // Draw points
        glDrawArrays(GL_POINTS, 0, mol->AtomCount());
        //glDrawArrays(GL_POINTS, 0, 1);

        // Disable arrays for attributes
        glDisableVertexAttribArray(vertexPos);
        glDisableVertexAttribArray(vertexColor);

        // Disable sphere shader
        this->sphereShaderGeo.Disable();
    }
    else { // Use point sprites
        // Enable sphere shader
        this->sphereShader.Enable();
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        // set shader variables
        glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
        glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
        glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());

        // Draw points
        glVertexPointer(4, GL_FLOAT, 0, posInter);
        glColorPointer(3, GL_FLOAT, 0, this->atomColor.PeekElements());
        glDrawArrays(GL_POINTS, 0, mol->AtomCount());
        //glDrawArrays(GL_POINTS, 0, 1);

        // disable sphere shader
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        this->sphereShader.Disable();
    }

    delete[] pos0;
    delete[] pos1;
    delete[] posInter;

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // Render rectangle

    float curVP[4];
    glGetFloatv(GL_VIEWPORT, curVP);

    // Draw rectangle if draging is enabled
    if((this->drag)&&(this->showSelRect)) {

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glDisable(GL_CULL_FACE);
        glEnable(GL_BLEND);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();

        // This sets up the OpenGL window so that (0,0) corresponds to the top left corner
        //printf("Current viewport (%f %f %f %f)\n", curVP[0], curVP[1], curVP[2], curVP[3]); // DEBUG
        glOrtho(curVP[0], curVP[2], curVP[3], curVP[1], -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        // Draw transparent quad
        glPolygonMode(GL_FRONT_AND_BACK,  GL_FILL);
        glColor4f(1.0f, 1.0f, 1.0f, 0.2f);
        glBegin(GL_QUADS);
            glVertex2i(this->startSelect.X(), this->startSelect.Y());
            glVertex2i(this->endSelect.X(), this->startSelect.Y());
            glVertex2i(this->endSelect.X(),   this->endSelect.Y());
            glVertex2i(this->startSelect.X(),   this->endSelect.Y());
        glEnd();

        // Draw outline
        glPolygonMode(GL_FRONT_AND_BACK,  GL_LINE);
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        glBegin(GL_QUADS);
            glVertex2i(this->startSelect.X(), this->startSelect.Y());
            glVertex2i(this->endSelect.X(), this->startSelect.Y());
            glVertex2i(this->endSelect.X(),   this->endSelect.Y());
            glVertex2i(this->startSelect.X(),   this->endSelect.Y());
        glEnd();

        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);

        glEnable(GL_CULL_FACE);
        glDisable(GL_BLEND);
    }

    return true;
}


/*
 * protein::SphereRendererMouse::release
 */
void protein::SphereRendererMouse::release(void) {
    // intentionally empty
}


/*
 * protein::SphereRendererMouse::MouseEvent
 */
bool protein::SphereRendererMouse::MouseEvent(int x, int y, core::view::MouseFlags flags) {

    this->mouseX = x;
    this->mouseY = y;

    //printf("POS (%i %i)\n", this->mouseX, this->mouseY); // DEBUG

    if ((flags & core::view::MOUSEFLAG_BUTTON_LEFT_DOWN) != 0) {
        if(!this->drag) { // Start dragging
            this->endSelect.Set(this->mouseX, this->mouseY);
            this->startSelect.Set(this->mouseX, this->mouseY);
            this->resetSelection = true;
            this->drag = true;
        }
        else { // Dragging is now enabled
            this->endSelect.Set(this->mouseX, this->mouseY);
        }
    }
    else {
        this->drag = false;
    }
    return true;
}
