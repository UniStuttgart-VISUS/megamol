/*
 * CartoonRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "CartoonRenderer.h"
#include "mmcore/CoreInstance.h"
#include "Color.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/math/Quaternion.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/AbstractOpenGLShader.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/StringConverter.h"
#include "vislib/math/Matrix.h"
#include <GL/glu.h>
#include <omp.h>

#define RENDER_ATOMS_AS_SPHERES 1

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

/*
 * protein::CartoonRenderer::CartoonRenderer (CTOR)
 */
CartoonRenderer::CartoonRenderer(void)
        : Renderer3DModuleDS(), 
        molDataCallerSlot("getData", "Connects the molecule rendering with molecule data storage"), 
        bsDataCallerSlot("getBindingSites", "Connects the molecule rendering with binding site data storage"),
        PositionSlot(0) {
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataCallerSlot);
    this->bsDataCallerSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable(&this->bsDataCallerSlot);

}

/*
 * protein::CartoonRenderer::~CartoonRenderer (DTOR)
 */
CartoonRenderer::~CartoonRenderer(void) {
    this->Release();
}

/*
 * protein::CartoonRenderer::release
 */
void CartoonRenderer::release(void) {

}

/*
 * protein::CartoonRenderer::create
 */
bool CartoonRenderer::create(void) {
    if (!ogl_IsVersionGEQ(2,0))
        return false;

    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions())
        return false;
    if (!vislib::graphics::gl::GLSLTesselationShader::InitialiseExtensions())
        return false;

    glEnable (GL_DEPTH_TEST);
    glDepthFunc (GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;
    ShaderSource geomSrc;
    ShaderSource tessContSrc;
    ShaderSource tessEvalSrc;

    // Load sphere shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
        "protein::std::sphereVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to load vertex shader source for sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
        "protein::std::sphereFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to load vertex shader source for sphere shader");
        return false;
    }
    try {
        if (!this->sphereShader.Create(vertSrc.Code(), vertSrc.Count(),
            fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__,
                __LINE__);
        }
    }
    catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to create sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    // Load cartoon shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("proteincartoon::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for cartoon shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("proteincartoon::tesscontrol", tessContSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load tessellation control shader source for cartoon shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("proteincartoon::tesseval", tessEvalSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load tessellation evaluation shader source for cartoon shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("proteincartoon::geometry", geomSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load geometry shader source for cartoon shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("proteincartoon::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for cartoon shader");
        return false;
    }
    try {
        // compile the shader
        if (!this->cartoonShader.Compile(vertSrc.Code(), vertSrc.Count(),
            tessContSrc.Code(), tessContSrc.Count(),
            tessEvalSrc.Code(), tessEvalSrc.Count(),
            0, 0,
            fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Could not compile cartoon shader. ", __FILE__, __LINE__);
        }
        // link the shader
        if (!this->cartoonShader.Link()){
            throw vislib::Exception("Could not link cartoon shader", __FILE__, __LINE__);
        }
    }
    catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create cartoon shader: %s\n", e.GetMsgA());
        return false;
    }

    return true;
}


/*
 * protein::CartoonRenderer::GetExtents
 */
bool CartoonRenderer::GetExtents(Call& call) {
    view::AbstractCallRender3D *cr3d =
            dynamic_cast<view::AbstractCallRender3D *>(&call);
    if (cr3d == NULL)
        return false;

    MolecularDataCall *mol =
            this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL)
        return false;
    if (!(*mol)(MolecularDataCall::CallForGetExtent))
        return false;

    float scale;
    if (!vislib::math::IsEqual(
            mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        scale = 2.0f
                / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    cr3d->AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
    cr3d->SetTimeFramesCount(mol->FrameCount());

    return true;
}

/**********************************************************************
 * 'render'-functions
 **********************************************************************/

/*
 * protein::CartoonRenderer::Render
 */
bool CartoonRenderer::Render(Call& call) {
    // cast the call to Render3D
    view::AbstractCallRender3D *cr3d =
            dynamic_cast<view::AbstractCallRender3D *>(&call);
    if (cr3d == NULL)
        return false;

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    float callTime = cr3d->Time();

    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL)
        return false;
    
    // get pointer to BindingSiteCall
    BindingSiteCall *bs = this->bsDataCallerSlot.CallAs<BindingSiteCall>();
    if (bs) {
        (*bs)(BindingSiteCall::CallForGetData);
    }

    //int cnt;

    // set call time
    mol->SetCalltime(callTime);
    // set frame ID and call data
    mol->SetFrameID(static_cast<int>(callTime));

    if (!(*mol)(MolecularDataCall::CallForGetData))
        return false;
    
    glPushMatrix();
    // compute scale factor and scale world
    float scale;
    if (!vislib::math::IsEqual(
            mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        scale = 2.0f
                / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef(scale, scale, scale);

    // ---------- update parameters ----------
    this->UpdateParameters(mol, bs);

    // TODO: ---------- render ----------
    if (this->positions.Count() != mol->MoleculeCount()) {
        this->positions.SetCount(mol->MoleculeCount());
    }
    unsigned int firstResIdx = 0;
    unsigned int lastResIdx = 0;
    unsigned int firstAtomIdx = 0;
    unsigned int lastAtomIdx = 0;
    unsigned int atomTypeIdx = 0;
    for (unsigned int molIdx = 0; molIdx < mol->MoleculeCount(); molIdx++){
        this->positions[molIdx].Clear();
        this->positions[molIdx].AssertCapacity(mol->Molecules()[molIdx].ResidueCount()*4);

        firstResIdx = mol->Molecules()[molIdx].FirstResidueIndex();
        lastResIdx = firstResIdx + mol->Molecules()[molIdx].ResidueCount();
        for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++){
            firstAtomIdx = mol->Residues()[resIdx]->FirstAtomIndex();
            lastAtomIdx = firstAtomIdx + mol->Residues()[resIdx]->AtomCount();
            for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx; atomIdx++){
                unsigned int atomTypeIdx = mol->AtomTypeIndices()[atomIdx];
                if (mol->AtomTypes()[atomTypeIdx].Name().Equals("CA")){
                    this->positions[molIdx].Add(mol->AtomPositions()[3 * atomIdx]);
                    this->positions[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 1]);
                    this->positions[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 2]);
                    this->positions[molIdx].Add(1.0f);
                }
            }
        }
    }

    this->cartoonShader.Enable();
    if (PositionSlot == 0) {
        glBindAttribLocation(this->cartoonShader.ProgramHandle(), PositionSlot, "Position");
    }
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    glBegin(GL_PATCHES);
    for (unsigned int i = 0; i < this->positions.Count(); i++) {
        for (unsigned int j = 0; j < this->positions[i].Count() / 4; j++) {
            glVertex3f(this->positions[i][4 * j], this->positions[i][4 * j + 1], this->positions[i][4 * j + 2]);
        }
    }
    glEnd();

    this->cartoonShader.Disable();


#if RENDER_ATOMS_AS_SPHERES
    glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
    // get viewpoint parameters for raycasting
    float viewportStuff[4] = { cameraInfo->TileRect().Left(),
        cameraInfo->TileRect().Bottom(), cameraInfo->TileRect().Width(),
        cameraInfo->TileRect().Height() };
    if (viewportStuff[2] < 1.0f)
        viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f)
        viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // enable sphere shader
    this->sphereShader.Enable();
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
    // set vertex and color pointers and draw them
    //glEnableClientState(GL_VERTEX_ARRAY);
    glBegin(GL_POINTS);
    for (unsigned int i = 0; i < this->positions.Count(); i++) {
        for (unsigned int j = 0; j < this->positions[i].Count()/4; j++) {
            glColor4f(0.75f, 0.5f, 0.1f, 1.0f);
            glVertex4f(this->positions[i][4 * j], this->positions[i][4 * j + 1], this->positions[i][4 * j + 2], 0.3f);
        }
    }
    for (unsigned int i = 0; i < mol->AtomCount(); i++) {
        unsigned int atomTypeIdx = mol->AtomTypeIndices()[i];
        if (mol->AtomTypes()[atomTypeIdx].Name().Equals("CA")){
            glColor4f(0.5f, 0.75f, 0.1f, 0.5f);
            glVertex4f(mol->AtomPositions()[3 * i], mol->AtomPositions()[3 * i + 1], mol->AtomPositions()[3 * i + 2], 1.0f);
        } else {
            glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
            glVertex4f(mol->AtomPositions()[3 * i], mol->AtomPositions()[3 * i + 1], mol->AtomPositions()[3 * i + 2], 0.5f);
        }

    }
    glEnd();
    //glVertexPointer(3, GL_FLOAT, 0, mol->AtomPositions());
    //glDrawArrays(GL_POINTS, 0, mol->AtomCount());
    //glDisableClientState(GL_VERTEX_ARRAY);
    // disable sphere shader
    this->sphereShader.Disable();
#endif

    glPopMatrix();

    // unlock the current frame
    mol->Unlock();

    return true;
}

/*
 * update parameters
 */
void CartoonRenderer::UpdateParameters(const MolecularDataCall *mol,
    const protein_calls::BindingSiteCall *bs) {
    // TODO
}
