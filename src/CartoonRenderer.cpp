/*
 * CartoonRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "vislib/graphics/gl/IncludeAllGL.h"

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

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::core::moldyn;

/*
 * protein::CartoonRenderer::CartoonRenderer (CTOR)
 */
CartoonRenderer::CartoonRenderer(void)
        : Renderer3DModuleDS(), 
        molDataCallerSlot("getData", "Connects the molecule rendering with molecule data storage"), 
        bsDataCallerSlot("getBindingSites", "Connects the molecule rendering with binding site data storage") {
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
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Unable to create sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    return true;
}

/*
 * protein::CartoonRenderer::GetCapabilities
 */
bool CartoonRenderer::GetCapabilities(Call& call) {
    view::AbstractCallRender3D *cr3d =
            dynamic_cast<view::AbstractCallRender3D *>(&call);
    if (cr3d == NULL)
        return false;

    cr3d->SetCapabilities(
            view::AbstractCallRender3D::CAP_RENDER
                    | view::AbstractCallRender3D::CAP_LIGHTING
                    | view::AbstractCallRender3D::CAP_ANIMATION);

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

    int cnt;

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

    glPopMatrix();

    // unlock the current frame
    mol->Unlock();

    return true;
}

/*
 * update parameters
 */
void CartoonRenderer::UpdateParameters(const MolecularDataCall *mol,
    const core::moldyn::BindingSiteCall *bs) {
    // TODO
}

