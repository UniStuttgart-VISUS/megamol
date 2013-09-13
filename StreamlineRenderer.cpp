//
// StreamlineRenderer.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 11, 2013
//     Author: scharnkn
//

#include "stdafx.h"
#include "StreamlineRenderer.h"

#if (defined(WITH_CUDA) && (WITH_CUDA))

#include "VBODataCall.h"
#include "VTIDataCall.h"
#include "cuda_error_check.h"
#include "ogl_error_check.h"
#include "CUDAFieldTopology.cuh"

#include "vislib/GLSLShader.h"
#include "vislib/Cuboid.h"

#include "param/FloatParam.h"
#include "param/IntParam.h"

#include <cuda_gl_interop.h>


using namespace megamol;
using namespace megamol::protein;
using namespace megamol::core;


/*
 * StreamlineRenderer::StreamlineRenderer
 */
StreamlineRenderer::StreamlineRenderer(void) : Renderer3DModuleDS(),
        /* Caller slots */
        vertexDataCallerSlot("getVertexData", "Connects the renderer with the vertex data"),
        volumeDataCallerSlot("getVolumeData", "Connects the renderer with the volume data"),
        /* Streamline integration parameters */
        streamlineMaxStepsSlot("streamlines::nSteps", "Set the number of steps for streamline integration"),
        streamlineStepSlot("streamlines::step","Set stepsize for the streamline integration"),
        streamlineEpsSlot("streamlines::eps","Set epsilon for the termination of the streamline integration"),
        vbo(0), vboSize(0),
        triggerComputeGradientField(true), triggerComputeStreamlines(true) {

    // Data caller for vertex data
    this->vertexDataCallerSlot.SetCompatibleCall<VBODataCallDescription>();
    this->MakeSlotAvailable(&this->vertexDataCallerSlot);

    // Data caller for volume data
    this->volumeDataCallerSlot.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->volumeDataCallerSlot);


    /* Streamline integration parameters */

    // Set the number of steps for streamline integration
    this->streamlineMaxSteps = 10;
    this->streamlineMaxStepsSlot.SetParameter(new core::param::IntParam(this->streamlineMaxSteps, 0));
    this->MakeSlotAvailable(&this->streamlineMaxStepsSlot);

    // Set the step size for streamline integration
    this->streamlineStep = 1.0f;
    this->streamlineStepSlot.SetParameter(new core::param::FloatParam(this->streamlineStep, 0.1f));
    this->MakeSlotAvailable(&this->streamlineStepSlot);

    // Set the step size for streamline integration
    this->streamlineEps = 0.01f;
    this->streamlineEpsSlot.SetParameter(new core::param::FloatParam(this->streamlineEps, 0.0f));
    this->MakeSlotAvailable(&this->streamlineEpsSlot);
}


/*
 * StreamlineRenderer::~StreamlineRenderer
 */
StreamlineRenderer::~StreamlineRenderer(void) {
    this->Release();
}


/*
 * StreamlineRenderer::create
 */
bool StreamlineRenderer::create(void) {

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    // Init extensions
    if (!glh_init_extensions("\
            GL_VERSION_2_0 GL_EXT_texture3D \
            GL_EXT_framebuffer_object \
            GL_ARB_draw_buffers \
            GL_ARB_vertex_buffer_object")) {
        return false;
    }
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    return true;
}


/*
 * StreamlineRenderer::createVbo
 */
bool StreamlineRenderer::createVbo(GLuint* vbo, size_t s, GLuint target) {
    glGenBuffersARB(1, vbo);
    glBindBufferARB(target, *vbo);
    glBufferDataARB(target, s, 0, GL_DYNAMIC_DRAW);
    glBindBufferARB(target, 0);
    return CheckForGLError();
}

/*
 * ComparativeSurfacePotentialRenderer::destroyVbo
 */
void StreamlineRenderer::destroyVbo(GLuint* vbo, GLuint target) {
    glBindBufferARB(target, *vbo);
    glDeleteBuffersARB(1, vbo);
    *vbo = 0;
    CheckForGLError();
}


/*
 * StreamlineRenderer::release
 */
void StreamlineRenderer::release(void) {
    if (!this->vbo) {
        this->destroyVbo(&this->vbo, GL_ARRAY_BUFFER);
    }
}


/*
 * StreamlineRenderer::GetCapabilities
 */
bool StreamlineRenderer::GetCapabilities(core::Call& call) {

    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    cr3d->SetCapabilities(core::view::AbstractCallRender3D::CAP_RENDER |
                          core::view::AbstractCallRender3D::CAP_LIGHTING |
                          core::view::AbstractCallRender3D::CAP_ANIMATION);

    return true;
}


/*
 * StreamlineRenderer::GetExtents
 */
bool StreamlineRenderer::GetExtents(core::Call& call) {

    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    if (cr3d == NULL) {
        return false;
    }

    // Extent of vertex data

    VBODataCall *vboCall = this->vertexDataCallerSlot.CallAs<VBODataCall>();
    if (vboCall == NULL) {
        return false;
    }

    if (!(*vboCall)(VBODataCall::CallForGetExtent)) {
        return false;
    }

    // Extent of volume data

    VTIDataCall *vtiCall = this->volumeDataCallerSlot.CallAs<VTIDataCall>();
    if (vtiCall == NULL) {
        return false;
    }

    if (!(*vtiCall)(VTIDataCall::CallForGetExtent)) {
         return false;
    }

    // Unite both bounding boxes and make scaled worl bounding box

    vislib::math::Cuboid<float> tempBox0 = vboCall->GetBBox().ObjectSpaceBBox();
    vislib::math::Cuboid<float> tempBox1(vtiCall->GetWholeExtent());
    tempBox0.Union(tempBox1);
    this->bbox.SetObjectSpaceBBox(tempBox0);
    float scale;
    if(!vislib::math::IsEqual(this->bbox.ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    this->bbox.MakeScaledWorld(scale);
    cr3d->AccessBoundingBoxes() = this->bbox;
    cr3d->SetTimeFramesCount(std::min(vboCall->GetFrameCnt(), vtiCall->FrameCount()));

    return true;
}


/*
 * StreamlineRenderer::Render
 */
bool StreamlineRenderer::Render(core::Call& call) {

    // Update parameters
    this->updateParams();

    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    VTIDataCall *vtiCall = this->volumeDataCallerSlot.CallAs<VTIDataCall>();
    if (vtiCall == NULL) {
        return false;
    }

    // Get volume data
    vtiCall->SetCalltime(cr3d->Time());
    if (!(*vtiCall)(VTIDataCall::CallForGetData)) {
         return false;
    }

    VBODataCall *vboCall = this->vertexDataCallerSlot.CallAs<VBODataCall>();
    if (vboCall == NULL) {
        return false;
    }

    // Get vertex data
    if (!(*vboCall)(VTIDataCall::CallForGetData)) {
         return false;
    }

    // (Re)compute gradient field if necessary
    if (this->triggerComputeGradientField) {
       if (!this->computeGradient(vtiCall)) {
           return false;
       }
       this->triggerComputeGradientField = false;
    }

    // (Re)compute streamlines if necessary
    if (this->triggerComputeStreamlines) {
        if (!this->computeStreamlines(vboCall, vtiCall)) {
            return false;
        }
        this->triggerComputeStreamlines = false;
    }

    return true;
}


/*
 * StreamlineRenderer::computeGradient
 */
bool StreamlineRenderer::computeGradient(VTIDataCall *vtiCall) {

    uint volSize = vtiCall->GetGridsize().X()*vtiCall->GetGridsize().Y()*
            vtiCall->GetGridsize().Z();

    // (Re)allocate device memory if necessary
    if (!CudaSafeCall(this->scalarField_D.Validate(volSize))) {
        return false;
    }
    if (!CudaSafeCall(this->gradientField_D.Validate(volSize*3))) {
        return false;
    }

    // Copy volume data to device memory
    if (!CudaSafeCall(cudaMemcpy(
            this->scalarField_D.Peek(),
            (const void*)vtiCall->GetPointDataByIdx(0, 0),
            sizeof(float)*volSize,
            cudaMemcpyHostToDevice))) {
        return false;
    }

    // Init grid parameters
    if (!CudaSafeCall(SetGridParams(
            make_uint3(vtiCall->GetGridsize().X(),
                    vtiCall->GetGridsize().Y(),
                    vtiCall->GetGridsize().Z()),
            make_float3(vtiCall->GetOrigin().X(),
                    vtiCall->GetOrigin().Y(),
                    vtiCall->GetOrigin().Z()),
            make_float3(vtiCall->GetOrigin().X()+(vtiCall->GetGridsize().X()-1)*vtiCall->GetSpacing().X(),
                        vtiCall->GetOrigin().Y()+(vtiCall->GetGridsize().Y()-1)*vtiCall->GetSpacing().Y(),
                        vtiCall->GetOrigin().Z()+(vtiCall->GetGridsize().Z()-1)*vtiCall->GetSpacing().Z()),
            make_float3(vtiCall->GetSpacing().X(),
                        vtiCall->GetSpacing().Y(),
                        vtiCall->GetSpacing().Z())))){
        return false;
    }

    // Init with zero and compute gradient
    if (!CudaSafeCall(this->gradientField_D.Set(0x00))) {
        return false;
    }
    if (!CudaSafeCall(CalcGradient(
            this->scalarField_D.Peek(),
            this->gradientField_D.Peek(),
            volSize))) {
        return false;
    }

    return true;
}


/*
 * StreamlineRenderer::computeStreamlines
 */
bool StreamlineRenderer::computeStreamlines(VBODataCall *vboCall, VTIDataCall *vtiCall) {

    // Calculate the theoretically needed size for the vbo
    size_t sizeVbo = vboCall->GetVertexCnt()*
            this->streamlineMaxSteps*2*3*sizeof(float) +
            3*sizeof(float); // This makes the cuda kernel less complicated

    // (Re)create vbo if necessary
    if (sizeVbo > this->vboSize) {
        if (!this->vbo) {
            this->destroyVbo(&this->vbo, GL_ARRAY_BUFFER);
        }
        // Create empty vbo to hold data for the surface
        if (!this->createVbo(&this->vbo, sizeVbo, GL_ARRAY_BUFFER)) {
            return false;
        }

        // Register memory with CUDA
        if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(&this->vboResource[0],
                this->vbo,
                cudaGraphicsMapFlagsNone))) {
            return false;
        }
        this->vboSize = sizeVbo;
    }

    // Map both ressources
//    this->vboResource[1] = *(vboCall->GetCudaRessourceHandle()); // <- TODO This causes stack overflow (?)
//    if (!CudaSafeCall(cudaGraphicsMapResources(2, this->vboResource, 0))) {
//        return false;
//    }


//    // Get mapped pointers to the vbos
//    float *vboPt_D, *vboStreamlinesPt_D;
//    size_t vboSize, vboStreamlinesSize;
//    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
//            reinterpret_cast<void**>(&vboPt_D), // The mapped pointer
//            &vboSize,             // The size of the accessible data
//            this->vboResource[0]))) {                   // The mapped resource
//        return false;
//    }
//    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
//            reinterpret_cast<void**>(&vboStreamlinesPt_D), // The mapped pointer
//            &vboStreamlinesSize,             // The size of the accessible data
//            this->vboResource[1]))) {                   // The mapped resource
//        return false;
//    }
//
//    // Init some const values
//    if (!CudaSafeCall(SetStreamlineParams(this->streamlineStep, this->streamlineMaxSteps))) {
//        return false;
//    }
//    if (!CudaSafeCall(SetNumberOfPos(vboCall->GetVertexCnt()))) {
//        return false;
//    }
//    if (!CudaSafeCall(SetGridParams(
//            make_uint3(vtiCall->GetGridsize().X(),
//                    vtiCall->GetGridsize().Y(),
//                    vtiCall->GetGridsize().Z()),
//            make_float3(vtiCall->GetOrigin().X(),
//                    vtiCall->GetOrigin().Y(),
//                    vtiCall->GetOrigin().Z()),
//            make_float3(vtiCall->GetOrigin().X()+(vtiCall->GetGridsize().X()-1)*vtiCall->GetSpacing().X(),
//                        vtiCall->GetOrigin().Y()+(vtiCall->GetGridsize().Y()-1)*vtiCall->GetSpacing().Y(),
//                        vtiCall->GetOrigin().Z()+(vtiCall->GetGridsize().Z()-1)*vtiCall->GetSpacing().Z()),
//            make_float3(vtiCall->GetSpacing().X(),
//                        vtiCall->GetSpacing().Y(),
//                        vtiCall->GetSpacing().Z())))){
//        return false;
//    }
//
//    // Init positions
//    if (!CudaSafeCall(cudaMemset((void*)vboStreamlinesPt_D, 0, sizeVbo))) {
//        return false;
//    }
////    if (!CudaSafeCall(InitStartPos(
////            vboPt_D,
////            vboStreamlinesPt_D,
////            vboCall->GetDataStride(),
////            vboCall->GetDataOffsPosition(),
////            vboCall->GetVertexCnt()))) {
////        return false;
////    }
//    // Compute streamline using RK4
//    for (uint i = 0; i < this->streamlineMaxSteps; ++i) {
////        if (!CudaSafeCall(UpdateStreamlinePos(
////                vboStreamlinesPt_D,
////                this->gradientField_D.Peek(),
////                vboCall->GetVertexCnt(),
////                i))) {
////            return false;
////        }
//    }
//

//    // Unmap both ressources
//    if (!CudaSafeCall(cudaGraphicsUnmapResources(2, this->vboResource, 0))) {
//        return false;
//    }

    return true;
}


/*
 * StreamlineRenderer::renderStreamlineBundleManual
 */
bool StreamlineRenderer::renderStreamlineBundleManual() {
    return true;
}


/*
 * StreamlineRenderer::renderStreamlines
 */
bool StreamlineRenderer::renderStreamlines() {
    return true;
}


/*
 * StreamlineRenderer::updateParams
 */
void StreamlineRenderer::updateParams() {

    /* Streamline integration parameters */

    // Set the number of steps for streamline integration
    if (this->streamlineMaxStepsSlot.IsDirty()) {
        this->streamlineMaxSteps = this->streamlineMaxStepsSlot.Param<core::param::IntParam>()->Value();
        this->streamlineMaxStepsSlot.ResetDirty();
        this->triggerComputeStreamlines = true;
    }

    // Set the step size for streamline integration
    if (this->streamlineStepSlot.IsDirty()) {
        this->streamlineStep = this->streamlineStepSlot.Param<core::param::FloatParam>()->Value();
        this->streamlineStepSlot.ResetDirty();
        this->triggerComputeStreamlines = true;
    }

    // Set the epsilon for the streamline termination
    if (this->streamlineEpsSlot.IsDirty()) {
        this->streamlineEps = this->streamlineEpsSlot.Param<core::param::FloatParam>()->Value();
        this->streamlineEpsSlot.ResetDirty();
        this->triggerComputeStreamlines = true;
    }

}


#endif // (defined(WITH_CUDA) && (WITH_CUDA))
