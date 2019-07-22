/*
 * SimplestSphereRenderer_2.cpp
 *
 * Copyright (C) 2018 by Karsten Schatz
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "SimplestSphereRenderer_2.h"
#include "CallSpheres.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/nextgen/CallRender3D_2.h"
#include "mmcore/nextgen/Camera_2.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/ShallowMatrix.h"

using namespace megamol;
using namespace megamol::megamol101;

/*
 * SimplestSphereRenderer_2::SimplestSphereRenderer_2
 */
SimplestSphereRenderer_2::SimplestSphereRenderer_2(void)
    : core::nextgen::Renderer3DModule_2()
    , sphereDataSlot("inData", "The input data slot for sphere data.")
    , sphereModeSlot("sphere rendering", "Switch for the pretty sphere rendering mode")
    , sizeScalingSlot("scaling factor", "Scaling factor for the size of the rendered GL_POINTS") {
    // TUTORIAL: A name and a description for each slot (CallerSlot, CalleeSlot, ParamSlot) has to be given in the
    // constructor initializer list

    // TUTORIAL: For each CallerSlot all compatible calls have to be set
    this->sphereDataSlot.SetCompatibleCall<CallSpheresDescription>();
    this->MakeSlotAvailable(&this->sphereDataSlot);

    // TUTORIAL: For each ParamSlot a default value has to be set
    this->sphereModeSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->sphereModeSlot);

    this->sizeScalingSlot.SetParameter(new core::param::FloatParam(1.0f, 0.01f, 1000.0f));
    this->MakeSlotAvailable(&this->sizeScalingSlot);

    // TUTORIAL: Each slot that shall be visible in the GUI has to be made available by this->MakeSlotAvailable(...)

    lastDataHash = 0;
    vbo = 0;
    va = 0;
}

/*
 * SimplestSphereRenderer_2::~SimplestSphereRenderer_2
 */
SimplestSphereRenderer_2::~SimplestSphereRenderer_2(void) {
    this->Release();
    // TUTORIAL: this->Release() should be called in each modules' destructor.
}

/*
 * SimplestSphereRenderer_2::create
 */
bool SimplestSphereRenderer_2::create(void) {

    // TUTORIAL Shader creation should always happen in the create method of a renderer.

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertSrc;
    ShaderSource fragSrc;
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("simplePoints::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for simple point shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("simplePoints::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for simple point shader");
        return false;
    }
    try {
        if (!this->simpleShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    ShaderSource prettyVertSrc;
    ShaderSource prettyGeomSrc;
    ShaderSource prettyFragSrc;
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("prettyPoints::vertex", prettyVertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for simple point shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("prettyPoints::geometry", prettyGeomSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load geometry shader source for simple point shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("prettyPoints::fragment", prettyFragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for simple point shader");
        return false;
    }
    try {
        if (!this->sphereShader.Compile(prettyVertSrc.Code(), prettyVertSrc.Count(), prettyGeomSrc.Code(),
                prettyGeomSrc.Count(), prettyFragSrc.Code(), prettyFragSrc.Count())) {

            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create sphere shader: %s\n", e.GetMsgA());
        return false;
    }
    try {
        if (!this->sphereShader.Link()) {
            throw vislib::Exception("Generic Linkage failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to link sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    return true;
}

/*
 * SimplestSphereRenderer_2::GetExtents
 */
bool SimplestSphereRenderer_2::GetExtents(core::nextgen::CallRender3D_2& call) {
    core::nextgen::CallRender3D_2* cr3d = dynamic_cast<core::nextgen::CallRender3D_2*>(&call);
    if (cr3d == nullptr) return false;

    CallSpheres* cs = this->sphereDataSlot.CallAs<CallSpheres>();
    if (cs == nullptr) return false;
    if (!(*cs)(CallSpheres::CallForGetExtent)) return false;

    cr3d->AccessBoundingBoxes() = cs->AccessBoundingBoxes();
    cr3d->SetTimeFramesCount(cs->FrameCount());

    return true;
}

/*
 * SimplestSphereRenderer_2::release
 */
void SimplestSphereRenderer_2::release(void) {
    if (va != 0) {
        glDeleteVertexArrays(1, &va);
    }
    if (vbo != 0) {
        glDeleteBuffers(1, &vbo);
    }
}

/*
 * SimplestSphereRenderer_2::Render
 */
bool SimplestSphereRenderer_2::Render(core::nextgen::CallRender3D_2& call) {
    core::nextgen::CallRender3D_2* cr3d = dynamic_cast<core::nextgen::CallRender3D_2*>(&call);
    if (cr3d == nullptr) return false;

    // before rendering, call all necessary data
    CallSpheres* cs = this->sphereDataSlot.CallAs<CallSpheres>();
    if (cs == nullptr) return false;
    if (!(*cs)(CallSpheres::CallForGetExtent)) return false;
    if (!(*cs)(CallSpheres::CallForGetData)) return false;
    auto sphereCount = cs->Count();
    bool renderMultipleColors = cs->HasColors();

    // only reload the vertex array if the data has changed
    if (cs->DataHash() != lastDataHash) {
        lastDataHash = cs->DataHash();

        if (va == 0 || vbo == 0) { // generate new buffers only if they do not exist
            glGenVertexArrays(1, &va);
            glGenBuffers(1, &vbo);
        }

        // get the data
        const float* spherePtr = cs->GetSpheres();
        const float* colorPtr = cs->GetColors();

        if (spherePtr == nullptr) return false;

        // load the data into the vertex buffer
        glBindVertexArray(va);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

        if (renderMultipleColors) {
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8 * sphereCount, nullptr,
                GL_STATIC_DRAW); // init the memory for vertices and colors
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * 4 * sphereCount, spherePtr); // write spheres to the gpu
            glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * 4 * sphereCount, sizeof(float) * 4 * sphereCount,
                colorPtr); // write colors to the gpu
        } else {
            std::vector<float> colVec(4 * sphereCount, 1.0f); // white color for everything
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8 * sphereCount, nullptr, GL_STATIC_DRAW);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * 4 * sphereCount, spherePtr); // write spheres to the gpu
            glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * 4 * sphereCount, sizeof(float) * 4 * sphereCount,
                colVec.data()); // write colors to the gpu
        }
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 4, 0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (GLvoid*)(sizeof(float) * 4 * sphereCount));

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    cr3d->AccessBoundingBoxes() = cs->AccessBoundingBoxes();

    core::nextgen::Camera_2 localCam;
    cr3d->GetCamera(localCam);

    cam_type::snapshot_type camsnap;
    cam_type::matrix_type viewCam, projCam;
    localCam.calc_matrices(camsnap, viewCam, projCam);

    glm::mat4 view = viewCam;
    glm::mat4 proj = projCam;
    glm::mat4 mvp = projCam * viewCam;

    this->GetLights();

    // start the rendering

    // Scale the point size with the parameter
    glPointSize(this->sizeScalingSlot.Param<core::param::FloatParam>()->Value());

    // Switch between shaders for rendering simple flat points or shaded spheres
    if (this->sphereModeSlot.Param<core::param::BoolParam>()->Value()) {
        this->sphereShader.Enable();
    } else {
        this->simpleShader.Enable();
    }

    glBindVertexArray(va);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    if (this->sphereModeSlot.Param<core::param::BoolParam>()->Value()) {
        // compute the necessary camera parameters from the modelview matrix
        auto invView = glm::inverse(view);

        // set all uniforms for the shaders
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(this->sphereShader.ParameterLocation("proj"), 1, GL_FALSE, glm::value_ptr(proj));
        glUniform3f(this->sphereShader.ParameterLocation("camRight"), camsnap.right_vector.x(), camsnap.right_vector.y(), camsnap.right_vector.z());
        glUniform3f(this->sphereShader.ParameterLocation("camUp"), camsnap.up_vector.x(), camsnap.up_vector.y(), camsnap.up_vector.z());
        glUniform3f(this->sphereShader.ParameterLocation("camPos"), camsnap.position.x(), camsnap.position.y(), camsnap.position.z());
        glUniform3f(this->sphereShader.ParameterLocation("camDir"), camsnap.view_vector.x(), camsnap.view_vector.y(), camsnap.view_vector.z());
        glUniform1f(this->sphereShader.ParameterLocation("scalingFactor"),
            this->sizeScalingSlot.Param<core::param::FloatParam>()->Value());

        if (this->lightMap.size() >= 1 && this->lightMap.begin()->second.lightType == core::view::light::POINTLIGHT) {
            auto lightPos = this->lightMap.begin()->second.pl_position;
            glUniform3f(this->sphereShader.ParameterLocation("lightPos"), lightPos[0], lightPos[1], lightPos[2]);
        }
    } else {
        glUniformMatrix4fv(this->simpleShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    }

    glEnable(GL_DEPTH_TEST);

    // draw one point for each sphere
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(sphereCount));
    glBindVertexArray(0);

    glDisable(GL_DEPTH_TEST);

    if (this->sphereModeSlot.Param<core::param::BoolParam>()->Value()) {
        this->sphereShader.Disable();
    } else {
        this->simpleShader.Disable();
    }

    return true;
}
