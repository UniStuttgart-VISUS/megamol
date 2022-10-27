/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#include "SimplestSphereRenderer.h"

#include "CallSpheres.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/ShallowMatrix.h"

using namespace megamol;
using namespace megamol::megamol101_gl;

/*
 * SimplestSphereRenderer::SimplestSphereRenderer
 */
SimplestSphereRenderer::SimplestSphereRenderer()
        : mmstd_gl::Renderer3DModuleGL()
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
 * SimplestSphereRenderer::~SimplestSphereRenderer
 */
SimplestSphereRenderer::~SimplestSphereRenderer() {
    this->Release();
    // TUTORIAL: this->Release() should be called in each modules' destructor.
}

/*
 * SimplestSphereRenderer::create
 */
bool SimplestSphereRenderer::create() {

    // TUTORIAL Shader creation should always happen in the create method of a renderer.

    using namespace megamol::core::utility::log;

    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    try {
        simpleShader = core::utility::make_glowl_shader("simplePoints", shader_options,
            "megamol101_gl/simple_points.vert.glsl", "megamol101_gl/simple_points.frag.glsl");
        sphereShader =
            core::utility::make_glowl_shader("prettyPoints", shader_options, "megamol101_gl/pretty_points.vert.glsl",
                "megamol101_gl/pretty_points.geom.glsl", "megamol101_gl/pretty_points.frag.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("SimplestSphereRenderer: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}

/*
 * SimplestSphereRenderer::GetExtents
 */
bool SimplestSphereRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    mmstd_gl::CallRender3DGL* cr3d = dynamic_cast<mmstd_gl::CallRender3DGL*>(&call);
    if (cr3d == nullptr)
        return false;

    CallSpheres* cs = this->sphereDataSlot.CallAs<CallSpheres>();
    if (cs == nullptr)
        return false;
    if (!(*cs)(CallSpheres::CallForGetExtent))
        return false;

    cr3d->AccessBoundingBoxes() = cs->AccessBoundingBoxes();
    cr3d->SetTimeFramesCount(cs->FrameCount());

    return true;
}

/*
 * SimplestSphereRenderer::release
 */
void SimplestSphereRenderer::release() {
    if (va != 0) {
        glDeleteVertexArrays(1, &va);
    }
    if (vbo != 0) {
        glDeleteBuffers(1, &vbo);
    }
}

/*
 * SimplestSphereRenderer::Render
 */
bool SimplestSphereRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    mmstd_gl::CallRender3DGL* cr3d = dynamic_cast<mmstd_gl::CallRender3DGL*>(&call);
    if (cr3d == nullptr)
        return false;

    // before rendering, call all necessary data
    CallSpheres* cs = this->sphereDataSlot.CallAs<CallSpheres>();
    if (cs == nullptr)
        return false;
    if (!(*cs)(CallSpheres::CallForGetExtent))
        return false;
    if (!(*cs)(CallSpheres::CallForGetData))
        return false;
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

        if (spherePtr == nullptr)
            return false;

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

    core::view::Camera cam = cr3d->GetCamera();

    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();
    auto mvp = proj * view;
    auto cam_pose = cam.get<core::view::Camera::Pose>();

    // start the rendering

    // Scale the point size with the parameter
    glPointSize(this->sizeScalingSlot.Param<core::param::FloatParam>()->Value());

    // Switch between shaders for rendering simple flat points or shaded spheres
    if (this->sphereModeSlot.Param<core::param::BoolParam>()->Value()) {
        this->sphereShader->use();
    } else {
        this->simpleShader->use();
    }

    glBindVertexArray(va);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    if (this->sphereModeSlot.Param<core::param::BoolParam>()->Value()) {
        // compute the necessary camera parameters from the modelview matrix
        auto invView = glm::inverse(view);

        // set all uniforms for the shaders
        this->sphereShader->setUniform("mvp", mvp);
        this->sphereShader->setUniform("view", view);
        this->sphereShader->setUniform("proj", proj);
        this->sphereShader->setUniform("camRight", cam_pose.right.x, cam_pose.right.y, cam_pose.right.z);
        this->sphereShader->setUniform("camUp", cam_pose.up.x, cam_pose.up.y, cam_pose.up.z);
        this->sphereShader->setUniform("camPos", cam_pose.position.x, cam_pose.position.y, cam_pose.position.z);
        this->sphereShader->setUniform("camDir", cam_pose.direction.x, cam_pose.direction.y, cam_pose.direction.z);
        this->sphereShader->setUniform(
            "scalingFactor", this->sizeScalingSlot.Param<core::param::FloatParam>()->Value());
    } else {
        this->simpleShader->setUniform("mvp", mvp);
    }

    glEnable(GL_DEPTH_TEST);

    // draw one point for each sphere
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(sphereCount));
    glBindVertexArray(0);

    glDisable(GL_DEPTH_TEST);

    glUseProgram(0);

    return true;
}
