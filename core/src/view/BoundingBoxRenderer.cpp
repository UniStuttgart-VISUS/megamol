/*
 * BoundingBoxRenderer.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/view/BoundingBoxRenderer.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/sys/Log.h"

using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::view;

/*
 * BoundingBoxRenderer::BoundingBoxRenderer
 */
BoundingBoxRenderer::BoundingBoxRenderer(void)
    : RendererModule<CallRender3D_2>()
    , enableBoundingBoxSlot("enableBoundingBox", "Enables the rendering of the bounding box")
    , boundingBoxColorSlot("boundingBoxColor", "Color of the bounding box")
    , smoothLineSlot("smoothLines", "Enables the smoothing of lines (may look strange on some setups)")
    , enableViewCubeSlot("enableViewCube", "Enables the rendering of the view cube")
    , vbo(0)
    , ibo(0)
    , va(0) {

    this->enableBoundingBoxSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->enableBoundingBoxSlot);

    this->boundingBoxColorSlot.SetParameter(new param::ColorParam("#ffffffff"));
    this->MakeSlotAvailable(&this->boundingBoxColorSlot);

    this->smoothLineSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->smoothLineSlot);

    this->enableViewCubeSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->enableViewCubeSlot);

    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * BoundingBoxRenderer::~BoundingBoxRenderer
 */
BoundingBoxRenderer::~BoundingBoxRenderer(void) { this->Release(); }

/*
 * BoundingBoxRenderer::create
 */
bool BoundingBoxRenderer::create(void) {
    // TODO the vislib shaders have to die a slow and painful death
    vislib::graphics::gl::ShaderSource bbVertSrc;
    vislib::graphics::gl::ShaderSource bbFragSrc;
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("boundingbox::vertex", bbVertSrc)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load vertex shader source for bounding box line shader");
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("boundingbox::fragment", bbFragSrc)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load fragment shader source for bounding box line shader");
    }
    try {
        if (!this->lineShader.Create(bbVertSrc.Code(), bbVertSrc.Count(), bbFragSrc.Code(), bbFragSrc.Count())) {
            throw vislib::Exception("Shader creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to create bounding box line shader: %s\n", e.GetMsgA());
        return false;
    }

    // the used vertex order resembles the one used in the old View3D
    std::vector<float> vertexCoords = {-1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
        -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<unsigned int> vertexIndices = {0, 2, 3, 1, 4, 5, 7, 6, 2, 6, 7, 3, 0, 1, 5, 4, 0, 4, 6, 2, 1, 3, 7, 5};

    glCreateBuffers(1, &this->vbo);
    glCreateBuffers(1, &this->ibo);
    glCreateVertexArrays(1, &this->va);

    glBindVertexArray(this->va);
    glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ibo);

    glEnableVertexAttribArray(0);

    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexCoords.size(), vertexCoords.data(), GL_STATIC_DRAW);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * vertexIndices.size(), vertexIndices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0);

    glBindVertexArray(0);

    return true;
}

/*
 * BoundingBoxRenderer::release
 */
void BoundingBoxRenderer::release(void) {
    this->lineShader.Release();
    if (this->va != 0) {
        glDeleteVertexArrays(1, &this->va);
        this->va = 0;
    }
    if (this->vbo != 0) {
        glDeleteBuffers(1, &this->vbo);
        this->vbo = 0;
    }
    if (this->ibo != 0) {
        glDeleteBuffers(1, &this->ibo);
        this->ibo = 0;
    }
}

/*
 * BoundingBoxRenderer::GetExtents
 */
bool BoundingBoxRenderer::GetExtents(CallRender3D_2& call) {
    CallRender3D_2* chainedCall = this->chainRenderSlot.CallAs<CallRender3D_2>();
    if (chainedCall == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "The BoundingBoxRenderer does not work without a renderer attached to its right");
        return false;
    }
    *chainedCall = call;
    bool retVal = (*chainedCall)(view::AbstractCallRender::FnGetExtents);
    call = *chainedCall;
    return retVal;
}

/*
 * BoundingBoxRenderer::Render
 */
bool BoundingBoxRenderer::Render(CallRender3D_2& call) {
    auto leftSlotParent = call.PeekCallerSlot()->Parent();
    std::shared_ptr<const view::AbstractView> viewptr =
        std::dynamic_pointer_cast<const view::AbstractView>(leftSlotParent);

    if (viewptr != nullptr) {
        // TODO move this behind the fbo magic?
        auto vp = call.GetViewport();
        glViewport(vp.Left(), vp.Bottom(), vp.Width(), vp.Height());
        auto backCol = call.BackgroundColor();
        glClearColor(backCol.x, backCol.y, backCol.z, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    CallRender3D_2* chainedCall = this->chainRenderSlot.CallAs<CallRender3D_2>();
    if (chainedCall == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "The BoundingBoxRenderer does not work without a renderer attached to its right");
        return false;
    }

    Camera_2 cam;
    call.GetCamera(cam);

    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewT, projT;

    cam.calc_matrices(snapshot, viewT, projT);

    glm::mat4 view = viewT;
    glm::mat4 proj = projT;
    glm::mat4 mvp = proj * view;

    auto boundingBoxes = chainedCall->AccessBoundingBoxes();
    auto smoothLines = this->smoothLineSlot.Param<param::BoolParam>()->Value();

    bool renderRes = true;
    if (this->enableBoundingBoxSlot.Param<param::BoolParam>()->Value()) {
        renderRes &= this->RenderBoundingBoxBack(mvp, boundingBoxes, smoothLines);
    }
    renderRes &= (*chainedCall)(view::AbstractCallRender::FnRender);
    if (this->enableBoundingBoxSlot.Param<param::BoolParam>()->Value()) {
        renderRes &= this->RenderBoundingBoxFront(mvp, boundingBoxes, smoothLines);
    }
    if (this->enableViewCubeSlot.Param<param::BoolParam>()->Value()) {
        renderRes &= this->RenderViewCube(call);
    }

    return renderRes;
}

/*
 * BoundingBoxRenderer::RenderBoundingBoxFront
 */
bool BoundingBoxRenderer::RenderBoundingBoxFront(const glm::mat4& mvp, const BoundingBoxes_2& bb, bool smoothLines) {
    glm::vec3 bbmin = glm::vec3(bb.BoundingBox().Left(), bb.BoundingBox().Bottom(), bb.BoundingBox().Back());
    glm::vec3 bbmax = glm::vec3(bb.BoundingBox().Right(), bb.BoundingBox().Top(), bb.BoundingBox().Front());

    auto colptr = this->boundingBoxColorSlot.Param<param::ColorParam>()->Value();

    this->lineShader.Enable();

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    if (smoothLines) glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.75f);
    glPolygonMode(GL_FRONT, GL_LINE);

    glBindVertexArray(this->va);
    glEnableVertexAttribArray(0);

    glUniformMatrix4fv(this->lineShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform3fv(this->lineShader.ParameterLocation("bbMin"), 1, glm::value_ptr(bbmin));
    glUniform3fv(this->lineShader.ParameterLocation("bbMax"), 1, glm::value_ptr(bbmax));
    glUniform3f(this->lineShader.ParameterLocation("color"), colptr[0], colptr[1], colptr[2]);

    glDrawElements(GL_QUADS, 24, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);

    glEnable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDisable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    if (smoothLines) glDisable(GL_LINE_SMOOTH);
    glLineWidth(1.0f);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT, GL_FILL);

    this->lineShader.Disable();

    return true;
}

/*
 * BoundingBoxRenderer::RenderBoundingBoxBack
 */
bool BoundingBoxRenderer::RenderBoundingBoxBack(const glm::mat4& mvp, const BoundingBoxes_2& bb, bool smoothLines) {
    glm::vec3 bbmin = glm::vec3(bb.BoundingBox().Left(), bb.BoundingBox().Bottom(), bb.BoundingBox().Back());
    glm::vec3 bbmax = glm::vec3(bb.BoundingBox().Right(), bb.BoundingBox().Top(), bb.BoundingBox().Front());

    auto color = this->boundingBoxColorSlot.Param<param::ColorParam>()->Value();

    this->lineShader.Enable();

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    if (smoothLines) glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.25f);
    glPolygonMode(GL_BACK, GL_LINE);

    glBindVertexArray(this->va);
    glEnableVertexAttribArray(0);

    glUniformMatrix4fv(this->lineShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform3fv(this->lineShader.ParameterLocation("bbMin"), 1, glm::value_ptr(bbmin));
    glUniform3fv(this->lineShader.ParameterLocation("bbMax"), 1, glm::value_ptr(bbmax));
    glUniform3f(this->lineShader.ParameterLocation("color"), color[0], color[1], color[2]);

    glDrawElements(GL_QUADS, 24, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);

    glEnable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDisable(GL_DEPTH_TEST);
    if (smoothLines) glDisable(GL_LINE_SMOOTH);
    glLineWidth(1.0f);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_BACK, GL_FILL);

    this->lineShader.Disable();

    return true;
}

/*
 * BoundingBoxRenderer::RenderViewCube
 */
bool BoundingBoxRenderer::RenderViewCube(CallRender3D_2& call) { return true; }
