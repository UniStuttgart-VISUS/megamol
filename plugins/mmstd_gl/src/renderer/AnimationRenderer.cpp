/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/renderer/AnimationRenderer.h"

#include "mmcore/MegaMolGraph.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/graphics/CameraUtils.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore_gl/utility/ShaderFactory.h"

megamol::mmstd_gl::AnimationRenderer::AnimationRenderer()
        : Renderer3DModuleGL()
        , snapshotSlot("make snapshot", "grab current representation of the source")
        , numberOfViewsSlot("number of Views", "how many snapshots are taken, using camera reset variants")
        , observedRendererSlot("observe", "Output that connects to the observed renderer")
        , observedRenderSlot("renderObservation", "Input from the camera view")
        , showLocalCoordSystems("orientation whiskers", "show local coordinate systems at each frame")
        , tex_inspector_({"Color", "Depth"}) {

    showLocalCoordSystems.SetParameter(new core::param::BoolParam(false));
    MakeSlotAvailable(&showLocalCoordSystems);

    snapshotSlot.SetParameter(new core::param::ButtonParam());
    MakeSlotAvailable(&snapshotSlot);

    numberOfViewsSlot.SetParameter(new core::param::IntParam(snaps_to_take.size()));
    MakeSlotAvailable(&numberOfViewsSlot);


    // the boring stuff, unfortunately, AKA InputCall
    observedRenderSlot.SetCallback(CallRender3DGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnKey), &AnimationRenderer::OnObservedKey);
    observedRenderSlot.SetCallback(CallRender3DGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnChar), &AnimationRenderer::OnObservedChar);
    observedRenderSlot.SetCallback(CallRender3DGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseButton),
        &AnimationRenderer::OnObservedMouseButton);
    observedRenderSlot.SetCallback(CallRender3DGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseMove),
        &AnimationRenderer::OnObservedMouseMove);
    observedRenderSlot.SetCallback(CallRender3DGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseScroll),
        &AnimationRenderer::OnObservedMouseScroll);

    // the interesting stuff we want to hook into
    observedRenderSlot.SetCallback(CallRender3DGL::ClassName(),
        core::view::AbstractCallRender::FunctionName(core::view::AbstractCallRender::FnRender),
        &AnimationRenderer::RenderObservation);
    observedRenderSlot.SetCallback(CallRender3DGL::ClassName(),
        core::view::AbstractCallRender::FunctionName(core::view::AbstractCallRender::FnGetExtents),
        &AnimationRenderer::GetObservationExtents);
    MakeSlotAvailable(&observedRenderSlot);

    observedRendererSlot.SetCompatibleCall<CallRender3DGLDescription>();
    MakeSlotAvailable(&observedRendererSlot);

    auto tex_inspector_slots = this->tex_inspector_.GetParameterSlots();
    for (auto& tex_slot : tex_inspector_slots) {
        this->MakeSlotAvailable(tex_slot);
    }
}


constexpr uint32_t xres = 1024, yres = 1024;
constexpr float aspect = xres / static_cast<float>(yres);

megamol::mmstd_gl::AnimationRenderer::~AnimationRenderer() {}


bool megamol::mmstd_gl::AnimationRenderer::create() {
    theGraph = const_cast<core::MegaMolGraph*>(&frontend_resources.get<core::MegaMolGraph>());
    theAnimation = const_cast<frontend_resources::AnimationEditorData*>(
        &frontend_resources.get<frontend_resources::AnimationEditorData>());

    approx_fbo = std::make_shared<glowl::FramebufferObject>(xres, yres);
    approx_fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

    // layout: xyz(RGBA)xyz(RGBA) etc. for each view.
    GLsizeiptr size = xres * yres * 4 * sizeof(float) * snaps_to_take.size();

    the_points = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, size, GL_DYNAMIC_COPY);
    animation_positions = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    animation_keys = std::make_unique<glowl::BufferObject>(GL_ELEMENT_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    animation_orientations = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &line_vao);
    glBindVertexArray(line_vao);
    glBindBuffer(GL_ARRAY_BUFFER, animation_positions->getName());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glGenVertexArrays(1, &keys_vao);
    glBindVertexArray(keys_vao);
    glBindBuffer(GL_ARRAY_BUFFER, animation_positions->getName());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, animation_keys->getName());

    glGenVertexArrays(1, &orientations_vao);
    glBindVertexArray(orientations_vao);
    glBindBuffer(GL_ARRAY_BUFFER, animation_positions->getName());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, animation_orientations->getName());
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);

    auto const shaderOptions =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    try {
        fbo_to_points_program = core::utility::make_glowl_shader(
            "fbo_to_points", shaderOptions, "mmstd_gl/animation/fbo_to_points.comp.glsl");
    } catch (std::exception& e) {
        core::utility::log::Log::DefaultLog.WriteError(
            ("AnimationRenderer: could not compile point generator shader: " + std::string(e.what())).c_str());
        return false;
    }

    try {
        render_points_program = core::utility::make_glowl_shader("fbo_to_points", shaderOptions,
            "mmstd_gl/animation/points.vert.glsl", "mmstd_gl/animation/points.frag.glsl");
    } catch (std::exception& e) {
        core::utility::log::Log::DefaultLog.WriteError(
            ("AnimationRenderer: could not compile point rendering shader: " + std::string(e.what())).c_str());
        return false;
    }

    try {
        campath_program = core::utility::make_glowl_shader(
            "campath", shaderOptions, "mmstd_gl/animation/campath.vert.glsl", "mmstd_gl/animation/campath.frag.glsl");
    } catch (std::exception& e) {
        core::utility::log::Log::DefaultLog.WriteError(
            ("AnimationRenderer: could not compile cam path rendering shader: " + std::string(e.what())).c_str());
        return false;
    }

    try {
        keys_program = core::utility::make_glowl_shader(
            "keys", shaderOptions, "mmstd_gl/animation/keys.vert.glsl", "mmstd_gl/animation/keys.frag.glsl");
    } catch (std::exception& e) {
        core::utility::log::Log::DefaultLog.WriteError(
            ("AnimationRenderer: could not compile key rendering shader: " + std::string(e.what())).c_str());
        return false;
    }

    try {
        orientations_program =
            core::utility::make_glowl_shader("orientation", shaderOptions, "mmstd_gl/animation/orientation.vert.glsl",
                "mmstd_gl/animation/orientation.geom.glsl", "mmstd_gl/animation/orientation.frag.glsl");
    } catch (std::exception& e) {
        core::utility::log::Log::DefaultLog.WriteError(
            ("AnimationRenderer: could not compile orientation rendering shader: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}


void megamol::mmstd_gl::AnimationRenderer::release() {
    glDeleteVertexArrays(1, &line_vao);
}


bool megamol::mmstd_gl::AnimationRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    // TODO: joint bbox of points and path!
    //call.AccessBoundingBoxes() = lastBBox;
    auto box = lastBBox.BoundingBox();
    for (int i = 0; i < trajectory_vertices.size() / 3; ++i) {
        box.GrowToPoint(vislib::math::Point<float, 3>(
            trajectory_vertices[i * 3 + 0], trajectory_vertices[i * 3 + 1], trajectory_vertices[i * 3 + 2]));
    }
    call.AccessBoundingBoxes().SetClipBox(box);
    call.AccessBoundingBoxes().SetBoundingBox(box);
    return true;
}


bool megamol::mmstd_gl::AnimationRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    bool make_snapshot = snapshotSlot.IsDirty();
    //make_snapshot = true;
    auto const incoming_fbo = call.GetFramebuffer();
    auto const actual_snaps_to_take =
        std::min<int>(snaps_to_take.size(), numberOfViewsSlot.Param<core::param::IntParam>()->Value());

    call_to_hijack = observedRendererSlot.CallAs<CallRender3DGL>();
    if (make_snapshot && call_to_hijack) {
        lastBBox = call_to_hijack->GetBoundingBoxes();
        for (int i = 0; i < actual_snaps_to_take; ++i) {
            approx_fbo->bind();
            glViewport(0, 0, approx_fbo->getWidth(), approx_fbo->getHeight());
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            call_to_hijack->SetFramebuffer(approx_fbo);

            core::view::Camera cam;
            core::view::Camera::PerspectiveParameters cam_intrinsics;
            cam_intrinsics.near_plane = 0.01f;
            cam_intrinsics.far_plane = 100.0f;
            cam_intrinsics.fovy = 0.5;
            cam_intrinsics.aspect = aspect;
            cam_intrinsics.image_plane_tile =
                core::view::Camera::ImagePlaneTile(); // view is in control -> no tiling -> use default tile values

            auto dv = snaps_to_take[i].first;
            auto dor = snaps_to_take[i].second;

            auto cam_orientation = core::utility::get_default_camera_orientation(dv, dor);
            auto cam_position =
                core::utility::get_default_camera_position(lastBBox, cam_intrinsics, cam_orientation, dv);
            core::view::Camera::Pose cam_pose(glm::vec3(cam_position), cam_orientation);

            auto min_max_dist = core::utility::get_min_max_dist_to_bbox(lastBBox, cam_pose);
            cam_intrinsics.far_plane = std::max(0.0f, min_max_dist.y);
            cam_intrinsics.near_plane = std::max(cam_intrinsics.far_plane / 10000.0f, min_max_dist.x);
            cam = core::view::Camera(cam_pose, cam_intrinsics);
            call_to_hijack->SetCamera(cam);

            // render and convert the result to "point geometry"
            call_to_hijack->operator()(core::view::AbstractCallRender::FnRender);
            // TODO: is that needed?
            glFlush();
            incoming_fbo->bind();
            fbo_to_points_program->use();
            the_points->bindAs(GL_SHADER_STORAGE_BUFFER, 1);
            glActiveTexture(GL_TEXTURE1);
            approx_fbo->bindColorbuffer(0);
            glActiveTexture(GL_TEXTURE2);
            approx_fbo->bindDepthbuffer();
            fbo_to_points_program->setUniform("view", cam.getViewMatrix());
            fbo_to_points_program->setUniform("projection", cam.getProjectionMatrix());
            fbo_to_points_program->setUniform("mvp", cam.getProjectionMatrix() * cam.getViewMatrix());
            fbo_to_points_program->setUniform("output_offset", i * xres * yres);
            glDispatchCompute(xres / 16, yres / 16, 1);
        }
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glUseProgram(0);

        // TODO: grab bbox of points from gpu
        // do we really need that? we have the original one anyway.
        this->ResetAllDirtyFlags();
    }

    core::view::Camera cam = call.GetCamera();
    glm::mat4 view = cam.getViewMatrix();
    glm::mat4 proj = cam.getProjectionMatrix();
    glm::mat4 mvp = proj * view;

    // draw points all the time
    incoming_fbo->bind();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CLIP_DISTANCE0);
    glViewport(0, 0, incoming_fbo->getWidth(), incoming_fbo->getHeight());
    render_points_program->use();
    the_points->bindAs(GL_SHADER_STORAGE_BUFFER, 1);
    render_points_program->setUniform("mvp", mvp);
    glDrawArrays(GL_POINTS, 0, xres * yres * actual_snaps_to_take);
    glDisable(GL_CLIP_DISTANCE0);

    auto anim_start = theAnimation->active_region.first;
    auto anim_end = theAnimation->active_region.second;
    auto anim_len = anim_end - anim_start + 1;

    if (theAnimation->pos_animation != nullptr) {
        auto anim = theAnimation->pos_animation;
        trajectory_vertices.clear();
        trajectory_vertices.reserve(3 * anim_len);
        glBindVertexArray(line_vao);
        for (auto t = anim_start; t <= anim_end; ++t) {
            auto v = anim->GetValue(t);
            trajectory_vertices.emplace_back(v[0]);
            trajectory_vertices.emplace_back(v[1]);
            trajectory_vertices.emplace_back(v[2]);
        }
        animation_positions->rebuffer(trajectory_vertices.data(), trajectory_vertices.size() * sizeof(float));

        campath_program->use();
        campath_program->setUniform("mvp", mvp);
        campath_program->setUniform("line_len", anim_len);
        glDrawArrays(GL_LINE_STRIP, 0, anim_len);

        auto keys = anim->GetAllKeys();
        key_indices.clear();
        key_indices.reserve(keys.size());
        for (auto key : keys) {
            key_indices.emplace_back(key);
        }
        glBindVertexArray(keys_vao);
        animation_keys->rebuffer(key_indices.data(), key_indices.size() * sizeof(int));

        keys_program->use();
        keys_program->setUniform("mvp", mvp);
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glDrawElements(GL_POINTS, keys.size(), GL_UNSIGNED_INT, nullptr);
        glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glBindVertexArray(0);
    }

    if (theAnimation->pos_animation != nullptr && theAnimation->orientation_animation != nullptr &&
        showLocalCoordSystems.Param<core::param::BoolParam>()->Value()) {
        auto anim = theAnimation->orientation_animation;
        trajectory_orientations.clear();
        trajectory_orientations.reserve(4 * anim_len);
        glBindVertexArray(orientations_vao);
        for (auto t = anim_start; t <= anim_end; ++t) {
            auto v = anim->GetValue(t);
            trajectory_orientations.emplace_back(v[0]);
            trajectory_orientations.emplace_back(v[1]);
            trajectory_orientations.emplace_back(v[2]);
            trajectory_orientations.emplace_back(v[3]);
        }
        animation_orientations->rebuffer(
            trajectory_orientations.data(), trajectory_orientations.size() * sizeof(float));

        orientations_program->use();
        orientations_program->setUniform("mvp", mvp);
        orientations_program->setUniform("direction_len", 0.5f);
        glDrawArrays(GL_POINTS, 0, anim_len);
    }

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);

    if (tex_inspector_.GetShowInspectorSlotValue()) {
        GLuint tex_to_show = 0;
        switch (tex_inspector_.GetSelectTextureSlotValue()) {
        case 0:
        default:
            tex_to_show = approx_fbo->getColorAttachment(0)->getName();
            break;
        case 1:
            tex_to_show = approx_fbo->getDepthStencil()->getName();
            break;
        }

        tex_inspector_.SetTexture((void*) (intptr_t) tex_to_show, xres, yres);
        tex_inspector_.ShowWindow();
    }

    return true;
}

bool megamol::mmstd_gl::AnimationRenderer::CheckObservedSlots(
    CallRender3DGL*& in, CallRender3DGL*& out, core::Call& call) {
    out = observedRendererSlot.CallAs<CallRender3DGL>();
    in = dynamic_cast<CallRender3DGL*>(&call);
    return out != nullptr && in != nullptr;
}

bool megamol::mmstd_gl::AnimationRenderer::GetObservationExtents(core::Call& call) {
    CallRender3DGL *in = nullptr, *out = nullptr;
    if (CheckObservedSlots(in, out, call)) {
        *out = *in;
        const auto res = (*out)(core::view::AbstractCallRender::FnGetExtents);
        *in = *out;
        return res;
    }
    return false;
}

bool megamol::mmstd_gl::AnimationRenderer::RenderObservation(core::Call& call) {
    CallRender3DGL *in = nullptr, *out = nullptr;
    if (CheckObservedSlots(in, out, call)) {
        *out = *in;
        const auto res = (*out)(core::view::AbstractCallRender::FnRender);
        *in = *out;
        return res;
    }
    return false;
}

bool megamol::mmstd_gl::AnimationRenderer::OnObservedMouseButton(core::Call& call) {
    CallRender3DGL *in = nullptr, *out = nullptr;
    if (CheckObservedSlots(in, out, call)) {
        out->SetInputEvent(in->GetInputEvent());
        return (*out)(core::view::AbstractCallRender::FnOnMouseButton);
    }
    return false;
}

bool megamol::mmstd_gl::AnimationRenderer::OnObservedMouseMove(core::Call& call) {
    CallRender3DGL *in = nullptr, *out = nullptr;
    if (CheckObservedSlots(in, out, call)) {
        out->SetInputEvent(in->GetInputEvent());
        return (*out)(core::view::AbstractCallRender::FnOnMouseMove);
    }
    return false;
}

bool megamol::mmstd_gl::AnimationRenderer::OnObservedMouseScroll(core::Call& call) {
    CallRender3DGL *in = nullptr, *out = nullptr;
    if (CheckObservedSlots(in, out, call)) {
        out->SetInputEvent(in->GetInputEvent());
        return (*out)(core::view::AbstractCallRender::FnOnMouseScroll);
    }
    return false;
}

bool megamol::mmstd_gl::AnimationRenderer::OnObservedChar(core::Call& call) {
    CallRender3DGL *in = nullptr, *out = nullptr;
    if (CheckObservedSlots(in, out, call)) {
        out->SetInputEvent(in->GetInputEvent());
        return (*out)(core::view::AbstractCallRender::FnOnChar);
    }
    return false;
}

bool megamol::mmstd_gl::AnimationRenderer::OnObservedKey(core::Call& call) {
    CallRender3DGL *in = nullptr, *out = nullptr;
    if (CheckObservedSlots(in, out, call)) {
        out->SetInputEvent(in->GetInputEvent());
        return (*out)(core::view::AbstractCallRender::FnOnKey);
    }
    return false;
}
