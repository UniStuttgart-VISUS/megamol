#include "mmstd_gl/renderer/AnimationRenderer.h"

#include "mmcore/MegaMolGraph.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/graphics/CameraUtils.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore/utility/log/Log.h"

megamol::mmstd_gl::AnimationRenderer::AnimationRenderer()
        : Renderer3DModuleGL()
        , approximationSourceSlot("approximation source", "renderer output to approximate as scene proxy")
        , snapshotSlot("make snapshot", "grab current representation of the source")
        , numberOfViewsSlot("number of Views", "how many snapshots are taken, using camera reset variants")
        , tex_inspector_({"Color", "Depth"}) {

    approximationSourceSlot.SetParameter(new core::param::StringParam(""));
    this->MakeSlotAvailable(&approximationSourceSlot);

    snapshotSlot.SetParameter(new core::param::ButtonParam());
    this->MakeSlotAvailable(&snapshotSlot);

    numberOfViewsSlot.SetParameter(new core::param::IntParam(snaps_to_take.size()));
    this->MakeSlotAvailable(&numberOfViewsSlot);

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
    glGenVertexArrays(1, &line_vao);
    glBindVertexArray(line_vao);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, animation_positions->getName());
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

    return true;
}


void megamol::mmstd_gl::AnimationRenderer::release() {
    glDeleteVertexArrays(1, &line_vao);
}


bool megamol::mmstd_gl::AnimationRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    // TODO: joint bbox of points and path!
    call.AccessBoundingBoxes() = lastBBox;
    return true;
}


bool megamol::mmstd_gl::AnimationRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    bool make_snapshot = snapshotSlot.IsDirty();
    if (make_snapshot) {
        glClear(GL_COLOR_BUFFER_BIT);
        snapshotSlot.ResetDirty();
    }
    auto const incoming_fbo = call.GetFramebuffer();
    auto const actual_snaps_to_take =
        std::min<int>(snaps_to_take.size(), numberOfViewsSlot.Param<core::param::IntParam>()->Value());
    if (AnyParameterDirty()) {
        const auto mod_name = approximationSourceSlot.Param<core::param::StringParam>()->Value();
        auto mod = theGraph->FindModule(mod_name);
        CallRender3DGL* call_to_hijack = nullptr;
        uint32_t callback_idx = 0;
        if (mod) {
            for (auto& c : theGraph->ListCalls()) {
                if (c.callPtr->PeekCalleeSlot()->Parent()->FullName() == mod->FullName()) {
                    for (uint32_t x = 0; x < c.callPtr->GetCallbackCount(); ++x) {
                        if (c.callPtr->GetCallbackName(x) == "Render") {
                            call_to_hijack = dynamic_cast<CallRender3DGL*>(c.callPtr.get());
                            callback_idx = x;
                            break;
                        }
                    }
                }
            }
        }
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
                auto cam_position = core::utility::get_default_camera_position(lastBBox, cam_intrinsics, cam_orientation, dv);
                core::view::Camera::Pose cam_pose(glm::vec3(cam_position), cam_orientation);

                auto min_max_dist = core::utility::get_min_max_dist_to_bbox(lastBBox, cam_pose);
                cam_intrinsics.far_plane = std::max(0.0f, min_max_dist.y);
                cam_intrinsics.near_plane = std::max(cam_intrinsics.far_plane / 10000.0f, min_max_dist.x);
                cam = core::view::Camera(cam_pose, cam_intrinsics);
                call_to_hijack->SetCamera(cam);

                // render and convert the result to "point geometry"
                call_to_hijack->operator()(callback_idx);
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
        }
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
    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CLIP_DISTANCE0);

    if (theAnimation->pos_animation != nullptr) {
        auto anim = theAnimation->pos_animation;
        auto vertices = std::vector<float>();
        vertices.reserve(3 * anim->GetLength());
        glBindVertexArray(line_vao);
        for (auto t = anim->GetStartTime(); t <= anim->GetEndTime(); ++t) {
            auto v = anim->GetValue(t);
            vertices.emplace_back(v[0]);
            vertices.emplace_back(v[1]);
            vertices.emplace_back(v[2]);
        }
        animation_positions->rebuffer(vertices.data(), vertices.size() * sizeof(float));
    }

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

        tex_inspector_.SetTexture((void*)(intptr_t)tex_to_show, xres, yres);
        tex_inspector_.ShowWindow();
    }

    return true;
}
