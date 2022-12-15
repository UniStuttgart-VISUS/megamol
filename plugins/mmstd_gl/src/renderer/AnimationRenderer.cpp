#include "mmstd_gl/renderer/AnimationRenderer.h"

#include "mmcore/MegaMolGraph.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/graphics/CameraUtils.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore/utility/log/Log.h"

megamol::mmstd_gl::AnimationRenderer::AnimationRenderer()
        : Renderer3DModuleGL()
        , approximationSourceSlot("approxSource", "renderer output to approximate as scene proxy")
        , debugHijackSlot("debugHijack", "you do not want to know")
        , tex_inspector_({"Color", "Depth"}) {

    approximationSourceSlot.SetParameter(new core::param::StringParam(""));
    this->MakeSlotAvailable(&approximationSourceSlot);

    debugHijackSlot.SetParameter(new core::param::ButtonParam());
    this->MakeSlotAvailable(&debugHijackSlot);

    auto tex_inspector_slots = this->tex_inspector_.GetParameterSlots();
    for (auto& tex_slot : tex_inspector_slots) {
        this->MakeSlotAvailable(tex_slot);
    }
}


constexpr uint32_t xres = 1024, yres = 1024;
constexpr float aspect = xres / static_cast<float>(yres);

megamol::mmstd_gl::AnimationRenderer::~AnimationRenderer() {}


bool megamol::mmstd_gl::AnimationRenderer::create() {
    // yuck
    theGraph = const_cast<core::MegaMolGraph*>(&frontend_resources.get<core::MegaMolGraph>());

    approx_fbo = std::make_shared<glowl::FramebufferObject>(xres, yres);
    approx_fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

    // layout: xyz(RGBA)xyz(RGBA) etc. for each view.
    GLsizeiptr size = xres * yres * 4 * sizeof(float) * 6;

    //struct point {
    //    float x;
    //    float y;
    //    float z;
    //    // packed unorm 4x8
    //    glm::uint col;
    //};
    //std::vector<point> hurz;
    //hurz.resize(size);
    //for (auto x = 0; x < size; ++x) {
    //    hurz[x].x = -1.0f;
    //    hurz[x].y = 2.0f;
    //    hurz[x].z = 12.0f;
    //    hurz[x].col = 0xFF00FF00;
    //}

    the_points = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, size, GL_DYNAMIC_COPY);

    try {
        auto const shaderOptions = core::utility::make_path_shader_options(
            frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
        fbo_to_points_program = core::utility::make_glowl_shader(
            "fbo_to_points", shaderOptions, "mmstd_gl/fbo_to_points.comp.glsl");
    } catch (std::exception& e) {
        core::utility::log::Log::DefaultLog.WriteError(
            ("AnimationRenderer: could not compile compute shader: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}


void megamol::mmstd_gl::AnimationRenderer::release() {}


bool megamol::mmstd_gl::AnimationRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    return true;
}


bool megamol::mmstd_gl::AnimationRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    // TODO just for testing, so we can splice this into a normal graph
    bool hijack = debugHijackSlot.IsDirty();
    hijack = true;
    if (hijack) {
        glClear(GL_COLOR_BUFFER_BIT);
        debugHijackSlot.ResetDirty();
    }


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
        if (hijack && call_to_hijack) {
            approx_fbo->bind();
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

            auto dv = core::utility::DefaultView::DEFAULTVIEW_FACE_FRONT;
            auto dor = core::utility::DEFAULTORIENTATION_TOP;

            auto cam_orientation = core::utility::get_default_camera_orientation(dv, dor);
            auto cam_position = core::utility::get_default_camera_position(
                call_to_hijack->GetBoundingBoxes(), cam_intrinsics, cam_orientation, dv);
            core::view::Camera::Pose cam_pose(glm::vec3(cam_position), cam_orientation);

            auto min_max_dist = core::utility::get_min_max_dist_to_bbox(call_to_hijack->GetBoundingBoxes(), cam_pose);
            cam_intrinsics.far_plane = std::max(0.0f, min_max_dist.y);
            cam_intrinsics.near_plane = std::max(cam_intrinsics.far_plane / 10000.0f, min_max_dist.x);
            cam = core::view::Camera(cam_pose, cam_intrinsics);
            call_to_hijack->SetCamera(cam);

            // render and convert the result to "point geometry"
            call_to_hijack->operator()(callback_idx);
            // some sync thingy?
            glFlush();
            call.GetFramebuffer()->bind();

            fbo_to_points_program->use();
            //the_points->bind();
            the_points->bindAs(GL_SHADER_STORAGE_BUFFER, 1);
            glActiveTexture(GL_TEXTURE1);
            approx_fbo->bindColorbuffer(0);
            glActiveTexture(GL_TEXTURE2);
            approx_fbo->bindDepthbuffer();

            glDispatchCompute(xres / 16, yres / 16, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            glUseProgram(0);

        }
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
