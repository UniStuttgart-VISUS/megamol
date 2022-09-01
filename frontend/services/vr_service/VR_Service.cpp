/*
 * VR_Service.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "VR_Service.hpp"

#ifdef MEGAMOL_USE_VR_INTEROP
#include <glm/gtc/matrix_transform.hpp>
#include <interop.hpp>

#include "ImageWrapper_to_GLTexture.hpp"
#include "ViewRenderInputs.h"

#include "mmcore/Module.h"
#include "mmcore/view/AbstractViewInterface.h"
#include "mmstd/view/AbstractView.h"
#endif // MEGAMOL_USE_VR_INTEROP

#include "mmcore/MegaMolGraph.h"

// local logging wrapper for your convenience until central MegaMol logger established
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/utility/log/Log.h"


static const std::string service_name = "VR_Service: ";
static void log(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log_error(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}

static void log_warning(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(msg.c_str());
}


namespace megamol {
namespace frontend {

VR_Service::VR_Service() {}

VR_Service::~VR_Service() {}

bool VR_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

bool VR_Service::init(const Config& config) {

    m_requestedResourcesNames = {
        "ImagePresentationEntryPoints",
        "MegaMolGraph",
#ifdef MEGAMOL_USE_VR_INTEROP
        "OpenGL_Context",
#endif // MEGAMOL_USE_VR_INTEROP
    };

    switch (config.mode) {
    case Config::Mode::Off:
        break;
#ifdef MEGAMOL_USE_VR_INTEROP
    case Config::Mode::UnityKolabBW:
        log("running Unity KolabBW VR mode. Duplicating and sending View3D Entry Points via Spout.");
        m_vr_device_ptr = std::make_unique<VR_Service::KolabBW>();
        break;
#endif // MEGAMOL_USE_VR_INTEROP
    default:
        log_error("Unknown VR Service Mode: " + std::to_string(static_cast<int>(config.mode)));
        return false;
        break;
    }

    log("initialized successfully");
    return true;
}

void VR_Service::close() {
    m_vr_device_ptr.reset();
}

std::vector<FrontendResource>& VR_Service::getProvidedResources() {
    m_providedResourceReferences = {};

    return m_providedResourceReferences;
}

const std::vector<std::string> VR_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

std::string VR_Service::vr_service_marker = "#vr_service";

auto mark = [](auto const& name) { return name + VR_Service::vr_service_marker; };

void VR_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    this->m_requestedResourceReferences = resources;

    auto& entry_points =
        m_requestedResourceReferences[0].getResource<frontend_resources::ImagePresentationEntryPoints>();

    // we register an entry point subscription at the image presentation
    // we will get notified of entry point changes and pass those subscription events
    // to the currently active IVR_Device

    using SubscriptionEvent = frontend_resources::ImagePresentationEntryPoints::SubscriptionEvent;

    entry_points.subscribe_to_entry_point_changes(
        [&](SubscriptionEvent const& event, std::vector<std::any> const& arguments) {
            std::string entry_point_name;

            if (arguments.size() > 0) {
                entry_point_name = std::any_cast<std::string>(arguments[0]);

                // if the entry point is not new to us, we ignore it

                if (entry_point_name.find(vr_service_marker) != std::string::npos) {
                    return;
                }

                if (entry_point_name.find("GUI Service") != std::string::npos) {
                    return;
                }
            }

            switch (event) {
            case SubscriptionEvent::Add:
                m_entry_points_registry.add_entry_point(
                    entry_point_name, std::any_cast<frontend_resources::EntryPointRenderFunctions const>(arguments[1]));
                break;
            case SubscriptionEvent::Remove:
                m_entry_points_registry.remove_entry_point(entry_point_name);
                break;
            case SubscriptionEvent::Rename:
                m_entry_points_registry.rename_entry_point(
                    entry_point_name, std::any_cast<std::string const>(arguments[1]));
                break;
            case SubscriptionEvent::Clear:
                m_entry_points_registry.clear_entry_points();
                break;
            default:
                log_error("unknown ImagePresentationEntryPoints::SubscriptionEvent type");
                break;
            }
        });

    // pass entry point changes to the currently active IVR_Device
    m_entry_points_registry.add_entry_point = [&](auto const& name, auto const& callbacks) -> bool {
        vr_device(add_entry_point(
            name, callbacks, const_cast<frontend_resources::ImagePresentationEntryPoints&>(entry_points)));

        return true;
    };
    m_entry_points_registry.remove_entry_point = [&](auto const& name) -> bool {
        vr_device(
            remove_entry_point(name, const_cast<frontend_resources::ImagePresentationEntryPoints&>(entry_points)));

        return true;
    };
    m_entry_points_registry.rename_entry_point = [&](auto const& name, auto const& newname) -> bool { return true; };
    m_entry_points_registry.clear_entry_points = [&]() -> void { vr_device(clear_entry_points()); };
    m_entry_points_registry.subscribe_to_entry_point_changes = [&](auto const& func) -> void {};
    m_entry_points_registry.get_entry_point = [&](std::string const& name) -> auto {
        return std::nullopt;
    };

    auto& megamol_graph = m_requestedResourceReferences[1].getResource<core::MegaMolGraph>();
#ifdef MEGAMOL_USE_VR_INTEROP
    // Unity Kolab wants to manipulate graph modules like clipping plane or views
    static_cast<VR_Service::KolabBW*>(m_vr_device_ptr.get())
        ->add_graph(const_cast<core::MegaMolGraph*>(&megamol_graph));
#endif // MEGAMOL_USE_VR_INTEROP
}

void VR_Service::updateProvidedResources() {}

void VR_Service::digestChangedRequestedResources() {}

void VR_Service::resetProvidedResources() {}

// let the IVR_Device do things before and after graph rendering
void VR_Service::preGraphRender() {
    vr_device(preGraphRender());
}

void VR_Service::postGraphRender() {
    vr_device(postGraphRender());
}

} // namespace frontend
} // namespace megamol

#ifdef MEGAMOL_USE_VR_INTEROP
namespace {
glm::vec4 toGlm(const interop::vec4& v) {
    return glm::vec4{v.x, v.y, v.z, v.w};
}
} // namespace

struct megamol::frontend::VR_Service::KolabBW::PimplData {
    interop::StereoCameraView stereoCameraView;
    interop::CameraProjection cameraProjection;
    interop::BoundingBoxCorners bboxCorners;
    bool has_camera_view = false;
    bool has_camera_proj = false;
    bool has_bbox = false;
    bool ep_handles_installed = false;

    std::string ep_name;
    frontend_resources::EntryPoint* left_ep = nullptr;
    frontend_resources::EntryPoint* right_ep = nullptr;

    frontend_resources::gl_texture left_ep_result;
    frontend_resources::gl_texture right_ep_result;

    interop::TextureSender left_texturesender;
    interop::TextureSender right_texturesender;
    interop::TexturePackageSender singlestereo_texturesender;

    interop::DataReceiver stereoCameraViewReceiver_relative;
    interop::DataReceiver cameraProjectionReceiver;
    interop::DataSender bboxSender;

    std::function<void()> update_entry_point_inputs;

    core::MegaMolGraph* graph_ptr = nullptr;

    interop::DataReceiver clip_plane_positionReceiver, clip_plane_normalReceiver, clip_plane_stateReceiver;
    bool has_plane_position = false, has_plane_normal = false, has_plane_state = false;
    interop::vec4 clip_plane_point, clip_plane_normal;
    bool clip_plane_state = false;
    core::Module* clip_module_ptr = nullptr;

    std::function<core::param::ParamSlot*(std::string const&, core::Module*)> get_param_slot =
        [&](std::string const& name, core::Module* module_ptr) -> core::param::ParamSlot* {
        return dynamic_cast<core::param::ParamSlot*>(module_ptr->FindSlot(name.c_str()));
    };

    // overwrite clip plane position/normal with state received from outside
    void update_clip_plane_state(interop::vec4 const& pos, interop::vec4 const& normal, bool const state) {
        if (!graph_ptr)
            return;

        if (!clip_module_ptr) {
            // find clip module
            auto& modules = graph_ptr->ListModules();
            auto find_it = std::find_if(modules.begin(), modules.end(),
                [&](core::ModuleInstance_t const& module) { return module.request.className == "ClipPlane"; });

            if (find_it != modules.end()) {
                clip_module_ptr = find_it->modulePtr.get(); // if graph gets cleared we are in trouble
                log("found and using ClipPlane module " + find_it->request.id);
            }
        }

        if (!clip_module_ptr)
            return;

        auto* color_slot = get_param_slot("colour", clip_module_ptr);
        auto* point_slot = get_param_slot("point", clip_module_ptr);
        auto* normal_slot = get_param_slot("normal", clip_module_ptr);
        auto* dist_slot = get_param_slot("dist", clip_module_ptr);
        auto* enable_slot = get_param_slot("enable", clip_module_ptr);

        if (!point_slot || !normal_slot || !enable_slot) {
            log_error("could not find point or normal or enable Parameter slot(s) in ClipPlane module " +
                      std::string{clip_module_ptr->FullName().PeekBuffer()});
            return;
        }

        vislib::math::Vector<float, 3> point{pos.x, pos.y, pos.z};
        vislib::math::Vector<float, 3> norml{normal.x, normal.y, normal.z};
        norml.Normalise();

        point_slot->Param<core::param::Vector3fParam>()->SetValue(point, true);
        normal_slot->Param<core::param::Vector3fParam>()->SetValue(norml, true);
        enable_slot->Param<core::param::BoolParam>()->SetValue(state, true);
    }

    interop::DataReceiver view_animation_stateReceiver;
    bool view_animation_state = false;
    bool has_animation_state = false;
    core::Module* view_module_ptr = nullptr;

    // play/stop view dataset animation depending on received animation state
    void update_view_animation_state(bool const state) {
        if (!view_module_ptr)
            return;

        auto* play_slot = get_param_slot("anim::play", view_module_ptr);

        if (!play_slot) {
            log_error("could not find anim::play slot in view module " +
                      std::string{view_module_ptr->FullName().PeekBuffer()});
            return;
        }

        play_slot->Param<core::param::BoolParam>()->SetValue(state, true);
    }

    // entry points bring callbacks with them which define how the entry point gets rendered ("executed")
    // we rig/replace the original entry point rendering/execution callbacks with our own to achieve two things:
    // 1) we copy the view rendering results into a texture that we own so we can send that rendering to the Unity Kolab process
    // 2) we override the entry point camera data with the VR rendering settings coming from Unity
    void rig_ep_rendering_result_texture_copy() {
        // when this gets called the entry points are not null

        auto rig_ep = [&](auto& original_execute, auto& result_gl_copy) {
            auto old_execute = original_execute;

            // our entry point execution callback copies the entry point rendering result into a separate texture
            // we need to copy the rendering result texture because the frontend only holds a reference
            // to the view FBO, thus after rendering the second eye the view FBO loses the rendering of the first eye
            auto new_execute = std::function<bool(void*, std::vector<megamol::frontend::FrontendResource> const&,
                megamol::frontend_resources::ImageWrapper&)>{
                [&, old_execute](void* module_ptr, auto& resources, auto& result_image) -> bool {
                    bool success = old_execute(module_ptr, resources, result_image);

                    frontend_resources::gl_texture tmp_tex{result_image}; // no copy
                    result_gl_copy = tmp_tex;                             // copy GL texture

                    return success;
                }};
            original_execute = new_execute;
        };

        rig_ep(left_ep->execute, left_ep_result);
        rig_ep(right_ep->execute, right_ep_result);
    }
};
#define pimpl (*m_pimpl)

megamol::frontend::VR_Service::KolabBW::KolabBW() {
    m_pimpl = std::unique_ptr<PimplData, std::function<void(PimplData*)>>(new PimplData,
        // destructor function for pimpl
        std::function<void(PimplData*)>([](PimplData* ptr) {
            ptr->cameraProjectionReceiver.stop();
            ptr->stereoCameraViewReceiver_relative.stop();
            ptr->bboxSender.stop();

            ptr->clip_plane_positionReceiver.stop();
            ptr->clip_plane_normalReceiver.stop();
            ptr->clip_plane_stateReceiver.stop();

            ptr->view_animation_stateReceiver.stop();

            ptr->left_texturesender.destroy();
            ptr->right_texturesender.destroy();
            ptr->singlestereo_texturesender.destroy();

            delete ptr;
        }));

    std::string default_name = "/UnityInterop/DefaultName";
    pimpl.left_texturesender.init(default_name + "Left");
    pimpl.right_texturesender.init(default_name + "Right");
    pimpl.singlestereo_texturesender.init(default_name + "SingleStereo", 400, 600);

    pimpl.stereoCameraViewReceiver_relative.start("tcp://localhost:12345", "StereoCameraViewRelative");
    pimpl.cameraProjectionReceiver.start("tcp://localhost:12345", "CameraProjection");
    pimpl.bboxSender.start("tcp://127.0.0.1:12346", "BoundingBoxCorners");

    pimpl.clip_plane_positionReceiver.start("tcp://localhost:12345", "CuttingPlanePoint");
    pimpl.clip_plane_normalReceiver.start("tcp://localhost:12345", "CuttingPlaneNormal");
    pimpl.clip_plane_stateReceiver.start("tcp://localhost:12345", "CuttingPlaneState");

    pimpl.view_animation_stateReceiver.start("tcp://localhost:12345", "AnimationRemoteControl");
}

megamol::frontend::VR_Service::KolabBW::~KolabBW() {}

void megamol::frontend::VR_Service::KolabBW::add_graph(void* ptr) {
    if (!m_pimpl)
        return;

    pimpl.graph_ptr = static_cast<core::MegaMolGraph*>(ptr);
}

void megamol::frontend::VR_Service::KolabBW::receive_camera_data() {
    if (!pimpl.left_ep || !pimpl.right_ep)
        return;

    pimpl.has_camera_view |=
        pimpl.stereoCameraViewReceiver_relative.getData<interop::StereoCameraView>(pimpl.stereoCameraView);
    pimpl.has_camera_proj |= pimpl.cameraProjectionReceiver.getData<interop::CameraProjection>(pimpl.cameraProjection);

    if (!pimpl.ep_handles_installed && pimpl.has_camera_view && pimpl.has_camera_proj)
        pimpl.update_entry_point_inputs();


    pimpl.has_plane_position |= pimpl.clip_plane_positionReceiver.getData<interop::vec4>(pimpl.clip_plane_point);
    pimpl.has_plane_normal |= pimpl.clip_plane_normalReceiver.getData<interop::vec4>(pimpl.clip_plane_normal);
    pimpl.has_plane_state |= pimpl.clip_plane_stateReceiver.getData<bool>(pimpl.clip_plane_state);

    if (pimpl.has_plane_position && pimpl.has_plane_normal && pimpl.has_plane_state) {
        pimpl.update_clip_plane_state(pimpl.clip_plane_point, pimpl.clip_plane_normal, pimpl.clip_plane_state);
    }

    pimpl.has_animation_state |= pimpl.view_animation_stateReceiver.getData<bool>(pimpl.view_animation_state);
    if (pimpl.has_animation_state) {
        pimpl.update_view_animation_state(pimpl.view_animation_state);
    }
}

void megamol::frontend::VR_Service::KolabBW::send_image_data() {
    if (!pimpl.left_ep || !pimpl.right_ep)
        return;

    auto& left = pimpl.left_ep_result;
    auto& right = pimpl.right_ep_result;

    pimpl.left_texturesender.sendTexture(left.as_gl_handle(), left.size.width, left.size.height);
    pimpl.right_texturesender.sendTexture(right.as_gl_handle(), right.size.width, right.size.height);

    pimpl.singlestereo_texturesender.sendTexturePackage(
        left.as_gl_handle(), right.as_gl_handle(), 0, 0, left.size.width, left.size.height);

    if (!pimpl.has_bbox) {
        auto maybe_bbox = static_cast<core::view::AbstractView*>(pimpl.left_ep->modulePtr)->GetBoundingBoxes();

        if (!maybe_bbox.IsBoundingBoxValid())
            return;

        auto bbox = maybe_bbox.BoundingBox();

        pimpl.bboxCorners.min = interop::vec4{
            bbox.GetLeftBottomBack().GetX(), bbox.GetLeftBottomBack().GetY(), bbox.GetLeftBottomBack().GetZ(), 0.0f};

        pimpl.bboxCorners.max = interop::vec4{
            bbox.GetRightTopFront().GetX(), bbox.GetRightTopFront().GetY(), bbox.GetRightTopFront().GetZ(), 0.0f};

        //pimpl.has_bbox = true;
    }

    pimpl.bboxSender.sendData<interop::BoundingBoxCorners>("BoundingBoxCorners", pimpl.bboxCorners);
}

// what we do with incoming view entry points is to register them a second time as entry point
// the view module in the graph will exist only one time, but two entry points in the image presentation will
// feed that view module with left/right eye camera configurations, which we set up here
// at the same time the entry points will write back the view rendering results to textures we own, which we send out to Unity
bool megamol::frontend::VR_Service::KolabBW::add_entry_point(std::string const& entry_point_name,
    frontend_resources::EntryPointRenderFunctions const& entry_point_callbacks,
    ImagePresentationEntryPoints& entry_points_registry) {

    // the incoming entry point seems to be a new one that is not marked as belonging to the VR service or the GUI
    // marked entry points simply mean "artificially added as entry point by the VR service"
    auto marked_entry_point_name = mark(entry_point_name);

    // add an entry point for the second eye
    // the original entry point will be the left eye, the new (marked) entry point will be the right eye
    // QUESTION: is it a good idea to switch between left/right camera parameters in the same view? will weird (slow?) things happen down the line?
    bool success = entry_points_registry.add_entry_point(marked_entry_point_name, entry_point_callbacks);

    if (!success)
        return false;

    pimpl.ep_name = entry_point_name;

    auto ep_left = entry_points_registry.get_entry_point(entry_point_name);
    auto ep_right = entry_points_registry.get_entry_point(marked_entry_point_name);

    if (!ep_left.has_value() || !ep_right.has_value()) {
        log_error("could not retrieve entry points " + entry_point_name + " and " + marked_entry_point_name);
        return false;
    }

    // this cast will never fail but the modulePtr should be valid nonetheless
    // no need to check right entry point because they represent the same view module
    auto* ptr = static_cast<megamol::core::Module*>(ep_left.value().get().modulePtr);
    if (!ptr) {
        log_error(
            "entry point " + entry_point_name + " does not seem to have a valid megamol::core::Module* (is nullptr).");
        return false;
    }

    // if the entry point is not a 3d view there is no point in doing stereo for it
    const auto* view = dynamic_cast<megamol::core::view::AbstractViewInterface*>(ptr);
    if (view == nullptr || view->GetViewDimension() != core::view::AbstractViewInterface::ViewDimension::VIEW_3D) {
        log_error("entry point " + entry_point_name +
                  " does not seem to be a supported 3D View Type. Not using it for stereo rendering.");
        return false;
    }

    pimpl.view_module_ptr = ptr;

    pimpl.left_ep = &ep_left.value().get();
    pimpl.right_ep = &ep_right.value().get();

    pimpl.rig_ep_rendering_result_texture_copy();

    // outputs a function that returns camera matrices built from current pimpl camera data
    auto make_matrices = [&](interop::CameraView const& iview, interop::CameraProjection const& iproj)
        -> std::function<std::optional<frontend_resources::RenderInput::CameraMatrices>()> {
        // captures iview and iproj by reference, directly accessing current pimpl data
        // note that these matrices do not account for any tiling!
        return [&]() {
            glm::mat4 view = glm::lookAt(
                glm::vec3(toGlm(iview.eyePos)), glm::vec3(toGlm(iview.lookAtPos)), glm::vec3(toGlm(iview.camUpDir)));

            glm::mat4 projection =
                glm::perspective(iproj.fieldOfViewY_rad, iproj.aspect, iproj.nearClipPlane, iproj.farClipPlane);

            return std::make_optional(frontend_resources::RenderInput::CameraMatrices{view, projection});
        };
    };

    auto make_view_projection_parameters = [&](interop::CameraView const& iview, interop::CameraProjection const& iproj)
        -> std::function<std::optional<frontend_resources::RenderInput::CameraViewProjectionParameters>()> {
        return [&]() {
            return std::make_optional(frontend_resources::RenderInput::CameraViewProjectionParameters{
                {// Pose
                    glm::vec3(toGlm(iview.eyePos)),
                    glm::normalize(glm::vec3(toGlm(iview.lookAtPos)) - glm::vec3(toGlm(iview.eyePos))),
                    glm::vec3(toGlm(iview.camUpDir))},
                {// Projection
                    frontend_resources::RenderInput::CameraViewProjectionParameters::ProjectionType::PERSPECTIVE,
                    iproj.fieldOfViewY_rad, iproj.aspect, iproj.nearClipPlane, iproj.farClipPlane}});
        };
    };

    auto fbo_size_handler = [&]() {
        return frontend_resources::UintPair{pimpl.cameraProjection.pixelHeight, pimpl.cameraProjection.pixelWidth};
    };

    auto tile_handler = [&, fbo_size_handler]() {
        return frontend_resources::ViewportTile{fbo_size_handler(), {0, 0}, {1, 1}};
    };

    // for the left/right entry point, replace the render input getter callbacks with our own
    // note that pimpl members are taken by reference by the lambdas (referencing updated camera config in pimpl)
    // while the lambdas themselves should be passed by value

    // it is ok to reference the pimpl data because view rendering and pimpl data updates DONT happen at the same time (see VR Service pre/postGraphRender)

    auto replace_ep_handlers = [&, fbo_size_handler, tile_handler](frontend_resources::EntryPoint& ep,
                                   auto make_matrices, auto make_view_projection_parameters) {
        // setting correct FBO size is important
        accessViewRenderInput(ep.entry_point_data).render_input_framebuffer_size_handler = fbo_size_handler;

        // tiling will be ignored by the view actually, because we probably overwrite the camera matrices directly
        accessViewRenderInput(ep.entry_point_data).render_input_tile_handler = tile_handler;

        // enforce our view/projection matrices
        // accessViewRenderInput(ep.entry_point_data).render_input_camera_handler = make_matrices;

        // enforcing view/projection matrices may actually lead to problems
        // so for now we pass the actual camera parametrization to the view
        accessViewRenderInput(ep.entry_point_data).render_input_camera_parameters_handler =
            make_view_projection_parameters;

        pimpl.ep_handles_installed = true;
    };

    // we dont replace the entry point render input callbacks now (during entry point creation) but later,
    // when the Kolab VR device received its first camera config state from the Unity process
    pimpl.update_entry_point_inputs = std::function<void()>([&, replace_ep_handlers]() {
        // view each entry point holds a resource with individual render inputs (frontend_resources::ViewRenderInpts)
        // the render inptus hold callbacks which tell them the next reder input configuration (e.g. fbo size, tiling config)
        // we overwrite those callbacks with our own, such that the view entry points get camera data as defined by us
        // when they look up render inputs for the next frame
        // the actual render input lookup/update (calling the render input callbacks)
        // happens in ImagePresentation.RenderNextFrame() right before view rendering
        replace_ep_handlers(*pimpl.left_ep, make_matrices(pimpl.stereoCameraView.leftEyeView, pimpl.cameraProjection),
            make_view_projection_parameters(pimpl.stereoCameraView.leftEyeView, pimpl.cameraProjection));
        replace_ep_handlers(*pimpl.right_ep, make_matrices(pimpl.stereoCameraView.rightEyeView, pimpl.cameraProjection),
            make_view_projection_parameters(pimpl.stereoCameraView.rightEyeView, pimpl.cameraProjection));
    });

    log("added entry point " + entry_point_name + " for Unity-KolabBW Stereo Rendering.");

    return true;
}

bool megamol::frontend::VR_Service::KolabBW::remove_entry_point(
    std::string const& entry_point_name, ImagePresentationEntryPoints& entry_points_registry) {

    if (pimpl.ep_name != entry_point_name) {
        return false;
    }

    auto marked_entry_point_name = mark(entry_point_name);

    bool success = entry_points_registry.remove_entry_point(marked_entry_point_name);

    this->clear_entry_points();

    log("remove entry points " + entry_point_name + ", " + marked_entry_point_name +
        " from Unity-KolabBW Stereo Rendering");

    if (!success)
        return false;

    return true;
}

void megamol::frontend::VR_Service::KolabBW::clear_entry_points() {
    pimpl.left_ep = nullptr;
    pimpl.right_ep = nullptr;
    pimpl.ep_name.clear();
}

void megamol::frontend::VR_Service::KolabBW::preGraphRender() {
    this->receive_camera_data();
}

void megamol::frontend::VR_Service::KolabBW::postGraphRender() {
    this->send_image_data();
}
#endif // MEGAMOL_USE_VR_INTEROP
