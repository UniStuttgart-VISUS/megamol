/*
 * VR_Service.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "VR_Service.hpp"

#ifdef WITH_VR_SERVICE_UNITY_KOLABBW
#include <glm/gtc/matrix_transform.hpp>
#include <include/interop.hpp>

#include "ImageWrapper_to_GLTexture.hpp"
#include "ViewRenderInputs.h"

#include "mmcore/Module.h"
#include "mmcore/view/View3D.h"
#include "mmcore_gl/view/View3DGL.h"
#endif // WITH_VR_SERVICE_UNITY_KOLABBW

// local logging wrapper for your convenience until central MegaMol logger established
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
#ifdef WITH_VR_SERVICE_UNITY_KOLABBW
        "OpenGL_Context",
#endif // WITH_VR_SERVICE_UNITY_KOLABBW
    };

    switch (config.mode) {
    case Config::Mode::Off:
        break;
#ifdef WITH_VR_SERVICE_UNITY_KOLABBW
    case Config::Mode::UnityKolabBW:
        log("running Unity KolabBW VR mode. Duplicating and sending View3D Entry Points via Spout.");
        m_vr_device_ptr = std::make_unique<VR_Service::KolabBW>();
        break;
#endif // WITH_VR_SERVICE_UNITY_KOLABBW
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

    using SubscriptionEvent = frontend_resources::ImagePresentationEntryPoints::SubscriptionEvent;

    entry_points.subscribe_to_entry_point_changes(
        [&](SubscriptionEvent const& event, std::vector<std::any> const& arguments) {
            std::string entry_point_name;

            if (arguments.size() > 0) {
                entry_point_name = std::any_cast<std::string>(arguments[0]);

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

    // VR stereo rendering: clone each view entry point in order to have two images rendered: left + right
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
}

void VR_Service::updateProvidedResources() {}

void VR_Service::digestChangedRequestedResources() {}

void VR_Service::resetProvidedResources() {}

void VR_Service::preGraphRender() {
    vr_device(preGraphRender());
}

void VR_Service::postGraphRender() {
    vr_device(postGraphRender());
}

} // namespace frontend
} // namespace megamol

#ifdef WITH_VR_SERVICE_UNITY_KOLABBW
namespace {
glm::vec4 toGlm(const interop::vec4& v) {
    return glm::vec4{v.x, v.y, v.z, v.w};
}
} // namespace

struct megamol::frontend::VR_Service::KolabBW::PimplData {
    struct Image {
        unsigned int gl_handle = 0;
        unsigned int width = 0;
        unsigned int height = 0;
    };

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

    frontend_resources::gl_texture left_ep_result{{}};
    frontend_resources::gl_texture right_ep_result{{}};

    interop::TextureSender left_texturesender;
    interop::TextureSender right_texturesender;

    interop::DataReceiver stereoCameraViewReceiver_relative;
    interop::DataReceiver cameraProjectionReceiver;
    interop::DataSender bboxSender;

    std::function<void()> update_entry_point_inputs;

    void rig_ep_execution() {
        // when this gets called the entry points are not null

        auto rig_ep = [&](auto& original_execute, auto& result_gl_copy) {
            auto old_execute = original_execute;

            auto new_execute = std::function<bool(void*, std::vector<megamol::frontend::FrontendResource> const&, megamol::frontend_resources::ImageWrapper&)>{
                [&,old_execute](void* module_ptr, auto& resources, auto& result_image) -> bool {
                        bool success = old_execute(module_ptr, resources, result_image);

                        frontend_resources::gl_texture tmp_tex{result_image}; // no copy
                        result_gl_copy = tmp_tex; // copy GL texture

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
    m_pimpl = std::unique_ptr<PimplData, std::function<void(PimplData*)>>(
        new PimplData, std::function<void(PimplData*)>([](PimplData* ptr) {
            ptr->cameraProjectionReceiver.stop();
            ptr->stereoCameraViewReceiver_relative.stop();
            ptr->bboxSender.stop();

            ptr->left_texturesender.destroy();
            ptr->right_texturesender.destroy();
            //ptr->texturepackageSender.destroy();

            delete ptr;
        }));

    std::string default_name = "/UnityInterop/DefaultName";
    pimpl.left_texturesender.init(default_name + "Left");
    pimpl.right_texturesender.init(default_name + "Right");

    pimpl.stereoCameraViewReceiver_relative.start("tcp://localhost:12345", "StereoCameraViewRelative");
    pimpl.cameraProjectionReceiver.start("tcp://localhost:12345", "CameraProjection");
    pimpl.bboxSender.start("tcp://127.0.0.1:12346", "BoundingBoxCorners");
}

megamol::frontend::VR_Service::KolabBW::~KolabBW() {}

void megamol::frontend::VR_Service::KolabBW::receive_camera_data() {
    if (!pimpl.left_ep || !pimpl.right_ep)
        return;

    pimpl.has_camera_view |=
        pimpl.stereoCameraViewReceiver_relative.getData<interop::StereoCameraView>(pimpl.stereoCameraView);
    pimpl.has_camera_proj |= pimpl.cameraProjectionReceiver.getData<interop::CameraProjection>(pimpl.cameraProjection);

    if (!pimpl.ep_handles_installed && pimpl.has_camera_view && pimpl.has_camera_proj)
        pimpl.update_entry_point_inputs();
}

void megamol::frontend::VR_Service::KolabBW::send_image_data() {
    if (!pimpl.left_ep || !pimpl.right_ep)
        return;

    auto& left = pimpl.left_ep_result;
    auto& right = pimpl.right_ep_result;

    pimpl.left_texturesender.sendTexture(left.as_gl_handle(), left.size.width, left.size.height);
    pimpl.right_texturesender.sendTexture(right.as_gl_handle(), right.size.width, right.size.height);

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

bool megamol::frontend::VR_Service::KolabBW::add_entry_point(std::string const& entry_point_name,
    frontend_resources::EntryPointRenderFunctions const& entry_point_callbacks,
    ImagePresentationEntryPoints& entry_points_registry) {

    // add an entry point for the second eye
    // the original entry point will be the left eye, the new (marked) entry point will be the right eye
    // QUESTION: is it a good idea to switch between left/right camera parameters in the same view? will weird (slow?) things happen down the line?
    bool success = entry_points_registry.add_entry_point(mark(entry_point_name), entry_point_callbacks);

    if (!success)
        return false;

    pimpl.ep_name = entry_point_name;

    frontend_resources::optional<frontend_resources::EntryPoint> ep_left =
        entry_points_registry.get_entry_point(entry_point_name);
    frontend_resources::optional<frontend_resources::EntryPoint> ep_right =
        entry_points_registry.get_entry_point(mark(entry_point_name));

    if (!ep_left.has_value() || !ep_right.has_value()) {
        log_error("could not retrieve entry points " + entry_point_name + " and " + mark(entry_point_name));
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
    auto* view3d = dynamic_cast<megamol::core::view::View3D*>(ptr);
    auto* view3dgl = dynamic_cast<megamol::core_gl::view::View3DGL*>(ptr);
    if (!view3d && !view3dgl) {
        log_error(
            "entry point " + entry_point_name +
            " does not seem to be a supported View Type (View3D or View3DGL). Not using it for stereo rendering.");
        return false;
    }

    pimpl.left_ep = &ep_left.value().get();
    pimpl.right_ep = &ep_right.value().get();

    pimpl.rig_ep_execution();

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

    bool success = entry_points_registry.remove_entry_point(mark(entry_point_name));

    this->clear_entry_points();

    log("remove entry points " + entry_point_name + ", " + mark(entry_point_name) +
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
#endif // WITH_VR_SERVICE_UNITY_KOLABBW
