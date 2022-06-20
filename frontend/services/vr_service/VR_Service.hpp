/*
 * VR_Service.hpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractFrontendService.hpp"

#include "ImagePresentationEntryPoints.h"

#include <memory>

namespace megamol {
namespace frontend {

// VR Service monitors Image Presentation Entry Points (Views, which output graph rendering results)
// and injects them with VR/AR data as it sees fit.
// this means we may take control over View Camera state
// (resolution, tiling, camera pose/projection) and inject our own data to achieve stereo rendering.
// we may also manipulate entry points set by the graph,
// delete or clone them to get the number of output renderings of the graph we need.

class VR_Service final : public AbstractFrontendService {
public:
    struct Config {
        enum class Mode {
            Off,
#ifdef WITH_VR_SERVICE_UNITY_KOLABBW
            UnityKolabBW,
#endif // WITH_VR_SERVICE_UNITY_KOLABBW
        };

        Mode mode = Mode::Off;
    };

    std::string serviceName() const override {
        return "VR_Service";
    }

    VR_Service();
    ~VR_Service();

    bool init(const Config& config);
    bool init(void* configPtr) override;
    void close() override;

    std::vector<FrontendResource>& getProvidedResources() override;
    const std::vector<std::string> getRequestedResourceNames() const override;
    void setRequestedResources(std::vector<FrontendResource> resources) override;

    void updateProvidedResources() override;
    void digestChangedRequestedResources() override;
    void resetProvidedResources() override;

    void preGraphRender() override;
    void postGraphRender() override;

    // from AbstractFrontendService
    //
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    //
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

    // we use the marker to mark entry point names as "belongs to vr service"
    static std::string vr_service_marker;

private:
    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;

    using ImagePresentationEntryPoints = megamol::frontend_resources::ImagePresentationEntryPoints;

    ImagePresentationEntryPoints m_entry_points_registry;

    // i believe the abstraction we want for VR rendering/interactions in MegaMol could be put into a "vr device"
    // the general interface of a "vr device" is that it wants to be notified of MegaMol entry point changes
    // and act on those entry points before/after Graph rendering (when the renderings of the entry points update)
    // so the VR_Device is not an actual piece of VR hardware, but a "handler" of how MegaMol does VR things
    // for example, the VR_Device would enforce some camera pose/projection for some entry point before rendering
    // and send the rendering result to the VR output after rendering
    struct IVR_Device {
        virtual ~IVR_Device() = default;

        bool virtual add_entry_point(std::string const& entry_point_name,
            frontend_resources::EntryPointRenderFunctions const& entry_point_callbacks,
            ImagePresentationEntryPoints& entry_points_registry) = 0;
        bool virtual remove_entry_point(
            std::string const& entry_point_name, ImagePresentationEntryPoints& entry_points_registry) = 0;
        void virtual clear_entry_points() = 0;

        void virtual preGraphRender() {}
        void virtual postGraphRender() {}
    };

    // the KolabBW implements VR capabilities by receiving reomte camera information from ZMQ channels
    // and sending out the corresponding MegaMol renderings via Spout
    // if a MegaMol View is registered as entry point in the Image Presentation Service
    // the KolabBW intercepts that entry point, uses it as the left eye for VR rendering,
    // and adds another entry point of his own, which will be used as right eye for VR rendering
    // to inject camera state for VR rendering, the entry point execution callbacks
    // (which define how entry points get their rendering data before rendering is triggered)
    // will be rigged with VR information
    struct KolabBW : public IVR_Device {
        KolabBW();
        ~KolabBW();

        // network communication
        void receive_camera_data();
        void send_image_data();

        bool add_entry_point(std::string const& entry_point_name,
            frontend_resources::EntryPointRenderFunctions const& entry_point_callbacks,
            ImagePresentationEntryPoints& entry_points_registry) override;
        bool remove_entry_point(
            std::string const& entry_point_name, ImagePresentationEntryPoints& entry_points_registry) override;
        void clear_entry_points();

        void preGraphRender() override;
        void postGraphRender() override;

        // to implement clipping plane and view animation manipulation we need access to the MegaMol Graph
        void add_graph(void* ptr);

        // actual implementation details are in the .cpp file
        struct PimplData;
        std::unique_ptr<PimplData, std::function<void(PimplData*)>> m_pimpl;
    };

    std::unique_ptr<IVR_Device> m_vr_device_ptr;
#define vr_device(do)        \
    if (m_vr_device_ptr) {   \
        m_vr_device_ptr->do; \
    }
};

} // namespace frontend
} // namespace megamol
