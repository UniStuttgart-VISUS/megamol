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

    static std::string vr_service_marker;

private:
    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;

    using ImagePresentationEntryPoints = megamol::frontend_resources::ImagePresentationEntryPoints;

    ImagePresentationEntryPoints m_entry_points_registry;

    // puts VR render inputs into entry point
    // whether left/right eye is rendered is done by the update handler
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

    struct KolabBW : public IVR_Device {
        KolabBW();
        ~KolabBW();

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

        void add_graph(void* ptr);

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
