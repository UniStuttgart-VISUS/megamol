/*
 * ImagePresentation_Service.hpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractFrontendService.hpp"

#include "ImagePresentationEntryPoints.h"
#include "ImageWrapper.h"

#include <list>

namespace megamol {
namespace frontend {

class ImagePresentation_Service final : public AbstractFrontendService {
public:

    struct Config {
    };

    std::string serviceName() const override { return "ImagePresentation_Service"; }

    ImagePresentation_Service();
    ~ImagePresentation_Service();

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

    // the Image Presentation Service is special in that it manages the objects (Graph Entry Points, or possibly other objects)
    // that are triggered to render something into images.
    // The resulting images are then presented in some appropriate way: drawn into a window, written to disk, sent via network, ...
    void RenderNextFrame();
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    //
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

    using ImageWrapper = megamol::frontend_resources::ImageWrapper;
    using EntryPointExecutionCallback =
        std::function<bool(
              void*
            , std::vector<megamol::frontend::FrontendResource> const&
            , ImageWrapper&
            )>;

private:

    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;

    // for each View in the MegaMol graph we create a GraphEntryPoint with corresponding callback for resource/input consumption
    // the ImagePresentation Service makes sure that the (lifetime and rendering) resources/dependencies requested by the module
    // are satisfied, which means that the execute() callback for the entry point is provided the requested
    // dependencies/resources for rendering
    megamol::frontend_resources::ImagePresentationEntryPoints m_entry_points_registry_resource; // resorce to add/remove entry points

    struct GraphEntryPoint {
        std::string moduleName;
        void* modulePtr = nullptr;
        std::vector<megamol::frontend::FrontendResource> entry_point_resources;

        EntryPointExecutionCallback execute;
        std::reference_wrapper<ImageWrapper> execution_result_image;
    };
    std::list<GraphEntryPoint> m_entry_points;

    bool add_entry_point(std::string name, void* module_raw_ptr);
    bool remove_entry_point(std::string name);
    bool rename_entry_point(std::string oldName, std::string newName);
    bool clear_entry_points();

    std::list<ImageWrapper> m_wrapped_images;

    std::vector<megamol::frontend::FrontendResource> map_resources(std::vector<std::string> const& requests);
    const std::vector<FrontendResource>* m_frontend_resources_ptr = nullptr;

};

} // namespace frontend
} // namespace megamol
