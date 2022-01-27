/*
 * ProjectLoader_Service.hpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractFrontendService.hpp"

#include "ProjectLoader.h"

#include "MegaMolProject.h"

namespace megamol {
namespace frontend {

class ProjectLoader_Service final : public AbstractFrontendService {
public:
    struct Config {};

    std::string serviceName() const override {
        return "ProjectLoader_Service";
    }

    ProjectLoader_Service();
    ~ProjectLoader_Service();

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

    bool load_file(std::filesystem::path const& filename) const;

    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    // bool shouldShutdown() const;// shutdown initially false
    // void setShutdown(const bool s = true);

private:
    megamol::frontend_resources::ProjectLoader m_loader;

    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;

    megamol::frontend_resources::MegaMolProject m_current_project;

    bool m_digestion_recursion = false;
};

} // namespace frontend
} // namespace megamol
