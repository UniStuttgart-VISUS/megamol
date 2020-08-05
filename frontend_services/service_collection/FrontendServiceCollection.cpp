/*
 * OpenGL_GLFW_Service.cpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "FrontendServiceCollection.hpp"

#include <algorithm>
#include <iostream>

namespace megamol {
namespace frontend {

	void FrontendServiceCollection::sortServices() {
        std::sort(m_services.begin(), m_services.end(), [](auto& lhs, auto& rhs) {
			return lhs.get().getPriority() < rhs.get().getPriority();
		});
	}

	void FrontendServiceCollection::add(AbstractFrontendService& service, void* service_config) {
		m_services.push_back({&service, service_config});
        this->sortServices();
	}

	#define for_each_service for (auto& service : m_services)

	bool FrontendServiceCollection::init() {
        bool some_failed = false;

		for_each_service{
			some_failed |= service.service->init(service.service_config);
		}

		return some_failed;
	}

	void FrontendServiceCollection::close() {
        // close services in reverse order,
        // hopeully avoiding dangling references to closed resources
        for(auto it = m_services.rbegin(); it != m_services.rend(); it++) {
            (*it).get().close();
		}
	}

	std::vector<ModuleResource>& FrontendServiceCollection::getProvidedResources() {
		return m_serviceResources;
	}

	// TODO: using the requested resource dependencies we can derive correct update order of services
	bool FrontendServiceCollection::assignRequestedResources() {
		// gather resources from all services
		for_each_service {
            auto& resources = service.get().getProvidedResources();

			// attention: ModuleResources are copyable because of the reference wrapper used internally
			// copying stuff should not break anything, but this frontend manager should be the only one
			// copying ModuleResources because he knows what he is doing
			for (auto& r: resources)
				m_serviceResources.push_back(r);
		}

		// for each servie, provide him with requested resources
        std::vector<ModuleResource> resources;
        for_each_service {
            resources.clear();
			auto request_names = service.get().getRequestedResourceNames();

			for (auto& name : request_names) {
                auto modulePtr = findResource(name);

				if (modulePtr) {
                    resources.push_back(*modulePtr);
                } else {
					// if a requested resource can not be found we fail and should stop program execution
                    std::cout << "could not find resource: " << name << " for service: " << service.get().serviceName() << std::endl;
                    return false;
                }
            }

			service.get().setRequestedResources(resources);
		}

		return true;
	}

	ModuleResource* FrontendServiceCollection::findResource(std::string const& name) {
        auto module_it = std::find_if(m_serviceResources.begin(), m_serviceResources.end(),
            [&](ModuleResource const& resource) { return name == resource.getIdentifier(); });

		if (module_it != m_serviceResources.end())
			return &(*module_it);

		return nullptr;
	}
	
	void FrontendServiceCollection::updateProvidedResources() {
        for_each_service {
			service.get().updateProvidedResources();
		}
	}

	void FrontendServiceCollection::digestChangedRequestedResources() {
        for_each_service {
			service.get().digestChangedRequestedResources();
		}
	}

	void FrontendServiceCollection::resetProvidedResources() {
        for_each_service {
			service.get().resetProvidedResources();
		}
	}

	void FrontendServiceCollection::preGraphRender() {
        for_each_service {
			service.get().preGraphRender();
		}
	}

	void FrontendServiceCollection::postGraphRender() {
        // traverse post update in reverse order
        for(auto it = m_services.rbegin(); it != m_services.rend(); it++) {
			(*it).get().postGraphRender();
		}
	}

	bool FrontendServiceCollection::shouldShutdown() const {
        bool shutdown = false;

        for (auto const& service: m_services){
			shutdown |= service.get().shouldShutdown();
		}

		return shutdown;
	}

} // namespace frontend
} // namespace megamol
