/*
 * FrontendResourcesLookup.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "FrontendResource.h"
#include "mmcore/utility/log/Log.h"

#include <iostream>
#include <vector>
#include <algorithm> // std::find_if

namespace megamol {
namespace frontend_resources {

struct FrontendResourcesLookup {

    FrontendResourcesLookup()
        : resources{}
    {}

    FrontendResourcesLookup(std::vector<frontend::FrontendResource> const& resources)
        : resources{resources}
    {}

    using ResourceLookupResult = std::tuple<bool, std::vector<megamol::frontend::FrontendResource>>;

    ResourceLookupResult get_requested_resources(std::vector<std::string> resource_requests) const {
        std::vector<megamol::frontend::FrontendResource> result_resources;
        result_resources.reserve(resource_requests.size());

        bool success = true;

        for (auto& request : resource_requests) {
            auto dependency_it = std::find_if(this->resources.begin(), this->resources.end(), [&](megamol::frontend::FrontendResource const& dependency) {
                return request.find(dependency.getIdentifier()) != std::string::npos;
            });

            bool resource_found = dependency_it != resources.end();

            success &= resource_found;

            if (resource_found) {
                auto& resource = *dependency_it;
                result_resources.push_back(resource);
            } else {
                auto msg = std::string("FrontendResourcesLookup: Fatal Error: ")
                    + "\n\tcould not find requested resource " + request;
                megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
            }
        }

        return {success, result_resources};
    }

    private:
    std::vector<frontend::FrontendResource> resources;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
