/*
 * FrontendResourcesLookup.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "FrontendResource.h"
#include "mmcore/utility/log/Log.h"

#include <algorithm> // std::find_if
#include <iostream>
#include <vector>

namespace megamol {
namespace frontend_resources {

struct FrontendResourcesLookup {

    FrontendResourcesLookup() : resources{} {}

    FrontendResourcesLookup(std::vector<frontend::FrontendResource> const& resources) : resources{resources} {}

    using ResourceLookupResult = std::tuple<bool, std::vector<megamol::frontend::FrontendResource>>;

    ResourceLookupResult get_requested_resources(std::vector<std::string> resource_requests) const {
        std::vector<megamol::frontend::FrontendResource> result_resources;
        result_resources.reserve(resource_requests.size());

        auto requests_optional = [](std::string const& request) -> bool {
            return request.find("optional<") == 0 && request.back() == '>';
        };

        bool success = true;

        for (auto& request : resource_requests) {
            auto dependency_it = std::find_if(this->resources.begin(), this->resources.end(),
                [&](megamol::frontend::FrontendResource const& dependency) {
                    auto find_pos = request.find(dependency.getIdentifier());
                    return find_pos == 0 || find_pos == 9;
                });

            bool resource_found = dependency_it != resources.end();
            bool resource_optional = requests_optional(request);

            success &= (resource_found || resource_optional);

            if (resource_found) {
                auto& resource = *dependency_it;
                result_resources.push_back(resource_optional ? resource.toOptional() : resource);
            } else {
                if (resource_optional) {
                    result_resources.push_back(megamol::frontend::FrontendResource{}.toOptional());
                } else {
                    auto msg = std::string("FrontendResourcesLookup: Fatal Error: ") +
                               "\n\tcould not find requested resource " + request;
                    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
                }
            }
        }

        return {success, result_resources};
    }

private:
    std::vector<frontend::FrontendResource> resources;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
