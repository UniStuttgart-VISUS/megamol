/*
 * FrontendResourcesMap.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "FrontendResource.h"

#include <vector>
#include <map>
#include <string>
#include <iostream>

namespace megamol {
namespace frontend_resources {

struct FrontendResourcesMap {
    FrontendResourcesMap() = default;

    FrontendResourcesMap (std::vector<frontend::FrontendResource> const& resources) {
        for (auto& resource : resources) {
            auto hash = resource.getHash();
            auto already_exists = this->resources.count(hash) != 0 && this->resources.at(hash).getIdentifier() != resource.getIdentifier();
            if (already_exists) {
                auto msg = std::string("FrontendResourcesMap: Fatal Error: ")
                    + "\n\tresource type hash " + std::to_string(hash)
                    + " already present in map, introduced by resource " + this->resources.at(hash).getIdentifier()
                    + "\n\tcan not add resource " + resource.getIdentifier()
                    + "\n\tbecause that would lead to ambiguous map entries for your resources "
                    + "\n\tstopping program execution ";

                std::cout << msg << std::endl;
                std::cerr << msg << std::endl;
                std::exit(1);
            } else {
                this->resources.insert({hash, resource});
            }
        }
    }

    // hash map lookup for resource using resource type id
    template <typename ResourceType>
    ResourceType const& get() const {
        return resources.at(typeid(ResourceType).hash_code()).getResource<ResourceType>();
    }

    private:
    std::map<std::size_t, frontend::FrontendResource> resources;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
