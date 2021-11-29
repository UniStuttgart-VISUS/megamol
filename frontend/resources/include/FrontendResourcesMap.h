/*
 * FrontendResourcesMap.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "FrontendResource.h"
#include "mmcore/utility/log/Log.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace megamol {
namespace frontend_resources {

struct FrontendResourcesMap {
    FrontendResourcesMap() = default;

    FrontendResourcesMap(std::vector<frontend::FrontendResource> const& resources) {
        for (auto& resource : resources) {
            auto hash = resource.getHash();
            auto already_exists = this->resources.count(hash) != 0 &&
                                  this->resources.at(hash).getIdentifier() != resource.getIdentifier();
            if (already_exists) {
                auto msg = std::string("FrontendResourcesMap: Fatal Error: ") + "\n\tresource type hash " +
                           std::to_string(hash) + " already present in map, introduced by resource " +
                           this->resources.at(hash).getIdentifier() + "\n\tcan not add resource " +
                           resource.getIdentifier() +
                           "\n\tbecause that would lead to ambiguous map entries for your resources " +
                           "\n\tstopping program execution ";

                megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
                std::exit(1);
            } else {
                this->resources.insert({hash, resource});
            }
        }
    }

    // hash map lookup for resource using resource type id
    template<typename ResourceType>
    ResourceType const& get() const {
        return resources.at(typeid(ResourceType).hash_code()).getResource<ResourceType>();
    }

    template<typename ResourceType>
    optional<const ResourceType> getOptional() const {
        auto key = typeid(ResourceType).hash_code();
        if (resources.count(key) > 0) {
            return resources.at(typeid(ResourceType).hash_code()).getOptionalResource<ResourceType>();
        } else {
            return std::nullopt;
        }
    }

private:
    std::map<std::size_t, frontend::FrontendResource> resources;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
