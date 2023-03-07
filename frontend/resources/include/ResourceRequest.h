/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <typeindex>
#include <typeinfo>
#include <vector>

#include "mmcore/utility/platform/TypeInfo.h"

namespace megamol::frontend_resources {
struct ResourceDescriptor {
    std::type_index type;
    bool optional;

    std::string TypeName() const {
        return core::utility::platform::DemangleTypeName(type.name());
    }
};

class ResourceRequest {
public:
    template<class T>
    void require() {
        resources_.emplace_back(ResourceDescriptor{typeid(T), false});
    }

    template<class T>
    void optional() {
        resources_.emplace_back(ResourceDescriptor{typeid(T), true});
    }

    const std::vector<ResourceDescriptor>& getResources() const {
        return resources_;
    }

private:
    std::vector<ResourceDescriptor> resources_;
};
} // namespace megamol::frontend_resources
