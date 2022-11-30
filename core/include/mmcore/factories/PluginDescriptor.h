/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include "AbstractPluginInstance.h"

namespace megamol::core::factories {
class AbstractPluginDescriptor {
public:
    [[nodiscard]] virtual std::shared_ptr<AbstractPluginInstance> create() const = 0;
};

template<class C>
class PluginDescriptor : public AbstractPluginDescriptor {
public:
    [[nodiscard]] std::shared_ptr<AbstractPluginInstance> create() const override {
        return std::make_shared<C>();
    }
};
} // namespace megamol::core::factories
