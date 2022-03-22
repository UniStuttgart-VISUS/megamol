/**
 * MegaMol
 * Copyright (c) 2020-2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_UTILITY_PLUGINS_PLUGINDESCRIPTOR_H_INCLUDED
#define MEGAMOLCORE_UTILITY_PLUGINS_PLUGINDESCRIPTOR_H_INCLUDED

#include <memory>

#include "AbstractPluginInstance.h"

namespace megamol::core::utility::plugins {
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
} // namespace megamol::core::utility::plugins

#endif // MEGAMOLCORE_UTILITY_PLUGINS_PLUGINDESCRIPTOR_H_INCLUDED
