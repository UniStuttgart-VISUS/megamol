/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <vector>

#include "PluginDescriptor.h"

#define REGISTERPLUGIN(classname) \
    [[maybe_unused]] static inline bool is_registered_ = megamol::core::factories::PluginRegister::add<classname>();

namespace megamol::core::factories {
class PluginRegister {
public:
    // Use PluginRegister only static!
    PluginRegister() = delete;
    ~PluginRegister() = delete;
    PluginRegister(const PluginRegister&) = delete;
    PluginRegister(PluginRegister&&) = delete;
    PluginRegister& operator=(const PluginRegister) = delete;
    PluginRegister& operator=(PluginRegister&&) = delete;

    template<class C>
    static bool add() noexcept {
        plugins_.emplace_back(std::make_shared<PluginDescriptor<C>>());
        return true;
    }

    [[nodiscard]] static std::shared_ptr<AbstractPluginDescriptor> get(std::size_t i) {
        if (i >= plugins_.size()) {
            throw std::out_of_range("Invalid plugin index!");
        }
        return plugins_[i];
    }

    [[nodiscard]] static auto empty() {
        return plugins_.empty();
    }
    [[nodiscard]] static auto size() {
        return plugins_.size();
    }
    [[nodiscard]] static const auto& getAll() {
        return plugins_;
    }

private:
    static inline std::vector<std::shared_ptr<AbstractPluginDescriptor>> plugins_;
};

} // namespace megamol::core::factories
