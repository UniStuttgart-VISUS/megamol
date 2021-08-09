/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

namespace megamol::gui {
class GuiPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(GuiPluginInstance)

public:
    GuiPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("gui", "Graphical User Interface Plugin"){};

    ~GuiPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {}
};
} // namespace megamol::gui
