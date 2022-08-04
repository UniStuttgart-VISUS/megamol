/**
 * MegaMol
 * Copyright (c) 2019-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "mmvtkm_gl/mmvtkmStreamlineRenderer.h"
//#include "mmvtkm_gl/mmvtkmRenderer.h"


namespace megamol::mmvtkm_gl {
class MmvtkmGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(MmvtkmGLPluginInstance)

public:
    MmvtkmGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  "vtkm_gl", "Plugin to read and render vtkm data."){};

    virtual ~MmvtkmGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::mmvtkm_gl::mmvtkmStreamlineRenderer>();
        // this->module_descriptions.RegisterAutoDescription<megamol::mmvtkm_gl::mmvtkmDataRenderer>();

        // register calls
    }
};
} // namespace megamol::mmvtkm_gl
