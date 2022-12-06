/**
 * MegaMol
 * Copyright (c) 2019-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "mmvtkm/mmvtkmDataCall.h"
#include "mmvtkm/mmvtkmDataSource.h"
#include "mmvtkm/mmvtkmFileLoader.h"
#include "mmvtkm/mmvtkmStreamLines.h"


namespace megamol::mmvtkm {
class MmvtkmPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(MmvtkmPluginInstance)

public:
    MmvtkmPluginInstance()
            : megamol::core::factories::AbstractPluginInstance("vtkm", "Plugin to read and render vtkm data."){};

    virtual ~MmvtkmPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::mmvtkm::mmvtkmDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mmvtkm::mmvtkmFileLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::mmvtkm::mmvtkmStreamLines>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::mmvtkm::mmvtkmDataCall>();
    }
};
} // namespace megamol::mmvtkm
