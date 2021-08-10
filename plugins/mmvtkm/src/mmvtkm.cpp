/**
 * MegaMol
 * Copyright (c) 2019-2021, MegaMol Dev Team
 * All rights reserved.
 */

// TODO: Vislib must die!!!
// Vislib includes Windows.h. This crashes when somebody else (i.e. zmq) is using Winsock2.h, but the vislib include
// is first without defining WIN32_LEAN_AND_MEAN. This define is the only thing we need from stdafx.h, include could be
// removed otherwise.
#include "stdafx.h"

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "mmvtkm/mmvtkmDataSource.h"
#include "mmvtkm/mmvtkmFileLoader.h"
#include "mmvtkm/mmvtkmMeshRenderTasks.h"
//#include "mmvtkm/mmvtkmRenderer.h"
#include "mmvtkm/mmvtkmDataCall.h"
#include "mmvtkm/mmvtkmStreamLines.h"


namespace megamol::mmvtkm {
class MmvtkmPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(MmvtkmPluginInstance)

public:
    MmvtkmPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("vtkm", "Plugin to read and render vtkm data."){};

    virtual ~MmvtkmPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::mmvtkm::mmvtkmDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mmvtkm::mmvtkmFileLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::mmvtkm::mmvtkmMeshRenderTasks>();
        // this->module_descriptions.RegisterAutoDescription<megamol::mmvtkm::mmvtkmDataRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::mmvtkm::mmvtkmStreamLines>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::mmvtkm::mmvtkmDataCall>();
    }
};
} // namespace megamol::mmvtkm
