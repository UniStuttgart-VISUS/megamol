/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "event/EventStorage.h"
#include "job/TickSwitch.h"
#include "misc/FileStreamProvider.h"
#include "misc/LuaHostSettingsModule.h"
#include "misc/ResourceTestModule.h"
#include "mmstd/data/DataWriterCtrlCall.h"
#include "mmstd/data/DataWriterJob.h"
#include "mmstd/data/DirectDataWriterCall.h"
#include "mmstd/event/EventCall.h"
#include "mmstd/flags/FlagCalls.h"
#include "mmstd/flags/FlagStorage.h"
#include "mmstd/job/JobThread.h"
#include "mmstd/job/TickCall.h"
#include "mmstd/light/AmbientLight.h"
#include "mmstd/light/CallLight.h"
#include "mmstd/light/DistantLight.h"
#include "mmstd/light/HDRILight.h"
#include "mmstd/light/PointLight.h"
#include "mmstd/light/QuadLight.h"
#include "mmstd/light/SpotLight.h"
#include "mmstd/light/TriDirectionalLighting.h"
#include "mmstd/param/GenericParamModule.h"
#include "mmstd/param/ParamCalls.h"
#include "mmstd/renderer/CallClipPlane.h"
#include "mmstd/renderer/CallGetTransferFunction.h"
#include "mmstd/renderer/CallRender3D.h"
#include "mmstd/renderer/CallTimeControl.h"
#include "mmstd/renderer/ClipPlane.h"
#include "mmstd/renderer/TransferFunction.h"
#include "mmstd/view/View3D.h"
#include "special/StubModule.h"

namespace megamol::mmstd {
class PluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(PluginInstance)
public:
    PluginInstance() : megamol::core::factories::AbstractPluginInstance("mmstd", "Core calls and modules."){};

    ~PluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<core::view::View3D>();
        this->module_descriptions.RegisterAutoDescription<core::FlagStorage>();
        this->module_descriptions.RegisterAutoDescription<core::view::light::AmbientLight>();
        this->module_descriptions.RegisterAutoDescription<core::view::light::DistantLight>();
        this->module_descriptions.RegisterAutoDescription<core::view::light::HDRILight>();
        this->module_descriptions.RegisterAutoDescription<core::view::light::PointLight>();
        this->module_descriptions.RegisterAutoDescription<core::view::light::QuadLight>();
        this->module_descriptions.RegisterAutoDescription<core::view::light::SpotLight>();
        this->module_descriptions.RegisterAutoDescription<core::view::light::TriDirectionalLighting>();
        this->module_descriptions.RegisterAutoDescription<core::special::StubModule>();
        this->module_descriptions.RegisterAutoDescription<core::job::DataWriterJob>();
        this->module_descriptions.RegisterAutoDescription<core::utility::LuaHostSettingsModule>();
        this->module_descriptions.RegisterAutoDescription<core::job::TickSwitch>();
        this->module_descriptions.RegisterAutoDescription<core::job::JobThread>();
        this->module_descriptions.RegisterAutoDescription<core::view::ClipPlane>();
        this->module_descriptions.RegisterAutoDescription<core::FileStreamProvider>();
        this->module_descriptions.RegisterAutoDescription<core::EventStorage>();
        this->module_descriptions.RegisterAutoDescription<core::ResourceTestModule>();
        this->module_descriptions.RegisterAutoDescription<core::param::FloatParamModule>();
        this->module_descriptions.RegisterAutoDescription<core::param::IntParamModule>();
        this->module_descriptions.RegisterAutoDescription<core::view::TransferFunction>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<core::FlagCallRead_CPU>();
        this->call_descriptions.RegisterAutoDescription<core::FlagCallWrite_CPU>();
        this->call_descriptions.RegisterAutoDescription<core::view::light::CallLight>();
        this->call_descriptions.RegisterAutoDescription<core::view::CallClipPlane>();
        this->call_descriptions.RegisterAutoDescription<core::view::CallTimeControl>();
        this->call_descriptions.RegisterAutoDescription<core::CallEvent>();
        this->call_descriptions.RegisterAutoDescription<core::view::CallRender3D>();
        this->call_descriptions.RegisterAutoDescription<core::param::FloatParamCall>();
        this->call_descriptions.RegisterAutoDescription<core::param::IntParamCall>();
        this->call_descriptions.RegisterAutoDescription<core::view::CallGetTransferFunction>();
        this->call_descriptions.RegisterAutoDescription<core::DataWriterCtrlCall>();
        this->call_descriptions.RegisterAutoDescription<core::job::TickCall>();
        this->call_descriptions.RegisterAutoDescription<core::DirectDataWriterCall>();
    }
};
} // namespace megamol::mmstd
