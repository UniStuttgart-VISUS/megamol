#include "LuaHostSettingsModule.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/sys/AutoLock.h"
#include <cassert>
#include <iostream>
#include <sstream>

using namespace megamol;

megamol::core::utility::LuaHostSettingsModule::LuaHostSettingsModule()
        : core::Module()
        , portSlot("port", "The local port number to open a socket on for listening for incoming connections")
        , enabledSlot("enabled", "Enables/disables remote parameter control") {

    portSlot.SetParameter(new core::param::StringParam("tcp://*:35421")); // bind string format for ZeroMQ
    portSlot.SetUpdateCallback(&LuaHostSettingsModule::portSlotChanged);
    MakeSlotAvailable(&portSlot);

    enabledSlot.SetParameter(new core::param::BoolParam(true));
    enabledSlot.SetUpdateCallback(&LuaHostSettingsModule::enabledSlotChanged);
    MakeSlotAvailable(&enabledSlot);
}

megamol::core::utility::LuaHostSettingsModule::~LuaHostSettingsModule() {
    Release();
}

bool megamol::core::utility::LuaHostSettingsModule::create() {
    LuaHostService* ser = getHostService();
    if (ser == nullptr)
        return false;

    portSlot.Param<core::param::StringParam>()->SetValue(ser->GetAddress().c_str(), false);
    enabledSlot.Param<core::param::BoolParam>()->SetValue(ser->IsEnabled(), false);

    return true;
}

void megamol::core::utility::LuaHostSettingsModule::release() {
    // intentionally empty
}

megamol::core::utility::LuaHostService* megamol::core::utility::LuaHostSettingsModule::getHostService() {
    core::AbstractService* s = GetCoreInstance()->GetInstalledService(LuaHostService::ID);
    return dynamic_cast<LuaHostService*>(s);
}

bool megamol::core::utility::LuaHostSettingsModule::portSlotChanged(core::param::ParamSlot& slot) {
    if (&slot != &portSlot)
        return false;
    LuaHostService* ser = getHostService();
    if (ser == nullptr)
        return false;

    ser->SetAddress(portSlot.Param<core::param::StringParam>()->Value());

    return true;
}

bool megamol::core::utility::LuaHostSettingsModule::enabledSlotChanged(core::param::ParamSlot& slot) {
    if (&slot != &enabledSlot)
        return false;
    LuaHostService* ser = getHostService();
    if (ser == nullptr)
        return false;

    if (enabledSlot.Param<core::param::BoolParam>()->Value()) {
        ser->Enable();
    } else {
        ser->Disable();
    }

    return true;
}
