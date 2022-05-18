#pragma once

#include "LuaHostService.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace core {
namespace utility {

class LuaHostSettingsModule : public core::Module {
public:
    static const char* ClassName() {
        return "LuaRemoteHost";
    }
    static const char* Description() {
        return "Host Modules for LuaRemote";
    }
    static bool IsAvailable() {
        return true;
    }

    LuaHostSettingsModule();
    virtual ~LuaHostSettingsModule();

protected:
    virtual bool create(void);
    virtual void release(void);

private:
    LuaHostService* getHostService();

    bool portSlotChanged(core::param::ParamSlot& slot);
    bool enabledSlotChanged(core::param::ParamSlot& slot);

    core::param::ParamSlot portSlot;
    core::param::ParamSlot enabledSlot;
};


} /* namespace utility */
} /* namespace core */
} /* namespace megamol */
