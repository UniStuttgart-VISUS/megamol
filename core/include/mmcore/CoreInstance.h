/*
 * CoreInstance.h
 *
 * Copyright (C) 2008, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "mmcore/AbstractSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/RootModuleNamespace.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
#include "mmcore/factories/ObjectDescription.h"
#include "mmcore/factories/ObjectDescriptionManager.h"
#include "mmcore/factories/PluginDescriptor.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/ParamUpdateListener.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/Array.h"
#include "vislib/IllegalStateException.h"
#include "vislib/Map.h"
#include "vislib/Pair.h"
#include "vislib/PtrArray.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/DynamicLinkLibrary.h"
#include "vislib/sys/Lockable.h"


namespace megamol::core {

class CoreInstance {
public:
    CoreInstance() = default;

    virtual ~CoreInstance() = default;

    inline uint32_t GetFrameID() {
        return this->frameID;
    }

    inline void SetFrameID(uint32_t frameID) {
        this->frameID = frameID;
    }

private:
    uint32_t frameID;
};

} // namespace megamol::core
