/*
 * RootModuleNamespace.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "mmcore/RootModuleNamespace.h"
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/AbstractNamedObjectContainer.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/assert.h"
#if defined(DEBUG) || defined(_DEBUG)
#include "vislib/sys/AutoLock.h"
#endif
#include "mmcore/utility/log/Log.h"
#include "vislib/Stack.h"
#include "vislib/String.h"
#include "vislib/Trace.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/sys/Thread.h"
#include <memory>

using namespace megamol::core;


/*
 * RootModuleNamespace::RootModuleNamespace
 */
RootModuleNamespace::RootModuleNamespace(void) : ModuleNamespace(""), lock() {
    // megamol::core::utility::log::Log::DefaultLog.WriteInfo("RootModuleNamespace Lock address: %x\n", std::addressof(this->lock));
    // intentionally empty ATM
}


/*
 * RootModuleNamespace::~RootModuleNamespace
 */
RootModuleNamespace::~RootModuleNamespace(void) {
    // intentionally empty ATM
}


/*
 * RootModuleNamespace::FullNamespace
 */
vislib::StringA RootModuleNamespace::FullNamespace(const vislib::StringA& base, const vislib::StringA& path) const {

    if (path.StartsWith("::")) {
        return path;
    }

    vislib::StringA retval = base;
    if (!retval.StartsWith("::")) {
        retval.Prepend("::");
    }
    if (!retval.EndsWith("::")) {
        retval.Append("::");
    }
    retval.Append(path);

    return retval;
}


/*
 * RootModuleNamespace::FindNamespace
 */
ModuleNamespace::ptr_type RootModuleNamespace::FindNamespace(
    const vislib::Array<vislib::StringA>& path, bool createMissing, bool quiet) {

    ModuleNamespace::ptr_type cns = dynamic_pointer_cast(this->shared_from_this());

    for (SIZE_T i = 0; i < path.Count(); i++) {
        AbstractNamedObject::ptr_type ano = cns->FindChild(path[i]);

        if (ano == NULL) {
            if (createMissing) {
                ModuleNamespace::ptr_type nns = std::make_shared<ModuleNamespace>(path[i]);
                cns->AddChild(nns);
                cns = ModuleNamespace::dynamic_pointer_cast(nns);
            } else {
                return NULL;
            }

        } else {
            ModuleNamespace* nns = dynamic_cast<ModuleNamespace*>(ano.get());
            if (nns != NULL) {
                cns = ModuleNamespace::dynamic_pointer_cast(ano);

            } else {
                if (!quiet) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("name conflicts with a namespace object\n");
                }
                return NULL;
            }
        }
    }

    return cns;
}


/*
 * RootModuleNamespace::ModuleGraphLock
 */
vislib::sys::AbstractReaderWriterLock& RootModuleNamespace::ModuleGraphLock(void) {
    return this->lock;
}


/*
 * RootModuleNamespace::ModuleGraphLock
 */
vislib::sys::AbstractReaderWriterLock& RootModuleNamespace::ModuleGraphLock(void) const {
    return this->lock;
}
