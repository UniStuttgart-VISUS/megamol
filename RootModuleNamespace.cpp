/*
 * RootModuleNamespace.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "RootModuleNamespace.h"
#include "vislib/Log.h"
#include "vislib/StackTrace.h"
#include "vislib/Trace.h"

using namespace megamol::core;


/*
 * RootModuleNamespace::RootModuleNamespace
 */
RootModuleNamespace::RootModuleNamespace(void) : ModuleNamespace("") {
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
vislib::StringA RootModuleNamespace::FullNamespace(const vislib::StringA& base,
        const vislib::StringA& path) const {

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
ModuleNamespace * RootModuleNamespace::FindNamespace(
        const vislib::Array<vislib::StringA>& path, bool createMissing) {

    ModuleNamespace *cns = this;

    for (SIZE_T i = 0; i < path.Count(); i++) {
        AbstractNamedObject *ano = cns->FindChild(path[i]);

        if (ano == NULL) {
            if (createMissing) {
                ModuleNamespace *nns = new ModuleNamespace(path[i]);
                cns->AddChild(nns);
                cns = nns;
            } else {
                return NULL;
            }

        } else {
            ModuleNamespace *nns = dynamic_cast<ModuleNamespace*>(ano);
            if (nns != NULL) {
                cns = nns;

            } else {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "name conflicts with a namespace object\n");
                return NULL;

            }
        }

    }

    return cns;
}


/*
 * RootModuleNamespace::LockModuleGraph
 */
void RootModuleNamespace::LockModuleGraph(bool write) {
    VLSTACKTRACE("LockModuleGraph", __FILE__, __LINE__);
//#if defined(DEBUG) || defined(_DEBUG)
//    unsigned int size;
//    vislib::StringA stack;
//    vislib::StackTrace::GetStackString(static_cast<char*>(NULL), size);
//    vislib::StackTrace::GetStackString(stack.AllocateBuffer(size), size);
//    VLTRACE(VISLIB_TRCELVL_INFO, "LockModuleGraph:\n%s\n", stack.PeekBuffer());
//#endif

    this->lock.Lock();
    // TODO: Implement
}


/*
 * RootModuleNamespace::UnlockModuleGraph
 */
void RootModuleNamespace::UnlockModuleGraph(void) {
    VLSTACKTRACE("UnlockModuleGraph", __FILE__, __LINE__);
//#if defined(DEBUG) || defined(_DEBUG)
//    unsigned int size;
//    vislib::StringA stack;
//    vislib::StackTrace::GetStackString(static_cast<char*>(NULL), size);
//    vislib::StackTrace::GetStackString(stack.AllocateBuffer(size), size);
//    VLTRACE(VISLIB_TRCELVL_INFO, "UnlockModuleGraph:\n%s\n", stack.PeekBuffer());
//#endif
    this->lock.Unlock();
    // TODO: Implement
}
