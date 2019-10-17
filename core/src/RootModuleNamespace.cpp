/*
 * RootModuleNamespace.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/RootModuleNamespace.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/AbstractNamedObjectContainer.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/assert.h"
#include "vislib/Array.h"
#if defined(DEBUG) || defined(_DEBUG)
#include "vislib/sys/AutoLock.h"
#endif
#include "vislib/sys/Log.h"
#include "vislib/Stack.h"
#include "vislib/String.h"
#include "vislib/sys/Thread.h"
#include "vislib/Trace.h"
#include "vislib/UTF8Encoder.h"
#include <memory>

using namespace megamol::core;


/*
 * RootModuleNamespace::RootModuleNamespace
 */
RootModuleNamespace::RootModuleNamespace(void) : ModuleNamespace(""), lock() {
    // vislib::sys::Log::DefaultLog.WriteInfo("RootModuleNamespace Lock address: %x\n", std::addressof(this->lock));
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
            ModuleNamespace *nns = dynamic_cast<ModuleNamespace*>(ano.get());
            if (nns != NULL) {
                cns = ModuleNamespace::dynamic_pointer_cast(ano);

            } else {
                if (!quiet) {
                    vislib::sys::Log::DefaultLog.WriteMsg(
                        vislib::sys::Log::LEVEL_ERROR,
                        "name conflicts with a namespace object\n");
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


/*
 * RootModuleNamespace::Serialize
 */
void RootModuleNamespace::SerializeGraph(vislib::RawStorage& outmem) {
    ASSERT(this->Parent() == NULL);

    // TODO: Only use module sub-graph containing the observed viewing module!!!

    vislib::Array<vislib::StringA> modClasses;
    vislib::Array<vislib::StringA> modNames;

    vislib::Array<vislib::StringA> callClasses;
    vislib::Array<vislib::StringA> callFrom;
    vislib::Array<vislib::StringA> callTo;

    vislib::Array<vislib::StringA> paramName;
    vislib::Array<vislib::StringA> paramValue;

    // collect data
    vislib::Stack<AbstractNamedObject::ptr_type> stack;

    stack.Push(this->shared_from_this());
    while (!stack.IsEmpty()) {
        AbstractNamedObject::ptr_type ano = stack.Pop();
        ASSERT(ano != NULL);
        AbstractNamedObjectContainer::ptr_type anoc = AbstractNamedObjectContainer::dynamic_pointer_cast(ano);
        Module *mod = dynamic_cast<Module *>(ano.get());
        //CalleeSlot *callee = dynamic_cast<CalleeSlot *>(ano.get());
        CallerSlot *caller = dynamic_cast<CallerSlot *>(ano.get());
        param::ParamSlot *param = dynamic_cast<param::ParamSlot *>(ano.get());

        if (anoc) {
            child_list_type::iterator i, e;
            i = anoc->ChildList_Begin();
            e = anoc->ChildList_End();
            for (; i != e; ++i) {
                stack.Push(*i);
            }
        }

        if (mod != NULL) {
            factories::ModuleDescription::ptr d;
            //ModuleDescriptionManager::DescriptionIterator i = ModuleDescriptionManager::Instance()->GetIterator();
            //while (i.HasNext()) {
            //    ModuleDescription *id = i.Next();
            for (auto id : this->core_inst->GetModuleDescriptionManager()) {
                if (id->IsDescribing(mod)) {
                    d = id;
                    break;
                }
            }
            ASSERT(d != NULL);

            modClasses.Append(d->ClassName());
            modNames.Append(mod->FullName());
        }

        if (caller != NULL) {
            Call *c = caller->CallAs<Call>();
            if (c == NULL) continue;
            factories::CallDescription::ptr d;
            //CallDescriptionManager::DescriptionIterator i = CallDescriptionManager::Instance()->GetIterator();
            //while (i.HasNext()) {
            //    CallDescription *id = i.Next();
            for (auto id : this->core_inst->GetCallDescriptionManager()) {
                if (id->IsDescribing(c)) {
                    d = id;
                    break;
                }
            }
            ASSERT(d != NULL);
            ASSERT(c->PeekCalleeSlot() != NULL);
            ASSERT(c->PeekCallerSlot() != NULL);

            callClasses.Append(d->ClassName());
            callFrom.Append(c->PeekCallerSlot()->FullName());
            callTo.Append(c->PeekCalleeSlot()->FullName());
        }

        if (param != NULL) {
            if (param->Parameter().IsNull()) continue;
            if (param->Param<param::ButtonParam>() != NULL) continue; // ignore button parameters (we do not want to press them)
            paramName.Append(param->FullName());
            vislib::TString v = param->Parameter()->ValueString();
            vislib::StringA vUTF8;
            vislib::UTF8Encoder::Encode(vUTF8, v);
            paramValue.Append(vUTF8);
        }

    }

    // serialize data
    ASSERT(modClasses.Count() == modNames.Count());
    ASSERT(callClasses.Count() == callFrom.Count());
    ASSERT(callClasses.Count() == callTo.Count());
    ASSERT(paramName.Count() == paramValue.Count());

    SIZE_T overallSize = 3 * sizeof(UINT64);
    for (SIZE_T i = 0; i < modClasses.Count(); i++) {
        overallSize += modClasses[i].Length() + 1;
        overallSize += modNames[i].Length() + 1;
    }
    for (SIZE_T i = 0; i < callClasses.Count(); i++) {
        overallSize += callClasses[i].Length() + 1;
        overallSize += callFrom[i].Length() + 1;
        overallSize += callTo[i].Length() + 1;
    }
    for (SIZE_T i = 0; i < paramName.Count(); i++) {
        overallSize += paramName[i].Length() + 1;
        overallSize += paramValue[i].Length() + 1;
    }
    outmem.EnforceSize(overallSize);
    outmem.As<UINT64>()[0] = static_cast<UINT64>(modClasses.Count());
    outmem.As<UINT64>()[1] = static_cast<UINT64>(callClasses.Count());
    outmem.As<UINT64>()[2] = static_cast<UINT64>(paramName.Count());
    overallSize = 3 * sizeof(UINT64);
    SIZE_T l;
    for (SIZE_T i = 0; i < modClasses.Count(); i++) {
        l = modClasses[i].Length() + 1;
        ::memcpy(outmem.At(overallSize), modClasses[i].PeekBuffer(), l);
        overallSize += l;
        l = modNames[i].Length() + 1;
        ::memcpy(outmem.At(overallSize), modNames[i].PeekBuffer(), l);
        overallSize += l;
    }
    for (SIZE_T i = 0; i < callClasses.Count(); i++) {
        l = callClasses[i].Length() + 1;
        ::memcpy(outmem.At(overallSize), callClasses[i].PeekBuffer(), l);
        overallSize += l;
        l = callFrom[i].Length() + 1;
        ::memcpy(outmem.At(overallSize), callFrom[i].PeekBuffer(), l);
        overallSize += l;
        l = callTo[i].Length() + 1;
        ::memcpy(outmem.At(overallSize), callTo[i].PeekBuffer(), l);
        overallSize += l;
    }
    for (SIZE_T i = 0; i < paramName.Count(); i++) {
        l = paramName[i].Length() + 1;
        ::memcpy(outmem.At(overallSize), paramName[i].PeekBuffer(), l);
        overallSize += l;
        l = paramValue[i].Length() + 1;
        ::memcpy(outmem.At(overallSize), paramValue[i].PeekBuffer(), l);
        overallSize += l;
    }

    ASSERT(overallSize == outmem.GetSize());
}
