/*
 * RootModuleNamespace.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "RootModuleNamespace.h"
#include "AbstractNamedObject.h"
#include "AbstractNamedObjectContainer.h"
#include "CallDescription.h"
#include "CallDescriptionManager.h"
#include "CallerSlot.h"
#include "CalleeSlot.h"
#include "Module.h"
#include "ModuleDescription.h"
#include "ModuleDescriptionManager.h"
#include "param/ButtonParam.h"
#include "param/ParamSlot.h"
#include "vislib/assert.h"
#include "vislib/Array.h"
#if defined(DEBUG) || defined(_DEBUG)
#include "vislib/AutoLock.h"
#endif
#include "vislib/Log.h"
#include "vislib/Stack.h"
#include "vislib/StackTrace.h"
#include "vislib/String.h"
#include "vislib/Thread.h"
#include "vislib/Trace.h"
#include "vislib/UTF8Encoder.h"

using namespace megamol::core;


#if defined(DEBUG) || defined(_DEBUG)

/*
 * RootModuleNamespace::lockedThreadLock
 */
vislib::sys::CriticalSection RootModuleNamespace::lockedThreadLock;


/*
 * RootModuleNamespace::lockedThread
 */
vislib::SingleLinkedList<unsigned int> RootModuleNamespace::lockedRThread;


/*
 * RootModuleNamespace::lockedThread
 */
vislib::SingleLinkedList<unsigned int> RootModuleNamespace::lockedWThread;

#endif


/*
 * RootModuleNamespace::RootModuleNamespace
 */
RootModuleNamespace::RootModuleNamespace(void) : ModuleNamespace(""), lock() {
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
        const vislib::Array<vislib::StringA>& path, bool createMissing, bool quiet) {

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
 * RootModuleNamespace::LockModuleGraph
 */
void RootModuleNamespace::LockModuleGraph(bool write) {
    VLSTACKTRACE("LockModuleGraph", __FILE__, __LINE__);
#if defined(DEBUG) || defined(_DEBUG)
    unsigned int size;
    vislib::StringA stack;
    vislib::StackTrace::GetStackString(static_cast<char*>(NULL), size);
    vislib::StackTrace::GetStackString(stack.AllocateBuffer(size), size);
    VLTRACE(vislib::Trace::LEVEL_VL_ANNOYINGLY_VERBOSE,
        "LockModuleGraph(%u;%s):\n%s\n",
        vislib::sys::Thread::CurrentID(),
        write ? "exclusive" : "shared",
        stack.PeekBuffer());

    vislib::sys::AutoLock lock(RootModuleNamespace::lockedThreadLock);
    ASSERT(!RootModuleNamespace::lockedRThread.Contains(vislib::sys::Thread::CurrentID())
        && !RootModuleNamespace::lockedWThread.Contains(vislib::sys::Thread::CurrentID()));
#endif
    if (write) {
        this->lock.LockExclusive();
#if defined(DEBUG) || defined(_DEBUG)
        RootModuleNamespace::lockedWThread.Add(vislib::sys::Thread::CurrentID());
#endif
    } else {
        this->lock.LockShared();
#if defined(DEBUG) || defined(_DEBUG)
        RootModuleNamespace::lockedRThread.Add(vislib::sys::Thread::CurrentID());
#endif
    }

}


/*
 * RootModuleNamespace::UnlockModuleGraph
 */
void RootModuleNamespace::UnlockModuleGraph(bool write) {
    VLSTACKTRACE("UnlockModuleGraph", __FILE__, __LINE__);
#if defined(DEBUG) || defined(_DEBUG)
    unsigned int size;
    vislib::StringA stack;
    vislib::StackTrace::GetStackString(static_cast<char*>(NULL), size);
    vislib::StackTrace::GetStackString(stack.AllocateBuffer(size), size);
    VLTRACE(vislib::Trace::LEVEL_VL_ANNOYINGLY_VERBOSE,
        "UnlockModuleGraph(%u;%s):\n%s\n",
        vislib::sys::Thread::CurrentID(),
        write ? "exclusive" : "shared",
        stack.PeekBuffer());

    vislib::sys::AutoLock lock(RootModuleNamespace::lockedThreadLock);
#endif
    if (write) {
        this->lock.UnlockExclusive();
#if defined(DEBUG) || defined(_DEBUG)
        ASSERT(RootModuleNamespace::lockedWThread.Contains(vislib::sys::Thread::CurrentID()));
        RootModuleNamespace::lockedWThread.RemoveAll(vislib::sys::Thread::CurrentID());
#endif
    } else {
        this->lock.UnlockShared();
#if defined(DEBUG) || defined(_DEBUG)
        ASSERT(RootModuleNamespace::lockedRThread.Contains(vislib::sys::Thread::CurrentID()));
        RootModuleNamespace::lockedRThread.RemoveAll(vislib::sys::Thread::CurrentID());
#endif
    }

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
    vislib::Stack<AbstractNamedObject *> stack;
    stack.Push(this);
    while (!stack.IsEmpty()) {
        AbstractNamedObject *ano = stack.Pop();
        ASSERT(ano != NULL);
        AbstractNamedObjectContainer *anoc = dynamic_cast<AbstractNamedObjectContainer *>(ano);
        Module *mod = dynamic_cast<Module *>(ano);
        //CalleeSlot *callee = dynamic_cast<CalleeSlot *>(ano);
        CallerSlot *caller = dynamic_cast<CallerSlot *>(ano);
        param::ParamSlot *param = dynamic_cast<param::ParamSlot *>(ano);

        if (anoc != NULL) {
            AbstractNamedObjectContainer::ChildList::Iterator anoccli = anoc->GetChildIterator();
            while (anoccli.HasNext()) {
                stack.Push(anoccli.Next());
            }
        }

        if (mod != NULL) {
            ModuleDescription *d = NULL;
            ModuleDescriptionManager::DescriptionIterator i = ModuleDescriptionManager::Instance()->GetIterator();
            while (i.HasNext()) {
                ModuleDescription *id = i.Next();
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
            CallDescription *d = NULL;
            CallDescriptionManager::DescriptionIterator i = CallDescriptionManager::Instance()->GetIterator();
            while (i.HasNext()) {
                CallDescription *id = i.Next();
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
