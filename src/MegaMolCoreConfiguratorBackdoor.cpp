/*
 * MegaMolCoreConfiguratorBackdoor.cpp
 * Copyright (C) 2014 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/api/MegaMolCore.h"

// these functions are evil, but only used by the configurator

#include <memory>
#include "mmcore/factories/ModuleDescriptionManager.h"
#include "factories/ModuleClassRegistry.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "factories/CallClassRegistry.h"
#include "mmcore/RootModuleNamespace.h"
#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/Stack.h"

namespace {
    static megamol::core::factories::ModuleDescriptionManager& backdoor_modules(void) {
        static std::shared_ptr<megamol::core::factories::ModuleDescriptionManager> i;
        if (!i) {
            i = std::make_shared<megamol::core::factories::ModuleDescriptionManager>();
            megamol::core::factories::register_module_classes(*i);
        }
        return *i;
    }
    static megamol::core::factories::CallDescriptionManager& backdoor_calls(void) {
        static std::shared_ptr<megamol::core::factories::CallDescriptionManager> i;
        if (!i) {
            i = std::make_shared<megamol::core::factories::CallDescriptionManager>();
            megamol::core::factories::register_call_classes(*i);
        }
        return *i;
    }
}


/*
 * mmcModuleCount
 */
MEGAMOLCORE_API int MEGAMOLCORE_CALL mmcModuleCount(void) {
    return static_cast<int>(::std::distance(
        backdoor_modules().begin(),
        backdoor_modules().end()));
}


/*
 * mmcModuleDescription
 */
MEGAMOLCORE_API void* MEGAMOLCORE_CALL mmcModuleDescription(int idx) {
    if (idx < 0) return nullptr;
    for (auto d : backdoor_modules()) {
        if (idx == 0) return const_cast<void*>(static_cast<const void *>(d.get()));
        idx--;
    }
    return nullptr;
}


/*
 * mmcCallCount
 */
MEGAMOLCORE_API int MEGAMOLCORE_CALL mmcCallCount(void) {
    return static_cast<int>(::std::distance(
        backdoor_calls().begin(),
        backdoor_calls().end()));
}


/*
 * mmcCallDescription
 */
MEGAMOLCORE_API void* MEGAMOLCORE_CALL mmcCallDescription(int idx) {
    if (idx < 0) return nullptr;
    for (auto d : backdoor_calls()) {
        if (idx == 0) return const_cast<void*>(static_cast<const void *>(d.get()));
        idx--;
    }
    return nullptr;
}


/**
 * TODO: Document me
 */
bool operator==(const mmcParamSlotDescription lhs, mmcParamSlotDescription rhs) {
    return ::memcmp(&lhs, &rhs, sizeof(mmcParamSlotDescription)) == 0;
}


/**
 * TODO: Document me
 */
bool operator==(const mmcCalleeSlotDescription lhs, mmcCalleeSlotDescription rhs) {
    return ::memcmp(&lhs, &rhs, sizeof(mmcCalleeSlotDescription)) == 0;
}


/**
 * TODO: Document me
 */
bool operator==(const mmcCallerSlotDescription lhs, mmcCallerSlotDescription rhs) {
    return ::memcmp(&lhs, &rhs, sizeof(mmcCallerSlotDescription)) == 0;
}


/*
 * mmcGetModuleSlotDescriptions
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcGetModuleSlotDescriptions(void * desc, 
        unsigned int *outCntParamSlots, mmcParamSlotDescription **outParamSlots,
        unsigned int *outCntCalleeSlots, mmcCalleeSlotDescription **outCalleeSlots,
        unsigned int *outCntCallerSlots, mmcCallerSlotDescription **outCallerSlots) {
    ASSERT(desc != NULL);
    ASSERT(outCntParamSlots != NULL);
    ASSERT(outParamSlots != NULL);
    ASSERT(outCntCalleeSlots != NULL);
    ASSERT(outCalleeSlots != NULL);
    ASSERT(outCntCallerSlots != NULL);
    ASSERT(outCallerSlots != NULL);

    megamol::core::RootModuleNamespace rms;
    megamol::core::factories::ModuleDescription *md = static_cast<megamol::core::factories::ModuleDescription*>(desc);
    ASSERT(md != NULL);

    megamol::core::Module *m = md->CreateModule(NULL);
    if (m == NULL) {
        *outCntParamSlots = 0;
        *outParamSlots = NULL;
        *outCntCalleeSlots = 0;
        *outCalleeSlots = NULL;
        *outCntCallerSlots = 0;
        *outCallerSlots = NULL;
        return;
    }
    rms.AddChild(m);

    vislib::Array<mmcParamSlotDescription> pa;
    vislib::Array<mmcCalleeSlotDescription> cea;
    vislib::Array<mmcCallerSlotDescription> cra;

    vislib::Stack<megamol::core::Module::ChildList::Iterator> stack;
    stack.Push(m->GetChildIterator());

    while (!stack.IsEmpty()) {
        megamol::core::Module::ChildList::Iterator iter = stack.Pop();
        while (iter.HasNext()) {
            megamol::core::AbstractNamedObject *ano = iter.Next();

            megamol::core::param::ParamSlot *ps = dynamic_cast<megamol::core::param::ParamSlot*>(ano);
            megamol::core::CalleeSlot *ces = dynamic_cast<megamol::core::CalleeSlot*>(ano);
            megamol::core::CallerSlot *crs = dynamic_cast<megamol::core::CallerSlot*>(ano);

            if (ps != NULL) {
                SIZE_T i = pa.Count();
                pa.SetCount(i + 1);

                vislib::StringA str = ps->FullName();
                vislib::StringA str2 = m->FullName() + "::";
                if (str.StartsWith(str2)) str.Remove(0, str2.Length());

                pa[i].name = new char[str.Length() + 1];
                ::memcpy(const_cast<char*>(pa[i].name), str.PeekBuffer(), str.Length() + 1);

                str = ps->Description();
                pa[i].desc = new char[str.Length() + 1];
                ::memcpy(const_cast<char*>(pa[i].desc), str.PeekBuffer(), str.Length() + 1);

                vislib::RawStorage blob;
                ps->Param< ::megamol::core::param::AbstractParam>()->Definition(blob);

                pa[i].typeInfoSize = static_cast<unsigned int>(blob.GetSize());
                pa[i].typeInfo = new unsigned char[pa[i].typeInfoSize];
                ::memcpy(const_cast<unsigned char*>(pa[i].typeInfo), blob, pa[i].typeInfoSize);

                str = ps->Parameter()->ValueString();
                pa[i].defVal = new char[str.Length() + 1];
                ::memcpy(const_cast<char*>(pa[i].defVal), str.PeekBuffer(), str.Length() + 1);

            } else if (ces != NULL) {
                SIZE_T i = cea.Count();
                cea.SetCount(i + 1);

                vislib::StringA str = ces->FullName();
                vislib::StringA str2 = m->FullName() + "::";
                if (str.StartsWith(str2)) str.Remove(0, str2.Length());

                cea[i].name = new char[str.Length() + 1];
                ::memcpy(const_cast<char*>(cea[i].name), str.PeekBuffer(), str.Length() + 1);

                str = ces->Description();
                cea[i].desc = new char[str.Length() + 1];
                ::memcpy(const_cast<char*>(cea[i].desc), str.PeekBuffer(), str.Length() + 1);

                cea[i].cntCallbacks = static_cast<unsigned int>(ces->GetCallbackCount());
                cea[i].callbackCallType = new const char*[cea[i].cntCallbacks];
                cea[i].callbackFuncName = new const char*[cea[i].cntCallbacks];

                for (unsigned int j = 0; j < cea[i].cntCallbacks; j++) {
                    str = ces->GetCallbackCallName(j);
                    cea[i].callbackCallType[j] = new char[str.Length() + 1];
                    ::memcpy(const_cast<char*>(cea[i].callbackCallType[j]), str.PeekBuffer(), str.Length() + 1);

                    str = ces->GetCallbackFuncName(j);
                    cea[i].callbackFuncName[j] = new char[str.Length() + 1];
                    ::memcpy(const_cast<char*>(cea[i].callbackFuncName[j]), str.PeekBuffer(), str.Length() + 1);
                }

            } else if (crs != NULL) {
                SIZE_T i = cra.Count();
                cra.SetCount(i + 1);

                vislib::StringA str = crs->FullName();
                vislib::StringA str2 = m->FullName() + "::";
                if (str.StartsWith(str2)) str.Remove(0, str2.Length());

                cra[i].name = new char[str.Length() + 1];
                ::memcpy(const_cast<char*>(cra[i].name), str.PeekBuffer(), str.Length() + 1);

                str = crs->Description();
                cra[i].desc = new char[str.Length() + 1];
                ::memcpy(const_cast<char*>(cra[i].desc), str.PeekBuffer(), str.Length() + 1);

                cra[i].cntCompCalls = static_cast<unsigned int>(crs->GetCompCallCount());
                cra[i].compCalls = new const char *[cra[i].cntCompCalls];
                for (unsigned int j = 0; j < cra[i].cntCompCalls; j++) {
                    str = crs->GetCompCallClassName(j);
                    cra[i].compCalls[j] = new char[str.Length() + 1];
                    ::memcpy(const_cast<char*>(cra[i].compCalls[j]), str.PeekBuffer(), str.Length() + 1);
                }

            }
        }
    }

    *outCntParamSlots = static_cast<unsigned int>(pa.Count());
    *outParamSlots = new mmcParamSlotDescription[*outCntParamSlots];
    ::memcpy(*outParamSlots, pa.PeekElements(), sizeof(mmcParamSlotDescription) * *outCntParamSlots);
    *outCntCalleeSlots = static_cast<unsigned int>(cea.Count());
    *outCalleeSlots = new mmcCalleeSlotDescription[*outCntCalleeSlots];
    ::memcpy(*outCalleeSlots, cea.PeekElements(), sizeof(mmcCalleeSlotDescription) * *outCntCalleeSlots);
    *outCntCallerSlots = static_cast<unsigned int>(cra.Count());
    *outCallerSlots = new mmcCallerSlotDescription[*outCntCallerSlots];
    ::memcpy(*outCallerSlots, cra.PeekElements(), sizeof(mmcCallerSlotDescription) * *outCntCallerSlots);

    rms.RemoveChild(m);
    m->SetAllCleanupMarks();
    m->PerformCleanup();
    delete m;
}


/*
 * mmcReleaseModuleSlotDescriptions
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcReleaseModuleSlotDescriptions(
        unsigned int outCntParamSlots, mmcParamSlotDescription **outParamSlots,
        unsigned int outCntCalleeSlots, mmcCalleeSlotDescription **outCalleeSlots,
        unsigned int outCntCallerSlots, mmcCallerSlotDescription **outCallerSlots) {
    ASSERT(outParamSlots != NULL);
    ASSERT(outCalleeSlots != NULL);
    ASSERT(outCallerSlots != NULL);

    for (unsigned int i = 0; i < outCntParamSlots; i++) {
        delete[] (*outParamSlots)[i].name;
        delete[] (*outParamSlots)[i].desc;
        delete[] (*outParamSlots)[i].typeInfo;
        delete[] (*outParamSlots)[i].defVal;
    }
    delete[] (*outParamSlots);
    *outParamSlots = NULL;

    for (unsigned int i = 0; i < outCntCalleeSlots; i++) {
        delete[] (*outCalleeSlots)[i].name;
        delete[] (*outCalleeSlots)[i].desc;
        for (unsigned int j = 0; j < (*outCalleeSlots)[i].cntCallbacks; j++) {
            delete[] (*outCalleeSlots)[i].callbackCallType[j];
            delete[] (*outCalleeSlots)[i].callbackFuncName[j];
        }
        delete[] (*outCalleeSlots)[i].callbackCallType;
        delete[] (*outCalleeSlots)[i].callbackFuncName;
    }
    delete[] (*outCalleeSlots);
    *outCalleeSlots = NULL;

    for (unsigned int i = 0; i < outCntCallerSlots; i++) {
        delete[] (*outCallerSlots)[i].name;
        delete[] (*outCallerSlots)[i].desc;
        for (unsigned int j = 0; j < (*outCallerSlots)[i].cntCompCalls; j++) {
            delete[] (*outCallerSlots)[i].compCalls[j];
        }
        delete[] (*outCallerSlots)[i].compCalls;
    }
    delete[] (*outCallerSlots);
    *outCallerSlots = NULL;

}


/*
 * mmcGetModuleDescriptionInfo
 */
MEGAMOLCORE_EXT_APICALL(mmcModuleDescriptionInfo*, mmcGetModuleDescriptionInfo)(void * desc) {
    megamol::core::factories::ModuleDescription *md = static_cast<megamol::core::factories::ModuleDescription*>(desc);
    ASSERT(md != NULL);
    mmcModuleDescriptionInfo *d = new mmcModuleDescriptionInfo();

    vislib::StringA str = md->ClassName();
    d->name = new char[str.Length() + 1];
    ::memcpy(const_cast<char*>(d->name), str.PeekBuffer(), str.Length() + 1);

    str = md->Description();
    d->desc = new char[str.Length() + 1];
    ::memcpy(const_cast<char*>(d->desc), str.PeekBuffer(), str.Length() + 1);

    return d;
}


/*
 * mmcReleaseModuleDescriptionInfo
 */
MEGAMOLCORE_EXT_APICALL(void, mmcReleaseModuleDescriptionInfo)(mmcModuleDescriptionInfo* desc) {
    delete[] desc->name;
    delete[] desc->desc;
    delete desc;
}


/*
 * mmcGetCallDescriptionInfo
 */
MEGAMOLCORE_EXT_APICALL(mmcCallDescriptionInfo*, mmcGetCallDescriptionInfo)(void * desc) {
    megamol::core::factories::CallDescription *cd = static_cast<megamol::core::factories::CallDescription*>(desc);
    ASSERT(cd != NULL);
    mmcCallDescriptionInfo *d = new mmcCallDescriptionInfo();

    vislib::StringA str = cd->ClassName();
    d->name = new char[str.Length() + 1];
    ::memcpy(const_cast<char*>(d->name), str.PeekBuffer(), str.Length() + 1);

    str = cd->Description();
    d->desc = new char[str.Length() + 1];
    ::memcpy(const_cast<char*>(d->desc), str.PeekBuffer(), str.Length() + 1);

    d->cntFunc = cd->FunctionCount();
    d->funcNames = new const char*[d->cntFunc];
    for (unsigned int i = 0; i < d->cntFunc; i++) {
        str = cd->FunctionName(i);
        d->funcNames[i] = new char[str.Length() + 1];
        ::memcpy(const_cast<char*>(d->funcNames[i]), str.PeekBuffer(), str.Length() + 1);
    }

    return d;
}


/*
 * mmcReleaseCallDescriptionInfo
 */
MEGAMOLCORE_EXT_APICALL(void, mmcReleaseCallDescriptionInfo)(mmcCallDescriptionInfo* desc) {
    delete[] desc->name;
    delete[] desc->desc;
    for (unsigned int i = 0; i < desc->cntFunc; i++) {
        delete[] desc->funcNames[i];
    }
    delete[] desc->funcNames;
    delete desc;
}


MEGAMOLCORE_EXT_APICALL(void, mmcPlugin200TestCompatInfo)(void* ci) {

}

MEGAMOLCORE_EXT_APICALL(const char*, mmcPlugin200GetName)(void* i) {

}

MEGAMOLCORE_EXT_APICALL(const char*, mmcPlugin200GetDesc)(void* i) {

}

MEGAMOLCORE_EXT_APICALL(int, mmcPlugin200GetModCnt)(void* i) {

}

MEGAMOLCORE_EXT_APICALL(mmcModuleDescriptionInfo*, mmcPlugin200GetModDesc)(void* i, int idx) {

}

MEGAMOLCORE_EXT_APICALL(int, mmcPlugin200GetCallCnt)(void* i) {

}

MEGAMOLCORE_EXT_APICALL(mmcCallDescriptionInfo*, mmcPlugin200GetCallDesc)(void* i, int idx) {

}
