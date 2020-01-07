/*
 * MegaMolCore.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/api/MegaMolCore.h"

#define _LOG_CORE_HASH_INFO 1
#define _SEND_CORE_HASH_INFO 1

#include "mmcore/mmd3d.h"
#include "mmcore/ApiHandle.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/JobDescription.h"
#include "mmcore/JobInstance.h"
#include "mmcore/factories/ObjectDescription.h"
#include "mmcore/factories/ObjectDescriptionManager.h"
#include "mmcore/versioninfo.h"
#include "mmcore/param/ParamHandle.h"
#include "mmcore/utility/Configuration.h"
#include "mmcore/ViewDescription.h"
#include "mmcore/ViewInstance.h"
#include "mmcore/view/AbstractTileView.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/ViewDirect3D.h"
#include "mmcore/job/AbstractJob.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/CallerSlot.h"

#include "vislib/assert.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/Console.h"
#include "vislib/sys/File.h"
#include "vislib/functioncast.h"
#include "vislib/sys/Log.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/MD5HashProvider.h"
#include "vislib/SHA1HashProvider.h"
#include "vislib/sys/Path.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/Trace.h"
#include "vislib/net/Socket.h"


#ifdef _WIN32
/* windows dll entry point */
#ifdef _MANAGED
#pragma managed(push, off)
#endif /* _MANAGED */

HMODULE mmCoreModuleHandle;

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call,
        LPVOID lpReserved) {
    mmCoreModuleHandle = hModule;
    switch (ul_reason_for_call) {
        case DLL_PROCESS_ATTACH:
        case DLL_THREAD_ATTACH:
        case DLL_THREAD_DETACH:
        case DLL_PROCESS_DETACH:
            break;
    }
    return TRUE;
}

#ifdef _MANAGED
#pragma managed(pop)
#endif /* _MANAGED */

#else /* _WIN32 */
/* linux shared object main */

extern "C" {

const char interp[] __attribute__((section(".interp"))) = 
"/lib/ld-linux.so.2";

void mmCoreMain(int argc, char *argv[]) {
    printf("Horst!\n");
    //printf("argc = %i (%u)\nargv = %p\n", argc, argc, argv);
    //printf("*argv = %s\n", *argv);
    exit(0);
}

}

#endif /* _WIN32 */


/*
 * mmcGetVersionInfo
 */
MEGAMOLCORE_API mmcBinaryVersionInfo* MEGAMOLCORE_CALL mmcGetVersionInfo(void) {
    mmcBinaryVersionInfo* rv = static_cast<mmcBinaryVersionInfo*>(malloc(sizeof(mmcBinaryVersionInfo)));

    rv->VersionNumber[0] = (const char*)MEGAMOL_CORE_MAJOR_VER;
    rv->VersionNumber[1] = (const char*)MEGAMOL_CORE_MINOR_VER;
    rv->VersionNumber[2] = MEGAMOL_CORE_COMP_REV;


    rv->SystemType = MMC_OSYSTEM_UNKNOWN;
#ifdef _WIN32
#if defined(WINVER)
#if (WINVER >= 0x0501)
    rv->SystemType = MMC_OSYSTEM_WINDOWS;
#endif /* (WINVER >= 0x0501) */
#endif /* defined(WINVER) */
#else /* _WIN32 */
    rv->SystemType = MMC_OSYSTEM_LINUX;
#endif /* _WIN32 */

    rv->HardwareArchitecture = MMC_HARCH_UNKNOWN;
#if defined(_WIN64) || defined(_LIN64)
    rv->HardwareArchitecture = MMC_HARCH_X64;
#else /* defined(_WIN64) || defined(_LIN64) */
    rv->HardwareArchitecture = MMC_HARCH_I86;
#endif /* defined(_WIN64) || defined(_LIN64) */

    rv->Flags = 0
#if defined(_DEBUG) || defined(DEBUG)
        | MMC_BFLAG_DEBUG
#endif /* defined(_DEBUG) || defined(DEBUG) */
#ifdef MEGAMOL_GLUT_ISDIRTY
        | MMC_BFLAG_DIRTY
#endif /* MEGAMOL_GLUT_ISDIRTY */
        ;

    const char *src = MEGAMOL_CORE_NAME;
    size_t buf_len = ::strlen(src);
    char *buf = static_cast<char*>(::malloc(buf_len + 1));
    ::memcpy(buf, src, buf_len);
    buf[buf_len] = 0;
    rv->NameStr = buf;

    src = MEGAMOL_CORE_COPYRIGHT;
    buf_len = ::strlen(src);
    buf = static_cast<char*>(::malloc(buf_len + 1));
    ::memcpy(buf, src, buf_len);
    buf[buf_len] = 0;
    rv->CopyrightStr = buf;

    src = MEGAMOL_CORE_COMMENTS;
    buf_len = ::strlen(src);
    buf = static_cast<char*>(::malloc(buf_len + 1));
    ::memcpy(buf, src, buf_len);
    buf[buf_len] = 0;
    rv->CommentStr = buf;

    return rv;
}


/*
 * mmvFreeVersionInfo
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcFreeVersionInfo(mmcBinaryVersionInfo* info) {
    if (info == nullptr) return;
    if (info->NameStr != nullptr) {
        ::free(const_cast<char*>(info->NameStr));
        info->NameStr = nullptr;
    }
    if (info->CopyrightStr != nullptr) {
        ::free(const_cast<char*>(info->CopyrightStr));
        info->CopyrightStr = nullptr;
    }
    if (info->CommentStr != nullptr) {
        ::free(const_cast<char*>(info->CommentStr));
        info->CommentStr = nullptr;
    }
    ::free(info);
}


/*
 * mmcGetHandleSize
 */
MEGAMOLCORE_API unsigned int MEGAMOLCORE_CALL mmcGetHandleSize(void) {
    return megamol::core::ApiHandle::GetHandleSize();
}


/*
 * mmcDisposeHandle
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcDisposeHandle(void *hndl) {
    megamol::core::ApiHandle::DestroyHandle(hndl);
}


/*
 * mmcIsHandleValid
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcIsHandleValid(void *hndl) {
    return (megamol::core::ApiHandle::InterpretHandle<
        megamol::core::ApiHandle>(hndl) != NULL);
}


/*
 * mmcGetHandleType
 */
MEGAMOLCORE_API mmcHandleType MEGAMOLCORE_CALL mmcGetHandleType(void *hndl) {

    if (megamol::core::ApiHandle::InterpretHandle<
            megamol::core::ApiHandle>(hndl) == NULL) {
        return MMC_HTYPE_INVALID;

    } else if (megamol::core::ApiHandle::InterpretHandle<
            megamol::core::CoreInstance>(hndl) == NULL) {
        return MMC_HTYPE_COREINSTANCE;

    } else if (megamol::core::ApiHandle::InterpretHandle<
            megamol::core::ViewInstance>(hndl) == NULL) {
        return MMC_HTYPE_VIEWINSTANCE;

    } else if (megamol::core::ApiHandle::InterpretHandle<
            megamol::core::JobInstance>(hndl) == NULL) {
        return MMC_HTYPE_JOBINSTANCE;

    } else if (megamol::core::ApiHandle::InterpretHandle<
            megamol::core::param::ParamHandle>(hndl) == NULL) {
        return MMC_HTYPE_PARAMETER;

    } else {
        return MMC_HTYPE_UNKNOWN;
    }
}


/*
 * mmcCreateCoreInstance
 */
MEGAMOLCORE_API mmcErrorCode MEGAMOLCORE_CALL mmcCreateCore(void *hCore) {
    if (mmcIsHandleValid(hCore) != 0) {
        return MMC_ERR_HANDLE; // handle was already valid.
    }
    if (*static_cast<unsigned char*>(hCore) != 0) {
        return MMC_ERR_MEMORY; // memory pointer seams to be invalid.
    }

//    { // self test for licencing
//        vislib::MD5HashProvider hash;
//        void *apifunctions[] = {
//__ALL_API_FUNCS
//        };
//        void *rp = function_cast<void*>(mmcCreateCore);
//        SIZE_T r = reinterpret_cast<SIZE_T>(rp);
//        SIZE_T d = UINT_MAX;
//        for (unsigned int i = 0; apifunctions[i] != NULL; i++) {
//            SIZE_T t = r - reinterpret_cast<SIZE_T>(apifunctions[i]);
//            if ((t > 0) && (t < d)) d = t;
//        }
//#ifdef _LOG_CORE_HASH_INFO
//        UINT logLevel = vislib::sys::Log::DefaultLog.GetLevel();
//        vislib::sys::Log::DefaultLog.SetLevel(800);
//        vislib::sys::Log::DefaultLog.WriteMsg(800, "Calculating core Hash using %d bytes\n", d);
//        vislib::sys::Log::DefaultLog.SetLevel(logLevel);
//#endif /* _LOG_CORE_HASH_INFO */
//        hash.Initialise();
//        hash.ComputeHash(NULL, r, static_cast<BYTE*>(rp), d);
//        BYTE *hashVal = new BYTE[r];
//        SIZE_T otherr = r;
//        hash.GetHashValue(hashVal, otherr);
//        ASSERT(otherr == r);
//
//        // Test against manifest hash value
//
//        vislib::StringA s, tmp;
//        for (d = 0; d < r; d++) {
//            if ((d % 4) == 0) tmp.Append("-");
//            s.Format("%02x", /*(int)*/hashVal[d]);
//            tmp.Append(s);
//        }
//        delete[] hashVal;
//        s.Format("%s-%s%d%s%s",
//            vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(),
//#ifdef _WIN32
//#if defined(WINVER)
//#if (WINVER >= 0x0501)
//            "Win",
//#endif /* (WINVER >= 0x0501) */
//#endif /* defined(WINVER) */
//#else /* _WIN32 */
//            "Lin",
//#endif /* _WIN32 */
//#if defined(_WIN64) || defined(_LIN64)
//            64,
//#else /* defined(_WIN64) || defined(_LIN64) */
//            32,
//#endif /* defined(_WIN64) || defined(_LIN64) */
//#if defined(_DEBUG) || defined(DEBUG)
//            "d",
//#else /* defined(_DEBUG) || defined(DEBUG) */
//            "",
//#endif /* defined(_DEBUG) || defined(DEBUG) */
//            tmp.PeekBuffer());
//        s.ToLowerCase();
//        tmp.ToLowerCase();
//
//#ifdef _LOG_CORE_HASH_INFO
//        logLevel = vislib::sys::Log::DefaultLog.GetLevel();
//        vislib::sys::Log::DefaultLog.SetLevel(800);
//        vislib::sys::Log::DefaultLog.WriteMsg(800, "Core Hash: %s\n", tmp.PeekBuffer() + 1);
//        vislib::sys::Log::DefaultLog.SetLevel(logLevel);
//#endif /* _LOG_CORE_HASH_INFO */
//
//#ifdef _SEND_CORE_HASH_INFO
//        // send infos
//        unsigned short s1, s2, s3, s4;
//        ::mmcGetVersion(&s1, &s2, &s3, &s4);
//
//        tmp.Format("-%d-%d-%d-%d", s1, s2, s3, s4);
//        s.Append(tmp);
//
//        vislib::net::Socket::Startup();
//        try {
//            vislib::net::Socket socket;
//            socket.Create(vislib::net::Socket::FAMILY_INET, vislib::net::Socket::TYPE_STREAM, vislib::net::Socket::PROTOCOL_TCP);
//            try {
//                vislib::net::IPEndPoint endPoint = vislib::net::IPEndPoint::CreateIPv4("www.vis.uni-stuttgart.de", 80);
//                socket.Connect(endPoint);
//                try {
//                    tmp = "GET /~grottel/megamol/corehashreg.php?hash=";
//                    tmp.Append(s);
//                    // socket.SetSndTimeo(5000); // Does not work under linux, why?
//                    socket.Send(tmp.PeekBuffer(), tmp.Length());
//                } catch(...) {
//                }
//                socket.Shutdown();
//            } catch(...) {
//            }
//            socket.Close();
//        } catch(...) {
//        }
//        vislib::net::Socket::Cleanup();
//
//#endif /* _SEND_CORE_HASH_INFO */
//
//    }

#if !(defined(DEBUG) || defined(_DEBUG))
    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_VL);
#endif /* !(defined(DEBUG) || defined(_DEBUG)) */
    megamol::core::CoreInstance *inst = new megamol::core::CoreInstance();
    if (inst == NULL) {
        return MMC_ERR_MEMORY; // out of memory or initialisation failed.
    }

    if (megamol::core::ApiHandle::CreateHandle(hCore, inst)) {
        return MMC_ERR_NO_ERROR;
    } else {
        vislib::sys::Log::DefaultLog.WriteError("mmcCreateCore: could not create ApiHandle");
        return MMC_ERR_UNKNOWN;
    }
}


/*
 * mmcSetInitialisationValue
 */
MEGAMOLCORE_API mmcErrorCode
MEGAMOLCORE_CALL mmcSetInitialisationValue(void *hCore, mmcInitValue key, 
        mmcValueType type, const void* value) {

    megamol::core::CoreInstance *inst
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);

    if (inst == NULL) { return MMC_ERR_INVALID_HANDLE; }
    try {

        return inst->SetInitValue(key, type, value);

    } catch(vislib::IllegalStateException) {
        return MMC_ERR_STATE;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteError("mmcSetInitialisationValue: exception");
        return MMC_ERR_UNKNOWN;
    }

    return MMC_ERR_UNKNOWN;
}


/*
 * mmcInitialiseCoreInstance
 */
MEGAMOLCORE_API mmcErrorCode
MEGAMOLCORE_CALL mmcInitialiseCoreInstance(void *hCore) {
    megamol::core::CoreInstance *inst
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (inst == NULL) { return MMC_ERR_INVALID_HANDLE; }
    try {
        inst->Initialise();
        return MMC_ERR_NO_ERROR;
    } catch (vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Failed to initialise core instance: %s (%s; %i)\n", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Failed to initialise core instance: %s\n", "unknown exception");
    }
    vislib::sys::Log::DefaultLog.WriteError("mmcInitialiseCoreInstance: exception");
    return MMC_ERR_UNKNOWN;
}


/*
 * mmcGetConfigurationValueA
 */
MEGAMOLCORE_API const void * MEGAMOLCORE_CALL mmcGetConfigurationValueA(
        void *hCore, mmcConfigID id, const char *name, 
        mmcValueType *outType) {
    megamol::core::CoreInstance *inst
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (inst == NULL) { return NULL; }
    return inst->Configuration().GetValue(id, name, outType);
}


/*
 * mmcGetConfigurationValueW
 */
MEGAMOLCORE_API const void * MEGAMOLCORE_CALL mmcGetConfigurationValueW(
        void *hCore, mmcConfigID id, const wchar_t *name, 
        mmcValueType *outType) {
    megamol::core::CoreInstance *inst
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (inst == NULL) { return NULL; }
    return inst->Configuration().GetValue(id, name, outType);
}


/*
 * mmcSetConfigurationValueA
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSetConfigurationValueA(
        void *hCore, mmcConfigID id, const char *name, const char* val) {
    megamol::core::CoreInstance *inst = megamol::core::ApiHandle::InterpretHandle<megamol::core::CoreInstance>(hCore);
    if (inst == NULL) { return false; }
    return const_cast<megamol::core::utility::Configuration&>(inst->Configuration()).SetValue(id, name, val);
}


/*
* mmcSetConfigurationValueW
*/
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSetConfigurationValueW(
        void *hCore, mmcConfigID id, const wchar_t *name, const wchar_t* val) {
    megamol::core::CoreInstance *inst = megamol::core::ApiHandle::InterpretHandle<megamol::core::CoreInstance>(hCore);
    if (inst == NULL) { return false; }
    return const_cast<megamol::core::utility::Configuration&>(inst->Configuration()).SetValue(id, name, val);
}


/*
 * mmcRequestAllInstances
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcRequestAllInstances(void *hCore) {
    printf("Request all instances\n");
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return;
    core->RequestAllInstantiations();
}


/*
 * mmcRequestInstanceA
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcRequestInstanceA(
        void *hCore, const char *name, const char *id) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return;

    std::shared_ptr<const megamol::core::ViewDescription> vd = core->FindViewDescription(name);
    if (vd != NULL) {
        core->RequestViewInstantiation(vd.get(), vislib::StringA(id));
        return;
    }
    std::shared_ptr<const megamol::core::JobDescription> jd = core->FindJobDescription(name);
    if (jd != NULL) {
        core->RequestJobInstantiation(jd.get(), vislib::StringA(id));
        return;
    }

    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
        "Unable to queue instantiation of \"%s\": "
        "Description \"%s\" has not been found.", id, name);
}


/*
 * mmcRequestInstanceW
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcRequestInstanceW(
        void *hCore, const wchar_t *name, const wchar_t *id) {
    mmcRequestInstanceA(hCore, vislib::StringA(name).PeekBuffer(),
        vislib::StringA(id).PeekBuffer());
}


/*
 * mmcHasPendingViewInstantiationRequests
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcHasPendingViewInstantiationRequests(
        void *hCore) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return false;
    return core->HasPendingViewInstantiationRequests();
}


/*
 * mmcGetPendingViewInstanceName
 */
MEGAMOLCORE_API const char* MEGAMOLCORE_CALL mmcGetPendingViewInstanceName(void *hCore) {
    static vislib::StringA name;
    megamol::core::CoreInstance *core = megamol::core::ApiHandle::InterpretHandle<megamol::core::CoreInstance>(hCore);
    if (core == nullptr) return nullptr;
    name = core->GetPendingViewName();
    return name.PeekBuffer();
}


/*
 * mmcInstantiatePendingView
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcInstantiatePendingView(void *hCore,
        void *hView) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return false;

    if (megamol::core::ApiHandle::InterpretHandle<
        megamol::core::ViewInstance>(hView) != NULL) return false;

    megamol::core::ViewInstance *view = dynamic_cast<megamol::core::ViewInstance*>(core->InstantiatePendingView().get());
    if (view == NULL) return false;

    if (megamol::core::ApiHandle::CreateHandle(hView, view)) {
        megamol::core::ApiHandle::SetDeallocator(hView, core,
            megamol::core::CoreInstance::ViewJobHandleDalloc);
        return true;
    }
    return false;
}


/*
 * mmcHasPendingJobInstantiationRequests
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcHasPendingJobInstantiationRequests(
        void *hCore) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return false;
    return core->HasPendingJobInstantiationRequests();
}


/*
 * mmcInstantiatePendingJob
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcInstantiatePendingJob(void *hCore,
        void *hJob) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return false;

    if (megamol::core::ApiHandle::InterpretHandle<
        megamol::core::JobInstance>(hJob) != NULL) return false;

    megamol::core::JobInstance *job = dynamic_cast<megamol::core::JobInstance*>(core->InstantiatePendingJob().get());
    if (job == NULL) return false;

    if (megamol::core::ApiHandle::CreateHandle(hJob, job)) {
        megamol::core::ApiHandle::SetDeallocator(hJob, core,
            megamol::core::CoreInstance::ViewJobHandleDalloc);
        return true;
    }
    return false;
}


/*
 * mmcRenderView
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcRenderView(void *hView,
        mmcRenderViewContext *context) {
    megamol::core::ViewInstance *view
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::ViewInstance>(hView);
    ASSERT(context != NULL);
    // If the following assert explodes, some one has added a new member to the
    // context structure in the core and not everything has been rebuilt. Most
    // likely, you should update the frontend and rebuild it.
    ASSERT(sizeof(mmcRenderViewContext) == context->Size);

    if (view != NULL) {
        vislib::sys::AutoLock lock(view->ModuleGraphLock());

#ifdef MEGAMOLCORE_WITH_DIRECT3D11
        /* Pass in the D3D device that we created in the Viewer DLL. */
        megamol::core::view::ViewDirect3D *vd3d 
            = dynamic_cast<megamol::core::view::ViewDirect3D *>(view->View());
        if (vd3d != NULL) {
            ASSERT(context->Direct3DDevice != NULL);
            vd3d->UpdateFromContext(context);
        }
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */

        //megamol::core::view::AbstractTileView *atv
        //    = dynamic_cast<megamol::core::view::AbstractTileView *>(view->View());
        //if (atv != NULL) {
        //    atv->AdjustTileFromContext(context);
        //}

        if (view->View() != NULL) {
            //double it = context->Time;

            //if (it <= 0.0) {
            //    // If we did not get a time via the context, determine the time
            //    // by using the standard method.
            //    // Note: 'it' being *exactly* zero is a special case that we
            //    // want to use the instance time and not store it; if 'it' is
            //    // negative, we want to use the instance time, too, but also in
            //    // the following frames until the caller resets it.
            //    it = view->View()->GetCoreInstance()->GetCoreInstanceTime();

            //    if (context->Time != 0) {
            //        // The viewer module wants to reuse this time until it
            //        // resets 'SynchronisedTime' to -1.
            //        context->Time = it;
            //    }
            //}

            double it = view->View()->GetCoreInstance()->GetCoreInstanceTime();
            context->Time = view->View()->DefaultTime(it);
            context->InstanceTime = it; 

            view->View()->Render(*context);
            context->ContinuousRedraw = true; // TODO: Implement the real thing
        }
    }
}


/*
 * mmcRegisterViewCloseRequestFunction
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcRegisterViewCloseRequestFunction(
        void *hView, mmcViewCloseRequestFunction func, void *data) {
    megamol::core::ViewInstance *view
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::ViewInstance>(hView);
    if (view != NULL) {
        view->SetCloseRequestCallback(func, data);
    }
}


/*
 * mmcResizeView
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcResizeView(void *hView,
        unsigned int width, unsigned int height) {
    megamol::core::ViewInstance *view
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::ViewInstance>(hView);
    if ((view != NULL) && (view->View() != NULL)) {
        view->View()->Resize(width, height);
    }
}


/*
 * mmcSendKeyEvent
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSendKeyEvent(void *hView,
        mmcInputKey key, mmcInputKeyAction act, mmcInputModifiers mods) {
    megamol::core::ViewInstance* view = megamol::core::ApiHandle::InterpretHandle<megamol::core::ViewInstance>(hView);
    if ((view != NULL) && (view->View() != NULL)) {
        return view->View()->OnKey(static_cast<megamol::core::view::Key>(key),
            static_cast<megamol::core::view::KeyAction>(act), megamol::core::view::Modifiers(mods));
    }
    return false;
}


/*
 * mmcSendCharEvent
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSendCharEvent(void *hView, unsigned int cp) {
    megamol::core::ViewInstance* view = megamol::core::ApiHandle::InterpretHandle<megamol::core::ViewInstance>(hView);
    if ((view != NULL) && (view->View() != NULL)) {
        return view->View()->OnChar(cp);
    }
    return false;
}


/*
 * mmcSendMouseButtonEvent
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSendMouseButtonEvent(void *hView,
        mmcInputButton btn, mmcInputButtonAction act, mmcInputModifiers mods) {
    megamol::core::ViewInstance* view = megamol::core::ApiHandle::InterpretHandle<megamol::core::ViewInstance>(hView);
    if ((view != NULL) && (view->View() != NULL)) {
        return view->View()->OnMouseButton(static_cast<megamol::core::view::MouseButton>(btn),
            static_cast<megamol::core::view::MouseButtonAction>(act), megamol::core::view::Modifiers(mods));
    }
    return false;
}


/*
 * mmcSendMouseMoveEvent
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSendMouseMoveEvent(void *hView,
        float x, float y) {
    megamol::core::ViewInstance* view = megamol::core::ApiHandle::InterpretHandle<megamol::core::ViewInstance>(hView);
    if ((view != NULL) && (view->View() != NULL)) {
        return view->View()->OnMouseMove(x, y);
    }
    return false;
}


/*
 * mmcSendMouseScrollEvent
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSendMouseScrollEvent(void *hView,
	float dx, float dy) {
    megamol::core::ViewInstance* view = megamol::core::ApiHandle::InterpretHandle<megamol::core::ViewInstance>(hView);
    if ((view != NULL) && (view->View() != NULL)) {
        return view->View()->OnMouseScroll(dx, dy);
    }
    return false;
}


/*
 * mmcDesiredViewWindowConfig
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcDesiredViewWindowConfig(void *hView,
        int *x, int *y, int *w, int *h, bool *nd) {
    megamol::core::ViewInstance *view
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::ViewInstance>(hView);
    if ((view != NULL) && (view->View() != NULL)) {
        return view->View()->DesiredWindowPosition(x, y, w, h, nd);
    }
    return false;
}


/*
 * mmcIsJobRunning
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcIsJobRunning(void *hJob) {
    megamol::core::JobInstance *job
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::JobInstance>(hJob);
    if ((job != NULL) && (job->Job() != NULL)) {
        return job->Job()->IsRunning();
    }
    return false;
}


/*
 * mmcIsViewRunning
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcIsViewRunning(void *hView) {
    megamol::core::ViewInstance *view
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::ViewInstance>(hView);
    return (view != NULL) && (view->View() != NULL);
}


/*
 * mmcStartJob
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcStartJob(void *hJob) {
    megamol::core::JobInstance *job
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::JobInstance>(hJob);
    if ((job != NULL) && (job->Job() != NULL)) {
        return job->Job()->Start();
    }
    return false;
}


/*
 * mmcTerminateJob
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcTerminateJob(void *hJob) {
    megamol::core::JobInstance *job
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::JobInstance>(hJob);
    if ((job != NULL) && (job->Job() != NULL)) {
        job->Job()->Terminate();
    }
}


/*
 * mmcSetParameterA
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcSetParameterValueA(void *hParam,
        const char *value) {
    megamol::core::param::ParamHandle *param
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::param::ParamHandle>(hParam);
    if (param == NULL) return;
    if (param->GetParameter()->ParseValue(A2T(value))) {
        vislib::StringA name;
        param->GetIDString(name);
        // TODO: Change text if it is a button parameter
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "Setting parameter \"%s\" to \"%s\"",
            name.PeekBuffer(), vislib::StringA(
            param->GetParameter()->ValueString()).PeekBuffer());
    } else {
        vislib::StringA name;
        param->GetIDString(name);
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to set parameter \"%s\": Failed to parse value \"%s\"",
            name.PeekBuffer(), value);
    }
}


/*
 * mmcSetParameterW
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcSetParameterValueW(void *hParam,
        const wchar_t *value) {
    megamol::core::param::ParamHandle *param
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::param::ParamHandle>(hParam);
    if (param == NULL) return;
    if (param->GetParameter()->ParseValue(W2T(value))) {
        vislib::StringA name;
        param->GetIDString(name);
        // TODO: Change text if it is a button parameter
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "Setting parameter \"%s\" to \"%s\"",
            name.PeekBuffer(), vislib::StringA(
            param->GetParameter()->ValueString()).PeekBuffer());
    } else {
        vislib::StringA name;
        param->GetIDString(name);
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to set parameter \"%s\": Failed to parse value \"%s\"",
            name.PeekBuffer(), vislib::StringA(value).PeekBuffer());
    }
}


/*
 * mmcLoadProjectA
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcLoadProjectA(void *hCore,
        const char *filename) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return;
    core->LoadProject(filename);
}


/*
 * mmcLoadProjectW
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcLoadProjectW(void *hCore,
        const wchar_t *filename) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return;
    core->LoadProject(filename);
}


/*
 * mmcGetParameterA
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcGetParameterA(void *hCore,
        const char *name, void *hParam, bool bCreate) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return false;

    vislib::SmartPtr<megamol::core::param::AbstractParam>
        param = core->FindParameter(name, false, bCreate);
    if (param.IsNull()) return false;

    if (mmcIsHandleValid(hParam) != 0) {
        return false; // handle was already valid.
    }
    if (*static_cast<unsigned char*>(hParam) != 0) {
        return false; // memory pointer seams to be invalid.
    }

    return megamol::core::ApiHandle::CreateHandle(hParam,
        new megamol::core::param::ParamHandle(*core, param));
}


/*
 * mmcGetParameterW
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcGetParameterW(void *hCore,
        const wchar_t *name, void *hParam, bool bCreate) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return false;

    vislib::SmartPtr<megamol::core::param::AbstractParam>
        param = core->FindParameter(name, false, bCreate);
    if (param.IsNull()) return false;

    if (mmcIsHandleValid(hParam) != 0) {
        return false; // handle was already valid.
    }
    if (*static_cast<unsigned char*>(hParam) != 0) {
        return false; // memory pointer seams to be invalid.
    }

    return megamol::core::ApiHandle::CreateHandle(hParam,
        new megamol::core::param::ParamHandle(*core, param));
}


/*
 * mmcGetParameterValueA
 */
MEGAMOLCORE_API const char * MEGAMOLCORE_CALL mmcGetParameterValueA(
        void *hParam) {
    static vislib::StringA retval;
    megamol::core::param::ParamHandle *param
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::param::ParamHandle>(hParam);
    if (param == NULL) return NULL;
    retval = param->GetParameter()->ValueString();
    return retval.PeekBuffer();

}


/*
 * mmcGetParameterValueW
 */
MEGAMOLCORE_API const wchar_t * MEGAMOLCORE_CALL mmcGetParameterValueW(
        void *hParam) {
    static vislib::StringW retval;
    megamol::core::param::ParamHandle *param
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::param::ParamHandle>(hParam);
    if (param == NULL) return NULL;
    retval = param->GetParameter()->ValueString();
    return retval.PeekBuffer();
}


/*
 * mmcEnumParametersA
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcEnumParametersA(void *hCore,
        mmcEnumStringAFunction func, void *data) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return;
    core->EnumParameters(func, data);
}


/**
 * Helper struct for unicode parameter enumeration
 */
typedef struct _EnumParamDataHelperW_t {
    mmcEnumStringWFunction func;
    void *data;
} EnumParamDataHelperW;


extern "C" {

/**
 * Helper function for unicode parameter enumeration
 *
 * @param name The parameter name
 * @param data The enumeration data
 */
static void
#ifdef _WIN32
__stdcall
#endif /* _WIN32 */
EnumParamsW(const char *name, void *data) {
    EnumParamDataHelperW *context
        = reinterpret_cast<EnumParamDataHelperW*>(data);
    context->func(vislib::StringW(name).PeekBuffer(), context->data);
}

}


/*
 * mmcEnumParametersW
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcEnumParametersW(void *hCore,
        mmcEnumStringWFunction func, void *data) {
    EnumParamDataHelperW context;
    context.func = func;
    context.data = data;
    ::mmcEnumParametersA(hCore, ::EnumParamsW, &context);
}


/*
 * mmcGetInstanceIDA
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcGetInstanceIDA(void *hInst,
        char *buf, unsigned int *len) {
    if (len == NULL) return;

    vislib::StringA id;

    megamol::core::ViewInstance *vi
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::ViewInstance>(hInst);
    megamol::core::JobInstance *ji
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::JobInstance>(hInst);
    megamol::core::param::ParamHandle *ph
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::param::ParamHandle>(hInst);

    if (vi != NULL) {
        id = vi->Name();
    } else if (ji != NULL) {
        id = ji->Name();
    } else if (ph != NULL){
        ph->GetIDString(id);
    }

    if (buf == NULL) {
        *len = id.Length() + 1;
    } else {
        memcpy(buf, id.PeekBuffer(),
            vislib::math::Min<SIZE_T>(id.Length() + 1, *len));
    }
}


/*
 * mmcGetInstanceIDW
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcGetInstanceIDW(void *hInst,
        wchar_t *buf, unsigned int *len) {
    if (len == NULL) return;

    char *bufA = new char[*len];
    ::mmcGetInstanceIDA(hInst, bufA, len);
    vislib::StringW id(vislib::StringA(bufA, *len));
    delete[] bufA;

    if (buf == NULL) {
        *len = id.Length() + 1;
    } else {
        memcpy(buf, id.PeekBuffer(), sizeof(wchar_t)
            * vislib::math::Min<SIZE_T>(id.Length() + 1, *len));
    }
}


/*
 * mmcIsParameterRelevant
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcIsParameterRelevant(void *hInst,
        void *hParam) {
    megamol::core::param::ParamHandle *param
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::param::ParamHandle>(hParam);
    if (param == NULL) return false;

    megamol::core::JobInstance *ji
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::JobInstance>(hInst);
    if ((ji != NULL) && (ji->Job() != NULL)) {
        return ji->Job()->IsParamRelevant(param->GetParameter());
    }

    megamol::core::ViewInstance *vi
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::ViewInstance>(hInst);
    if ((vi != NULL) && (vi->View() != NULL)) {
        return vi->View()->IsParamRelevant(param->GetParameter());
    }

    return false;
}


/*
 * mmcGetParameterTypeDescription
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcGetParameterTypeDescription(
        void *hParam, unsigned char *buf, unsigned int *len) {
    megamol::core::param::ParamHandle *param
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::param::ParamHandle>(hParam);

    if (len == NULL) return;
    if (param != NULL) {
        vislib::RawStorage rs;
        param->GetParameter()->Definition(rs);
        if (buf != NULL) {
            unsigned int s = vislib::math::Min<unsigned int>(
                static_cast<unsigned int>(rs.GetSize()), *len);
            ::memcpy(buf, rs.As<unsigned char>(), s);
            *len = s;
        } else {
            *len = static_cast<unsigned int>(rs.GetSize());
        }
    } else {
        *len = 0;
    }
}

MEGAMOLCORE_API size_t MEGAMOLCORE_CALL mmcGetGlobalParameterHash(void * hCore) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return 0;
    return core->GetGlobalParameterHash();
}


/*
 * mmcFreezeOrUpdateView
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcFreezeOrUpdateView(
        void *hView, bool freeze) {
    megamol::core::ViewInstance *view
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::ViewInstance>(hView);
    if ((view != NULL) && (view->View() != NULL)) {
        view->View()->UpdateFreeze(freeze);
    }
}


/*
 * mmcQuickstartA
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcQuickstartA(void *hCore, const char *filename) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return;
    core->Quickstart(A2T(filename));
}


/*
 * mmcQuickstartW
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcQuickstartW(void *hCore, const wchar_t *filename) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return;
    core->Quickstart(W2T(filename));
}


/*
 * mmcQuickstartRegistryA
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcQuickstartRegistryA(void *hCore,
        const char *frontend, const char *feparams,
        const char *filetype, bool unreg, bool overwrite) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return;
    core->QuickstartRegistry(A2T(frontend), A2T(feparams), A2T(filetype), unreg, overwrite);
}


/*
 * mmcQuickstartRegistryW
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcQuickstartRegistryW(void *hCore,
        const wchar_t *frontend, const wchar_t *feparams,
        const wchar_t *filetype, bool unreg, bool overwrite) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return;
    core->QuickstartRegistry(W2T(frontend), W2T(feparams), W2T(filetype), unreg, overwrite);
}


/*
 * mmcWriteStateToXML
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcWriteStateToXMLA(void *hCore, const char *outFilename) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core == NULL) return;
    core->WriteStateToXML(outFilename);
}


MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcPerformGraphUpdates(void *hCore) {
    megamol::core::CoreInstance *core
        = megamol::core::ApiHandle::InterpretHandle<
        megamol::core::CoreInstance>(hCore);
    if (core != NULL) {
        core->PerformGraphUpdates();
    }
}