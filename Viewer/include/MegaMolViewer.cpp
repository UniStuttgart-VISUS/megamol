/*
 * MegaMolViewer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MegaMolViewer.h"
#include "ApiHandle.h"
#include "CallbackSlot.h"
#include "Viewer.h"
#include "Window.h"
#include "vislib/memutils.h"
#include "vislib/String.h"


#ifdef _WIN32
/* windows dll entry point */
#ifdef _MANAGED
#pragma managed(push, off)
#endif /* _MANAGED */

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
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

#include "stdlib.h"
#include "stdio.h"

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
 * mmvGetVersion
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvGetVersion)(
        unsigned short *outMajorVersion, unsigned short *outMinorVersion, 
        unsigned short *outMajorRevision, unsigned short *outMinorRevision) {
    // Set version data
    // TODO: Implement correctly
    if (outMajorVersion != NULL) *outMajorVersion = 0;
    if (outMinorVersion != NULL) *outMinorVersion = 3;
    if (outMajorRevision != NULL) *outMajorRevision = 0;
    if (outMinorRevision != NULL) *outMinorRevision = 0;
}


/*
 * mmvGetViewerTypeInfo
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvGetViewerTypeInfo)(
        mmcOSys *outSys, mmcHArch *outArch, int *outDebug) {

    if (outSys != NULL) {
        *outSys = MMC_OSYSTEM_UNKNOWN;
#ifdef _WIN32
#if defined(WINVER)
#if (WINVER >= 0x0501)
        *outSys = MMC_OSYSTEM_WINDOWS;
#endif /* (WINVER >= 0x0501) */
#endif /* defined(WINVER) */
#else /* _WIN32 */
        *outSys = MMC_OSYSTEM_LINUX;
#endif /* _WIN32 */
    }

    if (outArch != NULL) {
        *outArch = MMC_HARCH_UNKNOWN;
#if defined(_WIN64) || defined(_LIN64)
        *outArch = MMC_HARCH_X64;
#else /* defined(_WIN64) || defined(_LIN64) */
        *outArch = MMC_HARCH_I86;
#endif /* defined(_WIN64) || defined(_LIN64) */
    }

    if (outDebug != NULL) {
#if defined(_DEBUG) || defined(DEBUG)
        *outDebug = 1; // debug version
#else /* defined(_DEBUG) || defined(DEBUG) */
        *outDebug = 0; // release version
#endif /* defined(_DEBUG) || defined(DEBUG) */
    }
}


/*
 * mmvGetHandleSize
 */
MEGAMOLVIEWER_API unsigned int MEGAMOLVIEWER_CALL(mmvGetHandleSize)(void) {
    return megamol::viewer::ApiHandle::GetHandleSize();
}


/*
 * mmvDisposeHandle
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvDisposeHandle)(void *hndl) {
    megamol::viewer::ApiHandle::DestroyHandle(hndl);
}


/*
 * mmvIsHandleValid
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvIsHandleValid)(void *hndl) {
    return (megamol::viewer::ApiHandle::InterpretHandle<
        megamol::viewer::ApiHandle>(hndl) != NULL);
}


/*
 * mmvGetHandleType
 */
MEGAMOLVIEWER_API mmvHandleType MEGAMOLVIEWER_CALL(mmvGetHandleType)(void *hndl) {

    if (megamol::viewer::ApiHandle::InterpretHandle<
            megamol::viewer::ApiHandle>(hndl) == NULL) {
        return MMV_HTYPE_INVALID;

    } else if (megamol::viewer::ApiHandle::InterpretHandle<
            megamol::viewer::Viewer>(hndl) == NULL) {
        return MMV_HTYPE_VIEWCOREINSTANCE;

    } else {
        return MMV_HTYPE_UNKNOWN;
    }
}


/*
 * mmvCreateViewerHandle
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvCreateViewerHandle)(void *hView) {
    if (mmvIsHandleValid(hView) != 0) {
        return false; // handle was already valid.
    }
    if (*static_cast<unsigned char*>(hView) != 0) {
        return false; // memory pointer seams to be invalid.
    }

    megamol::viewer::Viewer *viewer = new megamol::viewer::Viewer();
    if (viewer == NULL) {
        return false; // out of memory or initialisation failed.
    }
    if (!viewer->Initialise()) {
        delete viewer;
        return false; // initialisation failed.
    }

    return megamol::viewer::ApiHandle::CreateHandle(hView, viewer);
}


/*
 * mmvProcessEvents
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvProcessEvents)(void *hView) {
    megamol::viewer::Viewer *viewer 
        = megamol::viewer::ApiHandle::InterpretHandle<
        megamol::viewer::Viewer>(hView);
    if (viewer == NULL) return false;

    return viewer->ProcessEvents();
}


/*
 * mmvCreateWindow
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvCreateWindow)(void *hView,
        void *hWnd) {
    megamol::viewer::Viewer *viewer 
        = megamol::viewer::ApiHandle::InterpretHandle<
        megamol::viewer::Viewer>(hView);
    if (viewer == NULL) return false;

    megamol::viewer::Window *win
        = new megamol::viewer::Window(*viewer);
    if (win == NULL) return false;

    if (megamol::viewer::ApiHandle::CreateHandle(hWnd, win)) {
        viewer->OwnWindow(win);
        return true;
    }
    // DO NOT DELET win. Has already be deleted by 'CreateHandle' as sfx
    return false;
}


/*
 * mmvSetUserData
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvSetUserData)(void *hndl,
        void *userData) {
    megamol::viewer::ApiHandle *obj 
        = megamol::viewer::ApiHandle::InterpretHandle<
        megamol::viewer::ApiHandle>(hndl);
    if (obj != NULL) {
        obj->UserData = userData;
    }
}


/*
 * mmvGetUserData
 */
MEGAMOLVIEWER_API void * MEGAMOLVIEWER_CALL(mmvGetUserData)(void *hndl) {
    megamol::viewer::ApiHandle *obj 
        = megamol::viewer::ApiHandle::InterpretHandle<
        megamol::viewer::ApiHandle>(hndl);
    return (obj != NULL) ? obj->UserData : NULL;
}


/*
 * mmvRegisterWindowCallback
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvRegisterWindowCallback)(
        void *hWnd, mmvWindowCallbacks slot, mmvCallback function) {
    megamol::viewer::Window *win 
        = megamol::viewer::ApiHandle::InterpretHandle<
        megamol::viewer::Window>(hWnd);
    if (win == NULL) return;

    megamol::viewer::CallbackSlot *cbs = win->Callback(slot);
    if (cbs != NULL) {
        cbs->Register(function);
    }
}


/*
 * mmvUnregisterWindowCallback
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvUnregisterWindowCallback)(
        void *hWnd, mmvWindowCallbacks slot, mmvCallback function) {
    megamol::viewer::Window *win 
        = megamol::viewer::ApiHandle::InterpretHandle<
        megamol::viewer::Window>(hWnd);
    if (win == NULL) return;

    megamol::viewer::CallbackSlot *cbs = win->Callback(slot);
    if (cbs != NULL) {
        cbs->Unregister(function);
    }
}


/*
 * mmvInstallContextMenu
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvInstallContextMenu)(void *hWnd) {
    megamol::viewer::Window *win 
        = megamol::viewer::ApiHandle::InterpretHandle<
        megamol::viewer::Window>(hWnd);
    if (win == NULL) return false;
    win->InstallContextMenu();
    return true;
}


/*
 * mmvSetWindowSize
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvSetWindowSize)(void *hWnd,
        unsigned int width, unsigned int height) {
    megamol::viewer::Window *win 
        = megamol::viewer::ApiHandle::InterpretHandle<
        megamol::viewer::Window>(hWnd);
    if (win == NULL) return;
    if (width <= 0) width = 1;
    if (height <= 0) height = 1;
    win->ResizeWindow(width, height);
}
