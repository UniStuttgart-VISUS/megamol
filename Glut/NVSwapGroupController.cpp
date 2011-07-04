/*
 * NVSwapGroupController.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "NVSwapGroupController.h"
#include "vislib/Log.h"
#include "vislib/String.h"
#ifdef _WIN32
#include <windows.h>
#include <GL/gl.h>
#include "GL/wglext.h"
#include "glh/glh_extensions.h"
#endif /* _WIN32 */

using namespace megamol::viewer;

#ifdef _WIN32
typedef BOOL (* PF_WGLJOINSWAPGROUPNV)(HDC hDC, GLuint group);
typedef BOOL (* PF_WGLBINDSWAPBARRIERNV)(GLuint group, GLuint barrier);
typedef BOOL (* PF_WGLQUERYSWAPGROUPNV)(HDC hDC, GLuint *group, GLuint *barrier);
typedef BOOL (* PF_WGLQUERYMAXSWAPGROUPSNV)(HDC hDC, GLuint *maxGroups, GLuint *maxBarriers);
typedef BOOL (* PF_WGLQUERYFRAMECOUNTNV)(HDC hDC, GLuint *count);
typedef BOOL (* PF_WGLRESETFRAMECOUNTNV)(HDC hDC);
static PF_WGLJOINSWAPGROUPNV wglJoinSwapGroupNV = NULL;
static PF_WGLBINDSWAPBARRIERNV wglBindSwapBarrierNV = NULL;
static PF_WGLQUERYSWAPGROUPNV wglQuerySwapGroupNV = NULL;
static PF_WGLQUERYMAXSWAPGROUPSNV wglQueryMaxSwapGroupsNV = NULL;
static PF_WGLQUERYFRAMECOUNTNV wglQueryFrameCountNV = NULL;
static PF_WGLRESETFRAMECOUNTNV wglResetFrameCountNV = NULL;
#endif /* _WIN32 */

/*
 * NVSwapGroupController::Instance
 */
NVSwapGroupController& NVSwapGroupController::Instance(void) {
    static NVSwapGroupController i;
    return i;
}


/*
 * NVSwapGroupController::JoinGlutWindow
 */
void NVSwapGroupController::JoinGlutWindow(void) {
    using vislib::sys::Log;
    if (this->group == 0) return;

    Log::DefaultLog.SetEchoOutTarget(&Log::EchoTargetStream::StdOut);
    Log::DefaultLog.SetEchoLevel(Log::LEVEL_ALL);

#ifdef _WIN32
    this->assertExtensions();

    GLuint maxGrp, maxBar;
    ::wglQueryMaxSwapGroupsNV(::wglGetCurrentDC(), &maxGrp, &maxBar);
    if (this->group > maxGrp) {
        Log::DefaultLog.WriteWarn("Swap group %u clamped to max. group %u\n", this->group, maxGrp);
        this->group = maxGrp;
    }
    if (this->barrier > maxBar) {
        Log::DefaultLog.WriteWarn("Swap barrier %u clamped to max. barrier %u\n", this->barrier, maxBar);
        this->barrier = maxBar;
    }

    if (this->group > 0) {
        if (::wglJoinSwapGroupNV(::wglGetCurrentDC(), this->group) != GL_FALSE) {
            Log::DefaultLog.WriteInfo("Joined swap group %u\n", this->group);
        } else {
            Log::DefaultLog.WriteError("Unable to join swap group %u\n", this->group);
        }
        if (this->barrier > 0) {
            if (::wglBindSwapBarrierNV(this->group, this->barrier) != GL_FALSE) {
                Log::DefaultLog.WriteInfo("Swap group %u added to swap barrier %u\n", this->group, this->barrier);
            } else {
                Log::DefaultLog.WriteInfo("Unable to add swap group %u to swap barrier %u\n", this->group, this->barrier);
            }
        }
    }

#endif /* _WIN32 */
}


/*
 * NVSwapGroupController::NVSwapGroupController
 */
NVSwapGroupController::NVSwapGroupController(void) : group(0), barrier(0),
        extensionInitialized(false) {
    // Intentionally empty
}


/*
 * NVSwapGroupController::~NVSwapGroupController
 */
NVSwapGroupController::~NVSwapGroupController(void) {
    // Intentionally empty
}


/*
 * NVSwapGroupController::assertExtensions
 */
void NVSwapGroupController::assertExtensions(void) {
#ifdef _WIN32
    using vislib::sys::Log;
    if (this->extensionInitialized) return;
    this->extensionInitialized = true;

    vislib::StringA extStr(reinterpret_cast<const char *>(::glGetString(GL_EXTENSIONS)));
    extStr.Prepend("\n");
    extStr.Replace(" ", "\n");
    extStr.Append("\n");
    PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB = 0;
    wglGetExtensionsStringARB = (PFNWGLGETEXTENSIONSSTRINGARBPROC)::wglGetProcAddress("wglGetExtensionsStringARB");
    if(wglGetExtensionsStringARB != NULL) {
        vislib::StringA s(reinterpret_cast<const char*>(wglGetExtensionsStringARB(::wglGetCurrentDC())));
        s.Replace(" ", "\n");
        extStr.Append(s);
    }
    extStr.Append("\n");

    if (extStr.Contains("\nWGL_NV_swap_group\n")) {
        Log::DefaultLog.WriteInfo("NV_swap_group extension found");
    } else {
        Log::DefaultLog.WriteWarn("NV_swap_group extension not found");
        this->group = 0;
        return;
    }

    wglJoinSwapGroupNV = reinterpret_cast<PF_WGLJOINSWAPGROUPNV>(::wglGetProcAddress("wglJoinSwapGroupNV"));
    wglBindSwapBarrierNV = reinterpret_cast<PF_WGLBINDSWAPBARRIERNV>(::wglGetProcAddress("wglBindSwapBarrierNV"));
    wglQuerySwapGroupNV = reinterpret_cast<PF_WGLQUERYSWAPGROUPNV>(::wglGetProcAddress("wglQuerySwapGroupNV"));
    wglQueryMaxSwapGroupsNV = reinterpret_cast<PF_WGLQUERYMAXSWAPGROUPSNV>(::wglGetProcAddress("wglQueryMaxSwapGroupsNV"));
    wglQueryFrameCountNV = reinterpret_cast<PF_WGLQUERYFRAMECOUNTNV>(::wglGetProcAddress("wglQueryFrameCountNV"));
    wglResetFrameCountNV = reinterpret_cast<PF_WGLRESETFRAMECOUNTNV>(::wglGetProcAddress("wglResetFrameCountNV"));

    if ((wglJoinSwapGroupNV == NULL) || (wglBindSwapBarrierNV == NULL) || (wglQuerySwapGroupNV == NULL) || 
            (wglQueryMaxSwapGroupsNV == NULL) || (wglQueryFrameCountNV == NULL) || (wglResetFrameCountNV == NULL)) {
        Log::DefaultLog.WriteWarn("Unable to initialize NV_swap_group extension");
        this->group = 0;
        return;
    }

    Log::DefaultLog.WriteInfo("NV_swap_group extension initialized\n");
#endif /* _WIN32 */
}
