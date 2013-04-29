/*
 * NVSwapGroup.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "NVSwapGroup.h"
#include "vislib/Log.h"
#include "vislib/String.h"
#include "vislib/sysfunctions.h"
#ifdef _WIN32
#include "glh/glh_extensions.h"
#include <windows.h>
#include <GL/gl.h>
#include "GL/wglext.h"
#endif /* _WIN32 */

//extern "C" {

#ifdef _WIN32

// TODO: Can glh initialize wgl extensions?

static PFNWGLJOINSWAPGROUPNVPROC wglJoinSwapGroupNV = NULL;
static PFNWGLBINDSWAPBARRIERNVPROC wglBindSwapBarrierNV = NULL;
static PFNWGLQUERYSWAPGROUPNVPROC wglQuerySwapGroupNV = NULL;
static PFNWGLQUERYMAXSWAPGROUPSNVPROC wglQueryMaxSwapGroupsNV = NULL;
static PFNWGLQUERYFRAMECOUNTNVPROC wglQueryFrameCountNV = NULL;
static PFNWGLRESETFRAMECOUNTNVPROC wglResetFrameCountNV = NULL;
#endif /* _WIN32 */

//}

/*
 * NVSwapGroup::Instance
 */
NVSwapGroup& NVSwapGroup::Instance(void) {
    static NVSwapGroup i;
    return i;
}


bool NVSwapGroup::JoinSwapGroup(unsigned int group) {
    this->assertExtensions();
    if (wglJoinSwapGroupNV == NULL) return false;
    return wglJoinSwapGroupNV(::wglGetCurrentDC(), group) == GL_TRUE;
}

bool NVSwapGroup::BindSwapBarrier(unsigned int group, unsigned int barrier) {
    this->assertExtensions();
    if (wglBindSwapBarrierNV == NULL) return false;
    return wglBindSwapBarrierNV(group, barrier) == GL_TRUE;
}

bool NVSwapGroup::QuerySwapGroup(unsigned int &group, unsigned int &barrier) {
    this->assertExtensions();
    if (wglQuerySwapGroupNV == NULL) return false;
    return wglQuerySwapGroupNV(::wglGetCurrentDC(), &group, &barrier) == GL_TRUE;
}

bool NVSwapGroup::QueryMaxSwapGroups(unsigned int &maxGroups, unsigned int &maxBarriers) {
    this->assertExtensions();
    if (wglQueryMaxSwapGroupsNV == NULL) return false;
    return wglQueryMaxSwapGroupsNV(::wglGetCurrentDC(), &maxGroups, &maxBarriers) == GL_TRUE;
}

bool NVSwapGroup::QueryFrameCount(unsigned int &count) {
    this->assertExtensions();
    if (wglQueryFrameCountNV == NULL) return false;
    return wglQueryFrameCountNV(::wglGetCurrentDC(), &count) == GL_TRUE;
}

bool NVSwapGroup::ResetFrameCount(void) {
    this->assertExtensions();
    if (wglResetFrameCountNV == NULL) return false;
    return wglResetFrameCountNV(::wglGetCurrentDC()) == GL_TRUE;
}


/*
 * NVSwapGroup::NVSwapGroup
 */
NVSwapGroup::NVSwapGroup(void) : extensionInitialized(false) {
    // Intentionally empty
}


/*
 * NVSwapGroup::~NVSwapGroup
 */
NVSwapGroup::~NVSwapGroup(void) {
    // Intentionally empty
}


/*
 * NVSwapGroup::assertExtensions
 */
void NVSwapGroup::assertExtensions(void) {
#ifdef _WIN32
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

    //vislib::sys::WriteTextFile(vislib::StringA("V:\\keshiki.oglext.txt"), extStr);
    //printf("OpenGL extensions:\n%s\n", extStr.PeekBuffer());

    if (extStr.Contains("\nWGL_NV_swap_group\n")) {
        fprintf(stdout, "NV_swap_group extension found\n");
    } else {
        fprintf(stderr, "NV_swap_group extension not found\n");
        return;
    }

    wglJoinSwapGroupNV = reinterpret_cast<PFNWGLJOINSWAPGROUPNVPROC>(::wglGetProcAddress("wglJoinSwapGroupNV"));
    wglBindSwapBarrierNV = reinterpret_cast<PFNWGLBINDSWAPBARRIERNVPROC>(::wglGetProcAddress("wglBindSwapBarrierNV"));
    wglQuerySwapGroupNV = reinterpret_cast<PFNWGLQUERYSWAPGROUPNVPROC>(::wglGetProcAddress("wglQuerySwapGroupNV"));
    wglQueryMaxSwapGroupsNV = reinterpret_cast<PFNWGLQUERYMAXSWAPGROUPSNVPROC>(::wglGetProcAddress("wglQueryMaxSwapGroupsNV"));
    wglQueryFrameCountNV = reinterpret_cast<PFNWGLQUERYFRAMECOUNTNVPROC>(::wglGetProcAddress("wglQueryFrameCountNV"));
    wglResetFrameCountNV = reinterpret_cast<PFNWGLRESETFRAMECOUNTNVPROC>(::wglGetProcAddress("wglResetFrameCountNV"));

    if ((wglJoinSwapGroupNV == NULL) || (wglBindSwapBarrierNV == NULL) || (wglQuerySwapGroupNV == NULL) || 
            (wglQueryMaxSwapGroupsNV == NULL) || (wglQueryFrameCountNV == NULL) || (wglResetFrameCountNV == NULL)) {
        fprintf(stderr, "Unable to initialize NV_swap_group extension\n");
        return;
    }

    fprintf(stdout, "NOW TESTING\n");
    PFNWGLENABLEGENLOCKI3DPROC ft1 = reinterpret_cast<PFNWGLENABLEGENLOCKI3DPROC>(::wglGetProcAddress("wglEnableGenlockI3D"));
    fprintf(stdout, "wglEnableGenlockI3D = %d\n", reinterpret_cast<int>(ft1));
    PFNWGLENABLEFRAMELOCKI3DPROC ft2 = reinterpret_cast<PFNWGLENABLEFRAMELOCKI3DPROC>(::wglGetProcAddress("wglEnableFrameLockI3D"));
    fprintf(stdout, "wglEnableFrameLockI3D = %d\n", reinterpret_cast<int>(ft2));


    fprintf(stdout, "NV_swap_group extension initialized\n");
#endif /* _WIN32 */
}
