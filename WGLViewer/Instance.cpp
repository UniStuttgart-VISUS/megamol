/*
 * Instance.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include <tchar.h>
#include <cstdio>
#include "Instance.h"
#include "dwmapi.h"
#include "Window.h"
#include "vislib/sys/Thread.h"

using namespace megamol::wgl;


/*
 * Instance::WindowClassName
 */
const TCHAR* Instance::WindowClassName = _T("MMWGLWNDCLS");


/*
 * Instance::Instance
 */
Instance::Instance(void) : ApiHandle(), running(true), renderStartEvent(true, false) {
    // Intentionally empty
}


/*
 * Instance::~Instance
 */
Instance::~Instance(void) {
    if (Instance::wndClsRegistered > 0) {
        Instance::wndClsRegistered--;
        if (Instance::wndClsRegistered == 0) {
            ::UnregisterClass(WindowClassName, Instance::hInst);
        }
    }

}


/*
 * Instance::StartRender
 */
vislib::sys::Event *Instance::GetRenderStartEvent(void) {
	return &renderStartEvent;
}


/*
 * Instance::Init
 */
bool Instance::Init(HINSTANCE hInst) {
    Instance::hInst = hInst;
    this->deactivateCompositeDesktop();

    if (Instance::wndClsRegistered == 0) {
        WNDCLASS wc;
        ZeroMemory(&wc, sizeof(WNDCLASS));

        wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
        wc.lpfnWndProc = Window::GetWindowProcedure();
        wc.cbClsExtra = 0;
        wc.cbWndExtra = 0;
        wc.hInstance = Instance::HInstance();
        wc.hIcon = LoadIcon(Instance::HInstance(), MAKEINTRESOURCE(100));
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);
        wc.hbrBackground = NULL;
        wc.lpszMenuName = NULL;
        wc.lpszClassName = Instance::WindowClassName;

        if (!::RegisterClass(&wc)) {
            fprintf(stderr, "Failed to register window class\n");
            return false;
        }
    }
    Instance::wndClsRegistered++;

    return true;
}


/*
 * Instance::ProcessEvents
 */
bool Instance::ProcessEvents(void) {

    MSG msg;
    if (::PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            this->running = false;

        } else {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);

        }

    } else {
		StartRender();
        vislib::sys::Thread::Reschedule();

    }

    return this->running;
}


/*
 * Instance::StartRender
 */
void Instance::StartRender(void) {
	renderStartEvent.Set();
}


/*
 * Instance::hInst
 */
HINSTANCE Instance::hInst = NULL;


/*
 * Instance::wndClsRegistered
 */
unsigned int Instance::wndClsRegistered = 0;


/*
 * Instance::deactivateCompositeDesktop
 */
void Instance::deactivateCompositeDesktop(void) {
    BOOL retval = FALSE;

    if (FAILED(::DwmIsCompositionEnabled(&retval))) {
        fprintf(stderr, "DwmIsCompositionEnabled failed.\n");
    }

    if (retval == FALSE) return;

    if (FAILED(::DwmEnableComposition(DWM_EC_DISABLECOMPOSITION))) {
        fprintf(stderr, "DwmEnableComposition failed.\n");
    } else {
        fprintf(stdout, "Desktop composition deactivated\n");
    }


}
