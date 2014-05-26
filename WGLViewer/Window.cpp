/*
 * Window.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "glh/glh_extensions.h"
#include "Window.h"
#include "MegaMolCore.h"
#include <tchar.h>
#include <cstdio>
#include <gl/GL.h>
//#include "NVSwapGroup.h"
#pragma warning(disable: 4996)
#include "glh/glh_extensions.h"
#pragma warning(default: 4996)

#include "vislib/Log.h"

using namespace megamol::wgl;

using namespace vislib::sys;

/*
 * Window::Window
 */
Window::Window(Instance& inst) : ApiHandle(), inst(inst), hWnd(NULL), hDC(NULL), hRC(NULL), w(0), h(0),
        renderCallback(), resizeCallback(), renderer(&Window::renderThread),
		affinityDC(NULL), affinityContext(NULL), renderStartEvent(nullptr) {

    DWORD style = WS_OVERLAPPEDWINDOW;
    DWORD styleEx = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;

    ::wglMakeCurrent(NULL, NULL); // paranoia

    this->hWnd = ::CreateWindowEx(styleEx, Instance::WindowClassName,
        _T("MegaMol™"), style | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
        NULL, NULL, Instance::HInstance(), static_cast<void*>(this));
    if (this->hWnd == NULL) {
        fprintf(stderr, "Failed to create window\n");
        return;
    }

    this->hDC = ::GetDC(this->hWnd);
    if (this->hDC == NULL) {
        this->Close();
        fprintf(stderr, "Failed to get device context\n");
        return;
    }

    const unsigned int colBits = 32;
    const unsigned int depthBits = 32;

    static PIXELFORMATDESCRIPTOR pfd = {
        sizeof(PIXELFORMATDESCRIPTOR), 1,
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
        PFD_TYPE_RGBA, colBits, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        depthBits, 0, 0, PFD_MAIN_PLANE, 0, 0, 0, 0
    };

    int pixelFormat = ::ChoosePixelFormat(this->hDC, &pfd);
    if (pixelFormat == 0) {
        this->Close();
        fprintf(stderr, "Failed to find matching pixel format\n");
        return;
    }

    if (!::SetPixelFormat(this->hDC, pixelFormat, &pfd)) {
        this->Close();
        fprintf(stderr, "Failed to set pixel format\n");
        return;
    }

    this->hRC = ::wglCreateContext(this->hDC);
    if (this->hRC == NULL) {
        this->Close();
        fprintf(stderr, "Failed to create rendering context\n");
        return;
    }

    ::wglMakeCurrent(NULL, NULL); // detach RC from thread

    // Context sharing!
    if (Window::mainCtxt == NULL) {

        Window::mainDC = this->hDC; // DuplicateHandle?
        Window::mainCtxt = ::wglCreateContext(Window::mainDC);
        if (Window::mainCtxt == NULL) {
            this->Close();
            fprintf(stderr, "Failed to create main rendering context\n");
            return;
        }
    }

    DWORD le = ::GetLastError();
    GLenum ge = ::glGetError();

    if (::wglShareLists(Window::mainCtxt, this->hRC) != TRUE) {
        le = ::GetLastError();
        ge = ::glGetError();
        fprintf(stderr, "Unable to share contexts between two contexts (GLE: %u; GGE: %u)\n",
            static_cast<unsigned int>(le), static_cast<unsigned int>(ge));
    }

    ::ShowWindow(this->hWnd, SW_SHOW);
    ::SetForegroundWindow(this->hWnd);
    ::SetFocus(this->hWnd);
    RECT r;
    ::GetClientRect(this->hWnd, &r);
    this->Resized(r.right - r.left, r.bottom - r.top);

	// The renderer will not actually render someone sets the instance's
	// renderStartEvent.
	this->renderStartEvent = inst.GetRenderStartEvent();
    this->renderer.Start(this);

    ::wglMakeCurrent(Window::mainDC, Window::mainCtxt);

}


/*
 * Window::~Window
 */
Window::~Window(void) {
    this->Close();
}


/*
 * Window::Close
 */
void Window::Close(void) {

	if (affinityContext != NULL) {
		wglMakeCurrent(NULL, NULL);
		wglDeleteContext(affinityContext);
		affinityContext = NULL;
	}

    if (this->hRC != NULL) {
        if (Window::mainCtxt == this->hRC) {
            Window::mainCtxt = NULL;
        }
        if (!::wglMakeCurrent(NULL, NULL)) {
            fprintf(stderr, "Unable to deactivate rendering context\n");
        }
        if (!::wglDeleteContext(this->hRC)) {
            fprintf(stderr, "Unable to delete rendering context\n");
        }
        this->hRC = NULL;
    }

	if (renderStartEvent != nullptr)
		renderStartEvent->Set();

    if (this->renderer.IsRunning()) {
        this->renderer.Join();
    }

	if (affinityDC != NULL) {
		PFNWGLDELETEDCNVPROC wglDeleteDCNV =
			(PFNWGLDELETEDCNVPROC)wglGetProcAddress("wglDeleteDCNV");

		if (wglDeleteDCNV != nullptr)
			wglDeleteDCNV(affinityDC);
		affinityDC = NULL;
	}

    if (this->hDC != NULL) {
        if (!::ReleaseDC(this->hWnd, this->hDC)) {
            fprintf(stderr, "Failed to release device context\n");
        }
        this->hDC = NULL;
    }

    if (this->hWnd != NULL) {
        if (!::DestroyWindow(this->hWnd)) {
            fprintf(stderr, "Failed to destroy window\n");
        }
        this->hWnd = NULL;
    }
}


/*
 * Window::Resized
 */
void Window::Resized(unsigned int w, unsigned int h) {
    this->w = w;
    this->h = h;
    unsigned int p[2] = { w, h };
    this->resizeCallback.Call(*this, p);
}


/*
 * Window::SetHint
 */
void Window::SetHint(unsigned int hint, bool f) {
    switch (hint) {

        case MMV_WINHINT_NODECORATIONS: {
            /* window style for decorations */
            LONG_PTR normWS = WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE | WS_OVERLAPPEDWINDOW;
            /* window style without decorations */
            LONG_PTR thinWS = WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE;
            ::SetWindowLongPtr(this->hWnd, GWL_STYLE, f ? thinWS : normWS);
        } break;

        case MMV_WINHINT_HIDECURSOR: {
            HCURSOR cur = f ? NULL : ::LoadCursor(NULL, IDC_ARROW);
            ::SetCursor(cur);
            ::SetClassLongPtr(this->hWnd, GCLP_HCURSOR, reinterpret_cast<LONG_PTR>(cur));
        } break;

        case MMV_WINHINT_STAYONTOP: {
            /* window style for decorations */
            ::SetWindowPos(this->hWnd,
                f ? HWND_TOPMOST : HWND_NOTOPMOST,
                0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE);
        } break;

        //case MMV_WINHINT_VSYNC: {
        //} break;

        default:
            fprintf(stderr, "Hint %u not supported\n", hint);
            break;
    }

}


/*
 * Window::mainCtxt
 */
HGLRC Window::mainCtxt = NULL;


/*
 * Window::mainDC
 */
HDC Window::mainDC = NULL;


/*
 * Window::WndProc
 */
LRESULT CALLBACK Window::WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    Window *that = reinterpret_cast<Window*>(::GetWindowLongPtr(hWnd, GWLP_USERDATA));

    switch (uMsg) {

        case WM_CREATE: {
            CREATESTRUCT *cs = reinterpret_cast<CREATESTRUCT *>(lParam);
            that = static_cast<Window*>(cs->lpCreateParams);
            ::SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(that));
        } break;

        //case WM_ACTIVATE:
        //    //if (!HIWORD(wParam)) {
        //    //    active=TRUE;
        //    //} else {
        //    //    active=FALSE;
        //    //}
        //    return 0;

        case WM_SYSCOMMAND:
            switch (wParam) {
            case SC_SCREENSAVE:
            case SC_MONITORPOWER:
                return 0;
            }
            break;

        case WM_CLOSE:
            ::PostQuitMessage(0);
            that->Close();
            return 0;

        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                ::PostQuitMessage(0);
                that->Close();
            }
            //keys[wParam] = TRUE;
            return 0;

        //case WM_KEYUP:
        //    //keys[wParam] = FALSE;
        //    return 0;

        case WM_PAINT:
            //if (that->paintOnPaint) {
            //    that->Paint();
            //}
            ::ValidateRect(that->hWnd, NULL);
            return 0;

        case WM_SIZE:
            that->Resized(LOWORD(lParam), HIWORD(lParam));
            return 0;

    }

    return ::DefWindowProc(hWnd, uMsg, wParam, lParam);
}


/*
 * Window::renderThread
 */
DWORD Window::renderThread(void *userData) {
    Window *that = static_cast<Window *>(userData);
    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Size = sizeof(context);
	context.Window = that->hWnd;

    // not too good, but ok for now
	if (that->renderStartEvent != nullptr)
		that->renderStartEvent->Wait();
	else
		vislib::sys::Thread::Sleep(2500);
	
	// The context must not be bound until ALL windows have been created
    ::wglMakeCurrent(that->hDC, that->hRC);

	// (Try to) setup the affinity.
	that->setupContextAffinity(context.Window);

    //if (NVSwapGroup::Instance().JoinSwapGroup(1) == GL_TRUE) {
    //    if (NVSwapGroup::Instance().BindSwapBarrier(1, 1) == GL_TRUE) {
    //        printf("NVSwapGroup set up (1, 1)\n");
    //    } else {
    //        fprintf(stderr, "Failed to bind NVSwapBarrier\n");
    //    }
    //} else {
    //    fprintf(stderr, "Failed to join NVSwapGroup\n");
    //}

    while (that->hRC) {
        ::glViewport(0, 0, that->w, that->h);

        ASSERT(context.SynchronisedTime == 0);  // Standard timing behaviour.
        that->renderCallback.Call(*that, &context);

        ::SwapBuffers(that->hDC);

        vislib::sys::Thread::Reschedule();
    }

    ::wglMakeCurrent(NULL, NULL);

    return 0;
}

/*
 * Window::setupContextAffinity
 */
bool Window::setupContextAffinity(HWND window) {

	if (window == NULL) {
		// We can't do anything useful without knowing where our window is.
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "No render window "
			"supplied, GPU affinity not set.");
		return false;
	}

	// Try to get the affinity extension functions. The required types should
	// be defined in wglext.h.
	PFNWGLENUMGPUSNVPROC wglEnumGpusNV =
		(PFNWGLENUMGPUSNVPROC)wglGetProcAddress("wglEnumGpusNV");
	PFNWGLENUMGPUDEVICESNVPROC wglEnumGpuDevicesNV =
		(PFNWGLENUMGPUDEVICESNVPROC)wglGetProcAddress("wglEnumGpuDevicesNV");
	PFNWGLCREATEAFFINITYDCNVPROC wglCreateAffinityDCNV =
		(PFNWGLCREATEAFFINITYDCNVPROC)wglGetProcAddress(
		"wglCreateAffinityDCNV");
	PFNWGLDELETEDCNVPROC wglDeleteDCNV =
		(PFNWGLDELETEDCNVPROC)wglGetProcAddress("wglDeleteDCNV");

	if (wglEnumGpusNV == nullptr || wglEnumGpuDevicesNV == nullptr ||
		wglCreateAffinityDCNV == nullptr || wglDeleteDCNV == nullptr)
	{
		// The affinity extension is unavailable.
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
			"WGL_NV_gpu_affinity extension is unavailable. GPU affinity not set.");
		return false;
	}

	RECT windowRect;
	GetWindowRect(window, &windowRect);

	int windowArea = (windowRect.right - windowRect.left) *
		(windowRect.bottom - windowRect.top);

	unsigned int gpuIndex = 0;
	HGPUNV gpuHandle = nullptr;
	GPU_DEVICE deviceInfo;

	while (wglEnumGpusNV(gpuIndex, &gpuHandle))	{
		unsigned int deviceIndex = 0;

		ZeroMemory(&deviceInfo, sizeof(deviceInfo));
		deviceInfo.cb = sizeof(deviceInfo);

		while (wglEnumGpuDevicesNV(gpuHandle, deviceIndex, &deviceInfo)) {
			RECT intersectRect;
			IntersectRect(&intersectRect, &windowRect,
				&deviceInfo.rcVirtualScreen);

			int intersectArea = (intersectRect.right - intersectRect.left) *
				(intersectRect.bottom - intersectRect.top);

			if (intersectArea * 2 >= windowArea) {
				// >= 50% overlap. Good enough. Let's pick this GPU for rendering.
				Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Setting affinity "
					"to GPU #%u device \"%s\" (\"%s\").\n", gpuIndex,
					deviceInfo.DeviceString, deviceInfo.DeviceName);

				HGPUNV gpuMask[2];
				gpuMask[0] = gpuHandle;
				gpuMask[1] = nullptr;

				affinityDC = wglCreateAffinityDCNV(gpuMask);
				if (affinityDC == NULL) {
					Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to "
						"create an affinity device context.");
					return false;
				}

				// Copy the pixel format from the device context to the
				// affinity context.
				PIXELFORMATDESCRIPTOR pixelFormat;
				ZeroMemory(&pixelFormat, sizeof(pixelFormat));
				pixelFormat.nSize = sizeof(pixelFormat);

				int pixelFormatIndex = GetPixelFormat(hDC);
				DescribePixelFormat(hDC, pixelFormatIndex, sizeof(pixelFormat),
					&pixelFormat);

				SetPixelFormat(affinityDC, pixelFormatIndex, &pixelFormat);

				// Create an affinity render context from the affinity device
				// context.
				affinityContext = wglCreateContext(affinityDC);
				if (affinityContext == NULL) {
					Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to "
						"create a render context from the affinity context.");

					wglDeleteDCNV(affinityDC);
					affinityDC = NULL;
					return false;
				}

				// Now make current the affinityContext and the original (!)
				// device context (because we want to render into the window
				// represented by the original context).
				if (!wglMakeCurrent(hDC, affinityContext)) {
					Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to "
						"make the affinity context current.");
					return false;
				}

				return true;
			}

			deviceIndex++;
		}
		gpuIndex++;
	}

	Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Render window (%d, %d) - "
		"(%d, %d) has no sufficient overlap with any GPU device. GPU "
		"affinity not set.\n", windowRect.left, windowRect.top,
		windowRect.right, windowRect.bottom);

	return false;
}
