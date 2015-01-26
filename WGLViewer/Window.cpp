/*
 * Window.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "Window.h"
#include "MegaMolCore.h"
#include <tchar.h>
#include <cstdio>
//#include "NVSwapGroup.h"


#include "vislib/sys/Log.h"

using namespace megamol::wgl;

using namespace vislib::sys;

/*
 * Window::Window
 */
Window::Window(Instance& inst) : ApiHandle(), inst(inst), hWnd(NULL),
		hDC(NULL), hRC(NULL), w(0), h(0), renderCallback(), resizeCallback(),
		renderer(&Window::renderThread), affinityDC(NULL),
        affinityContext(NULL), guiAffinityContext(NULL), hGpu(nullptr),
		renderStartEvent(nullptr) {

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

	// The renderer will not actually render until someone sets the instance's
	// renderStartEvent.
	this->renderStartEvent = inst.GetRenderStartEvent();
    this->renderer.Start(this);

    // ::wglMakeCurrent(Window::mainDC, Window::mainCtxt);
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

	wglMakeCurrent(NULL, NULL);

	if (guiAffinityContext != NULL) {
		wglDeleteContext(guiAffinityContext);
		guiAffinityContext = NULL;
	}

	if (affinityContext != NULL) {
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


static LRESULT CALLBACK DummyWndProc(HWND handle, UINT message, WPARAM wParam, LPARAM lParam) {
	return DefWindowProc(handle, message, wParam, lParam);
}

/**
 * Window::InitContext
 */
void Window::InitContext() {

	// We want to avoid selecting any context into a window's device context
	// before the affinity context is selected. Since we need a context for
	// retrieving the affinity extension functions, we create a dummy window,
	// dummy device context and dummy rendering context.
	WNDCLASS dummyClass;
	ZeroMemory(&dummyClass, sizeof(dummyClass));
	dummyClass.lpfnWndProc = DummyWndProc;
	dummyClass.hInstance = Instance::HInstance();
	dummyClass.lpszClassName = _T("WGLViewer::Window::DummyClass");

	RegisterClass(&dummyClass);

	HWND dummyWindow = ::CreateWindowEx(0, dummyClass.lpszClassName,
		NULL, WS_POPUP, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
		CW_USEDEFAULT, NULL, NULL, Instance::HInstance(), NULL);
	if (dummyWindow == NULL) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to create dummy "
			"window for GPU affinity initialization.");
		return;
	}

	HDC dummyDC = ::GetDC(dummyWindow);
	if (dummyDC == NULL) {
		DestroyWindow(dummyWindow);
		Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to create dummy "
			"device context for GPU affinity initialization.");
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

	int pixelFormat = ::ChoosePixelFormat(dummyDC, &pfd);
	if (pixelFormat == 0 || !::SetPixelFormat(dummyDC, pixelFormat, &pfd)) {
		ReleaseDC(dummyWindow, dummyDC);
		DestroyWindow(dummyWindow);
		Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to find pixel "
			"format for GPU affinity initialization.");
		return;
	}

	HGLRC dummyContext = ::wglCreateContext(dummyDC);
	if (dummyContext == NULL) {
		ReleaseDC(dummyWindow, dummyDC);
		DestroyWindow(dummyWindow);
		Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to create "
			"rendering context for GPU affinity initialization.");
		return;
	}

	::wglMakeCurrent(dummyDC, dummyContext);

	if (!setupContextAffinity(hWnd)) {

		// No affinity context has been set. Fall back to the context created in our
		// constructor.
		::wglMakeCurrent(mainDC, mainCtxt);
	}
	else {
		// An affinity context has been set.
	}

	wglDeleteContext(dummyContext);
	ReleaseDC(dummyWindow, dummyDC);
	DestroyWindow(dummyWindow);
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
//	context.Window = that->hWnd;

	// @scharnkn: The previous implementation was based on the assumption
	// that, on average, the world is a better place after 2.5 seconds. For
	// this new approach, plan A assumes that someone supplied a pointer to an
	// event object to our constructor and that this event will be set once
	// all is well. This event resides in the Instance class and is set once
	// the Console's main loop begins and has processed all initial messages.
	// If we don't have an event pointer (because a null pointer was supplied
	// to the constructor), we revert to the original strategy of assuming a
	// 2.5 second delay between now and a better time.
	if (that->renderStartEvent != nullptr)
		that->renderStartEvent->Wait();
	else
		vislib::sys::Thread::Sleep(2500);

	if (that->affinityContext != nullptr) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Using affinity context in "
			"render thread.\n");
		if (::wglMakeCurrent(that->hDC, that->affinityContext)) {
			Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Current is now dc %p, rc %p.",
				that->hDC, that->affinityContext);
		}
		else {
			Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Unable to make affinity render "
				"context current for render thread (%p, %p).", that->hDC, that->affinityContext);
		}
	}
	else {
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Using non-affinity context "
			"in render thread.\n");
		if (::wglMakeCurrent(that->hDC, that->hRC)) {
			Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Current is now dc %p, rc %p.",
				that->hDC, that->hRC);
		}
		else {
			Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Unable to make non-affinity render "
				"context current for render thread (%p, %p).", that->hDC, that->hRC);
		}
	}

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

        ASSERT(context.Time == 0);  // Standard timing behaviour.
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
    this->hGpu = nullptr;
	_GPU_DEVICE deviceInfo;

	while (wglEnumGpusNV(gpuIndex, &this->hGpu)) {
		unsigned int deviceIndex = 0;

		ZeroMemory(&deviceInfo, sizeof(deviceInfo));
		deviceInfo.cb = sizeof(deviceInfo);

		while (wglEnumGpuDevicesNV(this->hGpu, deviceIndex, &deviceInfo)) {
			RECT intersectRect;
			IntersectRect(&intersectRect, &windowRect,
				&deviceInfo.rcVirtualScreen);

			int intersectArea = (intersectRect.right - intersectRect.left) *
				(intersectRect.bottom - intersectRect.top);

			if (intersectArea * 2 >= windowArea) {
				// >= 50% overlap. Good enough. Let's pick this GPU for
				// rendering.
				Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Setting affinity "
					"to GPU #%u device \"%s\" (\"%s\").\n", gpuIndex,
					deviceInfo.DeviceString, deviceInfo.DeviceName);

				HGPUNV gpuMask[2] = { this->hGpu, nullptr };

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

				// affinityContext will be used in the render thread. We need
				// a separate context that the gui thread can load its data
				// into.
				guiAffinityContext = wglCreateContext(affinityDC);
				if (guiAffinityContext == NULL) {
					Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to "
						"create a second render context from the affinity "
						"context.");

					wglDeleteContext(affinityContext);
					affinityContext = NULL;
					wglDeleteDCNV(affinityDC);
					affinityDC = NULL;
					return false;
				}

				// For this to work, these contexts must be shared (which only
				// works because they use the same affinity mask).
				if (!wglShareLists(affinityContext, guiAffinityContext)) {
					Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to "
						"share affinity contexts.");

					wglDeleteContext(guiAffinityContext);
					guiAffinityContext = NULL;
					wglDeleteContext(affinityContext);
					affinityContext = NULL;
					wglDeleteDCNV(affinityDC);
					affinityDC = NULL;
					return false;
				}

				// Now make current the affinityContext and the original (!)
				// device context (because we want to render into the window
				// represented by the original context). We think the first
				// wglMakeCurrent call for this window MUST be with an
				// affinity context or affinity will not work even if an
				// affinity context will be selected later.
				if (!wglMakeCurrent(hDC, guiAffinityContext)) {
					Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Failed to "
						"make the affinity context current.");

					// Fallback to the original context.
					wglMakeCurrent(mainDC, mainCtxt);

					wglDeleteContext(guiAffinityContext);
					guiAffinityContext = NULL;
					wglDeleteContext(affinityContext);
					affinityContext = NULL;
					wglDeleteDCNV(affinityDC);
					affinityDC = NULL;
					return false;
				}

				Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Current is now dc %p, rc %p.",
					hDC, guiAffinityContext);

				return true;
			}

			deviceIndex++;
		}
		gpuIndex++;
	}

	Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Render window (%d, %d) - "
		"(%d, %d) has no sufficient overlap with any GPU device. GPU "
		"affinity not set.", windowRect.left, windowRect.top,
		windowRect.right, windowRect.bottom);

	return false;
}
