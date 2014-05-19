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

using namespace megamol::wgl;


/*
 * Window::Window
 */
Window::Window(Instance& inst, int windows) : ApiHandle(), inst(inst), /*hWnd(NULL), hDC(NULL), hRC(NULL), w(0), h(0),*/
        renderCallback(), resizeCallback(), renderer(&Window::renderThread) {

	if (windows <= 0)
	{
		// throw std::invalid_argument("windows must be positive");
		fprintf(stderr, "windows must be positive\n");
		return;
	}

	if (windows > maxWindowCount)
	{
		// throw std::invalid_argument("windows is too large");
		fprintf(stderr, "windows is too large\n");
		return;
	}

	ZeroMemory(this->tiles, sizeof(Tile) * maxWindowCount);

	for (int i = 0; i < windows; i++)
	{
		if (!tiles[i].Create(this, i))
		{
			Close();
			fprintf(stderr, "Window creation failed\n");
			return;
		}
	}

    // ::wglMakeCurrent(NULL, NULL); // detach RC from thread
	::wglMakeCurrent(tiles[0].dc, tiles[0].rc);

    // Context sharing!
    /*if (Window::mainCtxt == NULL) {

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
    ::SetFocus(this->hWnd);*/

	int activeWindows = windows;
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

	for (int i = 0; i < maxWindowCount; i++)
		tiles[i].End();
	
	if (this->renderer.IsRunning())
		this->renderer.Join();

	for (int i = 0; i < maxWindowCount; i++)
		tiles[i].Destroy();
}

bool Window::IsValid()
{
	for (int i = 0; i < activeWindows; i++) {
		if (!IsWindow(tiles[i].handle))
			return false;
	}
	return true;
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
			for (int i = 0; i < activeWindows; i++)
				::SetWindowLongPtr(this->tiles[i].handle, GWL_STYLE, f ? thinWS : normWS);
        } break;

        case MMV_WINHINT_HIDECURSOR: {
            HCURSOR cur = f ? NULL : ::LoadCursor(NULL, IDC_ARROW);
            ::SetCursor(cur);
			for (int i = 0; i < activeWindows; i++)
				::SetClassLongPtr(this->tiles[i].handle, GCLP_HCURSOR, reinterpret_cast<LONG_PTR>(cur));
        } break;

        case MMV_WINHINT_STAYONTOP: {
            /* window style for decorations */
			for (int i = 0; i < activeWindows; i++)
				::SetWindowPos(this->tiles[i].handle,
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

void Window::SetTitle(const char *title)
{
	for (int i = 0; i < activeWindows; i++)
		SetWindowTextA(tiles[i].handle, title);
}

void Window::SetTitle(const wchar_t *title)
{
	for (int i = 0; i < activeWindows; i++)
		SetWindowTextW(tiles[i].handle, title);
}

/*void Window::SetPosition(int x, int y, int width, int height)
{
	if (width <= 0)
		width = 0;
	if (height <= 0)
		height = 0;

	int x1 = x + width / 2;
	int y1 = y + height / 2;

	for (int i = 0; i < activeWindows; i++)
	{
		int tx;
		int ty;
		int tw;
		int th;
		if (i % 2 == 0) {
			tx = x;
			tw = x1 - x;
		}
		else
		{
			tx = x1;
			tw = x + width - x1;
		}
		if (i / 2 == 0)
		{
			ty = y;
			th = y1 - y;
		}
		else
		{
			ty = y1;
			th = y + height - y1;
		}
	}
}*/

/*
 * Window::mainCtxt
 */
//HGLRC Window::mainCtxt = NULL;


/*
 * Window::mainDC
 */
//HDC Window::mainDC = NULL;


/*Window::WindowData::WindowData(Window *window, int index)
{
	this->window = window;
	this->index = index;
}*/


bool Window::Tile::Create(Window *window, int index)
{
	const DWORD style = WS_OVERLAPPEDWINDOW;
	const DWORD styleEx = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
	const unsigned int colBits = 32;
	const unsigned int depthBits = 32;

	::wglMakeCurrent(NULL, NULL); // paranoia

	this->window = window;
	this->index = index;

	// Create the window itself.
	handle = ::CreateWindowEx(styleEx, Instance::WindowClassName,
		_T("MegaMol™"), style | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
		NULL, NULL, Instance::HInstance(), this);

	if (handle == NULL)
		return false;

	// Create a GDI device context for this window.
	dc = ::GetDC(handle);
	if (dc == NULL)
		return false;

	static PIXELFORMATDESCRIPTOR pfd = {
		sizeof(PIXELFORMATDESCRIPTOR), 1,
		PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
		PFD_TYPE_RGBA, colBits, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		depthBits, 0, 0, PFD_MAIN_PLANE, 0, 0, 0, 0
	};

	int pixelFormat = ::ChoosePixelFormat(dc, &pfd);
	if (pixelFormat == 0) {
		return false;
	}

	if (!::SetPixelFormat(dc, pixelFormat, &pfd))
		return false;

	// Create an OpenGL render context for the window device context.
	rc = ::wglCreateContext(dc);
	if (rc == NULL)
		return false;

	ShowWindow(handle, SW_SHOW);

	// Notify the window of its initial position.
	RECT clientRect;
	::GetClientRect(handle, &clientRect);
	Resized(clientRect.right - clientRect.left, clientRect.bottom - clientRect.top);

	return true;
}

void Window::Tile::Destroy()
{
	if (dc != NULL) {
		if (!::ReleaseDC(handle, dc)) {
			fprintf(stderr, "Failed to release device context\n");
		}
		dc = NULL;
	}

	if (handle != NULL) {
		if (!::DestroyWindow(handle)) {
			fprintf(stderr, "Failed to destroy window\n");
		}
		handle = NULL;
	}
}

void Window::Tile::End()
{
	if (rc != NULL) {
		if (!::wglMakeCurrent(NULL, NULL)) {
			fprintf(stderr, "Unable to deactivate rendering context\n");
		}
		if (!::wglDeleteContext(rc)) {
			fprintf(stderr, "Unable to delete rendering context\n");
		}
		rc = NULL;
	}
}

/*
* Window::Resized
*/
void Window::Tile::Resized(unsigned int w, unsigned int h) {
	this->w = w;
	this->h = h;
	unsigned int p[2] = { w, h };
	window->resizeCallback.Call(*window, p);
}


/*
 * Window::WndProc
 */
LRESULT CALLBACK Window::WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    Tile *that = reinterpret_cast<Tile*>(::GetWindowLongPtr(hWnd, GWLP_USERDATA));

    switch (uMsg) {

        case WM_CREATE: {
            CREATESTRUCT *cs = reinterpret_cast<CREATESTRUCT *>(lParam);
            that = static_cast<Tile*>(cs->lpCreateParams);
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
            that->window->Close();
            return 0;

        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                ::PostQuitMessage(0);
                that->window->Close();
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
            ::ValidateRect(that->handle, NULL);
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

	ZeroMemory(&context, sizeof(context));
	context.Size = sizeof(context);
	context.SynchronisedTime = -1.0;

    // not too good, but ok for now
    vislib::sys::Thread::Sleep(2500);
    
	while (that->tiles[0].rc != nullptr)
	{
		context.SynchronisedTime = -1.0;

		// Render all windows in turn.
		for (int i = 0; i < maxWindowCount; i++)
		{
			if (that->tiles[i].rc != nullptr)
			{
				// The context must not be bound until ALL windows have been created
				::wglMakeCurrent(that->tiles[i].dc, that->tiles[i].rc);

				::glViewport(0, 0, that->tiles[i].w, that->tiles[i].h);

				that->renderCallback.Call(*that, &context);

				::SwapBuffers(that->tiles[i].dc);
			}
		}

		vislib::sys::Thread::Reschedule();
	}

    ::wglMakeCurrent(NULL, NULL);

    return 0;
}
