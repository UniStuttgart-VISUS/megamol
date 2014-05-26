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
Window::Window(Instance& inst, int windows) : ApiHandle(), inst(inst),
        renderCallback(), resizeCallback(), renderer(&Window::renderThread),
		activeTiles(0), infoDc(NULL), infoRc(NULL) {

	if (windows <= 0)
	{
		// throw std::invalid_argument("windows must be positive");
		fprintf(stderr, "windows must be positive\n");
		return;
	}

	if (windows > maxTileCount)
	{
		// throw std::invalid_argument("windows is too large");
		fprintf(stderr, "windows is too large\n");
		return;
	}

	ZeroMemory(this->tiles, sizeof(Tile) * maxTileCount);

	for (int i = 0; i < windows; i++)
	{
		if (!tiles[i].Create(this, i))
		{
			Close();
			fprintf(stderr, "Window creation failed\n");
			return;
		}
	}

	// ::wglMakeCurrent(tiles[0].dc, tiles[0].rc);

	// Megamol seems to assume that once a window has been created, an
	// OpenGL context has been set for the calling thread. We don't want to
	// set any of our tile contexts because we'll need them for rendering in
	// the render thread (and they can't be current in two threads at the same
	// time). Therefore, we create a dummy context for the desktop and set
	// that.
	if (CreateRenderContext(NULL, infoDc, infoRc))
		::wglMakeCurrent(infoDc, infoRc);

	activeTiles = windows;
    this->renderer.Start(this);
}


/*
 * Window::~Window
 */
Window::~Window(void) {
    this->Close();

	if (infoRc != NULL)
		::wglDeleteContext(infoRc);
	if (infoDc != NULL)
		::ReleaseDC(NULL, infoDc);
}


/*
 * Window::Close
 */
void Window::Close(void) {

	for (int i = 0; i < maxTileCount; i++)
		tiles[i].End();
	
	if (this->renderer.IsRunning())
		this->renderer.Join();

	for (int i = 0; i < maxTileCount; i++)
		tiles[i].Destroy();
}

bool Window::IsValid()
{
	for (int i = 0; i < activeTiles; i++) {
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
			for (int i = 0; i < activeTiles; i++)
				::SetWindowLongPtr(this->tiles[i].handle, GWL_STYLE, f ? thinWS : normWS);
        } break;

        case MMV_WINHINT_HIDECURSOR: {
            HCURSOR cur = f ? NULL : ::LoadCursor(NULL, IDC_ARROW);
            ::SetCursor(cur);
			for (int i = 0; i < activeTiles; i++)
				::SetClassLongPtr(this->tiles[i].handle, GCLP_HCURSOR, reinterpret_cast<LONG_PTR>(cur));
        } break;

        case MMV_WINHINT_STAYONTOP: {
            /* window style for decorations */
			for (int i = 0; i < activeTiles; i++)
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
	for (int i = 0; i < activeTiles; i++)
		SetWindowTextA(tiles[i].handle, title);
}

void Window::SetTitle(const wchar_t *title)
{
	for (int i = 0; i < activeTiles; i++)
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

	for (int i = 0; i < activeTiles; i++)
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

Window::Tile::Tile() : window(nullptr), index(-1), handle(nullptr), dc(nullptr), rc(nullptr) {

}


bool Window::Tile::Create(Window *window, int index)
{
	const DWORD style = WS_OVERLAPPEDWINDOW;
	const DWORD styleEx = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;

	::wglMakeCurrent(NULL, NULL); // paranoia

	this->window = window;
	this->index = index;

	// Decide where this window is to be placed.
	int x = CW_USEDEFAULT;
	int y = CW_USEDEFAULT;
	int width = CW_USEDEFAULT;
	int height = CW_USEDEFAULT;

	bool useExtension = false;
	if (!useExtension)
	{
		int vsx = GetSystemMetrics(SM_XVIRTUALSCREEN);
		int vsy = GetSystemMetrics(SM_YVIRTUALSCREEN);
		int vsw = GetSystemMetrics(SM_CXVIRTUALSCREEN);
		int vsh = GetSystemMetrics(SM_CYVIRTUALSCREEN);

		if (index % 2 == 0) {
			x = vsx;
			width = vsw / 2;
		}
		else
		{
			x = vsw / 2;
			width = vsw - x;
		}
		if (index / 2 == 0)
		{
			y = vsy;
			height = vsh / 2;
		}
		else
		{
			y = vsh / 2;
			height = vsh - y;
		}
	}

	// Create the tile window.
	handle = ::CreateWindowEx(styleEx, Instance::WindowClassName,
		_T("MegaMol™"), style | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
		x, y, width, height,
		NULL, NULL, Instance::HInstance(), this);

	if (handle == NULL)
		return false;

	if (!CreateRenderContext(handle, dc, rc)) {
		DestroyWindow(handle);
		handle = NULL;
		return false;
	}

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


bool Window::CreateRenderContext(HWND window, HDC &dc, HGLRC &rc) {
	
	const unsigned int colBits = 32;
	const unsigned int depthBits = 32;

	dc = ::GetDC(window);
	if (dc == NULL)
		return false;

	// Initialize the pixel format.
	const static PIXELFORMATDESCRIPTOR pfd = {
		sizeof(PIXELFORMATDESCRIPTOR), 1,
		PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
		PFD_TYPE_RGBA, colBits, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		depthBits, 0, 0, PFD_MAIN_PLANE, 0, 0, 0, 0
	};

	int pixelFormat = ::ChoosePixelFormat(dc, &pfd);
	if (pixelFormat == 0 || !::SetPixelFormat(dc, pixelFormat, &pfd))
	{
		ReleaseDC(window, dc);
		dc = NULL;
		return false;
	}

	// Create an OpenGL render context for the window device context.
	rc = ::wglCreateContext(dc);
	if (rc == NULL)
	{
		ReleaseDC(window, dc);
		dc = NULL;
		return false;
	}

	return true;
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
		for (int i = 0; i < that->activeTiles; i++)
		{
			if (that->tiles[i].rc != nullptr)
			{
				// The context must not be bound until ALL windows have been created
				BOOL result1 = ::wglMakeCurrent(that->tiles[i].dc, that->tiles[i].rc);
				GLenum ge = ::glGetError();
				const GLubyte *result = glGetString(GL_EXTENSIONS);
				ge = ::glGetError();
				// GL_INVALID_VALUE

				// int v = glh_extension_supported("WGL_NV_gpu_affinity");

				::glViewport(0, 0, that->tiles[i].w, that->tiles[i].h);

				if (i == 0)
					that->renderCallback.Call(*that, &context);

				::SwapBuffers(that->tiles[i].dc);
			}
		}

		vislib::sys::Thread::Reschedule();
	}

    ::wglMakeCurrent(NULL, NULL);

    return 0;
}
