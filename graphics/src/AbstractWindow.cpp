/*
 * AbstractWindow.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 * Copyright (C) 2012 Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/AbstractWindow.h"

#include "vislib/assert.h"
#include "vislib/IllegalStateException.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemException.h"


/*
 * vislib::graphics::AbstractWindow::~AbstractWindow
 */
vislib::graphics::AbstractWindow::~AbstractWindow(void) {
    VLSTACKTRACE("AbstractWindow::~AbstractWindow", __FILE__, __LINE__);
    try {
        this->Close();
    } catch (...) { /* Ignore this.*/ }
}


/*
 * vislib::graphics::AbstractWindow::Close
 */
void vislib::graphics::AbstractWindow::Close(void) {
    VLSTACKTRACE("AbstractWindow::Close", __FILE__, __LINE__);

#ifdef _WIN32
    if (::DestroyWindow(this->hWnd) == FALSE) {
        throw sys::SystemException(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    throw MissingImplementationException("AbstractWindow::Close", 
        __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::graphics::AbstractWindow::Create(
 */
void vislib::graphics::AbstractWindow::Create(const vislib::StringA& title, 
        int left, int top, int width, int height) {
    VLSTACKTRACE("AbstractWindow::Create", __FILE__, __LINE__);

#ifdef _WIN32
    this->Create(A2W(title), left, top, width, height);

#else /* _WIN32 */
    throw MissingImplementationException("AbstractWindow::Create", 
        __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::graphics::AbstractWindow::Create(
 */
void vislib::graphics::AbstractWindow::Create(const vislib::StringW& title, 
        int left, int top, int width, int height) {
    VLSTACKTRACE("AbstractWindow::Create", __FILE__, __LINE__);

#ifdef _WIN32
    HINSTANCE hInstance = NULL;
    DWORD dwStyle = 0;
    DWORD dwExStyle = 0;
    RECT wndRect;

    /* Sanity checks. */
    if (this->hWnd != NULL) {
        throw vislib::IllegalStateException("The window has already been "
            "created.", __FILE__, __LINE__);
    }

    /* Get instance handle. */
    hInstance = ::GetModuleHandle(NULL);
    if (hInstance == NULL) {
        throw vislib::sys::SystemException(__FILE__, __LINE__);
    }

    /* Register window class. */
    StringW wndClass = this->registerWindowClassW(hInstance);

    /* Set default styles. */
    dwStyle = WS_OVERLAPPEDWINDOW | WS_VISIBLE;
    dwExStyle = 0;

    /* Allow style customisation by subclasses. */
    this->onCreating(dwStyle, dwExStyle);

    /* Adjust size such that rectangle defines client area. */
    wndRect.left = left;
    wndRect.top = top;
    wndRect.right = left + width;
    wndRect.bottom = top + height;
    if (!::AdjustWindowRectEx(&wndRect, dwStyle, FALSE, dwExStyle)) {
        throw vislib::sys::SystemException(__FILE__, __LINE__); 
    }

    if ((this->hWnd = ::CreateWindowExW(dwExStyle, 
            wndClass.PeekBuffer(),
            title.PeekBuffer(), 
            dwStyle, 
            wndRect.left, 
            wndRect.top,
            wndRect.right - wndRect.left, 
            wndRect.bottom - wndRect.top,
            NULL,       // Parent window
            NULL,       // Menu
            hInstance, 
            NULL)) == NULL) {
        throw vislib::sys::SystemException(__FILE__, __LINE__);
    }

    ::SetWindowLongPtr(this->hWnd, GWLP_USERDATA,
        reinterpret_cast<LONG_PTR>(this));

    this->onCreated(this->hWnd);

#else /* _WIN32 */
    this->Create(W2A(title), left, top, width, height);
#endif /* _WIN32 */
}


/*
 * vislib::graphics::AbstractWindow::GetPosition
 */
inline vislib::graphics::AbstractWindow::Point 
vislib::graphics::AbstractWindow::GetPosition(void) const {
    VLSTACKTRACE("AbstractWindow::GetPosition", __FILE__, __LINE__);
#ifdef _WIN32
    RECT rect;
    if (::GetWindowRect(this->hWnd, &rect) == FALSE) {
        throw vislib::sys::SystemException(__FILE__, __LINE__);
    }
    return Point(rect.left, rect.top);

#else /* _WIN32 */
    throw MissingImplementationException("AbstractWindow::GetPosition",
        __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::graphics::AbstractWindow::GetSize
 */
inline vislib::graphics::AbstractWindow::Dimension 
vislib::graphics::AbstractWindow::GetSize(void) const {
    VLSTACKTRACE("AbstractWindow::GetSize", __FILE__, __LINE__);
#ifdef _WIN32
    RECT rect;
    if (::GetWindowRect(this->hWnd, &rect) == FALSE) {
        throw vislib::sys::SystemException(__FILE__, __LINE__);
    }
    return Dimension(rect.right - rect.left, rect.bottom - rect.top);

#else /* _WIN32 */
    throw MissingImplementationException("AbstractWindow::GetSize", 
        __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::graphics::AbstractWindow::Move
 */
void vislib::graphics::AbstractWindow::Move(const int x, const int y) {
    VLSTACKTRACE("AbstractWindow::Move", __FILE__, __LINE__);
#ifdef _WIN32
    Dimension size = this->GetSize();
    if (::MoveWindow(this->hWnd, x, y, size.Width(), size.Height(), TRUE) 
            == FALSE) {
        throw vislib::sys::SystemException(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    throw MissingImplementationException("AbstractWindow::Move", __FILE__, 
        __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::graphics::AbstractWindow::Resize
 */
void vislib::graphics::AbstractWindow::Resize(const int width, 
        const int height) {
    VLSTACKTRACE("AbstractWindow::Resize", __FILE__, __LINE__);
#ifdef _WIN32
    Point position = this->GetPosition();
    if (::MoveWindow(this->hWnd, position.X(), position.Y(), width, height,
            TRUE) == FALSE) {
        throw vislib::sys::SystemException(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    throw MissingImplementationException("AbstractWindow::Resize", __FILE__,
        __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::graphics::AbstractWindow::AbstractWindow
 */
vislib::graphics::AbstractWindow::AbstractWindow(void) {
    VLSTACKTRACE("AbstractWindow::AbstractWindow", __FILE__, __LINE__);
#ifdef _WIN32
    this->hWnd = NULL;
#endif /* _WIN32 */
}


#ifdef _WIN32
/*
 * vislib::graphics::AbstractWindow::onCreated
 */
void vislib::graphics::AbstractWindow::onCreated(HWND hWnd) {
    VLSTACKTRACE("AbstractWindow::onCreated", __FILE__, __LINE__);
    // Nothing to do.
}
#endif /* _WIN32 */


#ifdef _WIN32
/*
 * vislib::graphics::AbstractWindow::onCreating(
 */
void vislib::graphics::AbstractWindow::onCreating(DWORD& inOutStyle, 
        DWORD& inOutExStyle) throw() {
    VLSTACKTRACE("AbstractWindow::onCreating", __FILE__, __LINE__);
    // Nothing to do.
}
#endif /* _WIN32 */


#ifdef _WIN32 
/*
 * vislib::graphics::AbstractWindow::onMessage
 */
LRESULT vislib::graphics::AbstractWindow::onMessage(bool& outHandled, 
        UINT msg, WPARAM wParam, LPARAM lParam) throw() {
    VLSTACKTRACE("AbstractWindow::onMessage", __FILE__, __LINE__);
    outHandled = false;
    return static_cast<LRESULT>(0);
}
#endif /* _WIN32 */


/*
 * vislib::graphics::AbstractWindow::onResized
 */
void vislib::graphics::AbstractWindow::onResized(const int width, 
        const int height) {
    VLSTACKTRACE("AbstractWindow::onResized", __FILE__, __LINE__);
    // Nothing to do.
}


#ifdef _WIN32
/*
 * vislib::graphics::AbstractWindow::onWindowClassRegistering
 */
void vislib::graphics::AbstractWindow::onWindowClassRegistering(
        WNDCLASSEX& inOutWndClass) throw() {
    VLSTACKTRACE("AbstractWindow::onWindowClassRegistering", __FILE__,
        __LINE__);
    // Nothing to do.
}
#endif /* _WIN32 */


#ifdef _WIN32
/*
 * vislib::graphics::AbstractWindow::onWindowClassRegistered
 */
void vislib::graphics::AbstractWindow::onWindowClassRegistered(
        const vislib::StringW className) throw() {
    VLSTACKTRACE("AbstractWindow::onWindowClassRegistered", __FILE__, __LINE__);
    // Nothing to do.
}
#endif /* _WIN32 */


#ifdef _WIN32
/*
 * vislib::graphics::AbstractWindow::wndProc
 */
LRESULT CALLBACK vislib::graphics::AbstractWindow::wndProc(HWND hWnd, UINT msg,
        WPARAM wParam, LPARAM lParam) {
    VLSTACKTRACE("AbstractWindow::wndProc", __FILE__, __LINE__);

#pragma warning(disable: 4312)
    AbstractWindow *wnd = reinterpret_cast<AbstractWindow*>(
        static_cast<LONG_PTR>(::GetWindowLongPtrW(hWnd,GWLP_USERDATA)));
#pragma warning(default: 4312)

    bool handled = false;
    LRESULT retval = 0;

    if (wnd != NULL) {
        retval = wnd->onMessage(handled, msg, wParam, lParam);
    }

    // Implementation note: onMessage() cannot prevent the built-in event 
    // handlers from being fired, only the default procedure.
    switch (msg) {
        case WM_CLOSE:
            ASSERT(wnd != NULL);
            wnd->hWnd = NULL;
            retval = 0;
            break;

        case WM_SIZE:
            if (wnd != NULL) {
                wnd->onResized(LOWORD(lParam), HIWORD(lParam));
            }
            retval = 0;
            break;

        default:
            if (!handled) {
                retval = ::DefWindowProc(hWnd, msg, wParam, lParam);
            }
            break;
    }

    return retval;
}
#endif /* _WIN32 */


/*
 * vislib::graphics::AbstractWindow::AbstractWindow
 */
vislib::graphics::AbstractWindow::AbstractWindow(const AbstractWindow& rhs) {
    VLSTACKTRACE("AbstractWindow::AbstractWindow", __FILE__, __LINE__);
    throw UnsupportedOperationException("AbstractWindow::AbstractWindow",
        __FILE__, __LINE__);
}


#ifdef _WIN32
/*
 * vislib::graphics::AbstractWindow::registerWindowClassW
 */
vislib::StringW vislib::graphics::AbstractWindow::registerWindowClassW(
        HINSTANCE hInstance) {
    VLSTACKTRACE("AbstractWindow::registerWindowClassW", __FILE__, __LINE__);

    WNDCLASSEXW wcex = { sizeof(WNDCLASSEXW) };
    ::ZeroMemory(&wcex, sizeof(WNDCLASSEXW));

    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.cbSize = sizeof(wcex);
    wcex.cbWndExtra = sizeof(LONG_PTR);
    wcex.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    wcex.hInstance = hInstance;
    wcex.lpfnWndProc = AbstractWindow::wndProc;
    wcex.lpszClassName = L"VISLIB_ABSTRACT_WINDOW_CLASS";

    this->onWindowClassRegistering(wcex);
    ASSERT(wcex.cbWndExtra == sizeof(LONG_PTR));
    ASSERT(wcex.hInstance != NULL);
    ASSERT(wcex.lpfnWndProc == AbstractWindow::wndProc);

    if (!::GetClassInfoExW(wcex.hInstance, wcex.lpszClassName, &wcex)) {
        if (!::RegisterClassExW(&wcex)) {
            throw vislib::sys::SystemException(__FILE__, __LINE__);
        }
    }

    vislib::StringW retval(wcex.lpszClassName);
    this->onWindowClassRegistered(retval);

    return retval;
}
#endif /* _WIN32 */


/*
 * vislib::graphics::AbstractWindow::operator =
 */
vislib::graphics::AbstractWindow& vislib::graphics::AbstractWindow::operator =(
        const AbstractWindow& rhs) {
    VLSTACKTRACE("AbstractWindow::operator =", __FILE__, __LINE__);
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }
    return *this;
}
