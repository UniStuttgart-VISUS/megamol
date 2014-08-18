/*
 * AbstractWindow.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2012 Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTWINDOW_H_INCLUDED
#define VISLIB_ABSTRACTWINDOW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#endif /* _WIN32 */

#include "vislib/Dimension.h"
#include "vislib/Point.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/StackTrace.h"
#include "vislib/String.h"


namespace vislib {
namespace graphics {


    /**
     * This class provides an interface for native windows.
     *
     * TODO: Currently, I need this only for a new DX11 window factory, so
     * I will only implement the Win32 part.
     */
    class AbstractWindow {

    public:

        /** The type to specify the window's size. */
        typedef math::Dimension<int, 2> Dimension;

        /** The type to specify the window's position. */
        typedef math::Point<int, 2> Point;

        /** Dtor. */
        virtual ~AbstractWindow(void);

        /**
         * Close the window.
         *
         * @throws SystemException In case the window could not be closed, e.g.
         *                         because it was not open.
         */
        void Close(void);

        /**
         * Create the window.
         *
         * @param title  The title of the window.
         * @param left   The desired left border of the window.
         * @param top    The desired top border of the window.
         * @param width  The desired width of the client area of the window.
         * @param height The desired height of the client area of the window.
         * 
         * @throws IllegalStateException If the window handle of the object has
         *                               already been set.
         * @throws SystemException If the instance handle could not be retrieved
         *                         or if the window rectangle could not be 
         *                         adjusted in size.
         */
        void Create(const vislib::StringA& title, int left, int top, 
            int width, int height);

        /**
         * Create the window.
         *
         * @param title  The title of the window.
         * @param left   The desired left border of the window.
         * @param top    The desired top border of the window.
         * @param width  The desired width of the client area of the window.
         * @param height The desired height of the client area of the window.
         * 
         * @throws IllegalStateException If the window handle of the object has
         *                               already been set.
         * @throws SystemException If the instance handle could not be retrieved
         *                         or if the window rectangle could not be 
         *                         adjusted in size.
         */
        void Create(const vislib::StringW& title, int left, int top, 
            int width, int height);

        /**
         * Answer the current position (left, top corner) of the window.
         *
         * @return The position of the window.
         *
         * @throws SystemException In case of an error.
         */
        Point GetPosition(void) const;

        /**
         * Answer the current size of the window.
         *
         * @return The size of the window.
         *
         * @throws SystemException In case of an error.
         */
        Dimension GetSize(void) const;

        /**
         * Makes the window invisible.
         */
        inline void Hide(void) {
            VLSTACKTRACE("AbstractWindow::Hide", __FILE__, __LINE__);
#ifdef _WIN32
            ::ShowWindow(this->hWnd, SW_HIDE);
#else /* _WIN32 */
            throw MissingImplementationException("AbstractWindow::Hide",
                __FILE__, __LINE__);
#endif /* _WIN32 */
        }

        /**
         * Move the window to the designated position.
         *
         * @param x The new left side of the window.
         * @param y The new top side of the window.
         *
         * @throws SystemExeption In case of an error.
         */
        void Move(const int x, const int y);

        /**
         * Resize the window.
         *
         * @param width The new width of the window.
         * @param height The new height of the window.
         *
         * @throws SystemExeption In case of an error.
         */
        void Resize(const int width, const int height);

        /**
         * Makes the window visible.
         */
        inline void Show(void) {
            VLSTACKTRACE("AbstractWindow::Show", __FILE__, __LINE__);
#ifdef _WIN32
            ::ShowWindow(this->hWnd, SW_SHOW);
#else /* _WIN32 */
            throw MissingImplementationException("AbstractWindow::Show",
                __FILE__, __LINE__);
#endif /* _WIN32 */
        }

    protected:

        /** Ctor. */
        AbstractWindow(void);

#ifdef _WIN32
        /**
         * This method is called once the window was created. It allows 
         * subclasses to perform additional initialisation tasks, e.g. creating
         * 3D resources.
         *
         * The default implementation does nothing.
         *
         * @param hWnd The handle of the window that was just created.
         */
        virtual void onCreated(HWND hWnd);
#endif /* _WIN32 */

#ifdef _WIN32
        /**
         * This method is called before a window is actually created and after 
         * the default styles have been assigned. It allows subclasses to change
         * the style of the window.
         *
         * The default implementation does nothing.
         */
        virtual void onCreating(DWORD& inOutStyle, DWORD& inOutExStyle) throw();
#endif /* _WIN32 */

#ifdef _WIN32
        /**
         * This method is called if a window message for the window was 
         * received. Subclasses can override the message to perform their own
         * operations instead of the default actions.
         *
         * The default implementation does nothing but setting 'outHandled' to
         * false and returning 0.
         *
         * @param outHandled If set true, the default actions (default window 
         *                   procedure) for the message are skipped (this does,
         *                   however, not include the event handlers provided
         *                   by the window itself). Otherwise, the object will 
         *                   perform the standard handling of the message.
         * @param msg
         * @param wParam
         * @param lParam
         * 
         * @return
         */
        virtual LRESULT onMessage(bool& outHandled, UINT msg, WPARAM wParam, 
            LPARAM lParam) throw();
#endif /* _WIN32 */

        /**
         * This method is called if the window was resized.
         *
         * The default implementation does nothing.
         *
         * @param width
         * @param height
         */
        virtual void onResized(const int width, const int height);

#ifdef _WIN32
        /**
         * This method is called once the WNDCLASSEX for registering the window
         * class was filled and before the window class is actually registered.
         * Subclasses have the possibility to adjust the window class in this
         * method. If subclasses change any of the settings, they must also
         * change the name of the window class, i.e. WNDCLASSEX.lpszClassName.
         *
         * The default implementation does nothing.
         *
         * @param inOutWndClass The description of the window class that is 
         *                      registered once this method returns.
         */
        virtual void onWindowClassRegistering(
            WNDCLASSEXW& inOutWndClass) throw();
#endif /* _WIN32 */

#ifdef _WIN32
        /**
         * This method is called if the window class was successfully 
         * registered.
         *
         * The default implementation does nothing.
         *
         * @param className The name of the window class that was registered.
         */
         virtual void onWindowClassRegistered(
             const vislib::StringW className) throw();
#endif /* _WIN32 */

#ifdef _WIN32
        /** The window handle. */
        HWND hWnd;
#endif /* _WIN32 */

    private:

#ifdef _WIN32
        /**
         * Process the messages for this window by directing them to the 
         * respective window object.
         *
         * @param hWnd
         * @param msg
         * @param wParam
         * @param lParam
         *
         * @return
         */
        static LRESULT CALLBACK wndProc(HWND hWnd, UINT msg, WPARAM wParam,
            LPARAM lParam);
#endif /* _WIN32 */

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        AbstractWindow(const AbstractWindow& rhs);

#ifdef _WIN32
        ///**
        // * Register the window class for the window, if it was not yet 
        // * registered.
        // *
        // * The method will call onWindowClassRegistering() immediately before
        // * the window class will be registered thus giving derived classes the
        // * opportunity to adjust the parameters.
        // *
        // * @param hInstance The instance handle of the application instance.         
        // *
        // * @return The name of the class that has been registered.
        // *
        // * @throws SystemException If the instance handle could not be 
        // *                         retrieved or the window class could not
        // *                         be registered.
        // */
        //inline vislib::StringA registerWindowClassA(HINSTANCE hInstance) {
        //    VLSTACKTRACE("AbstractWindow::registerWindowClassA", __FILE__, 
        //        __LINE__);
        //    return vislib::StringA(this->registerWindowClassW(hInstance));
        //}

        /**
         * Register the window class for the window, if it was not yet 
         * registered.
         *
         * The method will call onWindowClassRegistering() immediately before
         * the window class will be registered thus giving derived classes the
         * opportunity to adjust the parameters.
         *
         * @param hInstance The instance handle of the application instance.
         *
         * @return The name of the class that has been registered.
         *
         * @throws SystemException If the instance handle could not be 
         *                         retrieved or the window class could not
         *                         be registered.
         */
        vislib::StringW registerWindowClassW(HINSTANCE hInstance);
#endif /* _WIN32 */

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         *
         * @throws IllegalParamException if (this != &rhs).
         */
        AbstractWindow& operator =(const AbstractWindow& rhs);
    };

} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTWINDOW_H_INCLUDED */
