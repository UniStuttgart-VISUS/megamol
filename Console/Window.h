/*
 * Window.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_WINDOW_H_INCLUDED
#define MEGAMOLCON_WINDOW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CoreHandle.h"
#include "HotKeyAction.h"
#include "MegaMolCore.h"
#include "MegaMolViewer.h"
#include "vislib/KeyCode.h"
#include "vislib/Map.h"
#include "vislib/SmartPtr.h"


#ifdef _WIN32
#define MMAPI_CALLBACK __stdcall
#else /* _WIN32 */
#define MMAPI_CALLBACK
#endif /* _WIN32 */

namespace megamol {
namespace console {

    /**
     * Class of viewing windows.
     */
    class Window {
    public:

        /**
         * The viewer close callback.
         *
         * @param wnd The viewer window userdata, which is the window object.
         * @param params The callback parameters.
         */
        static void CloseCallback(void *wnd, void *params);

        /**
         * Answer whether any window has been closed since this method was 
         * called the last time.
         *
         * @return 'true' if at least one window had been closed.
         */
        static inline bool HasClosed(void) {
            if (!hasClosed) return false;
            hasClosed = false;
            return true;
        }

        /**
         * The viewer rendering callback.
         *
         * @param wnd The viewer window userdata, which is the window object.
         * @param params The callback parameters.
         */
        static void RenderCallback(void *wnd, void *params);

        /**
         * The viewer resize callback.
         *
         * @param wnd The viewer window userdata, which is the window object.
         * @param params The callback parameters.
         */
        static void ResizeCallback(void *wnd,
            unsigned int *params);

        /**
         * The viewer key callback.
         *
         * @param wnd The viewer window userdata, which is the window object.
         * @param params The callback parameters.
         */
        static void KeyCallback(void *wnd,
            mmvKeyParams *params);

        /**
         * The viewer mouse button callback.
         *
         * @param wnd The viewer window userdata, which is the window object.
         * @param params The callback parameters.
         */
        static void MouseButtonCallback(void *wnd,
            mmvMouseButtonParams *params);

        /**
         * The viewer mouse move callback.
         *
         * @param wnd The viewer window userdata, which is the window object.
         * @param params The callback parameters.
         */
        static void MouseMoveCallback(void *wnd,
            mmvMouseMoveParams *params);

        /**
         * Callback when a frozen update is required
         *
         * @param wnd The viewer window userdata, which is the window object.
         * @param params The callback parameters.
         */
        static void UpdateFreezeCallback(void *wnd, int *params);

        /**
         * The view close request callback.
         *
         * @param data The user data pointer.
         */
        static void MMAPI_CALLBACK CloseRequestCallback(void* data);

        /** Ctor */
        Window(void);

        /** Dtor */
        ~Window(void);

        /**
         * Gets the handle to the core view instance.
         *
         * @return The handle to the core view instance.
         */
        inline CoreHandle& HView(void) {
            return this->hView;
        }

        /**
         * Gets the handle to the viewer window.
         *
         * @return The handle to the viewer window.
         */
        inline CoreHandle& HWnd(void) {
            return this->hWnd;
        }

        /**
         * Answer whether this window is closed, or not.
         *
         * @return 'true' if this window is closed, 'false' if not.
         */
        inline bool IsClosed(void) const {
            return this->isClosed;
        }

        /**
         * Marks this window to be closed.
         */
        inline void MarkToClose(void) {
            this->isClosed = true;
            hasClosed = true;
        }

        /**
         * Registers hot keys at the viewer
         *
         * @param hCore The core handle
         */
        void RegisterHotKeys(CoreHandle& hCore);

        /**
         * Registers a hot key action at the window. The callee takes
         * ownership of the memory of action.
         *
         * @param key The key code for the action
         * @param action The hotkey action
         * @param target A human-readable name for the target action
         */
        void RegisterHotKeyAction(const vislib::sys::KeyCode& key,
            HotKeyAction *action, const vislib::StringA& target);

    private:

        /**
         * Class for internal window hot key actions
         */
        class InternalHotKey : public HotKeyAction {
        public:

            /**
             * Ctor
             *
             * @param owner The owning window
             * @param callback The object method callback pointer
             */
            InternalHotKey(Window *owner, void (Window::*callback)(void));

            /**
             * Dtor
             */
            virtual ~InternalHotKey(void);

            /**
             * Aaaaaaaand Action!
             */
            virtual void Trigger(void);

        private:

            /** The owning object */
            Window *owner;

            /** The object method callback pointer */
            void (Window::*callback)(void);

        };

        /** global flag if any window has been closed */
        static bool hasClosed;

        /**
         * Enumeration method for the core parameters
         *
         * @param paramName The name of the parameter
         * @param contextPtr Pointer to the context structure
         */
        static void MEGAMOLCORE_CALLBACK registerHotKeys(const char *paramName, void *contextPtr);

        /**
         * Sets the states of the modifier keys.
         *
         * @param alt The state of the alt key.
         * @param ctrl The state of the ctrl key.
         * @param shift The state of the shift key.
         */
        inline void setModifierStates(bool alt, bool ctrl, bool shift);

        /**
         * The exit action exits the whole application
         */
        void doExitAction(void);

        /** Handle to the core view instance */
        CoreHandle hView;

        /** Handle to the viewer window */
        CoreHandle hWnd;

        /** Flag whether this window is closed, or not. */
        bool isClosed;

        /** Flag whether the alt modifier is active */
        bool modAlt;

        /** Flag whether the ctrl modifier is active */
        bool modCtrl;

        /** Flag whether the shif modifier is active */
        bool modShift;

        /** The state of the three mouse buttons */
        bool mouseBtns[3];

        /** The hotkeys of this window */
        vislib::Map<vislib::sys::KeyCode, vislib::SmartPtr<HotKeyAction> > hotkeys;

    };

} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_WINDOW_H_INCLUDED */
