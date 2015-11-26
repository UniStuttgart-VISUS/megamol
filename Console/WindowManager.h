/*
 * WindowManager.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_WINDOWMANAGER_H_INCLUDED
#define MEGAMOLCON_WINDOWMANAGER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Window.h"
#include "vislib/Iterator.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"


namespace megamol {
namespace console {

    class WindowManager{
    public:

        /**
         * The singelton instance method.
         *
         * @return The only instance of this class.
         */
        static WindowManager* Instance(void);

        /** Dtor. */
        ~WindowManager(void);

        /**
         * Adds a window to the manager.
         *
         * @param window The window to be added.
         */
        inline void Add(Window *window) {
            vislib::SmartPtr<Window> wnd(window);
            this->Add(wnd);
        }

        /**
         * Adds a window to the manager.
         *
         * @param window The window to be added.
         */
        void Add(vislib::SmartPtr<Window>& window);

        /**
         * Cleans up the list of windows, by removing the windows marked to be
         * closed.
         */
        void Cleanup(void);

        /**
         * Closes all window handles
         */
        void CloseAll(void);

        /**
         * Gets the number of windows registered at the manager.
         *
         * @return The number of windows registered at the manager.
         */
        inline unsigned int Count(void) const {
            return static_cast<unsigned int>(this->windows.Count());
        }

        /**
         * Marks all windows to be closed on next cleanup.
         */
        void MarkAllForClosure(void);

        /**
         * Gets a const iterator for the list of all windows.
         *
         * @return A const iterator for the list of all windows.
         */
        inline vislib::SingleLinkedList<vislib::SmartPtr<
                Window> >::Iterator GetIterator(void) {
            return windows.GetIterator();
        }

        void UpdateAll(CoreHandle& hCore);

    private:

        /** Private ctor. */
        WindowManager(void);

        /** The viewing windows. */
        vislib::SingleLinkedList<vislib::SmartPtr<Window> > windows;

    };

} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_WINDOWMANAGER_H_INCLUDED */
