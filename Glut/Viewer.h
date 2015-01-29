/*
 * Viewer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVIEWER_VIEWER_H_INCLUDED
#define MEGAMOLVIEWER_VIEWER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/ApiHandle.h"
#include "CallbackSlot.h"
#include "vislib/SingleLinkedList.h"


namespace megamol {
namespace viewer {

    /** forward declaration of Window */
    class Window;

    /**
     * class of viewer instances.
     */
    class Viewer : public ApiHandle {
    public:

        /** ctor */
        Viewer(void);

        /** dtor */
        virtual ~Viewer(void);

        /**
         * Initialises the viewer object.
         *
         * @param hints initialization hints
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool Initialise(unsigned int hints);

        /**
         * The main loop trigger
         *
         * @return 'true' if the main loop is valid,
         *         'false' if the main loop terminated.
         */
        bool ProcessEvents(void);

        /**
         * Takes ownership of a window by placing this window in the list of
         * owned windows.
         *
         * @param window The window to be owned.
         */
        inline void OwnWindow(megamol::viewer::Window *window) {
            if (this->windows.Find(window) == NULL) {
                this->windows.Add(window);
            }
        }

        /**
         * Removes a window from the list of owned windows.
         *
         * @param window The window to be removed from the list of owned
         *               windows.
         */
        inline void UnownWindow(megamol::viewer::Window *window) {
            this->windows.RemoveAll(window);
        }

        /**
         * Closes all windows owned by this viewer.
         */
        void CloseAllWindows(void);

        /**
         * Requests termination of the whole application
         */
        void RequestAppTermination(void);

        /**
         * Gets the callback slot for application termination requests.
         *
         * @return The callback slot for application termination requests.
         */
        inline CallbackSlot& ApplicationTerminateCallbackSlot(void) {
            return this->appTerminate;
        }

        /**
         * Gets the first window of the list of windows.
         *
         * @return The first window of the list of windows
         */
        inline megamol::viewer::Window* FirstWindow(void) {
            return (this->windows.IsEmpty()) ? NULL : this->windows.First();
        }

        /**
         * Performs a synchronizes buffer swap for all windows in presentation
         * mode.
         */
        void PresentationModeSwap(void);

    private:

        /** list of owned windows. */
        vislib::SingleLinkedList<megamol::viewer::Window*> windows;

        /** The callback slot for application termination requests */
        CallbackSlot appTerminate;

    };

} /* end namespace viewer */
} /* end namespace megamol */

#endif /* MEGAMOLVIEWER_VIEWER_H_INCLUDED */
