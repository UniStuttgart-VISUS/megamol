/*
 * Instance.h
 *
 * Copyright (C) 2006 - 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_WGL_INSTANCE_H_INCLUDED
#define MEGAMOL_WGL_INSTANCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ApiHandle.h"
#include "vislib/sys/Event.h"


namespace megamol {
namespace wgl {

    /** 
     * Library instance
     */
    class Instance : public ApiHandle {
    public:

        /** The window class name */
        static const TCHAR* WindowClassName;

        /**
         * Gets the application instance handle
         *
         * @return The application instance handle
         */
        static HINSTANCE HInstance(void) {
            return hInst;
        }

        /** Ctor */
        Instance(void);

        /** Dtor */
        virtual ~Instance(void);

		/**
		 * Retrieves a pointer to an event that will be set once this
		 * instance's windows are to start rendering.
		 *
		 * @returns A pointer to the event
		 */
		vislib::sys::Event *GetRenderStartEvent(void);

        /**
         * Initializes the instance
         *
         * @param hInst The application instance handle
         *
         * @return True on success
         */
        bool Init(HINSTANCE hInst);

        /**
         * Processes Window event
         *
         * @return True if the application should continue
         */
        bool ProcessEvents(void);

		/**
		 * Starts the rendering loop if it is not already running (by setting
		 * the event that can be retrieved via GetRenderStartEvent).
		 */
		void StartRender(void);

    private:

        /** The application instance handle */
        static HINSTANCE hInst;

        /** Reference counter how many instances use the window class */
        static unsigned int wndClsRegistered;

        /**
         * Deactivates Desktop Window Composition
         */
        void deactivateCompositeDesktop(void);

        /** Flag whether or not the application is running */
        bool running;

		/**
		 * An event to be set when the instance's windows are to start
		 * rendering.
		 */
		vislib::sys::Event renderStartEvent;
    };


} /* end namespace wgl */
} /* end namespace megamol */

#endif /* MEGAMOL_WGL_INSTANCE_H_INCLUDED */
