/*
 * Window.h
 *
 * Copyright (C) 2006 - 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_WGL_WINDOW_H_INCLUDED
#define MEGAMOL_WGL_WINDOW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ApiHandle.h"
#include "CallbackSlot.h"
#include "Instance.h"
#include "vislib/sys/Thread.h"


namespace megamol {
namespace wgl {

    /** 
     * OpenGL rendering window
     */
    class Window : public ApiHandle {
    public:

        /**
         * Answer the window procedure
         *
         * @return The window procedure
         */
        static inline WNDPROC GetWindowProcedure(void) {
            return &Window::WndProc;
        }

        /** Ctor */
        Window(Instance& inst, int windows);

        /** Dtor */
        virtual ~Window(void);

        /**
		 * Closes this window, which closes all open tiles.
		 */
        void Close(void);

        /**
         * Gets the callback slot requested.
         *
         * @param slot The slot to be returned.
         *
         * @return The requested callback slot or NULL in case of an error.
         */
        inline CallbackSlot* Callback(mmvWindowCallbacks slot) {
            switch (slot) {
                case MMV_WINCB_RENDER: return &this->renderCallback;
                case MMV_WINCB_RESIZE: return &this->resizeCallback;
                //case MMV_WINCB_KEY: return &this->keyCallback;
                //case MMV_WINCB_MOUSEBUTTON: return &this->mouseButtonCallback;
                //case MMV_WINCB_MOUSEMOVE: return &this->mouseMoveCallback;
                //case MMV_WINCB_CLOSE: return &this->closeCallback;
                //case MMV_WINCB_COMMAND: return &this->commandCallback;
                //case MMV_WINCB_APPEXIT: return &this->owner.ApplicationTerminateCallbackSlot();
                //case MMV_WINCB_UPDATEFREEZE: return &this->updateFreeze;
                default: return NULL;
            }
            return NULL;
        }

        /**
         * Gets the window handle
         *
         * @return The window handle
         */
        /*inline HWND Handle(void) {
            return this->hWnd;
        }*/

        /**
         * Answer if this window is valid
         */
		bool IsValid(void);

        /**
         * Informs that the window has been resized
         *
         * @param w The new width
         * @param h The new height
         */
        void Resized(unsigned int w, unsigned int h);

        /**
         * Sets or resets a widnow hint
         *
         * @param hint The hint to set
         * @param f The value of the hint
         */
        void SetHint(unsigned int hint, bool f);

		// void SetPosition(int x, int y, int width, int height);

		void SetTitle(const char *title);

		void SetTitle(const wchar_t *title);

    private:

		struct Tile
		{
			Window *window;
			int index;
			HWND handle;
			HDC dc;
			HGLRC rc;

			/** The tile width */
			unsigned int w;

			/** The tile height */
			unsigned int h;

			Tile::Tile();

			bool Create(Window *window, int index);
			void End();
			void Destroy();
			void Resized(unsigned int w, unsigned int h);
		};

		static const int maxTileCount = 4;
		int activeTiles;

		HDC infoDc;
		HGLRC infoRc;

		static bool CreateRenderContext(HWND window, HDC &dc, HGLRC &rc);

        /**
         * The window event processing method
         *
         * @param hWnd The window handle
         * @param uMsg The window message to be processed
         * @param wParam The word parameter
         * @param lParam The long parameter
         *
         * @return The return value from the message processing
         */
        static LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

        /**
         * The rendering thread implementation
         *
         * @param userData Pointer to this object
         *
         * @return 0
         */
        static DWORD renderThread(void *userData);

        /** The main rendering context */
        // static HGLRC mainCtxt;

        /** The main device context */
        // static HDC mainDC;

        /** The library instance */
        Instance& inst;

		Tile tiles[maxTileCount];

        /** The window handles */
        // HWND hWnd[maxWindowCount];

        /** The device context handle */
        // HDC hDC;

        /** The rendering context handle */
        // HGLRC hRC;


        /** Callback used for rendering */
        CallbackSlot renderCallback;

        /** Callback used to inform about window size changes */
        CallbackSlot resizeCallback;

        /** The rendering thread */
        vislib::sys::Thread renderer;
    };


} /* end namespace wgl */
} /* end namespace megamol */

#endif /* MEGAMOL_WGL_WINDOW_H_INCLUDED */
