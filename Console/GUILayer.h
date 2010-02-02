/*
 * GUILayer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#ifdef WITH_TWEAKBAR
#ifndef _MEGAMOL_CONSOLE_GUILAYER_H_INCLUDED
#define _MEGAMOL_CONSOLE_GUILAYER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "MegaMolViewer.h"


namespace megamol {
namespace console {

    /**
     * Manager class for GUI layer
     */
    class GUILayer {
    public:

        /**
         * User class implementing a GUI layer factory with reference counting
         */
        class GUIClient {
        public:

            /** Ctor */
            GUIClient(void);

            /** Dtor */
            ~GUIClient(void);

            /**
             * Grants access to the gui layer
             *
             * @return The gui layer object
             */
            GUILayer& Layer(void);

            /**
             * Tells the gui client the window size
             *
             * @param w The width of the window
             * @param h The height of the window
             */
            void SetWindowSize(unsigned int w, unsigned int h);

        private:

            /** the only instance of the gui layer */
            static GUILayer* layer;

            /** The reference counter */
            static SIZE_T cntr;

            /** The last width told to the gui */
            static int lastWidth;

            /** The last height told to the gui */
            static int lastHeight;

            /** The width of the window */
            int width;

            /** The height of the window */
            int height;

        };

        /** Friend factory may access the object */
        friend class GUIClient;

        /** Draws the GUI */
        void Draw(void);

        /**
         * Informs the GUI that the mouse moved
         *
         * @param x The new mouse position
         * @param y The new mouse position
         *
         * @return True if the event was consumed by the gui
         */
        bool MouseMove(int x, int y);

        /**
         * Informs the GUI that a mouse button state changed
         *
         * @param btn The mouse button
         * @param down The new state flag
         *
         * @return True if the event was consumed by the gui
         */
        bool MouseButton(int btn, bool down);

    private:

        /** Ctor */
        GUILayer(void);

        /** Dtor */
        ~GUILayer(void);

    };

} /* end namespace console */
} /* end namespace megamol */

#endif /* _MEGAMOL_CONSOLE_GUILAYER_H_INCLUDED */
#endif /* WITH_TWEAKBAR */
