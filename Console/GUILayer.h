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

        private:

            /** the only instance of the gui layer */
            static GUILayer* layer;

            /** The reference counter */
            static SIZE_T cntr;

        };

        /** Friend factory may access the object */
        friend class GUIClient;

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
