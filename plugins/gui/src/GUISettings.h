/*
 * GUISettings.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#ifndef MEGAMOL_GUI_GUISETTINGS_H_INCLUDED
#define MEGAMOL_GUI_GUISETTINGS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <string>

#include "json.hpp"

#include "GUIUtility.h"


namespace megamol {
    namespace gui {

        /**
         * Managing window settings for GUI
         */
        class GUISettings : public GUIUtility {

        public:

            /**
             * Ctor
             */
            GUISettings();

            /**
             * Dtor
             */
            ~GUISettings(void);

            /** 
             * Set file name for writing the settings.
             */
            inline void SetFilename(std::string file) {
                this->file = file;
            }



        private:

            /** The file the settings */
            std::string file;

        };
 

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUISETTINGS_H_INCLUDED
