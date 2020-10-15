/*
 * GUIWrapper.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIWINDOWS_WRAPPER_H_INCLUDED
#define MEGAMOL_GUI_GUIWINDOWS_WRAPPER_H_INCLUDED

#ifdef _WIN32
#ifdef GUI_EXPORTS
#define GUI_API __declspec(dllexport)
#else
#define GUI_API __declspec(dllimport)
#endif
#else // _WIN32
#define GUI_API
#endif // _WIN32

#include "GUIWindows.h"

namespace megamol {
namespace gui {


    class GUI_API GUIWrapper {
    public:
        typedef std::shared_ptr<megamol::gui::GUIWindows> GUIWindowsPtr_t;
        GUIWrapper(void);
        ~GUIWrapper(void);
        const GUIWindowsPtr_t& Get(void) {
            return this->m_gui;
        }

    private:
        GUIWindowsPtr_t m_gui;
    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIWINDOWS_WRAPPER_H_INCLUDED
