/*
 * GUIWrapper.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIMANAGER_WRAPPER_H_INCLUDED
#define MEGAMOL_GUI_GUIMANAGER_WRAPPER_H_INCLUDED
#pragma once


#include "GUIManager.h"


namespace megamol {
namespace gui {


    class GUIWrapper {
    public:
        typedef std::shared_ptr<megamol::gui::GUIManager> GUIManagerPtr_t;

        GUIWrapper();
        ~GUIWrapper();

        const GUIManagerPtr_t& Get() const {
            return this->m_gui;
        }

    private:
        GUIManagerPtr_t m_gui;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIMANAGER_WRAPPER_H_INCLUDED
