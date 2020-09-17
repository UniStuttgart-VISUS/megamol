/*
 * GUI_Wrapper.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "stdafx.h"
#include "GUI_Wrapper.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::GUI_Wrapper::GUI_Wrapper(void) : m_gui(nullptr) {
    this->m_gui = std::make_shared<megamol::gui::GUIWindows>();
}

megamol::gui::GUI_Wrapper::~GUI_Wrapper(void) {}
