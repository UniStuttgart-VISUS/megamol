/*
 * GUIWrapper.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "gui-wrapper.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::GUIWrapper::GUIWrapper(void) : m_gui(nullptr) {
    this->m_gui = std::make_shared<megamol::gui::GUIWindows>();
}

megamol::gui::GUIWrapper::~GUIWrapper(void) {}
