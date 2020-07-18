/*
 * WidgetPicking_gl.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "WidgetPicking_gl.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::WidgetPicking::WidgetPicking(void) {}


void megamol::gui::WidgetPicking::ProcessMouseMove(double x, double y) {}


void megamol::gui::WidgetPicking::ProcessMouseClick(megamol::core::view::MouseButton button,
    megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) {}


bool megamol::gui::WidgetPicking::Enable(void) { return false; }


bool megamol::gui::WidgetPicking::Disable(void) { return false; }
