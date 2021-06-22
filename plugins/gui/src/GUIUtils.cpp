/*
 * GUIUtils.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "GUIUtils.h"


ImGuiID megamol::gui::gui_generated_uid = 0;
unsigned int megamol::gui::gui_context_count = 0;
std::vector<std::string> megamol::gui::gui_resource_paths;
megamol::gui::GUIScaling megamol::gui::gui_scaling;
