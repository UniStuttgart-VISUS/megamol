/*
 * ButtonWidgets.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_BUTTONWIDGETS_INCLUDED
#define MEGAMOL_GUI_BUTTONWIDGETS_INCLUDED
#pragma once


#include <string>


namespace megamol {
namespace gui {

    class Parameter;

    /**
     * Different button widgets.
     */
    class ButtonWidgets {
    public:
        /** "Point in Circle" Button for additional drop down Options. */
        static bool OptionButton(const std::string& id, const std::string& label = "", bool dirty = false);

        /** Knob button for 'circular' float value manipulation. */
        static bool KnobButton(const std::string& id, float size, float& inout_value, float minval, float maxval);

        /** Extended mode button. */
        // OptionButton with menu for 'Basic' and 'Expert' option
        static bool ExtendedModeButton(const std::string& id, bool& inout_extended_mode);

        /** Lua parameter command copy button. */
        static bool LuaButton(
            const std::string& id, const megamol::gui::Parameter& param, const std::string& param_fullname);

        /** Toggle Button */
        // https://github.com/ocornut/imgui/issues/1537#issuecomment-355569554
        static bool ToggleButton(const std::string& id, bool& inout_bool);

    private:
        ButtonWidgets(void) = default;

        ~ButtonWidgets(void) = default;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_BUTTONWIDGETS_INCLUDED
