/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once


#include <string>


namespace megamol::gui {


/** Forward declaration */
class Parameter;


/** ************************************************************************
 * Different button widgets
 */
class ButtonWidgets {
public:
    enum ButtonStyle {
        POINT_CIRCLE,
        GRID,
        LINES, // aka hamburger menu
        POINTS
    };

    /** "Point in Circle" Button for additional drop down Options. */
    static bool OptionButton(
        ButtonStyle button_style, const std::string& id, const std::string& label, bool dirty, bool read_only);

    /** Knob button for 'circular' float value manipulation. */
    static bool KnobButton(
        const std::string& id, float size, float& inout_value, float minval, float maxval, float step);

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
    ButtonWidgets() = default;
    ~ButtonWidgets() = default;
};


} // namespace megamol::gui
