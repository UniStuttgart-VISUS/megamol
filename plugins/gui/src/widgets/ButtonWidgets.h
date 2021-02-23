/*
 * ButtonWidgets.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_BUTTONWIDGETS_INCLUDED
#define MEGAMOL_GUI_BUTTONWIDGETS_INCLUDED


#include "GUIUtils.h"


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
        static bool ExtendedModeButton(const std::string& id, bool& inout_extended_mode);

        /** Lua parameter command copy button. */
        static bool LuaButton(const std::string& id, const megamol::gui::Parameter& param,
            const std::string& param_fullname, const std::string& module_fullname);

    private:
        ButtonWidgets(void) = default;

        ~ButtonWidgets(void) = default;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_BUTTONWIDGETS_INCLUDED
