/*
 * InterfaceSlot.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_INTERFACESLOT_PRESENTATION_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_INTERFACESLOT_PRESENTATION_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/HoverToolTip.h"


namespace megamol {
namespace gui {


// Forward declarations
class InterfaceSlot;


/** ************************************************************************
 * Defines GUI call slot presentation.
 */
class InterfaceSlotPresentation {
public:
    friend class InterfaceSlot;

    struct GroupState {
        ImGuiID uid;
        bool collapsed_view;
    };

    // VARIABLES --------------------------------------------------------------

    GroupState group;
    bool label_visible;


    // Widgets
    HoverToolTip tooltip;

    // FUNCTIONS --------------------------------------------------------------

    InterfaceSlotPresentation(void);
    ~InterfaceSlotPresentation(void);

    std::string GetLabel(void) { return this->label; }
    ImVec2 GetPosition(InterfaceSlot& inout_interfaceslot);
    inline bool IsGroupViewCollapsed(void) { return this->group.collapsed_view; }

    void SetPosition(ImVec2 pos) { this->position = pos; }

private:
    // VARIABLES --------------------------------------------------------------

    bool selected;
    std::string label;
    ImGuiID last_compat_callslot_uid;
    ImGuiID last_compat_interface_uid;
    bool compatible;
    // Absolute position including canvas offset and zooming
    ImVec2 position;

    // FUNCTIONS --------------------------------------------------------------

    void Present(megamol::gui::PresentPhase phase, InterfaceSlot& inout_interfaceslot, GraphItemsState_t& state);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_INTERFACESLOT_PRESENTATION_H_INCLUDED
