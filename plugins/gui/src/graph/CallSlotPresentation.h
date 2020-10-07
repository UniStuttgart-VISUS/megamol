/*
 * CallSlot.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALLSLOT_PRESENTATION_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALLSLOT_PRESENTATION_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/HoverToolTip.h"


namespace megamol {
namespace gui {


// Forward declarations
class CallSlot;


/** ************************************************************************
 * Defines GUI call slot presentation.
 */
class CallSlotPresentation {
public:
    friend class CallSlot;

    struct GroupState {
        InterfaceSlotPtr_t interfaceslot_ptr;
    };

    // VARIABLES --------------------------------------------------------------

    GroupState group;
    bool label_visible;
    bool visible;


    // FUNCTIONS --------------------------------------------------------------

    CallSlotPresentation(void);
    ~CallSlotPresentation(void);

    ImVec2 GetPosition(void) { return this->position; }

private:
    // VARIABLES --------------------------------------------------------------

    // Absolute position including canvas offset and zooming
    ImVec2 position;
    bool selected;
    bool update_once;
    bool show_modulestock;
    ImGuiID last_compat_callslot_uid;
    ImGuiID last_compat_interface_uid;
    bool compatible;

    // Widgets
    HoverToolTip tooltip;

    // FUNCTIONS --------------------------------------------------------------

    void Present(megamol::gui::PresentPhase phase, CallSlot& inout_callslot, GraphItemsState_t& state);
    void Update(CallSlot& inout_callslot, const GraphCanvas_t& in_canvas);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALLSLOT_PRESENTATION_H_INCLUDED
