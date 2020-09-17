/*
 * Group.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GROUP_PRESENTATION_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GROUP_PRESENTATION_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/RenamePopUp.h"

#include "Call.h"
#include "InterfaceSlot.h"
#include "Module.h"


namespace megamol {
namespace gui {


// Forward declarations
class Group;


/** ************************************************************************
 * Defines GUI group presentation.
 */
class GroupPresentation {
public:
    friend class Group;

    // VARIABLES --------------------------------------------------------------

    // Relative position without considering canvas offset and zooming
    ImVec2 position;
    // Relative size without considering zooming
    ImVec2 size;

    // FUNCTIONS --------------------------------------------------------------

    GroupPresentation(void);
    ~GroupPresentation(void);

    inline ImVec2 GetSize(void) { return this->size; }
    inline bool IsViewCollapsed(void) { return this->collapsed_view; }
    inline bool ModulesVisible(void) { return !this->collapsed_view; }
    inline void ForceUpdate(void) { this->update = true; }

private:
    // VARIABLES --------------------------------------------------------------

    bool collapsed_view;
    bool allow_selection;
    bool allow_context;
    bool selected;
    bool update;

    // Widgets
    RenamePopUp rename_popup;

    // FUNCTIONS --------------------------------------------------------------

    void Present(megamol::gui::PresentPhase phase, Group& inout_group, GraphItemsState_t& state);
    void UpdatePositionSize(Group& inout_group, const GraphCanvas_t& in_canvas);
    void SetPosition(Group& inout_group, const GraphCanvas_t& in_canvas, ImVec2 pos);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GROUP_PRESENTATION_H_INCLUDED
