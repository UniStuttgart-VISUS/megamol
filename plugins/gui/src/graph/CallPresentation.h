/*
 * Call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALL_PRESENTATION_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALL_PRESENTATION_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/HoverToolTip.h"


namespace megamol {
namespace gui {


// Forward declarations
class Call;


/** ************************************************************************
 * Defines GUI call presentation.
 */
class CallPresentation {
public:
    friend class Call;

    // VARIABLES --------------------------------------------------------------

    bool label_visible;

    // FUNCTIONS --------------------------------------------------------------

    CallPresentation(void);
    ~CallPresentation(void);

private:
    // VARIABLES --------------------------------------------------------------

    bool selected;

    // Widgets
    HoverToolTip tooltip;

    // FUNCTIONS --------------------------------------------------------------

    void Present(megamol::gui::PresentPhase phase, Call& inout_call, GraphItemsState_t& state);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALL_PRESENTATION_H_INCLUDED
