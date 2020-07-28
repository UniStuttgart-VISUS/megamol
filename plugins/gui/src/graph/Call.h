/*
 * Call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/HoverToolTip.h"


namespace megamol {
namespace gui {


// Forward declarations
class Call;
class CallSlot;
class Module;
class Parameter;
class CallSlot;
#ifndef _CALL_SLOT_TYPE_
enum CallSlotType { CALLEE, CALLER };
#    define _CALL_SLOT_TYPE_
#endif
typedef std::shared_ptr<Parameter> ParamPtrType;
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::shared_ptr<Module> ModulePtrType;

// Types
typedef std::shared_ptr<Call> CallPtrType;
typedef std::vector<CallPtrType> CallPtrVectorType;


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

    void Present(megamol::gui::PresentPhase phase, Call& inout_call, GraphItemsStateType& state);
};


/** ************************************************************************
 * Defines call data structure for graph.
 */
class Call {
public:
    struct StockCall {
        std::string class_name;
        std::string description;
        std::string plugin_name;
        std::vector<std::string> functions;
    };

    // VARIABLES --------------------------------------------------------------

    const ImGuiID uid;
    CallPresentation present;

    // Init when adding call from stock
    std::string class_name;
    std::string description;
    std::string plugin_name;
    std::vector<std::string> functions;

    // FUNCTIONS --------------------------------------------------------------

    Call(ImGuiID uid);
    ~Call();

    bool IsConnected(void);
    bool ConnectCallSlots(CallSlotPtrType callslot_1, CallSlotPtrType callslot_2);
    bool DisconnectCallSlots(ImGuiID calling_callslot_uid = GUI_INVALID_ID);
    const CallSlotPtrType& GetCallSlot(CallSlotType type);

    // Presentation ----------------------------------------------------

    inline void PresentGUI(megamol::gui::PresentPhase phase, GraphItemsStateType& state) {
        this->present.Present(phase, *this, state);
    }

private:
    // VARIABLES --------------------------------------------------------------

    std::map<CallSlotType, CallSlotPtrType> connected_callslots;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
