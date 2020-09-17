/*
 * Call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED


#include "CallPresentation.h"


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
typedef std::shared_ptr<Parameter> ParamPtr_t;
typedef std::shared_ptr<CallSlot> CallSlotPtr_t;
typedef std::shared_ptr<Module> ModulePtr_t;

// Types
typedef std::shared_ptr<Call> CallPtr_t;
typedef std::vector<CallPtr_t> CallPtrVector_t;


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
    bool ConnectCallSlots(CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2);
    bool DisconnectCallSlots(ImGuiID calling_callslot_uid = GUI_INVALID_ID);
    const CallSlotPtr_t& GetCallSlot(CallSlotType type);

    // Presentation ----------------------------------------------------

    inline void PresentGUI(megamol::gui::PresentPhase phase, GraphItemsState_t& state) {
        this->present.Present(phase, *this, state);
    }

private:
    // VARIABLES --------------------------------------------------------------

    std::map<CallSlotType, CallSlotPtr_t> connected_callslots;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
