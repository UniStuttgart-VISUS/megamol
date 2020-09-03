/*
 * CallSlot.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED


#include "CallSlotPresentation.h"


namespace megamol {
namespace gui {


// Forward declarations
class CallSlot;
class Call;
class Module;
class Parameter;
class InterfaceSlot;
typedef std::shared_ptr<Parameter> ParamPtr_t;
typedef std::shared_ptr<Call> CallPtr_t;
typedef std::shared_ptr<Module> ModulePtr_t;
typedef std::shared_ptr<InterfaceSlot> InterfaceSlotPtr_t;

// Types
#ifndef _CALL_SLOT_TYPE_
enum CallSlotType { CALLEE, CALLER };
#    define _CALL_SLOT_TYPE_
#endif
typedef std::shared_ptr<CallSlot> CallSlotPtr_t;
typedef std::vector<CallSlotPtr_t> CallSlotPtrVector_t;
typedef std::map<CallSlotType, CallSlotPtrVector_t> CallSlotPtrMap_t;


/** ************************************************************************
 * Defines call slot data structure for graph.
 */
class CallSlot {
public:
    struct StockCallSlot {
        std::string name;
        std::string description;
        std::vector<size_t> compatible_call_idxs;
        CallSlotType type;
    };

    // VARIABLES --------------------------------------------------------------

    const ImGuiID uid;
    CallSlotPresentation present;

    // Init when adding call slot from stock
    std::string name;
    std::string description;
    std::vector<size_t> compatible_call_idxs; // Storing only indices of compatible calls for faster comparison.
    CallSlotType type;

    // FUNCTIONS --------------------------------------------------------------

    CallSlot(ImGuiID uid);
    ~CallSlot();

    bool CallsConnected(void) const;
    bool ConnectCall(const CallPtr_t& call_ptr);
    bool DisconnectCall(ImGuiID call_uid);
    bool DisconnectCalls(void);
    const std::vector<CallPtr_t>& GetConnectedCalls(void);

    bool IsParentModuleConnected(void) const;
    bool ConnectParentModule(ModulePtr_t parent_module);
    bool DisconnectParentModule(void);
    const ModulePtr_t& GetParentModule(void);

    static ImGuiID GetCompatibleCallIndex(const CallSlotPtr_t& callslot_1, const CallSlotPtr_t& callslot_2);
    static ImGuiID GetCompatibleCallIndex(const CallSlotPtr_t& callslot, const CallSlot::StockCallSlot& stock_callslot);

    bool IsConnectionValid(CallSlot& callslot);

    // Presentation ----------------------------------------------------

    inline void PresentGUI(megamol::gui::PresentPhase phase, GraphItemsState_t& state) {
        this->present.Present(phase, *this, state);
    }
    inline void UpdateGUI(const GraphCanvas_t& in_canvas) { this->present.Update(*this, in_canvas); }

private:
    // VARIABLES --------------------------------------------------------------

    ModulePtr_t parent_module;
    std::vector<CallPtr_t> connected_calls;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED
