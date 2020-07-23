/*
 * CallSlot.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/HoverToolTip.h"


namespace megamol {
namespace gui {


// Forward declarations
class CallSlot;
class Call;
class Module;
class Parameter;
class InterfaceSlot;
typedef std::shared_ptr<Parameter> ParamPtrType;
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<Module> ModulePtrType;
typedef std::shared_ptr<InterfaceSlot> InterfaceSlotPtrType;

// Types
#ifndef _CALL_SLOT_TYPE_
enum CallSlotType { CALLEE, CALLER };
#    define _CALL_SLOT_TYPE_
#endif
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::vector<CallSlotPtrType> CallSlotPtrVectorType;
typedef std::map<CallSlotType, CallSlotPtrVectorType> CallSlotPtrMapType;


/** ************************************************************************
 * Defines GUI call slot presentation.
 */
class CallSlotPresentation {
public:
    friend class CallSlot;

    struct GroupState {
        InterfaceSlotPtrType interfaceslot_ptr;
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

    void Present(megamol::gui::PresentPhase phase, CallSlot& inout_callslot, GraphItemsStateType& state);
    void Update(CallSlot& inout_callslot, const GraphCanvasType& in_canvas);
};


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
    bool ConnectCall(const CallPtrType& call_ptr);
    bool DisconnectCall(ImGuiID call_uid);
    bool DisconnectCalls(void);
    const std::vector<CallPtrType>& GetConnectedCalls(void);

    bool IsParentModuleConnected(void) const;
    bool ConnectParentModule(ModulePtrType parent_module);
    bool DisconnectParentModule(void);
    const ModulePtrType& GetParentModule(void);

    static ImGuiID GetCompatibleCallIndex(const CallSlotPtrType& callslot_1, const CallSlotPtrType& callslot_2);
    static ImGuiID GetCompatibleCallIndex(
        const CallSlotPtrType& callslot, const CallSlot::StockCallSlot& stock_callslot);

    bool IsConnectionValid(CallSlot& callslot);

    // Presentation ----------------------------------------------------

    inline void PresentGUI(megamol::gui::PresentPhase phase, GraphItemsStateType& state) {
        this->present.Present(phase, *this, state);
    }
    inline void UpdateGUI(const GraphCanvasType& in_canvas) { this->present.Update(*this, in_canvas); }

private:
    // VARIABLES --------------------------------------------------------------

    ModulePtrType parent_module;
    std::vector<CallPtrType> connected_calls;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED
