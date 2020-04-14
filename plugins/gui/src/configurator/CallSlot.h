/*
 * CallSlot.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED


#include "GUIUtils.h"
#include "InterfaceSlot.h"


namespace megamol {
namespace gui {
namespace configurator {

// Forward declaration
class CallSlot;

///
class Call;
class Module;
class Parameter;
class InterfaceSlot;

// Pointer types to classes
#ifndef _CALL_SLOT_TYPE_
enum CallSlotType { CALLEE, CALLER };
#    define _CALL_SLOT_TYPE_
#endif
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::vector<CallSlotPtrType> CallSlotPtrVectorType;
typedef std::map<CallSlotType, CallSlotPtrVectorType> CallSlotPtrMapType;

///
typedef std::shared_ptr<Parameter> ParamPtrType;
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<Module> ModulePtrType;
typedef std::shared_ptr<InterfaceSlot> InterfaceSlotPtrType;


/**
 * Defines call slot data structure for graph.
 */
class CallSlot {
public:
    enum Presentations : size_t { DEFAULT = 0, _COUNT_ = 1 };

    struct StockCallSlot {
        std::string name;
        std::string description;
        std::vector<size_t> compatible_call_idxs;
        CallSlotType type;
    };

    CallSlot(ImGuiID uid);
    ~CallSlot();

    const ImGuiID uid;

    // Init when adding call slot from stock
    std::string name;
    std::string description;
    std::vector<size_t> compatible_call_idxs; /// (Storing only indices of compatible calls for faster comparison.)
    CallSlotType type;

    bool CallsConnected(void) const;
    bool ConnectCall(CallPtrType call);
    bool DisConnectCall(ImGuiID call_uid, bool called_by_call);
    bool DisConnectCalls(void);
    const std::vector<CallPtrType>& GetConnectedCalls(void);

    bool ParentModuleConnected(void) const;
    bool ConnectParentModule(ModulePtrType parent_module);
    bool DisConnectParentModule(void);
    const ModulePtrType GetParentModule(void);

    static ImGuiID CheckCompatibleAvailableCallIndex(const CallSlotPtrType call_slot_ptr, CallSlot& call_slot);

    static ImGuiID GetCompatibleCallIndex(const CallSlotPtrType call_slot_1, const CallSlotPtrType call_slot_2);
    static ImGuiID GetCompatibleCallIndex(
        const CallSlotPtrType call_slot, const CallSlot::StockCallSlot& stock_call_slot);

    // GUI Presentation -------------------------------------------------------

    inline void GUI_Present(GraphItemsStateType& state) { this->present.Present(*this, state); }

    inline void GUI_Update(const GraphCanvasType& in_canvas) { this->present.UpdatePosition(*this, in_canvas); }

    inline bool GUI_IsVisible(void) { return this->present.IsVisible(); }
    inline bool GUI_IsGroupInterface(void) { return (this->present.group.interface_ptr != nullptr); }
    inline ImVec2 GUI_GetPosition(void) { return this->present.GetPosition(); }
    inline bool GUI_IsLabelVisible(void) { return this->present.label_visible; }
    inline const InterfaceSlotPtrType& GUI_GetGroupInterface(void) { return this->present.group.interface_ptr; }

    inline void GUI_SetVisibility(bool is_visible) { this->present.SetVisibility(is_visible); }
    inline void GUI_SetGroupInterface(const InterfaceSlotPtrType& interface_ptr) {
        this->present.group.interface_ptr = interface_ptr;
    }
    inline void GUI_SetPresentation(CallSlot::Presentations present) { this->present.presentations = present; }
    inline void GUI_SetLabelVisibility(bool visible) { this->present.label_visible = visible; }

private:
    ModulePtrType parent_module;
    std::vector<CallPtrType> connected_calls;

    /** ************************************************************************
     * Defines GUI call slot presentation.
     */
    class Presentation {
    public:
        struct GroupState {
            InterfaceSlotPtrType interface_ptr;
        };

        Presentation(void);

        ~Presentation(void);

        void Present(CallSlot& inout_call_slot, GraphItemsStateType& state);

        void UpdatePosition(CallSlot& inout_call_slot, const GraphCanvasType& in_canvas);

        ImVec2 GetPosition(void) { return this->position; }
        bool IsVisible(void) { return this->visible; }

        void SetVisibility(bool is_visible) { this->visible = is_visible; }

        GroupState group;
        CallSlot::Presentations presentations;
        bool label_visible;

    private:
        // Absolute position including canvas offset and zooming
        ImVec2 position;

        GUIUtils utils;
        bool visible;
        bool selected;
        bool update_once;
        bool show_modulestock;

    } present;
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED
