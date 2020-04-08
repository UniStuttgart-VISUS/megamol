/*
 * CallSlot.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED


#include "GUIUtils.h"


namespace megamol {
namespace gui {
namespace configurator {

// Forward declaration
class Call;
class CallSlot;
class Module;
class Parameter;

// Pointer types to classes
typedef std::shared_ptr<Parameter> ParamPtrType;
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::shared_ptr<Module> ModulePtrType;

enum CallSlotType { CALLEE, CALLER };

typedef std::vector<CallSlotPtrType> CallSlotPtrVectorType;
typedef std::map<CallSlotType, CallSlotPtrVectorType> CallSlotPtrMapType;

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

    struct GroupState {
        bool is_interface;
        ImVec2 position;
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

    inline bool GUI_GetGroupInterface(void) { return this->present.group.is_interface; }
    inline ImVec2 GUI_GetGroupPosition(void) { return this->present.group.position; }
    inline ImVec2 GUI_GetPosition(void) { return this->present.GetPosition(*this); }
    inline bool GUI_GetLabelVisibility(void) { return this->present.label_visible; }

    inline void GUI_SetGroupInterface(bool is_interface) { this->present.group.is_interface = is_interface; }
    inline void GUI_SetGroupPosition(ImVec2 position) { this->present.group.position = position; }
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
        Presentation(void);

        ~Presentation(void);

        void Present(CallSlot& inout_call_slot, GraphItemsStateType& state);

        void UpdatePosition(CallSlot& inout_call_slot, const GraphCanvasType& in_canvas);

        ImVec2 GetPosition(CallSlot& inout_call_slot);

        GroupState group;
        CallSlot::Presentations presentations;
        bool label_visible;

    private:
        // Absolute position including canvas offset and zooming
        ImVec2 position;

        GUIUtils utils;
        bool selected;
        bool update_once;
        bool show_modulestock;

    } present;
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED
