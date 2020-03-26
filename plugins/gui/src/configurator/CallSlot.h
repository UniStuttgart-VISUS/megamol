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

/**
 * Defines call slot data structure for graph.
 */
class CallSlot {
public:
    enum CallSlotType { CALLEE, CALLER };

    struct StockCallSlot {
        std::string name;
        std::string description;
        std::vector<size_t> compatible_call_idxs;
        CallSlot::CallSlotType type;
    };

    enum Presentations : size_t { DEFAULT = 0, _COUNT_ = 1 };

    CallSlot(ImGuiID uid);
    ~CallSlot();

    const ImGuiID uid;

    // Init when adding call slot from stock
    std::string name;
    std::string description;
    std::vector<size_t> compatible_call_idxs; // (Storing only indices of compatible calls for faster comparison.)
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

    void GUI_Present(StateType& state) {
        this->present.Present(*this, state);
    }
    void GUI_Update(const CanvasType& in_canvas) { this->present.UpdatePosition(*this, in_canvas); }

    ImVec2 GUI_GetPosition(void) { return this->present.GetPosition(); }
    bool GUI_GetLabelVisibility(void) { return this->present.label_visible; }

    void GUI_SetPresentation(CallSlot::Presentations present) { this->present.presentations = present; }
    void GUI_SetLabelVisibility(bool visible) { this->present.label_visible = visible; }

private:
    ModulePtrType parent_module;
    std::vector<CallPtrType> connected_calls;

    /**
     * Defines GUI call slot presentation.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(CallSlot& inout_call_slot, StateType& state);

        void UpdatePosition(CallSlot& call_slot, const CanvasType& in_canvas);

        ImVec2 GetPosition(void) { return this->position; }

        CallSlot::Presentations presentations;
        bool label_visible;

    private:
        // Absolute position including canvas offset and zooming
        ImVec2 position;
        GUIUtils utils;
        bool selected;
        bool update_once;

    } present;
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALLSLOT_H_INCLUDED