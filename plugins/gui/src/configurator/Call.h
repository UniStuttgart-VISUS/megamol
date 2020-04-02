/*
 * Call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED


#include "CallSlot.h"
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


typedef std::vector<CallPtrType> CallPtrVectorType;


/**
 * Defines call data structure for graph.
 */
class Call {
public:
    enum Presentations : size_t { DEFAULT = 0, _COUNT_ = 1 };

    struct StockCall {
        std::string class_name;
        std::string description;
        std::string plugin_name;
        std::vector<std::string> functions;
    };

    Call(ImGuiID uid);
    ~Call();

    const ImGuiID uid;

    // Init when adding call from stock
    std::string class_name;
    std::string description;
    std::string plugin_name;
    std::vector<std::string> functions;

    bool IsConnected(void);
    bool ConnectCallSlots(CallSlotPtrType call_slot_1, CallSlotPtrType call_slot_2);
    bool DisConnectCallSlots(void);
    const CallSlotPtrType& GetCallSlot(CallSlotType type);

    // GUI Presentation -------------------------------------------------------

    inline void GUI_Present(GraphItemsStateType& state) { this->present.Present(*this, state); }

    inline void GUI_SetLabelVisibility(bool visible) { this->present.label_visible = visible; }
    inline void GUI_SetPresentation(Call::Presentations present) { this->present.presentations = present; }

private:
    std::map<CallSlotType, CallSlotPtrType> connected_call_slots;

    /** ************************************************************************
     * Defines GUI call presentation.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(Call& inout_call, GraphItemsStateType& state);

        Call::Presentations presentations;
        bool label_visible;

    private:
        GUIUtils utils;
        bool selected;

    } present;
};


} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
