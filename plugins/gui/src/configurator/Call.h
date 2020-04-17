/*
 * Call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED


#include "GUIUtils.h"


namespace megamol {
namespace gui {
namespace configurator {

// Forward declaration
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


// Pointer types to classes
typedef std::shared_ptr<Call> CallPtrType;
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
    bool ConnectCallSlots(CallSlotPtrType callslot_1, CallSlotPtrType callslot_2);
    bool DisconnectCallSlots(void);
    const CallSlotPtrType& GetCallSlot(CallSlotType type);

    // GUI Presentation -------------------------------------------------------

    inline void GUI_Present(megamol::gui::PresentPhase phase, GraphItemsStateType& state) {
        this->present.Present(phase, *this, state);
    }

    inline void GUI_SetLabelVisibility(bool visible) { this->present.label_visible = visible; }
    inline void GUI_SetPresentation(Call::Presentations present) { this->present.presentations = present; }

private:
    std::map<CallSlotType, CallSlotPtrType> connected_callslots;

    /** ************************************************************************
     * Defines GUI call presentation.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(megamol::gui::PresentPhase phase, Call& inout_call, GraphItemsStateType& state);

        Call::Presentations presentations;
        bool label_visible;

    private:
        bool selected;
        GUIUtils utils;

    } present;
};


} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
