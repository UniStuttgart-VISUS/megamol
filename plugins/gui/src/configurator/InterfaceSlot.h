/*
 * InterfaceSlot.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_INTERFACESLOT_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_INTERFACESLOT_H_INCLUDED


#include "CallSlot.h"


namespace megamol {
namespace gui {
namespace configurator {


// Forward declaration
class InterfaceSlot;

// Pointer types to classes
typedef std::shared_ptr<InterfaceSlot> InterfaceSlotPtrType;
typedef std::vector<InterfaceSlotPtrType> InterfaceSlotPtrVectorType;
typedef std::map<CallSlotType, InterfaceSlotPtrVectorType> InterfaceSlotPtrMapType;


/**
 * Defines group interface slots bundling and redirecting calls of compatible call slots.
 */
class InterfaceSlot {
public:

    InterfaceSlot();
    ~InterfaceSlot();

    bool AddCallSlot(CallSlotPtrType callslot_ptr);
    
    bool RemoveCallSlot(ImGuiID callslot_uid);
    
    bool ContainsCallSlot(ImGuiID callslot_uid);
    
    bool IsCallSlotCompatible(CallSlotPtrType callslot_ptr);
    
    bool IsEmpty(void) { return (callslots.size() == 0); }
    
    const CallSlotPtrVectorType& GetCallSlots(void) { return this->callslots; }

    // GUI Presentation -------------------------------------------------------

    inline void GUI_Present(GraphItemsStateType& state) { this->present.Present(*this, state); }

    inline void GUI_Update(const GraphCanvasType& in_canvas) { this->present.UpdatePosition(*this, in_canvas); }

private:

    CallSlotPtrVectorType callslots;

    /** ************************************************************************
     * Defines GUI call slot presentation.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(InterfaceSlot& inout_interfaceslot, GraphItemsStateType& state);

        void UpdatePosition(InterfaceSlot& inout_interfaceslot, const GraphCanvasType& in_canvas);

        ImVec2 GetPosition(InterfaceSlot& inout_interfaceslot);

    private:
        // Absolute position including canvas offset and zooming
        ImVec2 position;

        GUIUtils utils;
        bool selected;

    } present;
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_INTERFACESLOT_H_INCLUDED
