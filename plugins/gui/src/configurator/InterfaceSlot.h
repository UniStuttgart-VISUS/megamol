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

///
class CallSlot;
#ifndef _CALL_SLOT_TYPE_
    enum CallSlotType { CALLEE, CALLER };
    #define _CALL_SLOT_TYPE_
#endif
typedef std::vector<CallSlotPtrType> CallSlotPtrVectorType;

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

    bool AddCallSlot(const CallSlotPtrType& callslot_ptr, const InterfaceSlotPtrType& parent_interface_ptr);
    
    bool RemoveCallSlot(ImGuiID callslot_uid);
    
    bool ContainsCallSlot(ImGuiID callslot_uid);
    
    bool IsCallSlotCompatible(const CallSlotPtrType& callslot_ptr);
    
    bool IsEmpty(void);
    
    CallSlotPtrVectorType& GetCallSlots(void) { return this->callslots; }

    // GUI Presentation -------------------------------------------------------

    inline void GUI_Present(GraphItemsStateType& state, bool collapsed_view) { this->present.Present(*this, state, collapsed_view); }

    inline ImVec2 GUI_GetPosition(void) { return this->present.GetPosition(); }

    inline void GUI_SetPosition(ImVec2 pos) { this->present.SetPosition(pos); }

private:

    CallSlotPtrVectorType callslots;

    /** ************************************************************************
     * Defines GUI call slot presentation.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(InterfaceSlot& inout_interfaceslot, GraphItemsStateType& state, bool collapsed_view);

        inline ImVec2 GetPosition(void) { return this->position; }

        void SetPosition(ImVec2 pos) { this->position = pos; }

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
