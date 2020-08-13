/*
 * InterfaceSlot.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_INTERFACESLOT_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_INTERFACESLOT_H_INCLUDED


#include "InterfaceSlotPresentation.h"


namespace megamol {
namespace gui {


// Forward declarations
class InterfaceSlot;
class CallSlot;
#ifndef _CALL_SLOT_TYPE_
enum CallSlotType { CALLEE, CALLER };
#    define _CALL_SLOT_TYPE_
#endif
typedef std::vector<CallSlotPtr_t> CallSlotPtrVector_t;

// Types
typedef std::shared_ptr<InterfaceSlot> InterfaceSlotPtr_t;
typedef std::vector<InterfaceSlotPtr_t> InterfaceSlotPtrVector_t;
typedef std::map<CallSlotType, InterfaceSlotPtrVector_t> InterfaceSlotPtrMap_t;


/** ************************************************************************
 * Defines group interface slots bundling and redirecting calls of compatible call slots.
 */
class InterfaceSlot {
public:
    // VARIABLES --------------------------------------------------------------

    const ImGuiID uid;
    InterfaceSlotPresentation present;

    // FUNCTIONS --------------------------------------------------------------

    InterfaceSlot(ImGuiID uid, bool auto_create);
    ~InterfaceSlot();

    bool AddCallSlot(const CallSlotPtr_t& callslot_ptr, const InterfaceSlotPtr_t& parent_interfaceslot_ptr);
    bool RemoveCallSlot(ImGuiID callslot_uid);
    bool ContainsCallSlot(ImGuiID callslot_uid);
    bool IsConnectionValid(CallSlot& callslot);
    bool IsConnectionValid(InterfaceSlot& interfaceslot);
    bool GetCompatibleCallSlot(CallSlotPtr_t& out_callslot_ptr);
    CallSlotPtrVector_t& GetCallSlots(void) { return this->callslots; }
    bool IsConnected(void);
    CallSlotType GetCallSlotType(void);
    bool IsEmpty(void);
    bool IsAutoCreated(void) { return this->auto_created; }

    // Presentation ----------------------------------------------------

    inline void PresentGUI(megamol::gui::PresentPhase phase, GraphItemsState_t& state) {
        this->present.Present(phase, *this, state);
    }
    inline ImVec2 GetGUIPosition(void) { return this->present.GetPosition(*this); }

private:
    // VARIABLES --------------------------------------------------------------

    bool auto_created;
    CallSlotPtrVector_t callslots;

    // FUNCTIONS --------------------------------------------------------------

    bool is_callslot_compatible(CallSlot& callslot);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_INTERFACESLOT_H_INCLUDED
