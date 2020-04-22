/*
 * InterfaceSlot.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_INTERFACESLOT_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_INTERFACESLOT_H_INCLUDED


#include "GUIUtils.h"


namespace megamol {
namespace gui {
namespace configurator {


// Forward declaration
class InterfaceSlot;
class CallSlot;
#ifndef _CALL_SLOT_TYPE_
enum CallSlotType { CALLEE, CALLER };
#    define _CALL_SLOT_TYPE_
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
    InterfaceSlot(ImGuiID uid);
    ~InterfaceSlot();

    const ImGuiID uid;

    bool AddCallSlot(const CallSlotPtrType& callslot_ptr, const InterfaceSlotPtrType& parent_interfaceslot_ptr);
    bool RemoveCallSlot(ImGuiID callslot_uid);
    bool ContainsCallSlot(ImGuiID callslot_uid);
    bool IsCallSlotCompatible(const CallSlot& callslot);
    bool GetCompatibleCallSlot(CallSlotPtrType& out_callslot_ptr);
    CallSlotPtrVectorType& GetCallSlots(void) { return this->callslots; }
    bool IsConnected(void);
    CallSlotType GetCallSlotType(void);
    bool IsEmpty(void);

    // GUI Presentation -------------------------------------------------------

    inline void GUI_Present(megamol::gui::PresentPhase phase, GraphItemsStateType& state) {
        this->present.Present(phase, *this, state);
    }

    inline ImVec2 GUI_GetPosition(void) { return this->present.GetPosition(*this); }
    inline ImGuiID GUI_GetGroupUID(void) { return this->present.group.uid; }

    inline void GUI_SetPosition(ImVec2 pos) { this->present.SetPosition(pos); }
    inline void GUI_SetGroupView(bool collapsed_view) { this->present.group.collapsed_view = collapsed_view; }
    inline void GUI_SetGroupUID(ImGuiID uid) { this->present.group.uid = uid; }

private:
    CallSlotPtrVectorType callslots;

    /** ************************************************************************
     * Defines GUI call slot presentation.
     */
    class Presentation {
    public:
        struct GroupState {
            ImGuiID uid;
            bool collapsed_view;
        };

        Presentation(void);

        ~Presentation(void);

        void Present(megamol::gui::PresentPhase phase, InterfaceSlot& inout_interfaceslot, GraphItemsStateType& state);

        ImVec2 GetPosition(InterfaceSlot& inout_interfaceslot);

        void SetPosition(ImVec2 pos) { this->position = pos; }

        GroupState group;

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
