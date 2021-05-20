/*
 * InterfaceSlot.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_INTERFACESLOT_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_INTERFACESLOT_H_INCLUDED
#pragma once


#include "widgets/HoverToolTip.h"


namespace megamol {
namespace gui {


    // Forward declarations
    class InterfaceSlot;
    class CallSlot;
#ifndef _CALL_SLOT_TYPE_
    enum CallSlotType { CALLEE, CALLER };
#define _CALL_SLOT_TYPE_
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
        InterfaceSlot(ImGuiID uid, bool auto_create);
        ~InterfaceSlot();

        bool AddCallSlot(const CallSlotPtr_t& callslot_ptr, const InterfaceSlotPtr_t& parent_interfaceslot_ptr);
        bool RemoveCallSlot(ImGuiID callslot_uid);
        bool ContainsCallSlot(ImGuiID callslot_uid);
        bool IsConnectionValid(CallSlot& callslot);
        bool IsConnectionValid(InterfaceSlot& interfaceslot);
        CallSlotPtr_t GetCompatibleCallSlot(void);
        CallSlotPtrVector_t& CallSlots(void) {
            return this->callslots;
        }
        bool IsConnected(void);
        CallSlotType GetCallSlotType(void);
        bool IsEmpty(void);
        bool IsAutoCreated(void) {
            return this->auto_created;
        }

        void Draw(megamol::gui::PresentPhase phase, GraphItemsState_t& state);

        inline std::string Label(void) const {
            return this->gui_label;
        }
        inline const ImGuiID UID(void) const {
            return this->uid;
        }
        inline const ImGuiID GroupUID(void) const {
            return this->group_uid;
        }
        ImVec2 Position(void) {
            return this->Position(this->gui_group_collapsed_view);
        }
        ImVec2 Position(bool group_collapsed_view);

        inline bool IsGroupViewCollapsed(void) const {
            return this->gui_group_collapsed_view;
        }

        void SetPosition(ImVec2 pos) {
            this->gui_position = pos;
        }
        inline void SetGroupViewCollapsed(bool collapsed) {
            this->gui_group_collapsed_view = collapsed;
        }
        inline void SetGroupUID(ImGuiID uid) {
            this->group_uid = uid;
        }

    private:
        // VARIABLES --------------------------------------------------------------

        const ImGuiID uid;

        bool auto_created;
        CallSlotPtrVector_t callslots;
        ImGuiID group_uid;

        bool gui_selected;
        ImVec2 gui_position; /// Absolute position including canvas offset and zooming
        std::string gui_label;
        ImGuiID gui_last_compat_callslot_uid;
        ImGuiID gui_last_compat_interface_uid;
        bool gui_compatible;
        bool gui_group_collapsed_view;

        HoverToolTip gui_tooltip;

        // FUNCTIONS --------------------------------------------------------------

        bool is_callslot_compatible(CallSlot& callslot);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_INTERFACESLOT_H_INCLUDED
