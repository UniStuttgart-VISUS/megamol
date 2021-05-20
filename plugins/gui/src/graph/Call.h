/*
 * Call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
#pragma once


#include "widgets/HoverToolTip.h"


namespace megamol {
namespace gui {


    // Forward declarations
    class Call;
    class CallSlot;
    class Module;
    class Parameter;
    class CallSlot;
#ifndef _CALL_SLOT_TYPE_
    enum CallSlotType { CALLEE, CALLER };
#define _CALL_SLOT_TYPE_
#endif
    typedef std::shared_ptr<Parameter> ParamPtr_t;
    typedef std::shared_ptr<CallSlot> CallSlotPtr_t;
    typedef std::shared_ptr<Module> ModulePtr_t;

    // Types
    typedef std::shared_ptr<Call> CallPtr_t;
    typedef std::vector<CallPtr_t> CallPtrVector_t;


    /** ************************************************************************
     * Defines call data structure for graph.
     */
    class Call {
    public:
        struct StockCall {
            std::string class_name;
            std::string description;
            std::string plugin_name;
            std::vector<std::string> functions;
        };

        Call(ImGuiID uid, const std::string& class_name, const std::string& description, const std::string& plugin_name,
            const std::vector<std::string>& functions);
        ~Call();

        bool IsConnected(void);
        bool ConnectCallSlots(CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2);
        bool DisconnectCallSlots(ImGuiID calling_callslot_uid = GUI_INVALID_ID);
        const CallSlotPtr_t& CallSlotPtr(CallSlotType type);

        void Draw(megamol::gui::PresentPhase phase, GraphItemsState_t& state);

        inline const ImGuiID UID(void) const {
            return this->uid;
        }
        inline const std::string ClassName(void) const {
            return this->class_name;
        }
        const std::string SlotsLabel(void) const {
            return std::string(this->caller_slot_name + this->slot_name_separator + this->callee_slot_name);
        }

    private:
        // VARIABLES --------------------------------------------------------------

        const ImGuiID uid;
        const std::string class_name;
        const std::string description;
        const std::string plugin_name;
        const std::vector<std::string> functions;

        std::map<CallSlotType, CallSlotPtr_t> connected_callslots;

        bool gui_selected;
        const std::string slot_name_separator = " > ";
        std::string caller_slot_name;
        std::string callee_slot_name;

        HoverToolTip gui_tooltip;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
