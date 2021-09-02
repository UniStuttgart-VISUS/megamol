/*
 * Call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
#pragma once


#ifdef PROFILING
#include "mmcore/PerformanceHistory.h"
#endif
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
    typedef std::shared_ptr<CallSlot> CallSlotPtr_t;
    typedef std::shared_ptr<Module> ModulePtr_t;

    // Types
    typedef std::shared_ptr<Call> CallPtr_t;
    typedef std::vector<CallPtr_t> CallPtrVector_t;


    /** ************************************************************************
     * Defines call data structure for graph
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

        bool IsConnected();
        bool ConnectCallSlots(CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2);
        bool DisconnectCallSlots(ImGuiID calling_callslot_uid = GUI_INVALID_ID);
        const CallSlotPtr_t& CallSlotPtr(CallSlotType type);

        void Draw(megamol::gui::PresentPhase phase, GraphItemsState_t& state);

        inline ImGuiID UID() const {
            return this->uid;
        }
        inline std::string ClassName() const {
            return this->class_name;
        }
        std::string SlotsLabel() const {
            return std::string(this->caller_slot_name + this->slot_name_separator + this->callee_slot_name);
        }

#ifdef PROFILING

        struct Profiling {
            double lcput;
            double acput;
            uint32_t ncpus;
            std::array<float, core::PerformanceHistory::buffer_length> hcpu;
            double lgput;
            double agput;
            uint32_t ngpus;
            std::array<float, core::PerformanceHistory::buffer_length> hgpu;
            std::string name;
        };

        void SetProfilingValues(const std::vector<Profiling>& p) {
            this->profiling = p;
        }

#endif // PROFILING

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

#ifdef PROFILING

        std::vector<Profiling> profiling;
        bool show_profiling_data;
        void draw_profiling_data();

#endif // PROFILING
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
