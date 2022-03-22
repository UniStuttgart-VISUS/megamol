/*
 * Module.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
#pragma once


#include "CallSlot.h"
#include "ParameterGroups.h"
#include "mmcore/MultiPerformanceHistory.h"
#include "widgets/HoverToolTip.h"
#include "widgets/PopUps.h"

namespace megamol {
namespace gui {


// Forward declarations
class Module;
class Call;
class CallSlot;
class Parameter;
typedef std::shared_ptr<Call> CallPtr_t;
typedef std::shared_ptr<CallSlot> CallSlotPtr_t;

// Types
typedef std::shared_ptr<Module> ModulePtr_t;
typedef std::vector<ModulePtr_t> ModulePtrVector_t;


/** ************************************************************************
 * Defines module data structure for graph
 */
class Module {
public:
    struct StockModule {
        std::string class_name;
        std::string description;
        std::string plugin_name;
        bool is_view;
        std::vector<Parameter::StockParameter> parameters;
        std::map<CallSlotType, std::vector<CallSlot::StockCallSlot>> callslots;
    };

    static ImVec2 GetDefaultModulePosition(const GraphCanvas_t& canvas);

    Module(ImGuiID uid, const std::string& class_name, const std::string& description, const std::string& plugin_name,
        bool is_view);
    ~Module();

    bool AddCallSlot(CallSlotPtr_t callslot);
    bool DeleteCallSlots();

    void Draw(megamol::gui::PresentPhase phase, GraphItemsState_t& state);
    void Update(const GraphItemsState_t& state);

    inline bool DrawParameters(const std::string& in_search, bool in_extended, bool in_indent,
        megamol::gui::Parameter::WidgetScope in_scope, std::shared_ptr<TransferFunctionEditor> tfeditor_ptr,
        ImGuiID in_override_header_state, megamol::core::utility::PickingBuffer* inout_picking_buffer) {

        return this->gui_param_groups.Draw(this->parameters, in_search, in_extended, in_indent, in_scope, tfeditor_ptr,
            in_override_header_state, inout_picking_buffer);
    }

    bool ParametersVisible();

    bool StateToJSON(nlohmann::json& inout_json);

    bool StateFromJSON(const nlohmann::json& in_json);

    // GET ----------------------------------------------------------------

    CallSlotPtr_t CallSlotPtr(ImGuiID callslot_uid);
    const CallSlotPtrVector_t& CallSlots(CallSlotType type) {
        return this->callslots[type];
    }
    const CallSlotPtrMap_t& CallSlots() {
        return this->callslots;
    }

    inline ImGuiID UID() const {
        return this->uid;
    }
    inline std::string ClassName() const {
        return this->class_name;
    }
    bool IsGraphEntry() const {
        return (!this->graph_entry_name.empty());
    }
    inline std::string Name() const {
        return this->name;
    }
    inline std::string FullName() const {
        std::string fullname = "::" + this->name;
        if (!this->group_name.empty()) {
            fullname = "::" + this->group_name + fullname;
        }
        return fullname;
    }
    inline ImVec2 Position() const {
        return this->gui_position;
    }
    inline ImVec2 Size() const {
        return this->gui_size;
    }
    inline ImGuiID GroupUID() const {
        return this->group_uid;
    }
    inline ParamVector_t& Parameters() {
        return this->parameters;
    }
    inline bool IsView() const {
        return this->is_view;
    }
    inline std::string GroupName() const {
        return this->group_name;
    }
    inline std::string Description() const {
        return this->description;
    }
    inline std::string GraphEntryName() const {
        return this->graph_entry_name;
    }
    inline bool IsHidden() const {
        return this->gui_hidden;
    }

    // SET ----------------------------------------------------------------

    void SetName(const std::string& mod_name);

    void SetGroupName(const std::string& gr_name);

    inline void SetGraphEntryName(const std::string& graph_entry) {
        this->graph_entry_name = graph_entry;
    }
    inline void SetGroupUID(ImGuiID guid) {
        this->group_uid = guid;
    }
    inline void SetHidden(bool hidden) {
        this->gui_hidden = hidden;
    }

    inline void SetPosition(ImVec2 pos) {
        this->gui_position = pos;
    }
    void SetSelectedSlotPosition() {
        this->gui_set_selected_slot_position = true;
    }
    void SetScreenPosition(ImVec2 pos) {
        this->gui_set_screen_position = pos;
    }
#ifdef PROFILING
    void SetProfilingData(void* ptr, frontend_resources::PerformanceManager* perf_manager) {
        this->profiling_parent_pointer = ptr;

        auto handles = perf_manager->lookup_timers(ptr);
        for (auto& h : handles) {
            auto& conf = perf_manager->lookup_config(h);
            switch (conf.api) {
            case frontend_resources::PerformanceManager::query_api::CPU:
                this->cpu_perf_history.emplace(std::make_pair(h, core::MultiPerformanceHistory()));
                this->cpu_perf_history[h].set_name(conf.name);
                break;
            case frontend_resources::PerformanceManager::query_api::OPENGL:
                this->gl_perf_history.emplace(std::make_pair(h, core::MultiPerformanceHistory()));
                this->gl_perf_history[h].set_name(conf.name);
                break;
            }
        }
    }

    void* GetProfilingParent() {
        return this->profiling_parent_pointer;
    }

    void AppendPerformanceData(frontend_resources::PerformanceManager::frame_type frame,
        const frontend_resources::PerformanceManager::timer_entry& entry);

    bool ShowProfiling() {
        return this->show_profiling_data;
    }

    void DrawProfiling(GraphItemsState_t& state);

    ImVec2 GetProfilingButtonPosition() {
        return this->profiling_button_position;
    }

#endif

private:
    // VARIABLES --------------------------------------------------------------

    const ImGuiID uid;
    // TODO place StockModule (Properties?) here instead
    const std::string class_name;
    const std::string description;
    const std::string plugin_name;
    const bool is_view;

    ParamVector_t parameters;
    CallSlotPtrMap_t callslots;

    std::string name;
    std::string graph_entry_name; // Do not change in Module class, only via Graph class

    ImGuiID group_uid;
    std::string group_name;

    ParameterGroups gui_param_groups;

    bool gui_selected;
    ImVec2 gui_position; /// Relative position without considering canvas offset and zooming
    ImVec2 gui_size;     /// Relative size without considering zooming
    bool gui_update;
    bool gui_param_child_show;
    ImVec2 gui_set_screen_position;
    bool gui_set_selected_slot_position;
    bool gui_hidden;
    bool gui_other_item_hovered;

    HoverToolTip gui_tooltip;
    PopUps gui_rename_popup;

#ifdef PROFILING

    std::unordered_map<frontend_resources::PerformanceManager::handle_type, core::MultiPerformanceHistory>
        cpu_perf_history;
    std::unordered_map<frontend_resources::PerformanceManager::handle_type, core::MultiPerformanceHistory>
        gl_perf_history;
    void* profiling_parent_pointer;
    float profiling_window_height; // determine dynamically
    bool show_profiling_data;
    ImageWidget gui_profiling_button;
    ImageWidget gui_profiling_run_button;
    bool pause_profiling_history_update;
    ImVec2 profiling_button_position;

#endif // PROFILING

    // FUNCTIONS --------------------------------------------------------------

    inline bool found_uid(UIDVector_t& modules_uid_vector, ImGuiID module_uid) const {
        return (
            std::find(modules_uid_vector.begin(), modules_uid_vector.end(), module_uid) != modules_uid_vector.end());
    }

    inline void erase_uid(UIDVector_t& modules_uid_vector, ImGuiID module_uid) const {
        for (auto iter = modules_uid_vector.begin(); iter != modules_uid_vector.end(); iter++) {
            if ((*iter) == module_uid) {
                modules_uid_vector.erase(iter);
                return;
            }
        }
    }

    inline void add_uid(UIDVector_t& modules_uid_vector, ImGuiID module_uid) const {
        if (!this->found_uid(modules_uid_vector, module_uid)) {
            modules_uid_vector.emplace_back(module_uid);
        }
    }
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
