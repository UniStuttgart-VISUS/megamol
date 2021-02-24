/*
 * Module.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/HoverToolTip.h"
#include "widgets/RenamePopUp.h"

#include "CallSlot.h"
#include "ParameterGroups.h"


namespace megamol {
namespace gui {


    // Forward declarations
    class Module;
    class Call;
    class CallSlot;
    class Parameter;
    typedef std::shared_ptr<Parameter> ParamPtr_t;
    typedef std::shared_ptr<Call> CallPtr_t;
    typedef std::shared_ptr<CallSlot> CallSlotPtr_t;

    // Types
    typedef std::shared_ptr<Module> ModulePtr_t;
    typedef std::vector<ModulePtr_t> ModulePtrVector_t;


    /** ************************************************************************
     * Defines module data structure for graph.
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

        Module(ImGuiID uid, const std::string& class_name, const std::string& description,
            const std::string& plugin_name, const ParamVector_t& parameters);
        ~Module();

        bool AddCallSlot(CallSlotPtr_t callslot);
        bool DeleteCallSlots(void);

        void Draw(megamol::gui::PresentPhase phase, GraphItemsState_t& state);
        void Update(const GraphCanvas_t& in_canvas);

        // GET ----------------------------------------------------------------

        CallSlotPtr_t GetCallSlot(ImGuiID callslot_uid);
        const CallSlotPtrVector_t& GetCallSlots(CallSlotType type) {
            return this->callslots[type];
        }
        const CallSlotPtrMap_t& GetCallSlots(void) {
            return this->callslots;
        }

        inline const ImGuiID UID(void) const {
            return this->uid;
        }
        inline const std::string ClassName(void) const {
            return this->class_name;
        }
        bool IsGraphEntry(void) {
            return (!this->graph_entry_name.empty());
        }
        inline const std::string Name(void) const {
            return this->name;
        }
        const inline std::string FullName(void) const {
            std::string fullname = "::" + this->name;
            if (!this->group_name.empty()) {
                fullname = "::" + this->group_name + fullname;
            }
            return fullname;
        }
        inline ImVec2 Position(void) const {
            return this->gui_position;
        }
        inline ImVec2 Size(void) const {
            return this->gui_size;
        }
        inline ImGuiID GroupUID(void) const {
            return this->group_uid;
        }
        inline bool IsLabelVisible(void) const {
            return this->gui_label_visible;
        }

        // SET ----------------------------------------------------------------

        inline void SetGroupUID(ImGuiID uid) {
            this->group_uid = uid;
        }
        inline void SetGroupVisible(bool visible) {
            this->gui_group_visible = visible;
        }
        inline void SetGroupName(const std::string& name) {
            this->group_name = name;
        }
        inline void SetPosition(ImVec2 pos) {
            this->gui_position = pos;
        }
        void SetSelectedSlotPosition(void) {
            this->gui_set_selected_slot_position = true;
        }
        void SetScreenPosition(ImVec2 pos) {
            this->gui_set_screen_position = pos;
        }

    private:
        // VARIABLES --------------------------------------------------------------

        const ImGuiID uid;
        const std::string class_name;
        const std::string description;
        const std::string plugin_name;
        const ParamVector_t parameters;

        CallSlotPtrMap_t callslots;

        bool is_view;
        std::string name;
        std::string graph_entry_name;
        ImGuiID group_uid;
        std::string group_name;

        bool gui_label_visible;
        ImVec2 gui_position; /// Relative position without considering canvas offset and zooming
        ParameterGroups gui_param_groups;
        ImVec2 gui_size; /// Relative size without considering zooming
        bool gui_selected;
        bool gui_update;
        bool gui_param_child_show;
        ImVec2 gui_set_screen_position;
        bool gui_set_selected_slot_position;
        bool gui_group_visible;

        HoverToolTip gui_tooltip;
        RenamePopUp gui_rename_popup;

        // FUNCTIONS --------------------------------------------------------------

        inline bool found_uid(UIDVector_t& modules_uid_vector, ImGuiID module_uid) const {
            return (std::find(modules_uid_vector.begin(), modules_uid_vector.end(), module_uid) !=
                    modules_uid_vector.end());
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
