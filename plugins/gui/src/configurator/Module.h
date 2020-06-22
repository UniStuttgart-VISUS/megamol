/*
 * Module.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED


#include "CallSlot.h"
#include "Parameter.h"


namespace megamol {
namespace gui {
namespace configurator {

// Forward declarations
class Module;
class Call;
class CallSlot;
class Parameter;
typedef std::shared_ptr<Parameter> ParamPtrType;
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<CallSlot> CallSlotPtrType;

// Types
typedef std::shared_ptr<Module> ModulePtrType;
typedef std::vector<ModulePtrType> ModulePtrVectorType;


/** ************************************************************************
 * Defines GUI module presentation.
 */
class ModulePresentation {
public:
    friend class Module;

    struct GroupState {
        ImGuiID uid;
        bool visible;
        std::string name;
    };

    // VARIABLES --------------------------------------------------------------

    GroupState group;
    bool label_visible;
    // Relative position without considering canvas offset and zooming
    ImVec2 position;

    // FUNCTIONS --------------------------------------------------------------

    ModulePresentation(void);
    ~ModulePresentation(void);

    static ImVec2 GetDefaultModulePosition(const GraphCanvasType& canvas);

    inline ImVec2 GetSize(void) { return this->size; }

    void SetSelectedSlotPosition(void) { this->set_selected_slot_position = true; }
    void SetScreenPosition(ImVec2 pos) { this->set_screen_position = pos; }

private:
    // VARIABLES --------------------------------------------------------------

    // Relative size without considering zooming
    ImVec2 size;
    GUIUtils utils;
    bool selected;
    bool update;
    bool show_params;
    ImVec2 set_screen_position;
    bool set_selected_slot_position;

    // FUNCTIONS --------------------------------------------------------------

    void Present(megamol::gui::PresentPhase phase, Module& inout_module, GraphItemsStateType& state);
    void Update(Module& inout_module, const GraphCanvasType& in_canvas);

    inline bool found_uid(UIDVectorType& modules_uid_vector, ImGuiID module_uid) const {
        return (
            std::find(modules_uid_vector.begin(), modules_uid_vector.end(), module_uid) != modules_uid_vector.end());
    }

    inline void erase_uid(UIDVectorType& modules_uid_vector, ImGuiID module_uid) const {
        for (auto iter = modules_uid_vector.begin(); iter != modules_uid_vector.end(); iter++) {
            if ((*iter) == module_uid) {
                modules_uid_vector.erase(iter);
                return;
            }
        }
    }

    inline void add_uid(UIDVectorType& modules_uid_vector, ImGuiID module_uid) const {
        if (!this->found_uid(modules_uid_vector, module_uid)) {
            modules_uid_vector.emplace_back(module_uid);
        }
    }
};


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

    // VARIABLES --------------------------------------------------------------

    const ImGuiID uid;
    ModulePresentation present;

    // Init when adding module from stock
    std::string class_name;
    std::string description;
    std::string plugin_name;
    bool is_view;
    std::vector<Parameter> parameters;

    // Init when adding module to graph
    std::string name;
    bool is_view_instance;

    // FUNCTIONS --------------------------------------------------------------

    Module(ImGuiID uid);
    ~Module();

    bool AddCallSlot(CallSlotPtrType callslot);
    bool DeleteCallSlots(void);
    bool GetCallSlot(ImGuiID callslot_uid, CallSlotPtrType& out_callslot_ptr);
    const CallSlotPtrVectorType& GetCallSlots(CallSlotType type) { return this->callslots[type]; }
    const CallSlotPtrMapType& GetCallSlots(void) { return this->callslots; }

    const inline std::string FullName(void) const {
        std::string fullname = "::" + this->name;
        if (!this->present.group.name.empty()) {
            fullname = "::" + this->present.group.name + fullname;
        }
        return fullname;
    }

    // Presentation ----------------------------------------------------

    inline void PresentGUI(megamol::gui::PresentPhase phase, GraphItemsStateType& state) {
        this->present.Present(phase, *this, state);
    }
    inline void UpdateGUI(const GraphCanvasType& in_canvas) { this->present.Update(*this, in_canvas); }

private:
    // VARIABLES --------------------------------------------------------------

    CallSlotPtrMapType callslots;
};


} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
