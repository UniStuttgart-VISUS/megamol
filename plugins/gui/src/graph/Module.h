/*
 * Module.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED


#include "ModulePresentation.h"


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

    // VARIABLES --------------------------------------------------------------

    const ImGuiID uid;
    ModulePresentation present;

    // Init when adding module from stock
    std::string class_name;
    std::string description;
    std::string plugin_name;
    bool is_view;
    ParamVector_t parameters;

    // Init when adding module to graph
    std::string name;
    bool is_view_instance;

    // FUNCTIONS --------------------------------------------------------------

    Module(ImGuiID uid);
    ~Module();

    bool AddCallSlot(CallSlotPtr_t callslot);
    bool DeleteCallSlots(void);
    bool GetCallSlot(ImGuiID callslot_uid, CallSlotPtr_t& out_callslot_ptr);
    const CallSlotPtrVector_t& GetCallSlots(CallSlotType type) { return this->callslots[type]; }
    const CallSlotPtrMap_t& GetCallSlots(void) { return this->callslots; }

    const inline std::string FullName(void) const {
        std::string fullname = "::" + this->name;
        if (!this->present.group.name.empty()) {
            fullname = "::" + this->present.group.name + fullname;
        }
        return fullname;
    }

    // Presentation ----------------------------------------------------

    inline void PresentGUI(megamol::gui::PresentPhase phase, GraphItemsState_t& state) {
        this->present.Present(phase, *this, state);
    }
    inline void UpdateGUI(const GraphCanvas_t& in_canvas) { this->present.Update(*this, in_canvas); }

private:
    // VARIABLES --------------------------------------------------------------

    CallSlotPtrMap_t callslots;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
