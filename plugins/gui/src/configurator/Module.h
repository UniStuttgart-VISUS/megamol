/*
 * Module.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED


#include "GUIUtils.h"
#include "Parameter.h"
#include "CallSlot.h"


namespace megamol {
namespace gui {
namespace configurator {

// Forward declaration
class Call;
class CallSlot;
class Module;
class Parameter;

// Pointer types to classes
typedef std::shared_ptr<Parameter> ParamPtrType;
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::shared_ptr<Module> ModulePtrType;

/**
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
        std::map<CallSlot::CallSlotType, std::vector<CallSlot::StockCallSlot>> call_slots;
    };

    enum Presentations : size_t { DEFAULT = 0, _COUNT_ = 1 };

    Module(ImGuiID uid);
    ~Module();

    const ImGuiID uid;

    // Init when adding module from stock
    std::string class_name;
    std::string description;
    std::string plugin_name;
    bool is_view;
    std::vector<Parameter> parameters;

    // Init when adding module to graph
    std::string name;
    std::string name_space;
    bool is_view_instance;

    bool AddCallSlot(CallSlotPtrType call_slot);
    bool RemoveAllCallSlots(void);
    const CallSlotPtrType GetCallSlot(ImGuiID call_slot_uid);
    const std::vector<CallSlotPtrType>& GetCallSlots(CallSlot::CallSlotType type);
    const std::map<CallSlot::CallSlotType, std::vector<CallSlotPtrType>>& GetCallSlots(void);

    const inline std::string FullName(void) const { 
        std::string fullname = "::" + this->name;
        if (!this->name_space.empty()) {
            fullname = "::" + this->name_space + fullname;
        }
        return fullname; 
    }

    // GUI Presentation -------------------------------------------------------

    void GUI_Present(
        const CanvasType& in_canvas, HotKeyArrayType& inout_hotkeys, InteractType& interact_state) {
        this->present.Present(*this, in_canvas, inout_hotkeys, interact_state);
    }
    void GUI_Update(const CanvasType& in_canvas) { this->present.UpdateSize(*this, in_canvas); }

    void GUI_SetLabelVisibility(bool visible) { this->present.label_visible = visible; }
    void GUI_SetPresentation(Module::Presentations present) { this->present.presentations = present; }
    void GUI_SetPosition(ImVec2 pos) { this->present.SetPosition(pos); }
    
    ImVec2 GUI_GetPosition(void) { return this->present.GetPosition(); }
    ImVec2 GUI_GetSize(void) { return this->present.GetSize(); }

private:
    std::map<CallSlot::CallSlotType, std::vector<CallSlotPtrType>> call_slots;

    /**
     * Defines GUI module presentation.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(Module& inout_mod, const CanvasType& in_canvas, HotKeyArrayType& inout_hotkeys, InteractType& interact_state);

        void UpdateSize(Module& mod, const CanvasType& in_canvas);

        void SetPosition(ImVec2 pos) { this->position = pos; }

        ImVec2 GetPosition(void) { return this->position; }
        ImVec2 GetSize(void) { return this->size; }

        Module::Presentations presentations;
        bool label_visible;

    private:
        // Relative position without considering canvas offset and zooming
        ImVec2 position;
        // Relative size without considering zooming
        ImVec2 size;
        std::string class_label;
        std::string name_label;
        GUIUtils utils;
        bool selected;
        bool update_once;

    } present;
};


} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED