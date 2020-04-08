/*
 * Module.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED


#include "CallSlot.h"
#include "GUIUtils.h"
#include "Parameter.h"


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
typedef std::vector<ModulePtrType> ModulePtrVectorType;


/**
 * Defines module data structure for graph.
 */
class Module {
public:
    enum Presentations : size_t { DEFAULT = 0, _COUNT_ = 1 };

    struct StockModule {
        std::string class_name;
        std::string description;
        std::string plugin_name;
        bool is_view;
        std::vector<Parameter::StockParameter> parameters;
        std::map<CallSlotType, std::vector<CallSlot::StockCallSlot>> call_slots;
    };

    struct GroupState {
        ImGuiID member;
        bool visible;
        std::string name;
    };

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
    bool is_view_instance;

    bool AddCallSlot(CallSlotPtrType call_slot);
    bool RemoveAllCallSlots(void);
    bool GetCallSlot(ImGuiID callslot_uid, CallSlotPtrType& out_callslot_ptr);
    const CallSlotPtrVectorType& GetCallSlots(CallSlotType type);
    const CallSlotPtrMapType& GetCallSlots(void);

    const inline std::string FullName(void) const {
        std::string fullname = "::" + this->name;
        if (!this->present.group.name.empty()) {
            fullname = "::" + this->present.group.name + fullname;
        }
        return fullname;
    }

    // GUI Presentation -------------------------------------------------------

    inline void GUI_Present(GraphItemsStateType& state) { this->present.Present(*this, state); }

    inline void GUI_Update(const GraphCanvasType& in_canvas) { this->present.UpdateSize(*this, in_canvas); }

    inline ImGuiID GUI_GetGroupMembership(void) { return this->present.group.member; }
    inline bool GUI_GetGroupVisibility(void) { return this->present.group.visible; }
    inline std::string GUI_GetGroupName(void) { return this->present.group.name; }
    inline bool GUI_GetLabelVisibility(void) { return this->present.label_visible; }
    inline ImVec2 GUI_GetPosition(void) { return this->present.GetPosition(); }
    inline ImVec2 GUI_GetSize(void) { return this->present.GetSize(); }

    inline void GUI_SetGroupMembership(ImGuiID member) { this->present.group.member = member; }
    inline void GUI_SetGroupVisibility(bool visible) { this->present.group.visible = visible; }
    inline void GUI_SetGroupName(const std::string& name) { this->present.group.name = name; }
    inline void GUI_SetLabelVisibility(bool visible) { this->present.label_visible = visible; }
    inline void GUI_SetPresentation(Module::Presentations present) { this->present.presentations = present; }
    inline void GUI_SetPosition(ImVec2 pos) { this->present.SetPosition(pos); }

    static ImVec2 GUI_GetInitModulePosition(const GraphCanvasType& canvas) {
        return Module::Presentation::GetInitModulePosition(canvas);
    }

private:
    CallSlotPtrMapType call_slots;

    /** ************************************************************************
     * Defines GUI module presentation.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(Module& inout_module, GraphItemsStateType& state);

        void UpdateSize(Module& inout_module, const GraphCanvasType& in_canvas);

        inline void SetPosition(ImVec2 pos) { this->position = pos; }

        inline ImVec2 GetPosition(void) { return this->position; }
        inline ImVec2 GetSize(void) { return this->size; }

        static ImVec2 GetInitModulePosition(const GraphCanvasType& canvas);

        GroupState group;
        Module::Presentations presentations;
        bool label_visible;

    private:
        // Relative position without considering canvas offset and zooming
        ImVec2 position;
        // Relative size without considering zooming
        ImVec2 size;

        GUIUtils utils;
        bool selected;
        bool update;
        bool other_item_hovered;
        bool show_params;

        inline bool found_uid(UIDVectorType& modules_uid_vector, ImGuiID module_uid) const {
            return (std::find(modules_uid_vector.begin(), modules_uid_vector.end(), module_uid) !=
                    modules_uid_vector.end());
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

    } present;
};


} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
