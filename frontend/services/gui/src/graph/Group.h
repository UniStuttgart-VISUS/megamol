/*
 * Group.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "Call.h"
#include "InterfaceSlot.h"
#include "Module.h"
#include "widgets/PopUps.h"


namespace megamol::gui {


// Forward declarations
class Group;

// Types
typedef std::shared_ptr<Group> GroupPtr_t;
typedef std::vector<GroupPtr_t> GroupPtrVector_t;


/** ************************************************************************
 * Defines module data structure for graph
 */
class Group {
public:
    explicit Group(ImGuiID uid);
    ~Group();

    bool AddModule(const ModulePtr_t& module_ptr);
    bool RemoveModule(ImGuiID module_uid, bool retore_interface = true);
    bool ContainsModule(ImGuiID module_uid);
    inline bool Empty() {
        return (this->modules.empty());
    }

    InterfaceSlotPtr_t AddInterfaceSlot(const CallSlotPtr_t& callslot_ptr, bool auto_add = true);
    InterfaceSlotPtr_t InterfaceSlotPtr(ImGuiID interfaceslot_uid);
    inline InterfaceSlotPtrMap_t& InterfaceSlots() {
        return this->interfaceslots;
    }
    inline InterfaceSlotPtrVector_t& InterfaceSlots(CallSlotType type) {
        return this->interfaceslots[type];
    }
    bool DeleteInterfaceSlot(ImGuiID interfaceslot_uid);
    bool InterfaceSlot_RemoveCallSlot(ImGuiID callslot_uid, bool force = false);
    bool InterfaceSlot_ContainsCallSlot(ImGuiID callslot_uid);

    void RestoreInterfaceslots();

    void Draw(megamol::gui::PresentPhase phase, GraphItemsState_t& state);
    void Update(const GraphCanvas_t& in_canvas);

    inline ImGuiID UID() const {
        return this->uid;
    }
    inline const ModulePtrVector_t& Modules() const {
        return this->modules;
    }
    inline std::string Name() const {
        return this->name;
    }
    inline ImVec2 Size() const {
        return this->gui_size;
    }
    inline void SetName(const std::string& group_name) {
        this->name = group_name;
    }
    void SetPosition(const GraphItemsState_t& state, ImVec2 pos);

    ImVec2 GetPositionBottomCenter() {
        return this->gui_position_bottom_center;
    }

private:
    // VARIABLES --------------------------------------------------------------

    const ImGuiID uid;

    std::string name;

    ModulePtrVector_t modules;
    InterfaceSlotPtrMap_t interfaceslots;

    bool gui_selected;
    ImVec2 gui_position; /// Relative position without considering canvas offset and zooming
    ImVec2 gui_size;     /// Relative size without considering zooming
    bool gui_collapsed_view;
    bool gui_allow_selection;
    bool gui_update;
    ImVec2 gui_position_bottom_center;
    PopUps gui_rename_popup;

    // FUNCTIONS --------------------------------------------------------------

    void spacial_sort_interfaceslots() {
        for (auto& interfaceslot_map : this->interfaceslots) {
            std::sort(interfaceslot_map.second.begin(), interfaceslot_map.second.end(),
                [](InterfaceSlotPtr_t isptr1, InterfaceSlotPtr_t isptr2) {
                    float y1 = -FLT_MAX;
                    if (isptr1 != nullptr)
                        y1 = isptr1->Position(false).y;
                    float y2 = -FLT_MAX;
                    if (isptr2 != nullptr)
                        y2 = isptr2->Position(false).y;
                    return (y1 < y2);
                });
        }
    }
};


} // namespace megamol::gui
