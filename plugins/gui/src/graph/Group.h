/*
 * Group.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED


#include "GroupPresentation.h"


namespace megamol {
namespace gui {


// Forward declarations
class Group;

// Types
typedef std::shared_ptr<Group> GroupPtr_t;
typedef std::vector<GroupPtr_t> GroupPtrVector_t;


/** ************************************************************************
 * Defines module data structure for graph.
 */
class Group {
public:
    // VARIABLES --------------------------------------------------------------

    const ImGuiID uid;
    GroupPresentation present;

    // Init when adding group to graph
    std::string name;

    // FUNCTIONS --------------------------------------------------------------

    Group(ImGuiID uid);
    ~Group();

    bool AddModule(const ModulePtr_t& module_ptr);
    bool RemoveModule(ImGuiID module_uid);
    bool ContainsModule(ImGuiID module_uid);
    inline const ModulePtrVector_t& GetModules(void) { return this->modules; }
    inline bool Empty(void) { return (this->modules.empty()); }

    ImGuiID AddInterfaceSlot(const CallSlotPtr_t& callslot_ptr, bool auto_add = true);
    bool GetInterfaceSlot(ImGuiID interfaceslot_uid, InterfaceSlotPtr_t& interfaceslot_ptr);
    inline InterfaceSlotPtrMap_t& GetInterfaceSlots(void) { return this->interfaceslots; }
    inline InterfaceSlotPtrVector_t& GetInterfaceSlots(CallSlotType type) { return this->interfaceslots[type]; }
    bool DeleteInterfaceSlot(ImGuiID interfaceslot_uid);
    bool ContainsInterfaceSlot(ImGuiID interfaceslot_uid);
    bool InterfaceSlot_RemoveCallSlot(ImGuiID callslot_uid, bool force = false);
    bool InterfaceSlot_ContainsCallSlot(ImGuiID callslot_uid);

    void RestoreInterfaceslots(void);

    // Presentation ----------------------------------------------------

    inline void PresentGUI(megamol::gui::PresentPhase phase, GraphItemsState_t& state) {
        this->present.Present(phase, *this, state);
    }
    inline void UpdateGUI(const GraphCanvas_t& in_canvas) { this->present.UpdatePositionSize(*this, in_canvas); }
    inline void SetGUIPosition(const GraphCanvas_t& in_canvas, ImVec2 pos) {
        this->present.SetPosition(*this, in_canvas, pos);
    }

private:
    // VARIABLES --------------------------------------------------------------

    ModulePtrVector_t modules;
    InterfaceSlotPtrMap_t interfaceslots;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED
