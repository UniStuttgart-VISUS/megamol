/*
 * Group.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED


#include "Call.h"
#include "InterfaceSlot.h"
#include "Module.h"
#include "widgets/RenamePopUp.h"


namespace megamol {
namespace gui {


// Forward declarations
class Group;

// Types
typedef std::shared_ptr<Group> GroupPtrType;
typedef std::vector<GroupPtrType> GroupPtrVectorType;


/** ************************************************************************
 * Defines GUI group presentation.
 */
class GroupPresentation {
public:
    friend class Group;

    // VARIABLES --------------------------------------------------------------

    // Relative position without considering canvas offset and zooming
    ImVec2 position;
    // Relative size without considering zooming
    ImVec2 size;

    // FUNCTIONS --------------------------------------------------------------

    GroupPresentation(void);
    ~GroupPresentation(void);

    inline ImVec2 GetSize(void) { return this->size; }
    inline bool IsViewCollapsed(void) { return this->collapsed_view; }
    inline bool ModulesVisible(void) { return !this->collapsed_view; }
    inline void ForceUpdate(void) { this->update = true; }

private:
    // VARIABLES --------------------------------------------------------------

    bool collapsed_view;
    bool allow_selection;
    bool allow_context;
    bool selected;
    bool update;

    // Widgets
    RenamePopUp rename_popup;

    // FUNCTIONS --------------------------------------------------------------

    void Present(megamol::gui::PresentPhase phase, Group& inout_group, GraphItemsStateType& state);
    void UpdatePositionSize(Group& inout_group, const GraphCanvasType& in_canvas);
    void SetPosition(Group& inout_group, const GraphCanvasType& in_canvas, ImVec2 pos);
};


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

    bool AddModule(const ModulePtrType& module_ptr);
    bool RemoveModule(ImGuiID module_uid);
    bool ContainsModule(ImGuiID module_uid);
    inline const ModulePtrVectorType& GetModules(void) { return this->modules; }
    inline bool Empty(void) { return (this->modules.empty()); }

    ImGuiID AddInterfaceSlot(const CallSlotPtrType& callslot_ptr, bool auto_add = true);
    bool GetInterfaceSlot(ImGuiID interfaceslot_uid, InterfaceSlotPtrType& interfaceslot_ptr);
    inline InterfaceSlotPtrMapType& GetInterfaceSlots(void) { return this->interfaceslots; }
    inline InterfaceSlotPtrVectorType& GetInterfaceSlots(CallSlotType type) { return this->interfaceslots[type]; }
    bool DeleteInterfaceSlot(ImGuiID interfaceslot_uid);
    bool ContainsInterfaceSlot(ImGuiID interfaceslot_uid);
    bool InterfaceSlot_RemoveCallSlot(ImGuiID callslot_uid, bool force = false);
    bool InterfaceSlot_ContainsCallSlot(ImGuiID callslot_uid);

    void RestoreInterfaceslots(void);

    // Presentation ----------------------------------------------------

    inline void PresentGUI(megamol::gui::PresentPhase phase, GraphItemsStateType& state) {
        this->present.Present(phase, *this, state);
    }
    inline void UpdateGUI(const GraphCanvasType& in_canvas) { this->present.UpdatePositionSize(*this, in_canvas); }
    inline void SetGUIPosition(const GraphCanvasType& in_canvas, ImVec2 pos) {
        this->present.SetPosition(*this, in_canvas, pos);
    }

private:
    // VARIABLES --------------------------------------------------------------

    ModulePtrVectorType modules;
    InterfaceSlotPtrMapType interfaceslots;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED
