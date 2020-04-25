/*
 * Group.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED


#include "Call.h"
#include "CallSlot.h"
#include "GUIUtils.h"
#include "Module.h"


namespace megamol {
namespace gui {
namespace configurator {

// Forward declaration
class Group;

// Pointer types to class
typedef std::shared_ptr<Group> GroupPtrType;
typedef std::vector<GroupPtrType> GroupPtrVectorType;


/**
 * Defines module data structure for graph.
 */
class Group {
public:
    Group(ImGuiID uid);
    ~Group();

    const ImGuiID uid;

    // Init when adding group to graph
    std::string name;

    bool AddModule(const ModulePtrType& module_ptr);
    bool RemoveModule(ImGuiID module_uid);
    bool ContainsModule(ImGuiID module_uid);
    inline const ModulePtrVectorType& GetModules(void) { return this->modules; }
    inline bool EmptyModules(void) { return (this->modules.size() == 0); }

    bool AddCallSlot(const CallSlotPtrType& callslot_ptr);
    bool RemoveCallSlot(ImGuiID callslot_uid);
    bool ContainsCallSlot(ImGuiID callslot_uid);
    inline const CallSlotPtrMapType& GetCallSlots(void) { return this->callslots; }

    // GUI Presentation -------------------------------------------------------

    inline void GUI_Present(GraphItemsStateType& state) { this->present.Present(*this, state); }

    inline void GUI_Update(const GraphCanvasType& in_canvas) { this->present.UpdatePositionSize(*this, in_canvas); }

private:
    // VARIABLES --------------------------------------------------------------

    ModulePtrVectorType modules;
    CallSlotPtrMapType callslots;

    /** ************************************************************************
     * Defines GUI group presentation.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(Group& inout_group, GraphItemsStateType& state);

        void UpdatePositionSize(Group& inout_group, const GraphCanvasType& in_canvas);
        inline bool ModuleVisible(void) { return !this->collapsed_view; }
        inline void ForceUpdate(void) { this->update = true; }

    private:
        const float border;

        // Relative position without considering canvas offset and zooming
        ImVec2 position;
        // Relative size without considering zooming
        ImVec2 size;

        GUIUtils utils;
        std::string name_label;
        bool collapsed_view;
        bool selected;
        bool update;

    } present;

    // FUNCTIONS --------------------------------------------------------------

    void restore_callslot_interface_sate(void);
};


} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED
