/*
 * Group.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/RenamePopUp.h"

#include "Call.h"
#include "InterfaceSlot.h"
#include "Module.h"


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
        Group(ImGuiID uid);
        ~Group();

        bool AddModule(const ModulePtr_t& module_ptr);
        bool RemoveModule(ImGuiID module_uid);
        bool ContainsModule(ImGuiID module_uid);
        inline bool Empty(void) {
            return (this->modules.empty());
        }

        ImGuiID AddInterfaceSlot(const CallSlotPtr_t& callslot_ptr, bool auto_add = true);
        InterfaceSlotPtr_t GetInterfaceSlot(ImGuiID interfaceslot_uid);
        inline InterfaceSlotPtrMap_t& GetInterfaceSlots(void) {
            return this->interfaceslots;
        }
        inline InterfaceSlotPtrVector_t& GetInterfaceSlots(CallSlotType type) {
            return this->interfaceslots[type];
        }
        bool DeleteInterfaceSlot(ImGuiID interfaceslot_uid);
        bool ContainsInterfaceSlot(ImGuiID interfaceslot_uid);
        bool InterfaceSlot_RemoveCallSlot(ImGuiID callslot_uid, bool force = false);
        bool InterfaceSlot_ContainsCallSlot(ImGuiID callslot_uid);

        void RestoreInterfaceslots(void);

        void Draw(megamol::gui::PresentPhase phase, GraphItemsState_t& state);
        void UpdatePositionSize(const GraphCanvas_t& in_canvas);

        inline const ImGuiID UID(void) const {
            return this->uid;
        }
        inline ImVec2 Size(void) {
            return this->gui_size;
        }
        inline bool IsViewCollapsed(void) {
            return this->gui_collapsed_view;
        }
        inline void ForceUpdate(void) {
            this->gui_update = true;
        }

        void SetPosition(const GraphCanvas_t& in_canvas, ImVec2 pos);

    private:
        // VARIABLES --------------------------------------------------------------

        const ImGuiID uid;

        std::string name;

        ModulePtrVector_t modules;
        InterfaceSlotPtrMap_t interfaceslots;

        ImVec2 gui_position; /// Relative position without considering canvas offset and zooming
        ImVec2 gui_size;     /// Relative size without considering zooming
        bool gui_collapsed_view;
        bool gui_allow_selection;
        bool gui_allow_context;
        bool gui_selected;
        bool gui_update;
        RenamePopUp gui_rename_popup;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED
