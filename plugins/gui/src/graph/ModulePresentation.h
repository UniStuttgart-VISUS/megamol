/*
 * Module.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_MODULE_PRESENTATION_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_MODULE_PRESENTATION_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/HoverToolTip.h"
#include "widgets/RenamePopUp.h"

#include "CallSlot.h"
#include "ParameterGroupsPresentation.h"


namespace megamol {
namespace gui {


// Forward declarations
class Module;


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
    ParameterGroupsPresentation param_groups;

    // FUNCTIONS --------------------------------------------------------------

    ModulePresentation(void);
    ~ModulePresentation(void);

    static ImVec2 GetDefaultModulePosition(const GraphCanvas_t& canvas);

    inline ImVec2 GetSize(void) { return this->size; }

    void SetSelectedSlotPosition(void) { this->set_selected_slot_position = true; }
    void SetScreenPosition(ImVec2 pos) { this->set_screen_position = pos; }

private:
    // VARIABLES --------------------------------------------------------------

    // Relative size without considering zooming
    ImVec2 size;
    bool selected;
    bool update;
    bool param_child_show;
    float param_child_height;
    ImVec2 set_screen_position;
    bool set_selected_slot_position;

    // Widgets
    HoverToolTip tooltip;
    RenamePopUp rename_popup;

    // FUNCTIONS --------------------------------------------------------------

    void Present(megamol::gui::PresentPhase phase, Module& inout_module, GraphItemsState_t& state);
    void Update(Module& inout_module, const GraphCanvas_t& in_canvas);

    inline bool found_uid(UIDVector_t& modules_uid_vector, ImGuiID module_uid) const {
        return (
            std::find(modules_uid_vector.begin(), modules_uid_vector.end(), module_uid) != modules_uid_vector.end());
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

#endif // MEGAMOL_GUI_GRAPH_MODULE_PRESENTATION_H_INCLUDED
