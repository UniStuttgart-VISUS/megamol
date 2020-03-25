/*
 * Group.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED


#include "Module.h"
#include "CallSlot.h"
#include "GUIUtils.h"


namespace megamol {
namespace gui {
namespace configurator {

// Forward declaration
class Group;

// Pointer types to class
typedef std::shared_ptr<Group> GroupPtrType;

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


    // GUI Presentation -------------------------------------------------------

    void GUI_Present(
        const CanvasType& in_canvas, HotKeyArrayType& inout_hotkeys, InteractType& interact_state) {
        this->present.Present(*this, in_canvas, inout_hotkeys, interact_state);
    }

private:

    /**
     * Defines GUI group presentation.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(Group& inout_mod, const CanvasType& in_canvas, HotKeyArrayType& inout_hotkeys, InteractType& interact_state);

    private:


    } present;
};


} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GROUP_H_INCLUDED