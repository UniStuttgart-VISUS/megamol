/*
 * Configurator.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
// Creating a node graph editor for ImGui
// Quick demo, not production code! This is more of a demo of how to use ImGui to create custom stuff.
// Better version by @daniel_collin here https://gist.github.com/emoon/b8ff4b4ce4f1b43e79f2
// See https://github.com/ocornut/imgui/issues/306
// v0.03: fixed grid offset issue, inverted sign of 'scrolling'
// Animated gif: https://cloud.githubusercontent.com/assets/8225057/9472357/c0263c04-4b4c-11e5-9fdf-2cd4f33f6582.gif

#ifndef MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
#define MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED

#include "vislib/sys/Log.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

#include <math.h> // fmodf

#include "mmcore/CoreInstance.h"
#include "mmcore/view//Input.h"

#include "GUIUtils.h"
#include "WindowManager.h"


namespace megamol {
namespace gui {

class Configurator {
public:
    /**
     * Initialises a new instance.
     */
    Configurator();

    /**
     * Finalises an instance.
     */
    virtual ~Configurator();

    /**
     * Draw configurator ImGui window.
     */
    bool Draw(WindowManager::WindowConfiguration& wc, megamol::core::CoreInstance* core_instance);

    // INPUT ------------------------------------------------------------------

    /**
     * This event handler can be reimplemented to receive key code events.
     *
     * @return true to stop propagation.
     */
    virtual bool OnKey(
        megamol::core::view::Key key, megamol::core::view::KeyAction action, megamol::core::view::Modifiers mods);

    /**
     * This event handler can be reimplemented to receive unicode events.
     *
     * @return Returns true if the event was accepted (stopping propagation), otherwise false.
     */
    virtual bool OnChar(unsigned int codePoint);

    /**
     * This event handler can be reimplemented to receive mouse button events.
     *
     * @return Returns true if the event was accepted (stopping propagation), otherwise false.
     */
    virtual bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
        megamol::core::view::Modifiers mods);

    /**
     * This event handler can be reimplemented to receive mouse move events.
     *
     * @return Returns true if the event was accepted (stopping propagation), otherwise false.
     */
    virtual bool OnMouseMove(double x, double y);

    /**
     * This event handler can be reimplemented to receive mouse scroll events.
     *
     * @return Returns true if the event was accepted (stopping propagation), otherwise false.
     */
    virtual bool OnMouseScroll(double dx, double dy);

private:
    class Node {
    public:
    private:
        struct ModuleData {
            std::string name;
            std::string description;
            // std::vector<> input_slots;
            // std::vector<> output_slots;

        } module_data;

        struct ImGuiData {


        } imgui_data;
    };

    // VARIABLES --------------------------------------------------------------

    std::list<Configurator::Node> nodes;

    // FUNCTIONS --------------------------------------------------------------

    void showNodeLinkGraph(bool* opened);

    void showModuleList(bool* opened);

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED