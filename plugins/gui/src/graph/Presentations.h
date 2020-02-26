/*
 * Presentations.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_PARAMETER_PRESENTATION_INCLUDED
#define MEGAMOL_GUI_GRAPH_PARAMETER_PRESENTATION_INCLUDED


#include "vislib/sys/Log.h"

#include "GUIUtils.h"
#include "Elements.h"


namespace megamol {
namespace gui {
namespace graph {

// GRAPH DATA STRUCTURE PRESENTATIONS -------------------------------------

/**
 * Defines GUI parameter presentations.
 */
class ParamPresentations : public megamol::gui::graph::Parameter {
public:
    ParamPresentations(int uid, megamol::gui::graph::Parameter::ParamType type);

    ~ParamPresentations(void);

    void Present();

private:
    enum Presentation { DEFAULT } presentation;
    bool read_only;
    bool visible;
};


/**
 * Defines GUI call slot presentations.
 */
class CallSlotPresentations : public megamol::gui::graph::CallSlot {
public:
    CallSlotPresentations(int uid);

    ~CallSlotPresentations(void);

    void Present();

    void UpdatePosition();

    ImVec2 position;

private:
    enum Presentation { DEFAULT } presentation;
    bool label_visible;

};


/**
 * Defines GUI call presentations.
 */
class CallPresentations : public megamol::gui::graph::Call {
public:
    CallPresentations(int uid);

    ~CallPresentations(void);

    void Present();

private:
    enum Presentation { DEFAULT } presentation;
    bool label_visible;
};


/**
 * Defines GUI module presentations.
 */
class ModulePresentations : public megamol::gui::graph::Module {
public:
    ModulePresentations(int uid);

    ~ModulePresentations(void);

    void Present();

    ImVec2 position;
    ImVec2 size;

private:
    enum Presentation { DEFAULT } presentation;
    bool label_visible;
};


} // namespace graph
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_PARAMETER_PRESENTATION_INCLUDED