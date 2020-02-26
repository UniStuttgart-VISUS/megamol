/*
 * Presentations.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETER_PRESENTATION_INCLUDED
#define MEGAMOL_GUI_PARAMETER_PRESENTATION_INCLUDED


#include "vislib/sys/Log.h"

#include "GUIUtils.h"
#include "Graph.h"


namespace megamol {
namespace gui {

/**
 * Defines GUI parameter presentations.
 */
class ParamPresentations {
public:
    ParamPresentations(void);

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
class CallSlotPresentations {
public:
    CallSlotPresentations(void);

    ~CallSlotPresentations(void);

    void Present();

    void UpdatePosition();


private:
    enum Presentation { DEFAULT } presentation;
    bool label_visible;

    ImVec2 position;
};


/**
 * Defines GUI call presentations.
 */
class CallPresentations {
public:
    CallPresentations(void);

    ~CallPresentations(void);

    void Present();

private:
    enum Presentation { DEFAULT } presentation;
    bool label_visible;
};


/**
 * Defines GUI module presentations.
 */
class ModulePresentations {
public:
    ModulePresentations(void);

    ~ModulePresentations(void);

    void Present();

private:
    enum Presentation { DEFAULT } presentation;
    bool label_visible;

    ImVec2 position;
    ImVec2 size;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETER_PRESENTATION_INCLUDED