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


namespace megamol {
namespace gui {
namespace graph {

// GRAPH DATA STRUCTURE PRESENTATIONS -------------------------------------

// Forward declaration
class Parameter;
class CallSlot;
class Call;
class Module;

// Pointer types to classes
typedef std::shared_ptr<Parameter> ParamPtrType;
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<Module> ModulePtrType;


/**
 * Defines GUI parameter presentations.
 */
class ParamPresentations {
public:
    ParamPresentations(ParamPtrType p);

    ~ParamPresentations(void);

    void Present();

private:
    enum Presentation { DEFAULT } presentation;
    bool read_only;
    bool visible;

    ParamPtrType parent;
};


/**
 * Defines GUI call slot presentations.
 */
class CallSlotPresentations {
public:
    CallSlotPresentations(CallSlotPtrType p);

    ~CallSlotPresentations(void);

    void Present();

    void UpdatePosition();

    ImVec2 position;

private:
    enum Presentation { DEFAULT } presentation;
    bool label_visible;

    CallSlotPtrType parent;
};


/**
 * Defines GUI call presentations.
 */
class CallPresentations {
public:
    CallPresentations(CallPtrType p);

    ~CallPresentations(void);

    void Present();

private:
    enum Presentation { DEFAULT } presentation;
    bool label_visible;

    CallPtrType parent;
};


/**
 * Defines GUI module presentations.
 */
class ModulePresentations {
public:
    ModulePresentations(ModulePtrType p);

    ~ModulePresentations(void);

    void Present();

    ImVec2 position;
    ImVec2 size;
    std::string class_label;
    std::string name_label;


private:
    enum Presentation { DEFAULT } presentation;
    bool label_visible;

    ModulePtrType parent;
};


} // namespace graph
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_PARAMETER_PRESENTATION_INCLUDED