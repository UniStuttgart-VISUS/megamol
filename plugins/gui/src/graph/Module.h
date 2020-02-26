/*
 * Module.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED


#include "vislib/sys/Log.h"

#include <map>
#include <vector>

#include "GUIUtils.h"
#include "Parameter.h"


namespace megamol {
namespace gui {
namespace graph {


// Forward declaration
class Call;
class CallSlot;
class Module;

// Pointer types to classes
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::shared_ptr<Module> ModulePtrType;

/**
 * Defines module data structure for graph.
 */
class Module {
public:
    Module(int uid);
    ~Module();

    const int uid;

    std::string class_name;
    std::string description;
    std::string plugin_name;
    bool is_view;

    std::vector<Parameter> parameters;

    std::string name;
    std::string full_name;
    bool is_view_instance;

    bool AddCallSlot(CallSlotPtrType call_slot);
    bool RemoveAllCallSlots(void);
    const std::vector<CallSlotPtrType>& GetCallSlots(CallSlot::CallSlotType type);
    const std::map<CallSlot::CallSlotType, std::vector<CallSlotPtrType>>& GetCallSlots(void);

    void Present(void);

    /**
     * Defines GUI module presentations.
     */
    class Presentations {
    public:
        Presentations(void);

        ~Presentations(void);

        void Present(Module& mod);

        ImVec2 position;
        ImVec2 size;
        std::string class_label;
        std::string name_label;


    private:
        enum Presentation { DEFAULT } presentation;
        bool label_visible;
    } presentations;

private:
    std::map<CallSlot::CallSlotType, std::vector<CallSlotPtrType>> call_slots;
};


} // namespace graph
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_MODULE_H_INCLUDED