/*
 * Call.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED


#include "vislib/sys/Log.h"

#include <map>
#include <vector>

#include "CallSlot.h"
#include "GUIUtils.h"


namespace megamol {
namespace gui {
namespace configurator {


// Forward declaration
class Call;
class CallSlot;
class Module;

// Pointer types to classes
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::shared_ptr<Module> ModulePtrType;

/**
 * Defines call data structure for graph.
 */
class Call {
public:
    Call(int uid);
    ~Call();

    const int uid;

    std::string class_name;
    std::string description;
    std::string plugin_name;
    std::vector<std::string> functions;

    bool IsConnected(void);
    bool ConnectCallSlots(CallSlotPtrType call_slot_1, CallSlotPtrType call_slot_2);
    bool DisConnectCallSlots(void);
    const CallSlotPtrType GetCallSlot(CallSlot::CallSlotType type);

    // GUI Presentation -------------------------------------------------------

    bool Present(void) { return this->present.Present(*this); }

private:
    std::map<CallSlot::CallSlotType, CallSlotPtrType> connected_call_slots;

    /**
     * Defines GUI call present.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        bool Present(Call& call);

    private:
        enum Presentations { DEFAULT } presentations;
        bool label_visible;

    } present;
};


} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_CALL_H_INCLUDED