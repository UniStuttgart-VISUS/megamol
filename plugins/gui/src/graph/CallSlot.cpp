/*
 * CallSlot.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "Call.h"
#include "CallSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui::graph;


megamol::gui::graph::CallSlot::CallSlot(int uid) : uid(uid), present() {
    this->parent_module.reset();
    connected_calls.clear();
}


megamol::gui::graph::CallSlot::~CallSlot() {

    // Call separately and check for reference count
    this->DisConnectCalls();
    this->DisConnectParentModule();
}


bool megamol::gui::graph::CallSlot::CallsConnected(void) const {

    /// TEMP Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
        }
    }
    return (!this->connected_calls.empty());
}


bool megamol::gui::graph::CallSlot::ConnectCall(megamol::gui::graph::CallPtrType call) {

    if (call == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->type == CallSlot::CallSlotType::CALLER) {
        if (this->connected_calls.size() > 0) {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Caller slots can only be connected to one call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            return false;
        }
    }
    this->connected_calls.emplace_back(call);
    return true;
}


bool megamol::gui::graph::CallSlot::DisConnectCall(int call_uid, bool called_by_call) {

    try {
        for (auto call_iter = this->connected_calls.begin(); call_iter != this->connected_calls.end(); call_iter++) {
            if ((*call_iter) == nullptr) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Call is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            } else {
                if ((*call_iter)->uid == call_uid) {
                    if (!called_by_call) {
                        (*call_iter)->DisConnectCallSlots();
                    }
                    (*call_iter).reset();
                    this->connected_calls.erase(call_iter);
                    return true;
                }
            }
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return false;
}


bool megamol::gui::graph::CallSlot::DisConnectCalls(void) {

    try {
        // Since connected calls operate on this list for disconnecting slots
        // a local copy of the connected calls is required.
        auto connected_calls_copy = this->connected_calls;
        for (auto& call_ptr : connected_calls_copy) {
            if (call_ptr == nullptr) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Call is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            } else {
                call_ptr->DisConnectCallSlots();
            }
        }
        this->connected_calls.clear();
        connected_calls_copy.clear();
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


const std::vector<megamol::gui::graph::CallPtrType>& megamol::gui::graph::CallSlot::GetConnectedCalls(void) {

    /// TEMP Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
        }
    }

    return this->connected_calls;
}


bool megamol::gui::graph::CallSlot::ParentModuleConnected(void) const { return (this->parent_module != nullptr); }


bool megamol::gui::graph::CallSlot::ConnectParentModule(megamol::gui::graph::ModulePtrType parent_module) {

    if (parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    if (this->parent_module != nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to parent module is already set. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module = parent_module;
    return true;
}


bool megamol::gui::graph::CallSlot::DisConnectParentModule(void) {

    if (parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to parent module is already nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module.reset();
    return true;
}


const megamol::gui::graph::ModulePtrType megamol::gui::graph::CallSlot::GetParentModule(void) {

    if (this->parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned pointer to parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->parent_module;
}


// CALL SLOT PRESENTATIONS ####################################################

megamol::gui::graph::CallSlot::Presentation::Presentation(void) : presentations(Presentation::DEFAULT) {}


megamol::gui::graph::CallSlot::Presentation::~Presentation(void) {}


bool megamol::gui::graph::CallSlot::Presentation::Present(CallSlot& call_slot) {
    return true;
}


void megamol::gui::graph::CallSlot::Presentation::UpdatePosition() {}
