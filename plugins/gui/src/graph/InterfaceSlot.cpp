/*
 * InterfaceSlot.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "InterfaceSlot.h"

#include "Call.h"
#include "CallSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::InterfaceSlot::InterfaceSlot(ImGuiID uid, bool auto_create) : uid(uid), auto_created(auto_create) {}


megamol::gui::InterfaceSlot::~InterfaceSlot(void) {

    // Remove all call slots from interface slot
    std::vector<ImGuiID> callslots_uids;
    for (auto& callslot_ptr : this->callslots) {
        callslots_uids.emplace_back(callslot_ptr->uid);
    }
    for (auto& callslot_uid : callslots_uids) {
        this->RemoveCallSlot(callslot_uid);
    }
    this->callslots.clear();
}


bool megamol::gui::InterfaceSlot::AddCallSlot(
    const CallSlotPtr_t& callslot_ptr, const InterfaceSlotPtr_t& parent_interfaceslot_ptr) {

    try {
        if (callslot_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        if (parent_interfaceslot_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to interface slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        if (this->is_callslot_compatible((*callslot_ptr))) {
            this->callslots.emplace_back(callslot_ptr);

            callslot_ptr->present.group.interfaceslot_ptr = parent_interfaceslot_ptr;
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Added call slot '%s' to interface slot of group.\n", callslot_ptr->name.c_str());
#endif // GUI_VERBOSE
            return true;
        }
    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slot '%s' is incompatible to interface slot
    /// of group. [%s, %s, line %d]\n", callslot_ptr->name.c_str(), __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::InterfaceSlot::RemoveCallSlot(ImGuiID callslot_uid) {

    try {
        for (auto iter = this->callslots.begin(); iter != this->callslots.end(); iter++) {
            if ((*iter)->uid == callslot_uid) {

                (*iter)->present.group.interfaceslot_ptr = nullptr;
#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[GUI] Removed call slot '%s' from interface slot of group.\n", (*iter)->name.c_str());
#endif // GUI_VERBOSE
                (*iter).reset();
                this->callslots.erase(iter);

                return true;
            }
        }
    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return false;
}


bool megamol::gui::InterfaceSlot::ContainsCallSlot(ImGuiID callslot_uid) {

    for (auto& callslot : this->callslots) {
        if (callslot_uid == callslot->uid) {
            return true;
        }
    }
    return false;
}


bool megamol::gui::InterfaceSlot::IsConnectionValid(InterfaceSlot& interfaceslot) {

    CallSlotPtr_t callslot_ptr_1;
    CallSlotPtr_t callslot_ptr_2;
    if (this->GetCompatibleCallSlot(callslot_ptr_1) && interfaceslot.GetCompatibleCallSlot(callslot_ptr_2)) {
        // Check for different group
        if (this->present.group.uid != interfaceslot.present.group.uid) {
            // Check for compatibility of call slots which are part of the interface slots
            return (callslot_ptr_1->IsConnectionValid((*callslot_ptr_2)));
        }
    }

    return false;
}


bool megamol::gui::InterfaceSlot::IsConnectionValid(CallSlot& callslot) {

    if (this->is_callslot_compatible(callslot)) {
        return true;
    } else {
        // Call slot can only be added if parent module is not part of same group
        if (callslot.GetParentModule() == nullptr) {
            /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots must have connceted parent
            /// module.
            /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        if (callslot.GetParentModule()->present.group.uid == this->present.group.uid) {
            /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Parent module of call slot should not be
            /// in same group as the interface. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        // Check for compatibility of call slots
        CallSlotPtr_t interface_callslot_ptr;
        if (this->GetCompatibleCallSlot(interface_callslot_ptr)) {
            if (interface_callslot_ptr->IsConnectionValid(callslot)) {
                return true;
            }
        }
    }
    return false;
}


bool megamol::gui::InterfaceSlot::GetCompatibleCallSlot(CallSlotPtr_t& out_callslot_ptr) {

    out_callslot_ptr.reset();
    if (!this->callslots.empty()) {
        out_callslot_ptr = this->callslots[0];
        return true;
    }
    return false;
}


bool megamol::gui::InterfaceSlot::IsConnected(void) {

    for (auto& callslot_ptr : this->callslots) {
        if (callslot_ptr->CallsConnected()) {
            return true;
        }
    }
    return false;
}


CallSlotType megamol::gui::InterfaceSlot::GetCallSlotType(void) {

    CallSlotType ret_type = CallSlotType::CALLER;
    if (!this->callslots.empty()) {
        return this->callslots[0]->type;
    }
    return ret_type;
}


bool megamol::gui::InterfaceSlot::IsEmpty(void) { return (this->callslots.empty()); }


bool megamol::gui::InterfaceSlot::is_callslot_compatible(CallSlot& callslot) {

    // Callee interface slots can only have one call slot
    if (this->callslots.size() > 0) {
        if ((this->GetCallSlotType() == CallSlotType::CALLEE)) {
            /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Callee interface slots can only have one
            /// call slot connceted.
            /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    }
    // Call slot can only be added if not already part of this interface
    if (this->ContainsCallSlot(callslot.uid)) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots can only be added if not already
        /// part of this interface.
        /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Call slot can only be added if not already part of other interface
    if (callslot.present.group.interfaceslot_ptr != nullptr) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots can only be added if not already
        /// part of other interface. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Call slot can only be added if parent module is part of same group
    if (callslot.GetParentModule() == nullptr) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots must have connceted parent module.
        /// [%s, %s, line %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (callslot.GetParentModule()->present.group.uid != this->present.group.uid) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Parent module of call slot should be in same
        /// group as the interface. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for compatibility (with all available call slots...)
    size_t compatible_slot_count = 0;
    for (auto& interface_callslot_ptr : this->callslots) {
        // Check for same type and same compatible call indices
        if ((callslot.type == interface_callslot_ptr->type) &&
            (callslot.compatible_call_idxs == interface_callslot_ptr->compatible_call_idxs)) {
            compatible_slot_count++;
        }
    }
    bool compatible = (compatible_slot_count == this->callslots.size());
    // Check for existing incompatible call slots
    if ((compatible_slot_count > 0) && (compatible_slot_count != this->callslots.size())) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Interface slot contains incompatible call
        /// slots.
        /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return compatible;
}
