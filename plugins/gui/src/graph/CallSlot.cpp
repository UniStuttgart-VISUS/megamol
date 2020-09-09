/*
 * CallSlot.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallSlot.h"

#include "Call.h"
#include "InterfaceSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::CallSlot::CallSlot(ImGuiID uid)
    : uid(uid), present(), name(), description(), compatible_call_idxs(), type(), parent_module(), connected_calls() {}


megamol::gui::CallSlot::~CallSlot() {

    // Disconnects calls and parent module
    this->DisconnectCalls();
    this->DisconnectParentModule();
}


bool megamol::gui::CallSlot::CallsConnected(void) const {

    /// Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to one of the connected calls is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }
    }
    return (!this->connected_calls.empty());
}


bool megamol::gui::CallSlot::ConnectCall(const megamol::gui::CallPtr_t& call_ptr) {

    if (call_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to given call is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->type == CallSlotType::CALLER) {
        if (this->connected_calls.size() > 0) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Caller slots can only be connected to one call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            return false;
        }
    }
    this->connected_calls.emplace_back(call_ptr);
    return true;
}


bool megamol::gui::CallSlot::DisconnectCall(ImGuiID call_uid) {

    try {
        for (auto call_iter = this->connected_calls.begin(); call_iter != this->connected_calls.end(); call_iter++) {
            if ((*call_iter) != nullptr) {
                if ((*call_iter)->uid == call_uid) {
                    (*call_iter)->DisconnectCallSlots(this->uid);
                    (*call_iter).reset();
                    if (call_iter != this->connected_calls.end()) {
                        this->connected_calls.erase(call_iter);
                    }
                    return true;
                }
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


bool megamol::gui::CallSlot::DisconnectCalls(void) {

    try {
        for (auto& call_ptr : this->connected_calls) {
            if (call_ptr != nullptr) {
                call_ptr->DisconnectCallSlots(this->uid);
            }
        }
        this->connected_calls.clear();

    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


const std::vector<megamol::gui::CallPtr_t>& megamol::gui::CallSlot::GetConnectedCalls(void) {

    /// Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to one of the connected calls is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }
    }
    return this->connected_calls;
}


bool megamol::gui::CallSlot::IsParentModuleConnected(void) const { return (this->parent_module != nullptr); }


bool megamol::gui::CallSlot::ConnectParentModule(megamol::gui::ModulePtr_t parent_module) {

    if (parent_module == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to given parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->parent_module != nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to parent module is already set. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module = parent_module;
    return true;
}


bool megamol::gui::CallSlot::DisconnectParentModule(void) {

    if (parent_module == nullptr) {
#ifdef GUI_VERBOSE
/// megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Pointer to parent module is already nullptr. [%s, %s,
/// line %d]\n", __FILE__,
/// __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
        return false;
    }
    this->parent_module.reset();
    return true;
}


const megamol::gui::ModulePtr_t& megamol::gui::CallSlot::GetParentModule(void) {

    if (this->parent_module == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Returned pointer to parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    }
    return this->parent_module;
}


ImGuiID megamol::gui::CallSlot::GetCompatibleCallIndex(
    const CallSlotPtr_t& callslot_1, const CallSlotPtr_t& callslot_2) {

    if ((callslot_1 != nullptr) && (callslot_2 != nullptr)) {
        if (callslot_1->GetParentModule() != callslot_2->GetParentModule() && (callslot_1->type != callslot_2->type)) {
            // Return first found compatible call index
            for (auto& comp_call_idx_1 : callslot_1->compatible_call_idxs) {
                for (auto& comp_call_idx_2 : callslot_2->compatible_call_idxs) {
                    if (comp_call_idx_1 == comp_call_idx_2) {
                        return static_cast<ImGuiID>(comp_call_idx_1);
                    }
                }
            }
        }
    }
    return GUI_INVALID_ID;
}


ImGuiID megamol::gui::CallSlot::GetCompatibleCallIndex(
    const CallSlotPtr_t& callslot, const CallSlot::StockCallSlot& stock_callslot) {

    if (callslot != nullptr) {
        if (callslot->type != stock_callslot.type) {
            // Return first found compatible call index
            for (auto& comp_call_idx_1 : callslot->compatible_call_idxs) {
                for (auto& comp_call_idx_2 : stock_callslot.compatible_call_idxs) {
                    if (comp_call_idx_1 == comp_call_idx_2) {
                        return static_cast<ImGuiID>(comp_call_idx_1);
                    }
                }
            }
        }
    }
    return GUI_INVALID_ID;
}


bool megamol::gui::CallSlot::IsConnectionValid(CallSlot& callslot) {

    // Check for different type
    if (this->type == callslot.type) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots must have different types. [%s,
        /// %s, line %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for present parent module
    if ((callslot.GetParentModule() == nullptr) || (this->GetParentModule() == nullptr)) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots must have a connected parent
        /// module.
        /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for different parent module
    if ((this->GetParentModule()->uid == callslot.GetParentModule()->uid)) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots must have different parent
        /// modules. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for at least one found compatible call index
    for (auto& selected_comp_callslot : callslot.compatible_call_idxs) {
        for (auto& current_comp_callslots : this->compatible_call_idxs) {
            if (selected_comp_callslot == current_comp_callslots) {
                return true;
            }
        }
    }
    return false;
}
