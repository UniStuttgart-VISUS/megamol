/*
 * Module.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Module.h"

#include "Call.h"
#include "CallSlot.h"
#include "InterfaceSlot.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::Module::Module(ImGuiID uid)
    : uid(uid)
    , present()
    , class_name()
    , description()
    , plugin_name()
    , is_view(false)
    , parameters()
    , name()
    , is_view_instance(false)
    , callslots() {

    this->callslots.emplace(megamol::gui::CallSlotType::CALLER, std::vector<CallSlotPtr_t>());
    this->callslots.emplace(megamol::gui::CallSlotType::CALLEE, std::vector<CallSlotPtr_t>());
}


megamol::gui::Module::~Module() {

    // Delete all call slots
    this->DeleteCallSlots();
}


bool megamol::gui::Module::AddCallSlot(megamol::gui::CallSlotPtr_t callslot) {

    if (callslot == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    auto type = callslot->type;
    for (auto& callslot_ptr : this->callslots[type]) {
        if (callslot_ptr == callslot) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to call slot already registered in modules call slot list. [%s, %s, line %d]\n",
                __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    }
    this->callslots[type].emplace_back(callslot);
    return true;
}


bool megamol::gui::Module::DeleteCallSlots(void) {

    try {
        for (auto& callslots_map : this->callslots) {
            for (auto callslot_iter = callslots_map.second.begin(); callslot_iter != callslots_map.second.end();
                 callslot_iter++) {
                (*callslot_iter)->DisconnectCalls();
                (*callslot_iter)->DisconnectParentModule();

                if ((*callslot_iter).use_count() > 1) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Unclean deletion. Found %i references pointing to call slot. [%s, %s, line %d]\n",
                        (*callslot_iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                }

                (*callslot_iter).reset();
            }
            callslots_map.second.clear();
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
    return true;
}


bool megamol::gui::Module::GetCallSlot(ImGuiID callslot_uid, megamol::gui::CallSlotPtr_t& out_callslot_ptr) {

    if (callslot_uid != GUI_INVALID_ID) {
        for (auto& callslot_map : this->GetCallSlots()) {
            for (auto& callslot : callslot_map.second) {
                if (callslot->uid == callslot_uid) {
                    out_callslot_ptr = callslot;
                    return true;
                }
            }
        }
    }
    return false;
}
