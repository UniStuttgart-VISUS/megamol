/*
 * Call.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Call.h"

#include "CallSlot.h"
#include "InterfaceSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::Call::Call(ImGuiID uid)
    : uid(uid), present(), class_name(), description(), plugin_name(), functions(), connected_callslots() {

    this->connected_callslots.emplace(CallSlotType::CALLER, nullptr);
    this->connected_callslots.emplace(CallSlotType::CALLEE, nullptr);
}


megamol::gui::Call::~Call() {

    // Disconnect call slots
    this->DisconnectCallSlots();
}


bool megamol::gui::Call::IsConnected(void) {

    unsigned int connected = 0;
    for (auto& callslot_map : this->connected_callslots) {
        if (callslot_map.second != nullptr) {
            connected++;
        }
    }
    if (connected != 2) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Call has only one connected call slot. [%s,
        /// %s, line %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return (connected == 2);
}


bool megamol::gui::Call::ConnectCallSlots(
    megamol::gui::CallSlotPtr_t callslot_1, megamol::gui::CallSlotPtr_t callslot_2) {

    if ((callslot_1 == nullptr) || (callslot_2 == nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if ((this->connected_callslots[callslot_1->type] != nullptr) ||
        (this->connected_callslots[callslot_2->type] != nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Call is already connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (callslot_1->IsConnectionValid((*callslot_2))) {
        this->connected_callslots[callslot_1->type] = callslot_1;
        this->connected_callslots[callslot_2->type] = callslot_2;
        return true;
    }
    return false;
}


bool megamol::gui::Call::DisconnectCallSlots(ImGuiID calling_callslot_uid) {

    try {
        for (auto& callslot_map : this->connected_callslots) {
            if (callslot_map.second != nullptr) {
                if (callslot_map.second->uid != calling_callslot_uid) {
                    callslot_map.second->DisconnectCall(this->uid);
                }
                callslot_map.second.reset();
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
    return true;
}


const megamol::gui::CallSlotPtr_t& megamol::gui::Call::GetCallSlot(megamol::gui::CallSlotType type) {

    if (this->connected_callslots[type] == nullptr) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Returned pointer to call slot is nullptr. [%s,
        /// %s, line %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
    }
    return this->connected_callslots[type];
}
