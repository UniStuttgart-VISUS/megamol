/*
 * CallerSlot.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/CallerSlot.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "stdafx.h"

using namespace megamol::core;


/*
 * CallerSlot::CallerSlot
 */
CallerSlot::CallerSlot(const vislib::StringA& name, const vislib::StringA& desc)
        : AbstractSlot(name, desc)
        , AbstractCallSlotPresentation()
        , call(NULL)
        , compDesc() {
    // intentionally empty
}


/*
 * CallerSlot::~CallerSlot
 */
CallerSlot::~CallerSlot(void) {
    SAFE_DELETE(this->call);
    this->compDesc.clear();
}


/*
 * CallerSlot::Call
 */
bool CallerSlot::Call(unsigned int func) {
    if (this->call != NULL) {
        try {
            return (*this->call)(func);
        } catch (...) {}
    }
    return false;
}


/*
 * CallerSlot::IsConnectedTo
 */
Call* CallerSlot::IsConnectedTo(CalleeSlot* target) {
    if ((this->call != NULL) && (this->call->PeekCalleeSlot() == target)) {
        return this->call;
    }
    return NULL;
}


/*
 * CallerSlot::IsConnectedTo
 */
const Call* CallerSlot::IsConnectedTo(const CalleeSlot* target) const {
    if ((this->call != NULL) && (this->call->PeekCalleeSlot() == target)) {
        return this->call;
    }
    return NULL;
}


/*
 * CallerSlot::ClearCleanupMark
 */
void CallerSlot::ClearCleanupMark(void) {
    if (!this->CleanupMark())
        return;

    AbstractNamedObject::ClearCleanupMark();
    if ((this->call != NULL) && (this->call->PeekCalleeSlot() != NULL)) {
        const_cast<CalleeSlot*>(this->call->PeekCalleeSlot())->ClearCleanupMark();
    }
}


/*
 * CallerSlot::DisconnectCalls
 */
void CallerSlot::DisconnectCalls(void) {
    if (this->CleanupMark() && (this->call != NULL)) {
        this->SetStatusDisconnected();
        //::megamol::core::Call *c = this->call;
        this->call->callee = nullptr;
        this->call->caller = nullptr;
        this->call = NULL;
        //delete c;
    }
}


/*
 * CallerSlot::IsParamRelevant
 */
bool CallerSlot::IsParamRelevant(vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
    const vislib::SmartPtr<param::AbstractParam>& param) const {

    if ((this->call != NULL) && (this->call->PeekCalleeSlot() != NULL)) {
        if (searched.Contains(this->call->PeekCalleeSlot())) {
            return false;
        } else {
            searched.Add(this->call->PeekCalleeSlot());
        }
        const_ptr_type ano = this->call->PeekCalleeSlot()->Parent();
        if (ano) {
            return ano->IsParamRelevant(searched, param);
        }
    }

    return false;
}
