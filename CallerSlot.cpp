/*
 * CallerSlot.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallerSlot.h"
#include "Call.h"
#include "CalleeSlot.h"

using namespace megamol::core;


/*
 * CallerSlot::CallerSlot
 */
CallerSlot::CallerSlot(const vislib::StringA& name,
        const vislib::StringA& desc) : AbstractSlot(name, desc),
        call(NULL), compDesc() {
    // intentionally empty
}


/*
 * CallerSlot::~CallerSlot
 */
CallerSlot::~CallerSlot(void) {
    SAFE_DELETE(this->call);
    for (unsigned int i = 0; i < this->compDesc.Count(); i++) {
        delete this->compDesc[i];
    }
    this->compDesc.Clear();
}


/*
 * CallerSlot::Call
 */
bool CallerSlot::Call(unsigned int func) {
    if (this->call != NULL) {
        try {
            return (*this->call)(func);
        } catch(...) {
        }
    }
    return false;
}


/*
 * CallerSlot::IsConnectedTo
 */
Call *CallerSlot::IsConnectedTo(CalleeSlot *target) {
    if ((this->call != NULL) && (this->call->PeekCalleeSlot() == target)) {
        return this->call;
    }
    return NULL;
}


/*
 * CallerSlot::IsConnectedTo
 */
const Call *CallerSlot::IsConnectedTo(const CalleeSlot *target) const {
    if ((this->call != NULL) && (this->call->PeekCalleeSlot() == target)) {
        return this->call;
    }
    return NULL;
}


/*
 * CallerSlot::ClearCleanupMark
 */
void CallerSlot::ClearCleanupMark(void) {
    if (!this->CleanupMark()) return;

    AbstractNamedObject::ClearCleanupMark();
    if ((this->call != NULL) && (this->call->PeekCalleeSlot() != NULL)) {
        const_cast<CalleeSlot *>(this->call->PeekCalleeSlot())->ClearCleanupMark();
    }
}


/*
 * CallerSlot::DisconnectCalls
 */
void CallerSlot::DisconnectCalls(void) {
    if (this->CleanupMark() && (this->call != NULL)) {
        this->SetStatusDisconnected();
        ::megamol::core::Call *c = this->call;
        this->call = NULL;
        delete c;
    }
}


/*
 * CallerSlot::IsParamRelevant
 */
bool CallerSlot::IsParamRelevant(
        vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
        const vislib::SmartPtr<param::AbstractParam>& param) const {

    if ((this->call != NULL) && (this->call->PeekCalleeSlot() != NULL)) {
        if (searched.Contains(this->call->PeekCalleeSlot())) {
            return false;
        } else {
            searched.Add(this->call->PeekCalleeSlot());
        }
        AbstractNamedObject *ano = const_cast<AbstractNamedObject *>(
            this->call->PeekCalleeSlot()->Parent());
        if (ano != NULL){
            return ano->IsParamRelevant(searched, param);
        }
    }

    return false;
}
