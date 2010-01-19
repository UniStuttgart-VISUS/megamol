/*
 * ParamSlot.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "param/ParamSlot.h"
#include "param/AbstractParam.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"

using namespace megamol::core;


/*
 * param::ParamSlot::ParamSlot
 */
param::ParamSlot::ParamSlot(const vislib::StringA& name,
        const vislib::StringA& desc) : AbstractSlot(name, desc),
        AbstractParamSlot(), callback(NULL) {
    // intentionally empty
}


/*
 * param::ParamSlot::~ParamSlot
 */
param::ParamSlot::~ParamSlot(void) {
    if (this->callback != NULL) {
        delete this->callback;
        this->callback = NULL;
    }
}


/*
 * param::ParamSlot::MakeAvailable
 */
void param::ParamSlot::MakeAvailable(void) {
    ASSERT(this->isParamSet());
    AbstractSlot::MakeAvailable();
}


/*
 * param::ParamSlot::IsParamRelevant
 */
bool param::ParamSlot::IsParamRelevant(
        vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
        const vislib::SmartPtr<param::AbstractParam>& param) const {
    return (this->Parameter() == param);
}


/*
 * param::ParamSlot::isSlotAvailable
 */
bool param::ParamSlot::isSlotAvailable(void) const {
    return (this->GetStatus() != AbstractSlot::STATUS_UNAVAILABLE);
}


/*
 * param::ParamSlot::update
 */
void param::ParamSlot::update(void) {
    bool oldDirty = this->IsDirty();
    AbstractParamSlot::update();
    if ((oldDirty != this->IsDirty()) && (this->callback != NULL)) {
        if (this->callback->Update(const_cast<Module*>(
                reinterpret_cast<const Module*>(this->Owner())), *this)) {
            this->ResetDirty();
        }
    }
}
