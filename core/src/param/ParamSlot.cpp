/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/param/ParamSlot.h"

#include "mmcore/Module.h"
#include "mmcore/param/AbstractParam.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"

using namespace megamol::core;


/*
 * param::ParamSlot::ParamSlot
 */
param::ParamSlot::ParamSlot(const vislib::StringA& name, const vislib::StringA& desc)
        : AbstractSlot(name, desc)
        , AbstractParamSlot()
        , callback(NULL) {
    // intentionally empty
}


/*
 * param::ParamSlot::~ParamSlot
 */
param::ParamSlot::~ParamSlot() {
    if (this->callback != NULL) {
        delete this->callback;
        this->callback = NULL;
    }
}


/*
 * param::ParamSlot::MakeAvailable
 */
void param::ParamSlot::MakeAvailable() {
    ASSERT(this->isParamSet());
    AbstractSlot::MakeAvailable();
}


/*
 * param::ParamSlot::isSlotAvailable
 */
bool param::ParamSlot::isSlotAvailable() const {
    return (this->GetStatus() != AbstractSlot::STATUS_UNAVAILABLE);
}

/*
 * param:::ParamSlot::QueueUpdateNotification
 */
void param::ParamSlot::QueueUpdateNotification(bool force) {
    if (this->Param<AbstractParam>()->ConsumeHasChanged() || force) {
        Module* m = dynamic_cast<Module*>(this->Parent().get());
        // TODO what to with this after core instance is removed?
        //if ((m != nullptr) && (m->GetCoreInstance() != nullptr)) {
        //    m->GetCoreInstance()->ParameterValueUpdate(*this);
        //}
    }
}

/*
 * param::ParamSlot::update
 */
void param::ParamSlot::update() {
    bool oldDirty = this->IsDirty();
    AbstractParamSlot::update();

    QueueUpdateNotification(true);

    if (oldDirty != this->IsDirty()) {
        if ((this->callback != NULL) &&
            (this->callback->Update(const_cast<Module*>(reinterpret_cast<const Module*>(this->Owner())), *this))) {
            this->ResetDirty();
        }
    }
}
