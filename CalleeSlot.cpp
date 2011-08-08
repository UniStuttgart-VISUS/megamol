/*
 * CalleeSlot.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CalleeSlot.h"
#include "AbstractNamedObject.h"
#include "AbstractNamedObjectContainer.h"
#include "CallerSlot.h"

using namespace megamol::core;


/*
 * CalleeSlot::CalleeSlot
 */
CalleeSlot::CalleeSlot(const vislib::StringA& name,
        const vislib::StringA& desc) : AbstractSlot(name, desc), callbacks() {
}


/*
 * CalleeSlot::~CalleeSlot
 */
CalleeSlot::~CalleeSlot(void) {
    for (unsigned int i = 0; i < this->callbacks.Count(); i++) {
        delete this->callbacks[i];
    }
    this->callbacks.Clear();
}


/*
 * CalleeSlot::InCall
 */
bool CalleeSlot::InCall(unsigned int func, Call& call) {
    if (func >= this->callbacks.Count()) return false;
    return this->callbacks[func]->CallMe(
        const_cast<Module*>(reinterpret_cast<const Module*>(this->Owner())),
        call);
}


/*
 * CalleeSlot::ClearCleanupMark
 */
void CalleeSlot::ClearCleanupMark(void) {
    if (!this->CleanupMark()) return;

    AbstractNamedObject::ClearCleanupMark();
    if (this->Parent() != NULL) {
        this->Parent()->ClearCleanupMark();
    }
}


/*
 * CalleeSlot::IsParamRelevant
 */
bool CalleeSlot::IsParamRelevant(
        vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
        const vislib::SmartPtr<param::AbstractParam>& param) const {
    if (searched.Contains(this)) {
        return false;
    } else {
        searched.Add(this);
    }

    this->LockModuleGraph(false);

    const AbstractNamedObject *ano = this->RootModule();
    const AbstractNamedObjectContainer *anoc
        = dynamic_cast<const AbstractNamedObjectContainer*>(ano);
    if (anoc == NULL) {
        this->UnlockModuleGraph(false);
        return false;
    }

    vislib::SingleLinkedList<const AbstractNamedObjectContainer *> heap;
    const CallerSlot* cs;
    heap.Append(anoc);
    while (!heap.IsEmpty()) {
        anoc = heap.First();
        heap.RemoveFirst();

        vislib::ConstIterator<AbstractNamedObjectContainer::ChildList::Iterator> iter
            = anoc->GetConstChildIterator();
        while (iter.HasNext()) {
            ano = iter.Next();
            anoc = dynamic_cast<const AbstractNamedObjectContainer*>(ano);
            if (anoc != NULL) {
                heap.Append(anoc);
                continue;
            }
            cs = dynamic_cast<const CallerSlot*>(ano);
            if ((cs != NULL) && cs->IsConnectedTo(this) && (ano->Parent() != NULL)) {
                if (ano->Parent()->IsParamRelevant(searched, param)) {
                    this->UnlockModuleGraph(false);
                    return true;
                }
            }
        }
    }

    this->UnlockModuleGraph(false);
    return false;
}
