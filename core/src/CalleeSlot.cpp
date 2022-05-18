/*
 * CalleeSlot.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/CalleeSlot.h"
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/AbstractNamedObjectContainer.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/profiler/Manager.h"
#include "stdafx.h"

using namespace megamol::core;

/****************************************************************************/

/*
 * CalleeSlot::ProfilingCallback::ProfilingCallback
 */
CalleeSlot::ProfilingCallback::ProfilingCallback(Callback* cb, profiler::Connection::ptr_type conn)
        : Callback(nullptr, nullptr)
        , cb(cb)
        , conn(conn) {
    // intentionally empty
}


/*
 * CalleeSlot::ProfilingCallback::~ProfilingCallback
 */
CalleeSlot::ProfilingCallback::~ProfilingCallback(void) {
    this->cb = NULL; // do not delete
    this->conn.reset();
}


/*
 * CalleeSlot::ProfilingCallback::CallMe
 */
bool CalleeSlot::ProfilingCallback::CallMe(Module* owner, Call& call) {
    ASSERT(this->conn);
    this->conn->begin_measure();
    bool rv = this->cb->CallMe(owner, call);
    this->conn->end_measure();
    return rv;
}

/****************************************************************************/


/*
 * CalleeSlot::CalleeSlot
 */
CalleeSlot::CalleeSlot(const vislib::StringA& name, const vislib::StringA& desc)
        : AbstractSlot(name, desc)
        , AbstractCallSlotPresentation()
        , callbacks() {}


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
 * CalleeSlot::ConnectCall
 */
bool CalleeSlot::ConnectCall(megamol::core::Call* call, factories::CallDescription::ptr call_description) {
    vislib::sys::AutoLock lock(this->Parent()->ModuleGraphLock());
    if (call == NULL) {
        this->SetStatusDisconnected(); // TODO: This is wrong! Reference counting!
        return true;
    }

    factories::CallDescriptionManager::description_ptr_type desc;
    // for the new MegaMolGraph, we don't want to handle ConreInstances.
    // so now we pass the call_description, which the graph holds anyway, to satisfy the code filling call->funcMap[]
    if (call_description == nullptr) {
        core::CoreInstance& coreInst = *this->GetCoreInstance();
        for (unsigned int i = 0; i < this->callbacks.Count(); i++) {
            if ((desc = coreInst.GetCallDescriptionManager().Find(this->callbacks[i]->CallName()))->IsDescribing(call))
                break;
            desc.reset();
        }
        if (!desc) {
            return false;
        }
    } else {
        desc = call_description;
    }

    vislib::StringA cn(desc->ClassName());
    for (unsigned int i = 0; i < desc->FunctionCount(); i++) {
        vislib::StringA fn(desc->FunctionName(i));
        for (unsigned int j = 0; j < this->callbacks.Count(); j++) {
            if (cn.Equals(this->callbacks[j]->CallName(), false) && fn.Equals(this->callbacks[j]->FuncName(), false)) {
                call->funcMap[i] = j;
                break;
            }
        }
    }
    call->callee = this;
    this->SetStatusConnected();
    return true;
}


/*
 * CalleeSlot::InCall
 */
bool CalleeSlot::InCall(unsigned int func, Call& call) {
    if (func >= this->callbacks.Count())
        return false;
    return this->callbacks[func]->CallMe(const_cast<Module*>(reinterpret_cast<const Module*>(this->Owner())), call);
}


/*
 * CalleeSlot::ClearCleanupMark
 */
void CalleeSlot::ClearCleanupMark(void) {
    if (!this->CleanupMark())
        return;

    AbstractNamedObject::ClearCleanupMark();
    if (this->Parent() != NULL) {
        this->Parent()->ClearCleanupMark();
    }
}


/*
 * CalleeSlot::IsParamRelevant
 */
bool CalleeSlot::IsParamRelevant(vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
    const vislib::SmartPtr<param::AbstractParam>& param) const {
    if (searched.Contains(this)) {
        return false;
    } else {
        searched.Add(this);
    }

    vislib::sys::AutoLock lock(this->ModuleGraphLock());

    const_ptr_type ano = this->RootModule();
    AbstractNamedObjectContainer::const_ptr_type anoc = AbstractNamedObjectContainer::dynamic_pointer_cast(ano);
    if (!anoc) {
        return false;
    }

    vislib::SingleLinkedList<AbstractNamedObjectContainer::const_ptr_type> heap;
    const CallerSlot* cs;
    heap.Append(anoc);
    while (!heap.IsEmpty()) {
        anoc = heap.First();
        heap.RemoveFirst();

        AbstractNamedObjectContainer::child_list_type::const_iterator iter, end;
        iter = anoc->ChildList_Begin();
        end = anoc->ChildList_End();
        for (; iter != end; ++iter) {
            ano = *iter;
            anoc = std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
            if (anoc != NULL) {
                heap.Append(anoc);
                continue;
            }
            cs = dynamic_cast<const CallerSlot*>(ano.get());
            if ((cs != NULL) && cs->IsConnectedTo(this) && (ano->Parent() != NULL)) {
                if (ano->Parent()->IsParamRelevant(searched, param)) {
                    return true;
                }
            }
        }
    }

    return false;
}


/*
 * CalleeSlot::IsCallProfiling
 */
bool CalleeSlot::IsCallProfiling(const Call* c) const {
    ASSERT(c->PeekCalleeSlot() == this);

    // find call description to know how many functions are required
    factories::CallDescription::ptr desc = nullptr;
    for (unsigned int i = 0; i < this->callbacks.Count(); i++) {
        if ((desc = this->GetCoreInstance()->GetCallDescriptionManager().Find(this->callbacks[i]->CallName()))
                ->IsDescribing(c))
            break;
        desc = nullptr;
    }
    ASSERT(desc != nullptr);

    // number of function callbacks
    unsigned int func_cnt = desc->FunctionCount();
    if (func_cnt == 0)
        return false; // WTF?

    // test if at least one function callback is a profiling callback
    for (unsigned int i = 0; i < func_cnt; i++) {
        unsigned int func_idx = c->funcMap[i];
        ASSERT(func_idx < this->callbacks.Count());

        Callback* cb = this->callbacks[func_idx];
        if (dynamic_cast<ProfilingCallback*>(cb) != nullptr) {
            // Callback cb is a profiling callback
            return true;
        }
    }

    return false;
}


/*
 * CalleeSlot::AddCallProfiling
 */
void CalleeSlot::AddCallProfiling(const Call* c) {
    ASSERT(this->IsCallProfiling(c) == false);
    ASSERT(c->PeekCalleeSlot() == this);

    // find call description to know how many functions are required
    factories::CallDescription::ptr desc = nullptr;
    for (unsigned int i = 0; i < this->callbacks.Count(); i++) {
        if ((desc = this->GetCoreInstance()->GetCallDescriptionManager().Find(this->callbacks[i]->CallName()))
                ->IsDescribing(c))
            break;
        desc = nullptr;
    }
    ASSERT(desc != nullptr);

    // number of function callbacks
    unsigned int func_cnt = desc->FunctionCount();
    if (func_cnt == 0)
        return; // WTF?

    // Change callbacks to be profiling
    for (unsigned int i = 0; i < func_cnt; i++) {
        unsigned int func_idx = c->funcMap[i];
        ASSERT(func_idx < this->callbacks.Count());

        ProfilingCallback* pcb = new ProfilingCallback(
            this->callbacks[func_idx], profiler::Connection::ptr_type(new profiler::Connection()));
        this->callbacks.Add(pcb);
        c->funcMap[i] = static_cast<unsigned int>(this->callbacks.Count() - 1);
        pcb->GetConnection()->set_call(c);
        pcb->GetConnection()->set_function_id(i);

        profiler::Manager::Instance().AddConnection(pcb->GetConnection());
    }
}


/*
 * CalleeSlot::RemoveCallProfiling
 */
void CalleeSlot::RemoveCallProfiling(const Call* c) {
    ASSERT(c->PeekCalleeSlot() == this);

    // find call description to know how many functions are required
    factories::CallDescription::ptr desc = nullptr;
    for (unsigned int i = 0; i < this->callbacks.Count(); i++) {
        if ((desc = this->GetCoreInstance()->GetCallDescriptionManager().Find(this->callbacks[i]->CallName()))
                ->IsDescribing(c))
            break;
        desc = nullptr;
    }
    ASSERT(desc != nullptr);

    // number of function callbacks
    unsigned int func_cnt = desc->FunctionCount();
    if (func_cnt == 0)
        return; // WTF?

    // switch from profiling callback to normal callbacks
    for (unsigned int i = 0; i < func_cnt; i++) {
        unsigned int func_idx = c->funcMap[i];
        ASSERT(func_idx < this->callbacks.Count());

        Callback* cb = this->callbacks[func_idx];
        if (dynamic_cast<ProfilingCallback*>(cb) != nullptr) {
            // Callback cb is a profiling callback

            Callback* tcb = cb;
            while (dynamic_cast<ProfilingCallback*>(tcb) != nullptr)
                tcb = dynamic_cast<ProfilingCallback*>(tcb)->GetCallback();
            // HAZARD: we assume this loop is run exactly one time! In other cases memory leaks appear.

            // now tcb is the 'real' non-profiling callback
            unsigned int t_func_idx(static_cast<unsigned int>(this->callbacks.Count() + 1));
            for (unsigned int j = 0; j < this->callbacks.Count(); j++) {
                if (this->callbacks[j] == tcb) {
                    t_func_idx = j;
                    break;
                }
            }
            ASSERT(t_func_idx < this->callbacks.Count() - 1);

            // found new target
            c->funcMap[i] = t_func_idx;
            this->callbacks.Remove(cb); // remove old profiling callback
            profiler::Manager::Instance().RemoveConnection(dynamic_cast<ProfilingCallback*>(cb)->GetConnection());
            delete cb;
        }
    }
}
