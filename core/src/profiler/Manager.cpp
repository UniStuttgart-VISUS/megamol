/*
 * profiler/Manager.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "mmcore/profiler/Manager.h"
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/AbstractNamedObjectContainer.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/Stack.h"
#include "vislib/sys/PerformanceCounter.h"

using namespace megamol;
using namespace megamol::core;


/*
 * profiler::Manager::Instance
 */
profiler::Manager& profiler::Manager::Instance(void) {
    static Manager man;
    return man;
}


/*
 * profiler::Manager::SetModus
 */
void profiler::Manager::SetMode(Mode mode) {
    if (this->mode != mode) {
        this->mode = mode;
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("Profiler modus set to %d", static_cast<int>(mode));

        if (mode == PROFILE_NONE) {
            this->UnselectAll();
        } else if (mode == PROFILE_SELECTED) {
            this->UnselectAll();
            // The call selection is not persistant.
            // So we do not select anything here.
            // Could be changed in future
        } else {
            ASSERT(mode == PROFILE_ALL);
            // collect all calls
            vislib::Stack<AbstractNamedObjectContainer::const_ptr_type> stack;
            stack.Push(ci->ModuleGraphRoot());
            while (!stack.IsEmpty()) {
                AbstractNamedObjectContainer::const_ptr_type node = stack.Pop();
                AbstractNamedObjectContainer::child_list_type::const_iterator children, childrenend;
                childrenend = node->ChildList_End();
                for (children = node->ChildList_Begin(); children != childrenend; ++children) {
                    AbstractNamedObject::const_ptr_type child = *children;
                    AbstractNamedObjectContainer::const_ptr_type anoc =
                        AbstractNamedObjectContainer::dynamic_pointer_cast(child);
                    if (anoc)
                        stack.Push(anoc); // continue
                    const CallerSlot* caller = dynamic_cast<const CallerSlot*>(child.get());
                    if (caller != NULL) {
                        Call* call = const_cast<CallerSlot*>(caller)->CallAs<Call>();
                        if (call != NULL) {
                            Select(caller->FullName());
                        }
                    }
                }
            }
        }
    }
}


/*
 * profiler::Manager::UnselectAll
 */
void profiler::Manager::UnselectAll(void) {
    this->connections.Lock();
    while (!this->connections.IsEmpty()) {
        const Call* call = this->connections[0]->get_call();
        this->connections.Unlock();
        const_cast<CalleeSlot*>(call->PeekCalleeSlot())->RemoveCallProfiling(call);
        this->connections.Lock();
    }
    this->connections.Unlock();
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("All calls removed from profiling");
}


/*
 * profiler::Manager::Select
 */
void profiler::Manager::Select(const vislib::StringA& caller) {
    const Call* call = NULL;

    vislib::Stack<AbstractNamedObjectContainer::const_ptr_type> stack;
    stack.Push(ci->ModuleGraphRoot());
    while (!stack.IsEmpty()) {
        AbstractNamedObjectContainer::const_ptr_type node = stack.Pop();
        AbstractNamedObjectContainer::child_list_type::const_iterator children, childrenend;
        childrenend = node->ChildList_End();
        for (children = node->ChildList_Begin(); children != childrenend; ++children) {
            AbstractNamedObject::const_ptr_type child = *children;
            AbstractNamedObjectContainer::const_ptr_type anoc =
                AbstractNamedObjectContainer::dynamic_pointer_cast(child);
            if (anoc)
                stack.Push(anoc); // continue
            const CallerSlot* callerSlot = dynamic_cast<const CallerSlot*>(child.get());
            if (callerSlot != NULL) {
                if (callerSlot->FullName().Equals(caller, false)) {
                    call = const_cast<CallerSlot*>(callerSlot)->CallAs<Call>();
                    if (call != NULL) {
                        if (call->PeekCalleeSlot() != NULL) {
                            break;
                        }
                    }
                }
            }
        }
    }

    if (call == NULL) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Failed to select call at %s for profiling: not found", caller.PeekBuffer());
        return;
    }
    ASSERT(call->PeekCalleeSlot() != NULL);
    ASSERT(call->PeekCallerSlot() != NULL);

    if (!call->PeekCalleeSlot()->IsCallProfiling(call)) {
        const_cast<CalleeSlot*>(call->PeekCalleeSlot())->AddCallProfiling(call);
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("Call at %s added to profiling", caller.PeekBuffer());
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "Call at %s not added to profiling: already profiling", caller.PeekBuffer());
    }
}


/*
 * profiler::Manager::AddConnection
 */
void profiler::Manager::AddConnection(Connection::ptr_type conn) {
    if (!this->connections.Contains(conn)) {
        this->connections.Add(conn);
    }
}


/*
 * profiler::Manager::RemoveConnection
 */
void profiler::Manager::RemoveConnection(Connection::ptr_type conn) {
    this->connections.RemoveAll(conn);
}


/*
 * profiler::Manager::Now
 */
double profiler::Manager::Now(void) const {
    UINT64 now = vislib::sys::PerformanceCounter::Query(true);
    UINT64 tc = now
#if defined(DEBUG) || defined(_DEBUG)
                - this->debugReportTime;
    if (this->mode != PROFILE_NONE) {
        if (vislib::sys::PerformanceCounter::ToMillis(tc) > 5000.0) {
            const_cast<Manager*>(this)->debugReportTime = now;
            const_cast<Manager*>(this)->Report();
        }
    }
    tc = now
#endif /* DEBUG || _DEBUG */
         - this->timeBase;
    return vislib::sys::PerformanceCounter::ToMillis(tc) * 0.001;
}


/*
 * profiler::Manager::Report
 */
void profiler::Manager::Report(void) {
    this->connections.Lock();
    try {

        // TODO: Implement some better reporting:

        if (!this->connections.IsEmpty()) {
            printf("Call Performance Profile:\n");
            for (SIZE_T i = 0; i < this->connections.Count(); i++) {
                Connection::ptr_type conn = this->connections[i];
                printf("\t%s(%u) = %f\n", conn->get_call()->PeekCallerSlot()->FullName().PeekBuffer(),
                    conn->get_function_id(), conn->get_mean());
            }
        }

        this->connections.Unlock();
    } catch (...) {
        this->connections.Unlock();
        throw;
    }
}


/*
 * profiler::Manager::Manager
 */
profiler::Manager::Manager(void) : mode(PROFILE_NONE), ci(NULL), connections(), timeBase(0), debugReportTime(0) {
    this->timeBase = vislib::sys::PerformanceCounter::Query(true);
}


/*
 * profiler::Manager::~Manager
 */
profiler::Manager::~Manager(void) {
    // this->UnselectAll(); Does not work, because calls will already be invalid. ...
    this->ci = NULL; // Do not delete
}
