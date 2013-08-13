/*
 * profiler/Manager.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "profiler/Manager.h"
#include "vislib/Log.h"
#include "vislib/Stack.h"
#include "AbstractNamedObjectContainer.h"
#include "CallerSlot.h"
#include "AbstractNamedObject.h"
#include "Call.h"

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
void profiler::Manager::SetModus(Modus modus) {
    if (this->modus != modus) {
        this->modus = modus;
        vislib::sys::Log::DefaultLog.WriteInfo("Profiler modus set to %d", static_cast<int>(modus));

        if (modus == PROFILE_NONE) {
            this->UnselectAll();
        } else if (modus == PROFILE_SELECTED) {
            this->UnselectAll();
            // The call selection is not persistant.
            // So we do not select anything here.
            // Could be changed in future
        } else {
            ASSERT(modus == PROFILE_ALL);
            // collect all calls
            vislib::Stack<const AbstractNamedObjectContainer*> stack;
            stack.Push(ci->ModuleGraphRoot());
            while (!stack.IsEmpty()) {
                const AbstractNamedObjectContainer* node = stack.Pop();
                vislib::ConstIterator<AbstractNamedObjectContainer::ChildList::Iterator> children = node->GetConstChildIterator();
                while (children.HasNext()) {
                    const AbstractNamedObject *child = children.Next();
                    const AbstractNamedObjectContainer *anoc = dynamic_cast<const AbstractNamedObjectContainer*>(child);
                    if (anoc != NULL) stack.Push(anoc); // continue
                    const CallerSlot *caller = dynamic_cast<const CallerSlot*>(child);
                    if (caller != NULL) {
                        Call *call = const_cast<CallerSlot*>(caller)->CallAs<Call>();
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

    // TODO: Implement

    vislib::sys::Log::DefaultLog.WriteInfo("All calls removed from profiling");
}


/*
 * profiler::Manager::Select
 */
void profiler::Manager::Select(const vislib::StringA& caller) {

    // TODO: Implement

    vislib::sys::Log::DefaultLog.WriteInfo("Call at %s added to profiling", caller.PeekBuffer());
}


/*
 * profiler::Manager::Manager
 */
profiler::Manager::Manager(void) : modus(PROFILE_NONE), ci(NULL) {

    // TODO: Implement

}


/*
 * profiler::Manager::~Manager
 */
profiler::Manager::~Manager(void) {
    this->ci = NULL; // Do not delete

    // TODO: Implement

}
