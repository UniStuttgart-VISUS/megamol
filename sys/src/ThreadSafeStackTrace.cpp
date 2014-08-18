/*
 * ThreadSafeStackTrace.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ThreadSafeStackTrace.h"
#include "vislib/assert.h"


/*
 * vislib::sys::ThreadSafeStackTrace::GetStackString
 */
void vislib::sys::ThreadSafeStackTrace::GetStackString(char *outStr,
        unsigned int &strSize) {
    vislib::StackTrace::GetStackString(outStr, strSize);
}


/*
 * vislib::sys::ThreadSafeStackTrace::GetStackString
 */
void vislib::sys::ThreadSafeStackTrace::GetStackString(wchar_t *outStr,
        unsigned int &strSize) {
    vislib::StackTrace::GetStackString(outStr, strSize);
}


/*
 * vislib::sys::ThreadSafeStackTrace::Initialise
 */
bool vislib::sys::ThreadSafeStackTrace::Initialise(
        vislib::SmartPtr<vislib::StackTrace> manager, bool force) {
    if (manager.IsNull()) {
        return vislib::StackTrace::Initialise(new ThreadSafeStackTrace(),
            force);
    }
    return vislib::StackTrace::Initialise(manager, force);
}


/*
 * vislib::sys::ThreadSafeStackTrace::Manager
 */
vislib::SmartPtr<vislib::StackTrace>
vislib::sys::ThreadSafeStackTrace::Manager(void) {
    return vislib::StackTrace::Manager();
}


/*
 * vislib::sys::ThreadSafeStackTrace::~ThreadSafeStackTrace
 */
vislib::sys::ThreadSafeStackTrace::~ThreadSafeStackTrace(void) {
    this->critSect.Lock();
    this->valid = false;
    ThreadStackRoot *r;
    StackElement *e;
    while (this->stacks) {
        r = this->stacks;
        this->stacks = r->next;
        while (r->stack) {
            e = r->stack;
            r->stack = e->next;
            e->func = NULL; // DO NOT DELETE
            e->file = NULL; // DO NOT DELETE
            e->next = NULL;
            delete e;
        }
        delete r;
    }
    this->critSect.Unlock();
}


/*
 * vislib::sys::ThreadSafeStackTrace::ThreadSafeStackTrace
 */
vislib::sys::ThreadSafeStackTrace::ThreadSafeStackTrace(void) : StackTrace(),
        critSect(), stacks(NULL), valid(true) {
    // Intentionally empty
}


/*
 * vislib::sys::ThreadSafeStackTrace::pop
 */
void vislib::sys::ThreadSafeStackTrace::pop(int id) {
    this->critSect.Lock();
    if (!this->valid) return;

    DWORD ctid = vislib::sys::Thread::CurrentID();

    ThreadStackRoot *r = this->stacks;
    while (r != NULL) {
        if (r->id == ctid) {

            this->stack = r->stack;
            StackTrace::pop(id);
            r->stack = this->stack;
            this->stack = NULL;

            if (r->stack == NULL) {
                if (this->stacks == r) {
                    this->stacks = r->next;
                } else {
                    ThreadStackRoot *p = this->stacks;
                    while ((p->next != NULL) && (p->next != r)) {
                        p = p->next;
                    }
                    ASSERT(p->next == r);
                    p->next = r->next;
                }
                delete r;
            }

            break;
        }
        r = r->next;
    }

    this->critSect.Unlock();
}


/*
 * vislib::sys::ThreadSafeStackTrace::push
 */
int vislib::sys::ThreadSafeStackTrace::push(const char* func, const char* file, const int line) {
    int retval = 0;
    this->critSect.Lock();
    if (!this->valid) return 0;

    DWORD ctid = vislib::sys::Thread::CurrentID();
    ThreadStackRoot *root = NULL;
    ThreadStackRoot *r = this->stacks;
    while (r != NULL) {
        if (r->id == ctid) {
            root = r;
            break;
        }
        r = r->next;
    }
    if (root == NULL) {
        root = new ThreadStackRoot;
        root->id = ctid;
        root->stack = NULL;
        root->next = this->stacks;
        this->stacks = root;
    }

    this->stack = root->stack;
    retval = StackTrace::push(func, file, line);
    root->stack = this->stack;
    this->stack = NULL;

    this->critSect.Unlock();
    return retval;
}


/*
 * vislib::sys::ThreadSafeStackTrace::startUseStack
 */
vislib::StackTrace::StackElement*
vislib::sys::ThreadSafeStackTrace::startUseStack(void) {
    this->critSect.Lock();
    if (!this->valid) return NULL;

    DWORD ctid = vislib::sys::Thread::CurrentID();
    ThreadStackRoot *r = this->stacks;
    while (r != NULL) {
        if (r->id == ctid) {
            return r->stack;
        }
        r = r->next;
    }

    return NULL;
}


/*
 * vislib::sys::ThreadSafeStackTrace::stopUseStack
 */
void vislib::sys::ThreadSafeStackTrace::stopUseStack(
        vislib::StackTrace::StackElement* stack) {
    // stack could be used for sanity check here
    this->critSect.Unlock();
}
