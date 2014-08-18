/*
 * StackTrace.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/StackTrace.h"
#include "vislib/StringConverter.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/vislibsymbolimportexport.inl"
#include <cmath>
#include <cstdio>


/*
 * __vl_stacktrace_manager
 */
VISLIB_STATICSYMBOL vislib::SmartPtr<vislib::StackTrace>
    __vl_stacktrace_manager;


/*
 * vislib::StackTrace::manager
 */
vislib::SmartPtr<vislib::StackTrace>&
    vislib::StackTrace::manager(__vl_stacktrace_manager);


/*
 * vislib::StackTrace::Marker::Marker
 */
vislib::StackTrace::Marker::Marker(const char *functionName, const char *file,
        const int line) : id(0) {
    if (!vislib::StackTrace::manager.IsNull()) {
        this->id = vislib::StackTrace::manager->push(functionName, file, line);
    }
}


/*
 * vislib::StackTrace::Marker::~Marker
 */
vislib::StackTrace::Marker::~Marker(void) {
    if ((this->id != 0) && !vislib::StackTrace::manager.IsNull()) {
        vislib::StackTrace::manager->pop(this->id);
    }
}


/*
 * vislib::StackTrace::Marker::Marker
 */
vislib::StackTrace::Marker::Marker(const vislib::StackTrace::Marker& src) {
    throw vislib::UnsupportedOperationException("Forbidden copy ctor",
        __FILE__, __LINE__);
}


/*
 * vislib::StackTrace::Marker::operator=
 */
vislib::StackTrace::Marker& vislib::StackTrace::Marker::operator=(
        const vislib::StackTrace::Marker& rhs) {
    throw vislib::UnsupportedOperationException(
        "Forbidden assignment operator", __FILE__, __LINE__);
}


/*
 * vislib::StackTrace::GetStackString
 */
void vislib::StackTrace::GetStackString(char *str, unsigned int &strSize) {
    if (vislib::StackTrace::manager.IsNull()) {
        if (str == NULL) {
            strSize = 1; // including the terminating zero
        } else {
            if (strSize > 0) {
                *str = '\0';
                strSize = 1;
            }
        }
    } else {
        vislib::StackTrace::manager->getStackString(str, strSize);
    }
}


/*
 * vislib::StackTrace::GetStackString
 */
void vislib::StackTrace::GetStackString(wchar_t *str, unsigned int &strSize) {
    if (vislib::StackTrace::manager.IsNull()) {
        if (str == NULL) {
            strSize = 1; // including the terminating zero
        } else {
            if (strSize > 0) {
                *str = L'\0';
                strSize = 1;
            }
        }
    } else {
        vislib::StackTrace::manager->getStackString(str, strSize);
    }
}


/*
 * vislib::StackTrace::Initialise
 */
bool vislib::StackTrace::Initialise(
        vislib::SmartPtr<vislib::StackTrace> manager, bool force) {
    if (!vislib::StackTrace::manager.IsNull()) {
        if (vislib::StackTrace::manager == manager) {
            return true;
        }
        if (!force) {
            return false;
        }
    }
    if (manager.IsNull()) {
        vislib::StackTrace::manager = new vislib::StackTrace();
    } else {
        vislib::StackTrace::manager = manager;
    }
    return true;
}


/*
 * vislib::StackTrace::Manager
 */
vislib::SmartPtr<vislib::StackTrace> vislib::StackTrace::Manager(void) {
    return vislib::StackTrace::manager;
}


/*
 * vislib::StackTrace::~StackTrace
 */
vislib::StackTrace::~StackTrace(void) {
    while (this->stack != NULL) {
        StackElement *e = this->stack;
        this->stack = e->next;
        e->func = NULL; // DO NOT DELETE
        e->file = NULL; // DO NOT DELETE
        e->next = NULL;
        delete e;
    }
}


/*
 * vislib::StackTrace::StackTrace
 */
vislib::StackTrace::StackTrace(void) : stack(NULL) {
    // Intentionally empty
}


/*
 * vislib::StackTrace::getStackString
 */
void vislib::StackTrace::getStackString(char *str, unsigned int &strSize) {
    StackElement *stck = this->startUseStack();
    StackElement *e = stck;
    unsigned int s = 1; // terminating zero
    unsigned int l1, l2, l3;
    char buf[1024];
    while (e != NULL) {
        s += static_cast<unsigned int>(::strlen(e->func));
        s += static_cast<unsigned int>(::strlen(e->file));
        s += static_cast<int>(ceilf(log10f(static_cast<float>(e->line + 1))));
        if (e->line == 0) s++;
        s += 5; // '%func [%file:%line]\n' => five additional characters
        e = e->next;
    }

    if (str == NULL) {
        strSize = s;
        this->stopUseStack(stck);
        return;
    }

    if (strSize == 0) {
        this->stopUseStack(stck);
        return;
    }

    if (s > strSize) {
        s = strSize - 1; // for the terminating zero
    }
    strSize = 0;

    e = stck;
    while (e != NULL) {

        l1 = static_cast<unsigned int>(::strlen(e->func));
        l2 = static_cast<unsigned int>(::strlen(e->file));
        l3 = static_cast<int>(ceilf(log10f(static_cast<float>(e->line + 1))));
        if (e->line == 0) l3++;

        if ((l1 + l2 + l3 + 5) > (s - 1)) break; // skip rest of stack

        ::memcpy(str, e->func, l1);
        str += l1;
        strSize += l1;
        s -= l1;

        ::memcpy(str, " [", 2);
        str += 2;
        strSize += 2;
        s -= 2;

        ::memcpy(str, e->file, l2);
        str += l2;
        strSize += l2;
        s -= l2;

        ::memcpy(str, ":", 1);
        str += 1;
        strSize += 1;
        s -= 1;

        l1 = e->line;
        for (int l = l3 - 1; l >= 0; l--) { // itoa (argl!)
            buf[l] = static_cast<char>('0' + (l1 % 10));
            l1 = l1 / 10;
        }
        ::memcpy(str, buf, l3);
        str += l3;
        strSize += l3;
        s -= l3;

        ::memcpy(str, "]\n", 2);
        str += 2;
        strSize += 2;
        s -= 2;

        e = e->next;
    }

    *str = '\0';
    strSize += 1;

    this->stopUseStack(stck);
}


/*
 * vislib::StackTrace::getStackString
 */
void vislib::StackTrace::getStackString(wchar_t *str,
        unsigned int &strSize) {
    if (str != NULL) {
        char *t = new char[strSize];
        this->getStackString(t, strSize);
        ::memcpy(str, A2W(t), strSize * sizeof(wchar_t));
        delete[] t;
    } else {
        this->getStackString((char*)NULL, strSize);
    }
}


/*
 * vislib::StackTrace::pop
 */
void vislib::StackTrace::pop(int id) {
    StackElement *e = this->stack;
    if (e == NULL) {
        VLTRACE(VISLIB_TRCELVL_ERROR, "Unable to Pop: Trace Stack is empty.\n");
        return;
    }

    this->stack = e->next;
    e->next = NULL;

    if (id != static_cast<int>((SIZE_T)e)) {
        VLTRACE(VISLIB_TRCELVL_WARN, "StackTrace::pop: Stack seems corrupted.\n");
    }

    e->func = NULL; // DO NOT DELETE
    e->file = NULL; // DO NOT DELETE
    e->next = NULL;
    delete e;
}


/*
 * vislib::StackTrace::push
 */
int vislib::StackTrace::push(const char* func, const char* file,
         const int line) {
    StackElement *e = new StackElement;
    e->func = func;
    e->file = file;
    e->line = line;
    e->next = this->stack;
    this->stack = e;
    return static_cast<int>((SIZE_T)e);
}


/*
 * vislib::StackTrace::startUseStack
 */
vislib::StackTrace::StackElement* vislib::StackTrace::startUseStack(void) {
    return this->stack;
}


/*
 * vislib::StackTrace::stopUseStack
 */
void vislib::StackTrace::stopUseStack(
        vislib::StackTrace::StackElement* stack) {
    // intentionally empty
}
