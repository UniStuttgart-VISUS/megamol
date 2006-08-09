/*
 * Trace.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */


#include <cstdio>

#include "vislib/IllegalParamException.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/Trace.h"


/*
 * vislib::Trace::GetInstance
 */
vislib::Trace& vislib::Trace::GetInstance(void) {
	if (vislib::Trace::instance == NULL) {
		vislib::Trace::instance = new Trace();
	}

	return *vislib::Trace::instance;
}


/*
 * vislib::Trace::~Trace
 */
vislib::Trace::~Trace(void) {
}


/*
 * vislib::Trace::operator ()
 */
void vislib::Trace::operator ()(const UINT level, const TCHAR *fmt, ...) {
	va_list list;
	va_start(list, fmt);
	this->trace(level, fmt, list);
	va_end(list);
}


/*
 * vislib::Trace::operator ()
 */
void vislib::Trace::operator ()(const TCHAR *fmt, ...) {
	va_list list;
	va_start(list, fmt);
	this->trace(0, fmt, list);
	va_end(list);
}


/*
 * vislib::Trace::instance
 */
vislib::Trace *vislib::Trace::instance = NULL;


/*
 * vislib::Trace::Trace
 */
vislib::Trace::Trace(void) : level(0) {
}


/*
 * vislib::Trace::Trace
 */
vislib::Trace::Trace(const Trace& rhs) {
    throw UnsupportedOperationException(_T("vislib::Trace::Trace"), __FILE__, 
        __LINE__);
}


/*
 * vislib::Trace::trace
 */
void vislib::Trace::trace(const UINT level, 
								const TCHAR *fmt, 
								va_list list) {
	if ((level <= this->level) && (fmt != NULL)) {
		::_ftprintf(stderr, _T("TRACE: "));
        ::_vftprintf(stderr, fmt, list);
		::fflush(stderr);
	}
}


/*
 * vislib::Trace::operator =
 */
vislib::Trace& vislib::Trace::operator =(const vislib::Trace &rhs) {
    if (this != &rhs) {
        throw IllegalParamException(_T("rhs"), __FILE__, __LINE__);
    }

    return *this;
}
