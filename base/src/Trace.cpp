/*
 * Trace.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#include "vislib/Trace.h"

#include <climits>

#include "vislib/IllegalParamException.h"
#include "vislib/UnsupportedOperationException.h"


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
 * vislib::Trace::LEVEL_ALL
 */
const UINT vislib::Trace::LEVEL_ALL = UINT_MAX;


/*
 * vislib::Trace::LEVEL_ERROR
 */
const UINT vislib::Trace::LEVEL_ERROR = 1;


/*
 * vislib::Trace::LEVEL_INFO
 */
const UINT vislib::Trace::LEVEL_INFO = 200;


/*
 * vislib::Trace::LEVEL_NONE
 */
const UINT vislib::Trace::LEVEL_NONE = 0;


/*
 * vislib::Trace::LEVEL_WARN
 */
const UINT vislib::Trace::LEVEL_WARN = 100;


/*
 * vislib::Trace::~Trace
 */
vislib::Trace::~Trace(void) {
    if (this->fp != NULL) {
        ::fclose(this->fp);
        this->fp = NULL;
    }
}


/*
 * vislib::Trace::EnableFileOutput
 */
bool vislib::Trace::EnableFileOutput(const char *filename) {
    if (this->fp != NULL) {
        ::fclose(this->fp);
    }

    if (filename != NULL) {
#if (_MSC_VER >= 1400)
        return (::fopen_s(&this->fp, filename, "w+") == 0);
#else /* (_MSC_VER >= 1400) */
        this->fp = ::fopen(filename, "w+");
        return (this->fp != NULL);
#endif /* (_MSC_VER >= 1400) */

    } else {
        this->fp = NULL;
        return true;
    }
}


/*
 * vislib::Trace::operator ()
 */
void vislib::Trace::operator ()(const UINT level, const char *fmt, ...) {
	va_list list;
	va_start(list, fmt);
	this->trace(level, fmt, list);
	va_end(list);
}


///*
// * vislib::Trace::operator ()
// */
//void vislib::Trace::operator ()(const char *fmt, ...) {
//	va_list list;
//	va_start(list, fmt);
//	this->trace(0, fmt, list);
//	va_end(list);
//}


/*
 * vislib::Trace::instance
 */
vislib::Trace *vislib::Trace::instance = NULL;


/*
 * vislib::Trace::Trace
 */
vislib::Trace::Trace(void) : fp(NULL), level(LEVEL_ERROR) {
}


/*
 * vislib::Trace::Trace
 */
vislib::Trace::Trace(const Trace& rhs) {
    throw UnsupportedOperationException("vislib::Trace::Trace", __FILE__, 
        __LINE__);
}


/*
 * vislib::Trace::trace
 */
void vislib::Trace::trace(const UINT level, const char *fmt, va_list list) {
	if ((level <= this->level) && (level > 0) && (fmt != NULL)) {
		::fprintf(stderr, "TRACE: ");
        ::vfprintf(stderr, fmt, list);
		::fflush(stderr);

        if (this->fp) {
            ::vfprintf(this->fp, fmt, list);
            ::fflush(this->fp);
        }
	}
}


/*
 * vislib::Trace::operator =
 */
vislib::Trace& vislib::Trace::operator =(const vislib::Trace &rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
