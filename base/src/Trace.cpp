/*
 * Trace.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#include "vislib/Trace.h"

#include <climits>
#include <ctime>

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/memutils.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::Trace::GetInstance
 */
vislib::Trace& vislib::Trace::GetInstance(void) {
	return vislib::Trace::instance;
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
    ARY_SAFE_DELETE(this->filename);

    if (this->fp != NULL) {
        ::fclose(this->fp);
        this->fp = NULL;
    }
}


/*
 * vislib::Trace::EnableFileOutput
 */
bool vislib::Trace::EnableFileOutput(const char *filename) {
    bool retval = true;

    if (filename != NULL) {
        if ((this->filename == NULL) 
                || (::strcmp(this->filename, filename) != 0)) {
            ARY_SAFE_DELETE(this->filename);
            SIZE_T len = ::strlen(filename) + 1;
            this->filename = new char[len];
            ASSERT(this->filename != NULL); // std::bad_alloc should have been thrown.
            ::memcpy(this->filename, filename, len * sizeof(char));

            if (this->fp != NULL) {
                ::fclose(this->fp);
                this->fp = NULL;
            }
        } 
        /* 
         * 'this->filename' is new trace file, 'this->fp' is NULL or the 
         * correct file is still open.
         */
        
        if (this->fp == NULL) {
#ifdef _WIN32
#pragma warning(disable: 4996)
#endif /* _WIN32 */
            if ((retval = ((this->fp = ::fopen(this->filename, "w")) != NULL))) {
                time_t now;
                ::time(&now);
                ::fprintf(this->fp, "Trace file opened at %s", 
                    ::asctime(::localtime(&now)));
#ifdef _WIN32
#pragma warning(default: 4996)
#endif /* _WIN32 */
            }
        }

    } else {
        /* Disable tracing to file. */
        ARY_SAFE_DELETE(this->filename);
        
        if (this->fp != NULL) {
            ::fclose(this->fp);
            this->fp = NULL;
        }
    } /* end if (filename != NULL) */

    return retval;
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
vislib::Trace vislib::Trace::instance;


/*
 * vislib::Trace::Trace
 */
vislib::Trace::Trace(void) : filename(NULL), fp(NULL), level(LEVEL_ERROR) {
#if defined(DEBUG) || defined(_DEBUG)
    this->level = LEVEL_ALL;
#endif /* defined(DEBUG) || defined(_DEBUG) */
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
