/*
 * Trace.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#include "vislib/Trace.h"

#include <climits>
#include <ctime>
#include <stdexcept>

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/memutils.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/vislibsymbolimportexport.inl"


/*
 * vislib::Trace::GetInstance
 */
vislib::Trace& vislib::Trace::GetInstance(void) {
    return *vislib::Trace::instance;
}


/*
 * vislib::Trace::OverrideInstance
 */
void vislib::Trace::OverrideInstance(vislib::Trace *inst) {
    ASSERT(inst != NULL);
    vislib::Trace::instance = inst; // no need to delete the old object
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
 * vislib::Trace::LEVEL_VL
 */
const UINT vislib::Trace::LEVEL_VL = 0x80000000;


/*
 * vislib::Trace::LEVEL_VL_ERROR
 */
const UINT vislib::Trace::LEVEL_VL_ERROR = vislib::Trace::LEVEL_VL + 1;


/*
 * vislib::Trace::LEVEL_VL_ANNOYINGLY_VERBOSE
 */
const UINT vislib::Trace::LEVEL_VL_ANNOYINGLY_VERBOSE = UINT_MAX;


/*
 * vislib::Trace::LEVEL_VL_INFO
 */
const UINT vislib::Trace::LEVEL_VL_INFO = vislib::Trace::LEVEL_VL + 200;


/*
 * vislib::Trace::LEVEL_VL_WARN
 */
const UINT vislib::Trace::LEVEL_VL_WARN = vislib::Trace::LEVEL_VL + 100;


/*
 * vislib::Trace::LEVEL_VL_VERBOSE
 */
const UINT vislib::Trace::LEVEL_VL_VERBOSE = vislib::Trace::LEVEL_VL + 10000;


/*
 * vislib::Trace::LEVEL_WARN
 */
const UINT vislib::Trace::LEVEL_WARN = 100;


/*
 * vislib::Trace::~Trace
 */
vislib::Trace::~Trace(void) {
    ARY_SAFE_DELETE(this->filename);
    this->SetPrefix(NULL);

    if (this->fp != NULL) {
        ::fclose(this->fp);
        this->fp = NULL;
    }
}


/*
 * vislib::Trace::EnableDebuggerOutput
 */
bool vislib::Trace::EnableDebuggerOutput(const bool useDebugger) {
    this->useDebugger = useDebugger;
    return true;
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
 * vislib::Trace::SetPrefix
 */
void vislib::Trace::SetPrefix(const char *prefix) {
    ARY_SAFE_DELETE(this->prefix);
    ASSERT(this->prefix == NULL);		// Ensure potential disabling.

    if (prefix != NULL) {
        SIZE_T len = ::strlen(prefix) + 1;
        this->prefix = new char[len];
        ASSERT(this->prefix != NULL);	// std::bad_alloc or OK.
        ::memcpy(this->prefix, prefix, len * sizeof(char));	
    }
}


/*
 * vislib::Trace::operator ()
 */
void vislib::Trace::operator ()(const UINT level, const char *fmt, ...) 
        throw() {
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
 * vislib::Trace::DEFAULT_PREFIX
 */
const char *vislib::Trace::DEFAULT_PREFIX = "TRACE: ";


/*
 * __vl_trace_instance
 */
VISLIB_STATICSYMBOL vislib::Trace __vl_trace_instance;


/*
 * vislib::Trace::instance
 */
vislib::Trace *vislib::Trace::instance(&__vl_trace_instance);


/*
 * vislib::Trace::Trace
 */
vislib::Trace::Trace(void) : filename(NULL), fp(NULL), prefix(NULL), 
        level(LEVEL_ERROR), useDebugger(true) {
#if defined(DEBUG) || defined(_DEBUG)
    this->level = LEVEL_ALL;
#endif /* defined(DEBUG) || defined(_DEBUG) */
    this->SetPrefix(DEFAULT_PREFIX);
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
void vislib::Trace::trace(const UINT level, const char *fmt, va_list list) 
        throw() {
    if ((level <= this->level) && (level > 0) && (fmt != NULL)) {
        if (this->prefix != NULL) {
            ::fprintf(stderr, this->prefix);
        }
        ::vfprintf(stderr, fmt, list);
        ::fflush(stderr);

        if (this->fp) {
            ::vfprintf(this->fp, fmt, list);
            ::fflush(this->fp);
        }

#ifdef _WIN32
        if (this->useDebugger) {
            try {
                int cnt = ::_vscprintf(fmt, list) + 1;
                char *tmp = new char[cnt];
#if (_MSC_VER >= 1400)
                ::_vsnprintf_s(tmp, cnt, cnt, fmt, list);
#else /* (_MSC_VER >= 1400) */
                ::vsnprintf(tmp, cnt, fmt, list);
#endif /* (_MSC_VER >= 1400) */
                
                ::OutputDebugStringA(tmp);
                ARY_SAFE_DELETE(tmp);
            } catch (std::bad_alloc) {
                ::fprintf(stderr, "OutputDebugStringA failed because of "
                    "insufficient system memory\n");
                ::fflush(stderr);
            }
        }
#endif /* _WIN32 */
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
