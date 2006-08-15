/*
 * DynamicLoadLibrary.cpp  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include <dlfcn.h>
#endif /* _WIN32 */

#include "vislib/DynamicLinkLibrary.h"
#include "vislib/IllegalParamException.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary
 */
vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary(void) : hModule(NULL) {
}


/*
 * vislib::sys::DynamicLinkLibrary::~DynamicLinkLibrary
 */
vislib::sys::DynamicLinkLibrary::~DynamicLinkLibrary(void) {
    this->Free();
}


/*
 * vislib::sys::DynamicLinkLibrary::Free
 */
void vislib::sys::DynamicLinkLibrary::Free(void) {
    if (this->hModule != NULL) {
#ifdef _WIN32
        if (::FreeLibrary(this->hModule) != TRUE) {
            throw SystemException(__FILE__, __LINE__);
        }
#else /* _WIN32 */
        int errorCode = ::dlclose(this->hModule);

        if (errorCode != 0) {
            throw SystemException(errorCode, __FILE__, __LINE__);
        }
#endif /* _WIN32 */
    }
}


/*
 * vislib::sys::DynamicLinkLibrary::GetProcAddress(
 */
#ifdef _WIN32
FARPROC vislib::sys::DynamicLinkLibrary::GetProcAddress(
        const CHAR *procName) const{
    return ::GetProcAddress(this->hModule, procName);
}
#else /* _WIN32 */
void *vislib::sys::DynamicLinkLibrary::GetProcAddress(
        const CHAR *procName) const {
    return ::dlsym(this->hModule, procName);
}
#endif /* _WIN32 */


/*
 * vislib::sys::DynamicLinkLibrary::Load
 */
bool vislib::sys::DynamicLinkLibrary::Load(const TCHAR *moduleName) {
    this->Free();

#ifdef _WIN32
    return ((this->hModule = ::LoadLibrary(moduleName)) != NULL);
#else /* _WIN32 */
    return ((this->hModule = ::dlopen(T2A(moduleName), RTLD_LAZY)) != NULL);
#endif /* _WIN32 */
}


/*
 * vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary
 */
vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary(
        const DynamicLinkLibrary& rhs) {
    throw UnsupportedOperationException(
        _T("vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary"), __FILE__, 
        __LINE__);
}


/*
 * vislib::sys::DynamicLinkLibrary::operator =
 */
vislib::sys::DynamicLinkLibrary& vislib::sys::DynamicLinkLibrary::operator =(
        const DynamicLinkLibrary& rhs) {
    if (this != &rhs) {
        throw IllegalParamException(_T("rhs"), __FILE__, __LINE__);
    }

    return *this;
}
