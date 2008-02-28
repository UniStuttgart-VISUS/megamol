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
#include "vislib/IllegalStateException.h"
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
        this->hModule = NULL;
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
    // TODO: Error handling using dlerror
    return ::dlsym(this->hModule, procName);
}
#endif /* _WIN32 */


/*
 * vislib::sys::DynamicLinkLibrary::Load
 */
bool vislib::sys::DynamicLinkLibrary::Load(const char *moduleName, bool dontResolveReferences) {
    if (this->IsLoaded()) {
        throw IllegalStateException("Call to DynamicLinLibrary::Load() when"
            "a library was already loaded.", __FILE__, __LINE__);
    }

#ifdef _WIN32
    return ((this->hModule = ::LoadLibraryExA(moduleName, NULL, 
        (dontResolveReferences) ? DONT_RESOLVE_DLL_REFERENCES : 0)) != NULL);
#else /* _WIN32 */
    // TODO: Error handling using dlerror
    return ((this->hModule = ::dlopen(moduleName, RTLD_LAZY)) != NULL);
#endif /* _WIN32 */
}


/*
 * vislib::sys::DynamicLinkLibrary::Load
 */
bool vislib::sys::DynamicLinkLibrary::Load(const wchar_t *moduleName, bool dontResolveReferences) {
    if (this->IsLoaded()) {
        throw IllegalStateException("Call to DynamicLinLibrary::Load() when"
            "a library was already loaded.", __FILE__, __LINE__);
    }

#ifdef _WIN32
    return ((this->hModule = ::LoadLibraryExW(moduleName, NULL, 
        (dontResolveReferences) ? DONT_RESOLVE_DLL_REFERENCES : 0)) != NULL);
#else /* _WIN32 */
    // Because we know, that Linux does not support a chefmäßige Unicode-API.
    return this->Load(W2A(moduleName));
#endif /* _WIN32 */
}


/*
 * vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary
 */
vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary(
        const DynamicLinkLibrary& rhs) {
    throw UnsupportedOperationException(
        "vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary", __FILE__, 
        __LINE__);
}


/*
 * vislib::sys::DynamicLinkLibrary::operator =
 */
vislib::sys::DynamicLinkLibrary& vislib::sys::DynamicLinkLibrary::operator =(
        const DynamicLinkLibrary& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
