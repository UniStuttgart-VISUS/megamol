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
#include "vislib/Path.h"
#include "vislib/StringConverter.h"
#ifdef _WIN32
#include "vislib/SystemMessage.h"
#endif /* _WIN32 */
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary
 */
vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary(void) : hModule(NULL),
        loadErrorMsg() {
    // intentionally empty
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
            throw DLLException(__FILE__, __LINE__);
        }
#else /* _WIN32 */
        int errorCode = ::dlclose(this->hModule);

        if (errorCode != 0) {
            throw DLLException(::dlerror(), __FILE__, __LINE__);
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
    // Using an exception is probably not useful because of the error-code
    // incompatibility.
    return ::dlsym(this->hModule, procName);
}
#endif /* _WIN32 */


/*
 * vislib::sys::DynamicLinkLibrary::Load
 */
bool vislib::sys::DynamicLinkLibrary::Load(const char *moduleName,
        bool dontResolveReferences, bool alternateSearchPath) {
    this->loadErrorMsg.Clear();
    if (this->IsLoaded()) {
        this->loadErrorMsg = "Call to DynamicLinLibrary::Load() when"
            "a library was already loaded.";
        throw IllegalStateException(this->loadErrorMsg, __FILE__, __LINE__);
    }

#ifdef _WIN32
    DWORD flags = 0;
    UINT oldErrorMode = ::SetErrorMode(SEM_FAILCRITICALERRORS);
    if (dontResolveReferences) flags |= DONT_RESOLVE_DLL_REFERENCES;
    if (alternateSearchPath && vislib::sys::Path::IsAbsolute(moduleName)) {
        flags |= LOAD_WITH_ALTERED_SEARCH_PATH;
    }
    this->hModule = ::LoadLibraryExA(moduleName, NULL, flags);
    ::SetErrorMode(oldErrorMode);
    if (this->hModule == NULL) {
        this->loadErrorMsg = vislib::sys::SystemMessage(::GetLastError());
    }
#else /* _WIN32 */

    this->hModule = ::dlopen(moduleName, RTLD_LAZY | RTLD_GLOBAL);
    if (this->hModule == NULL) {
        this->loadErrorMsg = ::dlerror();
    }
#endif /* _WIN32 */
    return (this->hModule != NULL);
}


/*
 * vislib::sys::DynamicLinkLibrary::Load
 */
bool vislib::sys::DynamicLinkLibrary::Load(const wchar_t *moduleName,
        bool dontResolveReferences, bool alternateSearchPath) {
    this->loadErrorMsg.Clear();
    if (this->IsLoaded()) {
        this->loadErrorMsg = "Call to DynamicLinLibrary::Load() when"
            "a library was already loaded.";
        throw IllegalStateException(this->loadErrorMsg, __FILE__, __LINE__);
    }

#ifdef _WIN32
    DWORD flags = 0;
    UINT oldErrorMode = ::SetErrorMode(SEM_FAILCRITICALERRORS);
    if (dontResolveReferences) flags |= DONT_RESOLVE_DLL_REFERENCES;
    if (alternateSearchPath && vislib::sys::Path::IsAbsolute(moduleName)) {
        flags |= LOAD_WITH_ALTERED_SEARCH_PATH;
    }
    this->hModule = ::LoadLibraryExW(moduleName, NULL, flags);
    ::SetErrorMode(oldErrorMode);
    if (this->hModule == NULL) {
        this->loadErrorMsg = vislib::sys::SystemMessage(::GetLastError());
    }
#else /* _WIN32 */
    // Because we know, that Linux does not support a chefmäßige Unicode-API.
    if (!this->Load(W2A(moduleName))) return false;
#endif /* _WIN32 */
    return (this->hModule!= NULL);

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
