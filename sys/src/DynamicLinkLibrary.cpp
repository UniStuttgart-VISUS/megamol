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
#include "the/argument_exception.h"
#include "the/invalid_operation_exception.h"
#include "vislib/Path.h"
#include "the/text/string_converter.h"
#ifdef _WIN32
#include "the/system/system_message.h"
#endif /* _WIN32 */
#include "the/not_supported_exception.h"


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
        const char *procName) const {
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
    this->loadErrorMsg.clear();
    if (this->IsLoaded()) {
        this->loadErrorMsg = "Call to DynamicLinLibrary::Load() when"
            "a library was already loaded.";
        throw the::invalid_operation_exception(this->loadErrorMsg.c_str(), __FILE__, __LINE__);
    }

#ifdef _WIN32
    unsigned int flags = 0;
    unsigned int oldErrorMode = ::SetErrorMode(SEM_FAILCRITICALERRORS);
    if (dontResolveReferences) flags |= DONT_RESOLVE_DLL_REFERENCES;
    if (alternateSearchPath && vislib::sys::Path::IsAbsolute(moduleName)) {
        flags |= LOAD_WITH_ALTERED_SEARCH_PATH;
    }
    this->hModule = ::LoadLibraryExA(moduleName, NULL, flags);
    ::SetErrorMode(oldErrorMode);
    if (this->hModule == NULL) {
        this->loadErrorMsg = the::astring(the::system::system_message(::GetLastError()).operator the::astring().c_str());
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
    this->loadErrorMsg.clear();
    if (this->IsLoaded()) {
        this->loadErrorMsg = "Call to DynamicLinLibrary::Load() when"
            "a library was already loaded.";
        throw the::invalid_operation_exception(this->loadErrorMsg.c_str(), __FILE__, __LINE__);
    }

#ifdef _WIN32
    unsigned int flags = 0;
    unsigned int oldErrorMode = ::SetErrorMode(SEM_FAILCRITICALERRORS);
    if (dontResolveReferences) flags |= DONT_RESOLVE_DLL_REFERENCES;
    if (alternateSearchPath && vislib::sys::Path::IsAbsolute(moduleName)) {
        flags |= LOAD_WITH_ALTERED_SEARCH_PATH;
    }
    this->hModule = ::LoadLibraryExW(moduleName, NULL, flags);
    ::SetErrorMode(oldErrorMode);
    if (this->hModule == NULL) {
        this->loadErrorMsg = static_cast<std::string>(the::system::system_message(::GetLastError())).c_str();
    }
#else /* _WIN32 */
    // Because we know, that Linux does not support a chefmäßige Unicode-API.
    if (!this->Load(THE_W2A(moduleName))) return false;
#endif /* _WIN32 */
    return (this->hModule!= NULL);

}


/*
 * vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary
 */
vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary(
        const DynamicLinkLibrary& rhs) {
    throw the::not_supported_exception(
        "vislib::sys::DynamicLinkLibrary::DynamicLinkLibrary", __FILE__, 
        __LINE__);
}


/*
 * vislib::sys::DynamicLinkLibrary::operator =
 */
vislib::sys::DynamicLinkLibrary& vislib::sys::DynamicLinkLibrary::operator =(
        const DynamicLinkLibrary& rhs) {
    if (this != &rhs) {
        throw the::argument_exception("rhs", __FILE__, __LINE__);
    }

    return *this;
}
