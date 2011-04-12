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

#ifndef _WIN32
#include "vislib/Array.h"
#include "vislib/DirectoryIterator.h"
#endif /* !_WIN32 */
#include "vislib/DynamicLinkLibrary.h"
#ifndef _WIN32
#include "vislib/Environment.h"
#include "vislib/File.h"
#endif /* !_WIN32 */
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/Path.h"
#include "vislib/StringConverter.h"
#ifndef _WIN32
#include "vislib/StringTokeniser.h"
#endif /* !_WIN32 */
#ifdef _WIN32
#include "vislib/SystemMessage.h"
#endif /* _WIN32 */
#ifndef _WIN32
#include "vislib/Trace.h"
#endif /* !_WIN32 */
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

    this->hModule = ::dlopen(moduleName, RTLD_LAZY/* | RTLD_GLOBAL*/);
    if (this->hModule == NULL) {
        this->loadErrorMsg = ::dlerror();
        //if ((!vislib::StringA(moduleName).Contains("/"))
        //        && alternateSearchPath) {
        //    vislib::SingleLinkedList<void*> secMods;
        //    try {
        //        this->hModule = this->searchAndLoadCrowbar(moduleName, NULL,
        //            secMods);
        //    } catch (Exception ex) {
        //        VLTRACE(Trace::LEVEL_VL_INFO,
        //            "searchAndLoadCrowbar failed: %s (%s, %d)\n",
        //            ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        //        this->hModule = NULL;
        //    } catch (...) {
        //        VLTRACE(Trace::LEVEL_VL_INFO,
        //            "searchAndLoadCrowbar failed with unknown exception\n");
        //        this->hModule = NULL;
        //    }
        //}
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

//#ifndef _WIN32
//
///*
// * vislib::sys::DynamicLinkLibrary::loadCrowbar
// */
//void * vislib::sys::DynamicLinkLibrary::loadCrowbar(
//        const char *modName, const char *searchPath,
//        vislib::SingleLinkedList<void *>& secMods) {
//    VLTRACE(Trace::LEVEL_VL_INFO, "DynamicLinkLibrary::loadCrowbar(%s)\n", modName);
//    ASSERT((modName != NULL) && (modName[0] == '/')); // modName is an absolut path
//    ASSERT(File::Exists(modName)); // modName exists!
//
//    void *hndl = NULL;
//
//    while (hndl == NULL) {
//        hndl = ::dlopen(modName, RTLD_LAZY | RTLD_GLOBAL | RTLD_DEEPBIND);
//        if (hndl == NULL) {
//            StringA errMsg(::dlerror());
//            if (errMsg.EndsWith(": cannot open shared object file: No such file or directory")) {
//                StringA::Size pos = errMsg.Find(":");
//                ASSERT(pos != StringA::INVALID_POS);
//
//                StringA secMod = errMsg.Substring(0, pos);
//                void *secHndl = this->searchAndLoadCrowbar(secMod,
//                    Path::GetDirectoryName(secMod), secMods);
//                if (secHndl == NULL) {
//                    this->loadErrorMsg = errMsg;
//                    break;
//                }
//                secMods.Add(secHndl);
//
//            } else {
//                this->loadErrorMsg = errMsg;
//                break;
//            }
//        }
//    }
//
//    return hndl;
//}
//
//
///*
// * vislib::sys::DynamicLinkLibrary::searchAndLoadCrowbar
// */
//void * vislib::sys::DynamicLinkLibrary::searchAndLoadCrowbar(
//        const char *modName, const char *searchPath,
//        vislib::SingleLinkedList<void *>& secMods) {
//    VLTRACE(Trace::LEVEL_VL_INFO,
//        "DynamicLinkLibrary::searchAndLoadCrowbar(%s)\n", modName);
//    ASSERT(StringA(modName).Contains("/") == false);
//    ASSERT(StringA(modName).Contains("\\") == false);
//    void *hndl = NULL;
//
//    // 1. DT_RPATH and DT_RUNPATH are not of any interest for me!
//
//    // 2. LD_LIBRARY_PATH
//    StringA ldLibPath(Environment::GetVariable("LD_LIBRARY_PATH", true));
//    if (!ldLibPath.IsEmpty()) {
//        Array<StringA> ldLibPaths(StringTokeniserA::Split(ldLibPath, ':',
//            true));
//        for (SIZE_T i = 0; i < ldLibPaths.Count(); i++) {
//            StringA path(Path::Concatenate(ldLibPaths[i], modName));
//
//            if (!File::Exists(path)) continue;
//            hndl = this->loadCrowbar(path, Path::GetDirectoryName(path),
//                secMods);
//            if (hndl != NULL) return hndl;
//        }
//    }
//
//    // 3. DR_RUNPATH is not of any interest for me!
//
//    // 4. ask /etc/ld.so.cache
//    FILE *p = popen("/sbin/ldconfig -p", "r");
//    if (p != NULL) {
//        StringA cache;
//        const SIZE_T bufSize = 1024 * 1024;
//        char buf[bufSize + 1];
//        if (!feof(p)) {
//            fgets(buf, bufSize, p); // skip first line
//        }
//        StringA mng(modName);
//        mng.Append(" ");
//        while (!feof(p)) {
//            StringA line(fgets(buf, bufSize, p));
//            line.TrimSpaces();
//
//            if (line.StartsWith(mng)) {
//                StringA::Size pos = line.Find("=>");
//                if (pos != StringA::INVALID_POS) {
//                    StringA path(line.Substring(pos + 2));
//                    path.TrimSpacesBegin();
//                    path = Path::Resolve(path);
//                    if (File::Exists(path)) {
//                        hndl = this->loadCrowbar(path,
//                            Path::GetDirectoryName(path), secMods);
//                        if (hndl != NULL) return hndl;
//                    }
//                }
//            }
//
//        }
//        pclose(p);
//    }
//
//    // 5.1. search in '/lib' and below
//    SingleLinkedList<StringA> pathFifo;
//    pathFifo.Add("/lib");
//    while (!pathFifo.IsEmpty()) {
//        StringA basepath(pathFifo.First());
//        pathFifo.RemoveFirst();
//
//        StringA path(Path::Concatenate(basepath, modName));
//        if (File::Exists(path)) {
//            hndl = this->loadCrowbar(path, basepath, secMods);
//            if (hndl != NULL) return hndl;
//        }
//
//        try {
//            DirectoryIteratorA idir(basepath);
//            while (idir.HasNext()) {
//                DirectoryIteratorA::Entry& dir = idir.Next();
//                if (dir.Type == DirectoryIteratorA::Entry::DIRECTORY) {
//                    pathFifo.Append(Path::Concatenate(basepath, dir.Path));
//                }
//            }
//        } catch (...) {
//        }
//    }
//
//    // 5.2. search in '/usr/lib' and below
//    pathFifo.Add("/usr/lib");
//    while (!pathFifo.IsEmpty()) {
//        StringA basepath(pathFifo.First());
//        pathFifo.RemoveFirst();
//
//        StringA path(Path::Concatenate(basepath, modName));
//        if (File::Exists(path)) {
//            hndl = this->loadCrowbar(path, basepath, secMods);
//            if (hndl != NULL) return hndl;
//        }
//
//        try {
//            DirectoryIteratorA idir(basepath);
//            while (idir.HasNext()) {
//                DirectoryIteratorA::Entry& dir = idir.Next();
//                if (dir.Type == DirectoryIteratorA::Entry::DIRECTORY) {
//                    pathFifo.Append(Path::Concatenate(basepath, dir.Path));
//                }
//            }
//        } catch (...) {
//        }
//    }
//
//    if (searchPath == NULL) {
//        // 6. search in searchPath
//        StringA path(Path::Concatenate(searchPath, modName));
//
//        if (File::Exists(path)) {
//            hndl = this->loadCrowbar(path, Path::GetDirectoryName(path),
//                secMods);
//            if (hndl != NULL) return hndl;
//        }
//    }
//
//    {
//        // 7. search in cwd
//        StringA path(Path::Concatenate(Path::GetCurrentDirectoryA(), modName));
//
//        if (File::Exists(path)) {
//            hndl = this->loadCrowbar(path, Path::GetDirectoryName(path),
//                secMods);
//            if (hndl != NULL) return hndl;
//        }
//    }
//
//    // all has failed
//    return NULL;
//}
//
//#endif /* !_WIN32 */
