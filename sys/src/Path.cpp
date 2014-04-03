/*
 * Path.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/Path.h"

#ifdef _WIN32
#include <Shlobj.h>
#include <Shlwapi.h>
#include <windows.h>
#else /* _WIN32 */
#include <climits>
#include <unistd.h>
#include <cstdlib> // for getenv
#include <sys/stat.h>
#include <sys/types.h>
#endif /* _WIN32 */

#include "the/assert.h"
#include "vislib/error.h"
#include "vislib/Environment.h"
#include "the/memory.h"
#include "the/system/system_exception.h"
#include "the/not_implemented_exception.h"
#include "vislib/Stack.h"
#include "vislib/File.h"
#include "vislib/DirectoryIterator.h"
#include "the/string.h"
#include "the/text/string_converter.h"


/*
 * vislib::sys::Path::Canonicalise
 */
the::astring vislib::sys::Path::Canonicalise(const the::astring& path) {
    const the::astring DOUBLE_SEPARATOR(2, Path::SEPARATOR_A);

#ifdef _WIN32
    the::astring retval(MAX_PATH, ' ');

    if (::PathCanonicalizeA(const_cast<char*>(retval.c_str()), path.c_str()) != TRUE) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    the::text::string_utility::replace(retval, DOUBLE_SEPARATOR, the::astring(1, SEPARATOR_A));
    the::text::string_utility::replace(retval, DOUBLE_SEPARATOR, the::astring(1, SEPARATOR_A));

    /* Ensure that a UNC path remains a UNC path. */
    if (the::text::string_utility::starts_with(path, DOUBLE_SEPARATOR)) {
        // Note: Double separator replacement above leaves at least one 
        // separator, so we must only prepend one additional one.
        retval.insert(0, 1, SEPARATOR_A);
    }

    return retval;

#else /* _WIN32 */
    const char *BACK_REF = "/..";
    const char *CUR_REF = "/.";             // Note: "./" does not work
    the::astring::size_type BACK_REF_LEN = ::strlen(BACK_REF);
    the::astring::size_type bwRefPos = 0;             // Position of back reference.
    the::astring::size_type remDirPos = 0;            // Position of directory to erase.
    the::astring retval(path);
    

    /* Remove backward references, first. */
    while ((bwRefPos = retval.find(BACK_REF)) != the::astring::npos) {

        if ((bwRefPos > 0) 
                && (remDirPos = retval.rfind(SEPARATOR_A, bwRefPos - 1))
                != the::astring::npos) {
            /* Found inner backward reference, so remove some parts. */
            retval.Remove(remDirPos, bwRefPos - remDirPos + BACK_REF_LEN);

        } else {
            /* 
             * No other path separator is before this one, so we can remove
             * everything before 'bwRefPos'.
             */
            retval.Remove(0, bwRefPos + BACK_REF_LEN);
        }
    }

    /*
     * Remove references to the current directory. This must be done after
     * removing backward references.
     */
    retval.Remove(CUR_REF);
    
    /* Remove odd and even number of repeated path separators. */
    the::text::string_utility::replace(retval, DOUBLE_SEPARATOR.c_str(), SEPARATOR_A);
    the::text::string_utility::replace(retval, DOUBLE_SEPARATOR.c_str(), SEPARATOR_A);

    return retval;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::Canonicalise
 */
the::wstring vislib::sys::Path::Canonicalise(const the::wstring& path) {
    const the::wstring DOUBLE_SEPARATOR(2, Path::SEPARATOR_W);

#ifdef _WIN32
    the::wstring retval(MAX_PATH, L' ');

    if (::PathCanonicalizeW(const_cast<wchar_t*>(retval.c_str()), path.c_str()) != TRUE) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    the::text::string_utility::replace(retval, DOUBLE_SEPARATOR, the::wstring(1, SEPARATOR_W));
    the::text::string_utility::replace(retval, DOUBLE_SEPARATOR, the::wstring(1, SEPARATOR_W));

    /* Ensure that a UNC path remains a UNC path. */
    if (the::text::string_utility::starts_with(path, DOUBLE_SEPARATOR)) {
        // Note: Double separator replacement above leaves at least one 
        // separator, so we must only prepend one additional one.
        retval.insert(0, 1, SEPARATOR_W);
    }

    return retval;

#else /* _WIN32 */
    const wchar_t *BACK_REF = L"/..";
    const wchar_t *CUR_REF = L"/.";         // Note: "./" does not work
    the::wstring::size_type BACK_REF_LEN = ::wcslen(BACK_REF);
    the::wstring::size_type bwRefPos = 0;             // Position of back reference.
    the::wstring::size_type remDirPos = 0;            // Position of directory to erase.
    the::wstring retval(path);
    

    /* Remove backward references, first. */
    while ((bwRefPos = retval.find(BACK_REF)) != the::wstring::npos) {

        if ((bwRefPos > 0) 
                && (remDirPos = retval.rfind(SEPARATOR_W, bwRefPos - 1))
                != the::wstring::npos) {
            /* Found inner backward reference, so remove some parts. */
            retval.Remove(remDirPos, bwRefPos - remDirPos + BACK_REF_LEN);

        } else {
            /* 
             * No other path separator is before this one, so we can remove
             * everything before 'bwRefPos'.
             */
            retval.Remove(0, bwRefPos + BACK_REF_LEN);
        }
    }

    /*
     * Remove references to the current directory. This must be done after
     * removing backward references.
     */
    retval.Remove(CUR_REF);
    
    /* Remove odd and even number of repeated path separators. */
    the::text::string_utility::replace(retval, DOUBLE_SEPARATOR.c_str(), SEPARATOR_W);
    the::text::string_utility::replace(retval, DOUBLE_SEPARATOR.c_str(), SEPARATOR_W);

    return retval;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::ChangeExtension
 */
the::astring vislib::sys::Path::ChangeExtension(const char *path,
        const char *extension) {
    const char EXT_SEP = '.';
    the::astring retval(path);
    the::astring::size_type extStart = retval.rfind(EXT_SEP);
    the::astring ext(extension);

    if (!ext.empty() && !the::text::string_utility::starts_with(ext, EXT_SEP)) {
        ext.insert(0, 1, EXT_SEP);
    }

    if (extStart != the::astring::npos) {
        /* Found a valid extension, remove that one before adding new. */
        retval.resize(extStart);
    }

    retval += ext;
    return retval;
}


/*
 * vislib::sys::Path::ChangeExtension
 */
the::wstring vislib::sys::Path::ChangeExtension(const wchar_t *path,
        const wchar_t *extension) {
    const wchar_t EXT_SEP = '.';
    the::wstring retval(path);
    the::wstring::size_type extStart = retval.rfind(EXT_SEP);
    the::wstring ext(extension);

    if (!ext.empty() && !the::text::string_utility::starts_with(ext, EXT_SEP)) {
        ext.insert(0, 1, EXT_SEP);
    }

    if (extStart != the::wstring::npos) {
        /* Found a valid extension, remove that one before adding new. */
        retval.resize(extStart);
    }

    retval += ext;
    return retval;
}


/*
 * vislib::sys::Path::Concatenate
 */
the::astring vislib::sys::Path::Concatenate(const the::astring& lhs,
        const the::astring& rhs, const bool canonicalise) {
    the::astring retval(lhs);

    if (the::text::string_utility::ends_with(lhs, SEPARATOR_A) && the::text::string_utility::starts_with(rhs, SEPARATOR_A)) {
        retval.append(rhs.c_str() + 1);

    } else if (!the::text::string_utility::ends_with(lhs, SEPARATOR_A) && !the::text::string_utility::starts_with(rhs, SEPARATOR_A)) {
        retval.append(1, SEPARATOR_A);
        retval.append(rhs);

    } else {
        retval.append(rhs);
    }

    return canonicalise ? Path::Canonicalise(retval) : retval;
}


/*
 * vislib::sys::Path::Concatenate
 */
the::wstring vislib::sys::Path::Concatenate(const the::wstring& lhs, 
        const the::wstring& rhs, const bool canonicalise) {
    the::wstring retval(lhs);

    if (the::text::string_utility::ends_with(lhs, SEPARATOR_W) && the::text::string_utility::starts_with(rhs, SEPARATOR_W)) {
        retval.append(rhs.c_str() + 1);

    } else if (!the::text::string_utility::ends_with(lhs, SEPARATOR_W) && !the::text::string_utility::starts_with(rhs, SEPARATOR_W)) {
        retval.append(1, SEPARATOR_W);
        retval.append(rhs);

    } else {
        retval.append(rhs);
    }

    return canonicalise ? Path::Canonicalise(retval) : retval;
}


/*
 * vislib::sys::Path::DeleteDirectory
 */
void vislib::sys::Path::DeleteDirectory(const the::astring& path, bool recursive) {
    if (!File::Exists(path.c_str())) return; // we don't delete non-existing stuff
    the::astring fullPath = Resolve(path);
    if (!the::text::string_utility::ends_with(fullPath, SEPARATOR_A)) fullPath += SEPARATOR_A;

    if (recursive) {
        // remove files and directory
        PurgeDirectory(fullPath, true);
    }

#ifdef _WIN32
    if (RemoveDirectoryA(fullPath.c_str()) == 0) {
#else /* _WIN32 */
    if (rmdir(fullPath) != 0) {
#endif /* _WIN32 */
        throw the::system::system_exception(__FILE__, __LINE__);
    }

}


/*
 * vislib::sys::Path::DeleteDirectory
 */
void vislib::sys::Path::DeleteDirectory(const the::wstring& path, bool recursive) {
#ifdef _WIN32
    if (!File::Exists(path.c_str())) return; // we don't delete non-existing stuff
    the::wstring fullPath = Resolve(path);
    if (!the::text::string_utility::ends_with(fullPath, SEPARATOR_W)) fullPath += SEPARATOR_W;

    if (recursive) {
        // remove files and directory
        PurgeDirectory(fullPath, true);
    }

    if (RemoveDirectoryW(fullPath.c_str()) == 0) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    // linux is stupid
    DeleteDirectory(THE_W2A(path), recursive);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::FindExecutablePath
 */
the::astring vislib::sys::Path::FindExecutablePath(
        const the::astring& filename) {
#ifdef _WIN32
    bool found = false;
    DWORD bufSize = MAX_PATH;
    char *buffer = new char[bufSize];

    // first try: "SearchPath"
    DWORD rv = ::SearchPathA(NULL, filename.c_str(), NULL, bufSize,
        buffer, NULL);
    if (rv > 0) {
        found = true;
        if (rv + 1 > bufSize) {
            bufSize = rv + 1;
            delete[] buffer;
            buffer = new char[bufSize];
            rv = ::SearchPathA(NULL, filename.c_str(), NULL, bufSize,
                buffer, NULL);
            if (rv == 0) { // failed
                found = false;
            }
        }
    }

    if (!found) {
        // second try: "AssocQueryString"
        // NOTE:
        //  AssocQueryString does not work as specified! It is not possible to ask
        // for the size of the string buffer holding the value returned. Therefore
        // this implementation increases the buffersize until the returned strings
        // no longer grow.
        DWORD bufLen = MAX_PATH;
        HRESULT hr;
        bufSize = MAX_PATH;
        
        do {
            hr = ::AssocQueryStringA(ASSOCF_INIT_BYEXENAME, ASSOCSTR_EXECUTABLE,
                filename.c_str(), NULL, buffer, &bufSize);
            if ((hr != E_POINTER) && (hr != S_OK)) { // error
                break;
            }
            if (bufSize == bufLen) {
                bufLen += MAX_PATH;
                bufSize = bufLen;
                delete[] buffer;
                buffer = new char[bufSize];
            } else {
                found = true;
            }
        } while (bufSize == bufLen);
    }

    if (found) {
        the::astring retval(buffer);
        delete[] buffer;
        return retval;
    } else {
        return "";
    }
#else /* _WIN32 */

    // Note:
    //  Do not use "Console::Run" or "Process" methods because they might use
    // this method to determine the full path of their binaries. So we avoid
    // calling cycles by just using "popen" directly
    the::astring cmd("which ");
    the::astring ret;
    cmd += filename;
    cmd += " 2> /dev/null";
    const int bufferSize = 1024;
    char buffer[bufferSize];
    FILE *which = ::popen(cmd.c_str(), "r");
    if (which == NULL) {
        return "";
    }
    while (!::feof(which)) {
        ret += fgets(buffer, bufferSize, which);
    }
    ::pclose(which);

    the::astring::size_type end = ret.find('\n');
    if (end != the::astring::npos) {
        ret.resize(end);
    }
    if (the::text::string_utility::ends_with(ret, filename)) {
        return ret;
    } else {
        return "";
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::FindExecutablePath
 */
the::wstring vislib::sys::Path::FindExecutablePath(
        const the::wstring& filename) {
#ifdef _WIN32
    bool found = false;
    DWORD bufSize = MAX_PATH;
    wchar_t *buffer = new wchar_t[bufSize];

    // first try: "SearchPath"
    DWORD rv = ::SearchPathW(NULL, filename.c_str(), NULL, bufSize,
        buffer, NULL);
    if (rv > 0) {
        found = true;
        if (rv + 1 > bufSize) {
            bufSize = rv + 1;
            delete[] buffer;
            buffer = new wchar_t[bufSize];
            rv = ::SearchPathW(NULL, filename.c_str(), NULL, bufSize,
                buffer, NULL);
            if (rv == 0) { // failed
                found = false;
            }
        }
    }

    if (!found) {
        // second try: "AssocQueryString"
        // NOTE:
        //  AssocQueryString does not work as specified! It is not possible to ask
        // for the size of the string buffer holding the value returned. Therefore
        // this implementation increases the buffersize until the returned strings
        // no longer grow.
        DWORD bufLen = MAX_PATH;
        HRESULT hr;
        bufSize = MAX_PATH;
        
        do {
            hr = ::AssocQueryStringW(ASSOCF_INIT_BYEXENAME, ASSOCSTR_EXECUTABLE,
                filename.c_str(), NULL, buffer, &bufSize);
            if ((hr != E_POINTER) && (hr != S_OK)) { // error
                break;
            }
            if (bufSize == bufLen) {
                bufLen += MAX_PATH;
                bufSize = bufLen;
                delete[] buffer;
                buffer = new wchar_t[bufSize];
            } else {
                found = true;
            }
        } while (bufSize == bufLen);
    }

    if (found) {
        the::wstring retval(buffer);
        delete[] buffer;
        return retval;
    } else {
        return L"";
    }
#else /* _WIN32 */
    // linux is stupid
    return THE_A2W(FindExecutablePath(THE_W2A(filename)));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::GetApplicationPathA
 */
the::astring vislib::sys::Path::GetApplicationPathA(void) {
    the::astring retval;
#ifdef _WIN32
    const DWORD nSize = MAX_PATH;
    retval = the::astring(nSize, ' ');
    if (::GetModuleFileNameA(NULL, const_cast<char*>(retval.c_str()), nSize)
            == ERROR_INSUFFICIENT_BUFFER) {
        retval.clear();
    } else {
        if (::GetLastError() != ERROR_SUCCESS) {
            retval.clear();
        }
    }

#else /* _WIN32 */
    // This is the best I got for now. Requires '/proc'
    the::astring pid;
    the::text::astring_builder::format_to(pid, "/proc/%d/exe", getpid());
    the::astring path;
    const size_t bufSize = 0xFFFF;
    char *buf = path.AllocateBuffer(bufSize);
    ssize_t size = readlink(pid.c_str(), buf, bufSize - 1);
    if (size >= 0) {
        buf[size] = 0;
        retval = buf;
    } else {
        retval.Clear();
    }

#endif /* _WIN32 */
    return retval;
}


/*
 * vislib::sys::Path::GetApplicationPathW
 */
the::wstring vislib::sys::Path::GetApplicationPathW(void) {
#ifdef _WIN32
    the::wstring retval;
    const DWORD nSize = MAX_PATH;
    retval = the::wstring(nSize, L' ');
    if (::GetModuleFileNameW(NULL, const_cast<wchar_t*>(retval.c_str()), nSize)
            == ERROR_INSUFFICIENT_BUFFER) {
        retval.clear();
    } else {
        if (::GetLastError() != ERROR_SUCCESS) {
            retval.clear();
        }
    }
    return retval;
#else /* _WIN32 */
    return THE_A2W(GetApplicationPathA());
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::GetCurrentDirectoryA
 */
the::astring vislib::sys::Path::GetCurrentDirectoryA(void) {
#ifdef _WIN32
    DWORD bufferSize = ::GetCurrentDirectoryA(0, NULL);
    char *buffer = new char[bufferSize];

    if (::GetCurrentDirectoryA(bufferSize, buffer) == 0) {
        the::safe_array_delete(buffer);
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    the::astring retval(buffer);
    the::safe_array_delete(buffer);

#else /* _WIN32 */
    const size_t BUFFER_GROW = 32;
    size_t bufferSize = 256;
    char *buffer = new char[bufferSize];

    while (::getcwd(buffer, bufferSize) == NULL) {
        the::safe_array_delete(buffer);

        if (errno == ERANGE) {
            bufferSize += BUFFER_GROW;
            buffer = new char[bufferSize];
        } else {
            throw the::system::system_exception(errno, __FILE__, __LINE__);
        }
    }

    the::astring retval(buffer);
    the::safe_array_delete(buffer);

#endif /* _WIN32 */

    if (!the::text::string_utility::ends_with(retval, SEPARATOR_A)) {
        retval += SEPARATOR_A;
    }

    return retval;
}


/*
 * vislib::sys::Path::GetCurrentDirectoryW
 */
the::wstring vislib::sys::Path::GetCurrentDirectoryW(void) {
#ifdef _WIN32
    DWORD bufferSize = ::GetCurrentDirectoryW(0, NULL);
    wchar_t *buffer = new wchar_t[bufferSize];

    if (::GetCurrentDirectoryW(bufferSize, buffer) == 0) {
        the::safe_array_delete(buffer);
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    the::wstring retval(buffer);
    the::safe_array_delete(buffer);

    if (!the::text::string_utility::ends_with(retval, SEPARATOR_W)) {
        retval += SEPARATOR_W;
    }
    return retval;

#else /* _WIN32 */
    return the::wstring(GetCurrentDirectoryA());
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::GetDirectoryName
 */
the::astring vislib::sys::Path::GetDirectoryName(const char *path) {
    the::astring retval(path);
    the::astring::size_type end = retval.rfind(SEPARATOR_A);

    if (end == the::astring::npos) {
        retval.clear();
    } else {
        retval.resize(end);
    }

    return retval;
}


/*
 * vislib::sys::Path::GetDirectoryName
 */
the::wstring vislib::sys::Path::GetDirectoryName(const wchar_t *path) {
    the::wstring retval(path);
    the::wstring::size_type end = retval.rfind(SEPARATOR_W);

    if (end == the::wstring::npos) {
        retval.clear();
    } else {
        retval.resize(end);
    }

    return retval;
}


/*
 * vislib::sys::Path::GetTempDirectoryA
 */
the::astring vislib::sys::Path::GetTempDirectoryA(void) {
#ifdef _WIN32
    char buffer[MAX_PATH + 1];
    buffer[MAX_PATH] = buffer[0] = 0;
    if (::GetTempPathA(MAX_PATH, buffer) > MAX_PATH) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
    return buffer;
//    return Environment::GetVariable("TEMP", false);
#else /* _WIN32 */
    return the::astring("/tmp");
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::GetTempDirectoryW
 */
the::wstring vislib::sys::Path::GetTempDirectoryW(void) {
#ifdef _WIN32
    wchar_t buffer[MAX_PATH + 1];
    buffer[MAX_PATH] = buffer[0] = 0;
    if (::GetTempPathW(MAX_PATH, buffer) > MAX_PATH) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
    return buffer;
//    return Environment::GetVariable(L"TEMP", false);
#else /* _WIN32 */
    return the::wstring(L"/tmp");
#endif /* _WIN32 */
}



/*
 * vislib::sys::Path::GetUserHomeDirectoryA
 */
the::astring vislib::sys::Path::GetUserHomeDirectoryA(void) {
#ifdef _WIN32
    the::astring retval(MAX_PATH, ' ');

    if (FAILED(::SHGetFolderPathA(NULL, CSIDL_PERSONAL, NULL, 0, 
            const_cast<char*>(retval.c_str())))) {
        throw the::system::system_exception(ERROR_NOT_FOUND, __FILE__, __LINE__);
    }

#else /* _WIN32 */
    char *path = getenv("HOME"); // Crowbar

    if (path == NULL) {
        throw the::system::system_exception(ENOENT, __FILE__, __LINE__);
    }

    the::astring retval(path);
#endif /* _WIN32 */

    if (!the::text::string_utility::ends_with(retval, SEPARATOR_A)) {
        retval += SEPARATOR_A;
    }

    return retval;
}


/*
 * vislib::sys::Path::GetUserHomeDirectoryW
 */
the::wstring vislib::sys::Path::GetUserHomeDirectoryW(void) {
#ifdef _WIN32
    the::wstring retval(MAX_PATH, L' ');

    if (FAILED(::SHGetFolderPathW(NULL, CSIDL_PERSONAL, NULL, 0, 
            const_cast<wchar_t*>(retval.c_str())))) {
        throw the::system::system_exception(ERROR_NOT_FOUND, __FILE__, __LINE__);
    }

    if (!the::text::string_utility::ends_with(retval, SEPARATOR_W)) {
        retval += SEPARATOR_W;
    }
    return retval;

#else /* _WIN32 */
    return the::wstring(GetUserHomeDirectoryA());

#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::IsRelative
 */
bool vislib::sys::Path::IsRelative(const the::astring& path) {
#ifdef _WIN32
    return (::PathIsRelativeA(path.c_str()) != FALSE)
        || path.empty()
        || ((path.c_str()[0] == SEPARATOR_A) && (path.c_str()[1] != SEPARATOR_A));
#else /* _WIN32 */
    return !the::text::string_utility::starts_with(path, SEPARATOR_A);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::IsRelative
 */
bool vislib::sys::Path::IsRelative(const the::wstring& path) {
#ifdef _WIN32
    return (::PathIsRelativeW(path.c_str()) != FALSE)
        || path.empty()
        || ((path.c_str()[0] == SEPARATOR_W) && (path.c_str()[1] != SEPARATOR_W));
#else /* _WIN32 */
    return !the::text::string_utility::starts_with(path, SEPARATOR_W);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::MakeDirectory
 */
void vislib::sys::Path::MakeDirectory(const the::astring& path) {
    Stack<the::astring> missingParts;
    the::astring firstBuilt;
    the::astring curPath = Resolve(path);

    while (!File::Exists(curPath.c_str())) {
        the::astring::size_type pos = curPath.rfind(SEPARATOR_A);
        if (pos != the::astring::npos) {
            missingParts.Push(curPath.substr(pos + 1));
            if (missingParts.Peek()->empty()) {
                // Remove empty directories as the incremental directory 
                // creation later on will fail for these.
                missingParts.Pop();
            }
            curPath.resize(pos);

        } else {
            // Problem: No Separators left, but directory still does not exist.
#ifdef _WIN32
            throw the::system::system_exception(ERROR_INVALID_NAME, __FILE__, __LINE__);
#else /* _WIN32 */
            throw the::system::system_exception(EINVAL, __FILE__, __LINE__);
#endif /* _WIN32 */
        }
    }

    // curPath exists
    if (!File::IsDirectory(curPath.c_str())) {
        // the latest existing directory is not a directory (may be a file?)
#ifdef _WIN32
        throw the::system::system_exception(ERROR_DIRECTORY, __FILE__, __LINE__);
#else /* _WIN32 */
        throw the::system::system_exception(EEXIST, __FILE__, __LINE__);
#endif /* _WIN32 */
    }

    while (!missingParts.empty()) {
        curPath += SEPARATOR_A;
        curPath += missingParts.Pop();

#ifdef _WIN32
        if (CreateDirectoryA(curPath.c_str(), NULL) != 0) {
#else /* _WIN32 */
        if (mkdir(curPath, S_IRWXG | S_IRWXO | S_IRWXU) == 0) { // TODO: Check
#endif /* _WIN32 */
            // success, so go on.
            if (firstBuilt.empty()) {
                firstBuilt = curPath;
            }

        } else {
            auto errorCode = GetLastError();

            try {
                // failure, so try to remove already created paths and throw exception.
                DeleteDirectory(firstBuilt, true);
            } catch(...) {
            }

            throw the::system::system_exception(errorCode, __FILE__, __LINE__);
        }
    }
    // we are done!
}


/*
 * vislib::sys::Path::MakeDirectory
 */
void vislib::sys::Path::MakeDirectory(const the::wstring& path) {
#ifdef _WIN32
    Stack<the::wstring> missingParts;
    the::wstring firstBuilt;
    the::wstring curPath = Resolve(path);

    while (!File::Exists(curPath.c_str())) {
        the::wstring::size_type pos = curPath.rfind(SEPARATOR_W);
        if (pos != the::wstring::npos) {
            missingParts.Push(curPath.substr(pos + 1));
            if (missingParts.Peek()->empty()) {
                // Remove empty directories as the incremental directory 
                // creation later on will fail for these.
                missingParts.Pop();
            }
            curPath.resize(pos);

        } else {
            // Problem: No Separators left, but directory still does not exist.
            throw the::system::system_exception(ERROR_INVALID_NAME, __FILE__, __LINE__);
        }
    }

    // curPath exists
    if (!File::IsDirectory(curPath.c_str())) {
        // the latest existing directory is not a directory (may be a file?)
        throw the::system::system_exception(ERROR_DIRECTORY, __FILE__, __LINE__);
    }

    while (!missingParts.empty()) {
        curPath += SEPARATOR_W;
        curPath += missingParts.Pop();

        if (CreateDirectoryW(curPath.c_str(), NULL) != 0) {
            // success, so go on.
            if (firstBuilt.empty()) {
                firstBuilt = curPath;
            }

        } else {
            DWORD errorCode = GetLastError();

            try {
                // failure, so try to remove already created paths and throw exception.
                DeleteDirectory(firstBuilt, true);
            } catch(...) {
            }

            throw the::system::system_exception(errorCode, __FILE__, __LINE__);
        }
    }
    // we are done!

#else /* _WIN32 */
    // linux is stupid
    MakeDirectory(THE_W2A(path));

#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::PurgeDirectory
 */
void vislib::sys::Path::PurgeDirectory(const the::astring& path, bool recursive) {
    the::astring fullpath = Resolve(path);
    if (!the::text::string_utility::ends_with(fullpath, SEPARATOR_A)) fullpath += SEPARATOR_A;
    DirectoryIteratorA iter(fullpath.c_str());

    while (iter.HasNext()) {
        DirectoryEntryA entry = iter.Next();
        if (entry.Type == DirectoryEntryA::FILE) {
            vislib::sys::File::Delete((fullpath + entry.Path).c_str());

        } else
        if (entry.Type == DirectoryEntryA::DIRECTORY) {
            if (recursive) {
                DeleteDirectory(fullpath + entry.Path, true);
            }

        } else {
            THE_ASSERT(false); // DirectoryEntry is something unknown to this 
                           // implementation. Check DirectoryIterator for 
                           // changes.
        }
    }
}


/*
 * vislib::sys::Path::PurgeDirectory
 */
void vislib::sys::Path::PurgeDirectory(const the::wstring& path, bool recursive) {
    the::wstring fullpath = Resolve(path);
    if (!the::text::string_utility::ends_with(fullpath, SEPARATOR_W)) fullpath += SEPARATOR_W;
    DirectoryIteratorW iter(fullpath.c_str());

    while (iter.HasNext()) {
        DirectoryEntryW entry = iter.Next();
        if (entry.Type == DirectoryEntryW::FILE) {
            vislib::sys::File::Delete((fullpath + entry.Path).c_str());

        } else
        if (entry.Type == DirectoryEntryW::DIRECTORY) {
            if (recursive) {
                DeleteDirectory(fullpath + entry.Path, true);
            }

        } else {
            THE_ASSERT(false); // DirectoryEntry is something unknown to this 
                           // implementation. Check DirectoryIterator for 
                           // changes.
        }
    }
}


/*
 * vislib::sys::Path::Resolve
 */
the::astring vislib::sys::Path::Resolve(the::astring path, the::astring basepath) {
    // TODO: Windows shell API resolve does not work in the expected
    // way, so we use the same manual approach for Windows and Linux.

#ifdef _WIN32
    /* Replace unchefm‰ﬂige path separators. */
    the::text::string_utility::replace(basepath, '/', SEPARATOR_A);
    the::text::string_utility::replace(path, '/', SEPARATOR_A);
#endif /* _WIN32 */

    if (Path::IsRelative(basepath)) {
        basepath = Resolve(basepath);
    }

    if (path.empty()) {
        /* Path is empty, i. e. return current working directory. */
        return Path::Canonicalise(basepath);
    
    } else if (Path::IsAbsolute(path)) {
        /* Path is absolute, just return it. */
        return Path::Canonicalise(path);

    } else if ((path[0] == MYDOCUMENTS_MARKER_A) 
            && ((path.size() == 1) || path[1] == SEPARATOR_A)) {
        /*
         * replace leading ~ with users home directory
         */
        the::text::string_utility::replace(path, MYDOCUMENTS_MARKER_A, Path::GetUserHomeDirectoryA(), 1);
        return Path::Canonicalise(path);

    } else if ((path[0] == SEPARATOR_A) && (path[1] != SEPARATOR_A)) {
        /*
         * Concatenate current drive and relative path, and canonicalise
         * the result.
         */
        return Path::Concatenate(basepath.substr(0, 2), path, true);

    } else {
        /*
         * Concatenate current directory and relative path, and canonicalise
         * the result.
         */
        return Path::Concatenate(basepath, path, true);
    }
}


/*
 * vislib::sys::Path::Resolve
 */
the::wstring vislib::sys::Path::Resolve(the::wstring path, the::wstring basepath) {
    // TODO: Windows shell API resolve does not work in the expected
    // way, so we use the same manual approach for Windows and Linux.

#ifdef _WIN32
    /* Replace unchefm‰ﬂige path separators. */
    the::text::string_utility::replace(basepath, L'/', SEPARATOR_W);
    the::text::string_utility::replace(path, L'/', SEPARATOR_W);
#endif /* _WIN32 */

    if (Path::IsRelative(basepath)) {
        basepath = Resolve(basepath);
    }

    if (path.empty()) {
        /* Path is empty, i. e. return current working directory. */
        return Path::Canonicalise(basepath);

    } else if (Path::IsAbsolute(path)) {
        /* Path is absolute, just return it. */
        return Path::Canonicalise(path);

    } else if ((path[0] == MYDOCUMENTS_MARKER_W) 
            && ((path.size() == 1) || path[1] == SEPARATOR_W)) {
        /*
         * replace leading ~ with users home directory
         */
        the::text::string_utility::replace(path, MYDOCUMENTS_MARKER_W, Path::GetUserHomeDirectoryW());
        return Path::Canonicalise(path);

    } else if ((path[0] == SEPARATOR_W) && (path[1] != SEPARATOR_W)) {
        /*
         * Concatenate current drive and relative path, and canonicalise
         * the result.
         */
        return Path::Concatenate(basepath.substr(0, 2), path, true);

    } else {
        /*
         * Concatenate current directory and relative path, and canonicalise
         * the result.
         */
        return Concatenate(basepath, path, true);
    }
}


/*
 * vislib::sys::Path::SetCurrentDirectory
 */
void vislib::sys::Path::SetCurrentDirectory(const the::astring& path) {
#ifdef _WIN32
    if (::SetCurrentDirectoryA(path.c_str()) != TRUE) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    if (::chdir(static_cast<const char *>(path)) == -1) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::SetCurrentDirectory
 */
void vislib::sys::Path::SetCurrentDirectory(const the::wstring& path) {
#ifdef _WIN32
    if (::SetCurrentDirectoryW(path.c_str()) != TRUE) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    SetCurrentDirectory(static_cast<the::astring>(path));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::MYDOCUMENTS_MARKER_A
 */
const char vislib::sys::Path::MYDOCUMENTS_MARKER_A = '~';


/*
 * vislib::sys::Path::MYDOCUMENTS_MARKER_W
 */
const char vislib::sys::Path::MYDOCUMENTS_MARKER_W = L'~';


/*
 * vislib::sys::Path::SEPARATOR_A
 */
#ifdef _WIN32
const char vislib::sys::Path::SEPARATOR_A = '\\';
#else /* _WIN32 */
const char vislib::sys::Path::SEPARATOR_A = '/';
#endif /* _WIN32 */



/*
 * vislib::sys::Path::SEPARATOR_W
 */
#ifdef _WIN32
const wchar_t vislib::sys::Path::SEPARATOR_W = L'\\';
#else /* _WIN32 */
const wchar_t vislib::sys::Path::SEPARATOR_W = L'/';
#endif /* _WIN32 */


/*
 * vislib::sys::Path::~Path
 */
vislib::sys::Path::~Path(void) {
}


/*
 * vislib::sys::Path::Path
 */
vislib::sys::Path::Path(void) {
}
