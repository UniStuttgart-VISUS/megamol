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

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/Environment.h"
#include "vislib/memutils.h"
#include "vislib/SystemException.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/Stack.h"
#include "vislib/File.h"
#include "vislib/DirectoryIterator.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"


/*
 * vislib::sys::Path::Canonicalise
 */
vislib::StringA vislib::sys::Path::Canonicalise(const StringA& path) {
    const StringA DOUBLE_SEPARATOR(Path::SEPARATOR_A, 2);

#ifdef _WIN32
    StringA retval;

    if (::PathCanonicalizeA(retval.AllocateBuffer(MAX_PATH), path) != TRUE) {
        throw SystemException(__FILE__, __LINE__);
    }

    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATOR_A);
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATOR_A);

    /* Ensure that a UNC path remains a UNC path. */
    if (path.StartsWith(DOUBLE_SEPARATOR)) {
        // Note: Double separator replacement above leaves at least one 
        // separator, so we must only prepend one additional one.
        retval.Prepend(SEPARATOR_A);
    }

    return retval;

#else /* _WIN32 */
    const char *BACK_REF = "/..";
    const char *CUR_REF = "/.";             // Note: "./" does not work
    StringA::Size BACK_REF_LEN = ::strlen(BACK_REF);
    StringA::Size bwRefPos = 0;             // Position of back reference.
    StringA::Size remDirPos = 0;            // Position of directory to erase.
    StringA retval(path);
    

    /* Remove backward references, first. */
    while ((bwRefPos = retval.Find(BACK_REF)) != StringA::INVALID_POS) {

        if ((bwRefPos > 0) 
                && (remDirPos = retval.FindLast(SEPARATOR_A, bwRefPos - 1))
                != StringA::INVALID_POS) {
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
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATOR_A);
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATOR_A);

    return retval;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::Canonicalise
 */
vislib::StringW vislib::sys::Path::Canonicalise(const StringW& path) {
    const StringW DOUBLE_SEPARATOR(Path::SEPARATOR_W, 2);

#ifdef _WIN32
    StringW retval;

    if (::PathCanonicalizeW(retval.AllocateBuffer(MAX_PATH), path) != TRUE) {
        throw SystemException(__FILE__, __LINE__);
    }

    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATOR_W);
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATOR_W);

    /* Ensure that a UNC path remains a UNC path. */
    if (path.StartsWith(DOUBLE_SEPARATOR)) {
        // Note: Double separator replacement above leaves at least one 
        // separator, so we must only prepend one additional one.
        retval.Prepend(SEPARATOR_W);
    }

    return retval;

#else /* _WIN32 */
    const wchar_t *BACK_REF = L"/..";
    const wchar_t *CUR_REF = L"/.";         // Note: "./" does not work
    StringW::Size BACK_REF_LEN = ::wcslen(BACK_REF);
    StringW::Size bwRefPos = 0;             // Position of back reference.
    StringW::Size remDirPos = 0;            // Position of directory to erase.
    StringW retval(path);
    

    /* Remove backward references, first. */
    while ((bwRefPos = retval.Find(BACK_REF)) != StringW::INVALID_POS) {

        if ((bwRefPos > 0) 
                && (remDirPos = retval.FindLast(SEPARATOR_W, bwRefPos - 1))
                != StringW::INVALID_POS) {
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
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATOR_W);
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATOR_W);

    return retval;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::ChangeExtension
 */
vislib::StringA vislib::sys::Path::ChangeExtension(const char *path,
        const char *extension) {
    const char EXT_SEP = '.';
    StringA retval(path);
    StringA::Size extStart = retval.FindLast(EXT_SEP);
    StringA ext(extension);

    if (!ext.IsEmpty() && !ext.StartsWith(EXT_SEP)) {
        ext.Prepend(EXT_SEP);
    }

    if (extStart != StringA::INVALID_POS) {
        /* Found a valid extension, remove that one before adding new. */
        retval.Truncate(extStart);
    }

    retval += ext;
    return retval;
}


/*
 * vislib::sys::Path::ChangeExtension
 */
vislib::StringW vislib::sys::Path::ChangeExtension(const wchar_t *path,
        const wchar_t *extension) {
    const wchar_t EXT_SEP = '.';
    StringW retval(path);
    StringW::Size extStart = retval.FindLast(EXT_SEP);
    StringW ext(extension);

    if (!ext.IsEmpty() && !ext.StartsWith(EXT_SEP)) {
        ext.Prepend(EXT_SEP);
    }

    if (extStart != StringW::INVALID_POS) {
        /* Found a valid extension, remove that one before adding new. */
        retval.Truncate(extStart);
    }

    retval += ext;
    return retval;
}


/*
 * vislib::sys::Path::Concatenate
 */
vislib::StringA vislib::sys::Path::Concatenate(const StringA& lhs,
        const StringA& rhs, const bool canonicalise) {
    StringA retval(lhs);

    if (lhs.EndsWith(SEPARATOR_A) && rhs.StartsWith(SEPARATOR_A)) {
        retval.Append(rhs.PeekBuffer() + 1);

    } else if (!lhs.EndsWith(SEPARATOR_A) && !rhs.StartsWith(SEPARATOR_A)) {
        retval.Append(SEPARATOR_A);
        retval.Append(rhs);

    } else {
        retval.Append(rhs);
    }

    return canonicalise ? Path::Canonicalise(retval) : retval;
}


/*
 * vislib::sys::Path::Concatenate
 */
vislib::StringW vislib::sys::Path::Concatenate(const StringW& lhs, 
        const StringW& rhs, const bool canonicalise) {
    StringW retval(lhs);

    if (lhs.EndsWith(SEPARATOR_W) && rhs.StartsWith(SEPARATOR_W)) {
        retval.Append(rhs.PeekBuffer() + 1);

    } else if (!lhs.EndsWith(SEPARATOR_W) && !rhs.StartsWith(SEPARATOR_W)) {
        retval.Append(SEPARATOR_W);
        retval.Append(rhs);

    } else {
        retval.Append(rhs);
    }

    return canonicalise ? Path::Canonicalise(retval) : retval;
}


/*
 * vislib::sys::Path::DeleteDirectory
 */
void vislib::sys::Path::DeleteDirectory(const StringA& path, bool recursive) {
    if (!File::Exists(path)) return; // we don't delete non-existing stuff
    StringA fullPath = Resolve(path);
    if (!fullPath.EndsWith(SEPARATOR_A)) fullPath += SEPARATOR_A;

    if (recursive) {
        // remove files and directory
        PurgeDirectory(fullPath, true);
    }

#ifdef _WIN32
    if (RemoveDirectoryA(fullPath) == 0) {
#else /* _WIN32 */
    if (rmdir(fullPath) != 0) {
#endif /* _WIN32 */
        throw vislib::sys::SystemException(__FILE__, __LINE__);
    }

}


/*
 * vislib::sys::Path::DeleteDirectory
 */
void vislib::sys::Path::DeleteDirectory(const StringW& path, bool recursive) {
#ifdef _WIN32
    if (!File::Exists(path)) return; // we don't delete non-existing stuff
    StringW fullPath = Resolve(path);
    if (!fullPath.EndsWith(SEPARATOR_W)) fullPath += SEPARATOR_W;

    if (recursive) {
        // remove files and directory
        PurgeDirectory(fullPath, true);
    }

    if (RemoveDirectoryW(fullPath) == 0) {
        throw vislib::sys::SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    // linux is stupid
    DeleteDirectory(W2A(path), recursive);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::FindExecutablePath
 */
vislib::StringA vislib::sys::Path::FindExecutablePath(
        const vislib::StringA& filename) {
#ifdef _WIN32
    bool found = false;
    DWORD bufSize = MAX_PATH;
    char *buffer = new char[bufSize];

    // first try: "SearchPath"
    DWORD rv = ::SearchPathA(NULL, filename.PeekBuffer(), NULL, bufSize,
        buffer, NULL);
    if (rv > 0) {
        found = true;
        if (rv + 1 > bufSize) {
            bufSize = rv + 1;
            delete[] buffer;
            buffer = new char[bufSize];
            rv = ::SearchPathA(NULL, filename.PeekBuffer(), NULL, bufSize,
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
                filename.PeekBuffer(), NULL, buffer, &bufSize);
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
        vislib::StringA retval(buffer);
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
    vislib::StringA cmd("which ");
    vislib::StringA ret;
    cmd += filename;
    cmd += " 2> /dev/null";
    const int bufferSize = 1024;
    char buffer[bufferSize];
    FILE *which = ::popen(cmd.PeekBuffer(), "r");
    if (which == NULL) {
        return "";
    }
    while (!::feof(which)) {
        ret += fgets(buffer, bufferSize, which);
    }
    ::pclose(which);

    vislib::StringA::Size end = ret.Find('\n');
    if (end != vislib::StringA::INVALID_POS) {
        ret.Truncate(end);
    }
    if (ret.EndsWith(filename)) {
        return ret;
    } else {
        return "";
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::FindExecutablePath
 */
vislib::StringW vislib::sys::Path::FindExecutablePath(
        const vislib::StringW& filename) {
#ifdef _WIN32
    bool found = false;
    DWORD bufSize = MAX_PATH;
    wchar_t *buffer = new wchar_t[bufSize];

    // first try: "SearchPath"
    DWORD rv = ::SearchPathW(NULL, filename.PeekBuffer(), NULL, bufSize,
        buffer, NULL);
    if (rv > 0) {
        found = true;
        if (rv + 1 > bufSize) {
            bufSize = rv + 1;
            delete[] buffer;
            buffer = new wchar_t[bufSize];
            rv = ::SearchPathW(NULL, filename.PeekBuffer(), NULL, bufSize,
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
                filename.PeekBuffer(), NULL, buffer, &bufSize);
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
        vislib::StringW retval(buffer);
        delete[] buffer;
        return retval;
    } else {
        return L"";
    }
#else /* _WIN32 */
    // linux is stupid
    return A2W(FindExecutablePath(W2A(filename)));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::GetApplicationPathA
 */
vislib::StringA vislib::sys::Path::GetApplicationPathA(void) {
    vislib::StringA retval;
#ifdef _WIN32
    const DWORD nSize = 0xFFFF;
    if (::GetModuleFileNameA(NULL, retval.AllocateBuffer(nSize), nSize)
            == ERROR_INSUFFICIENT_BUFFER) {
        retval.Clear();
    } else {
        if (::GetLastError() != ERROR_SUCCESS) {
            retval.Clear();
        }
    }

#else /* _WIN32 */
    // This is the best I got for now. Requires '/proc'
    vislib::StringA pid;
    pid.Format("/proc/%d/exe", getpid());
    vislib::StringA path;
    const SIZE_T bufSize = 0xFFFF;
    char *buf = path.AllocateBuffer(bufSize);
    ssize_t size = readlink(pid.PeekBuffer(), buf, bufSize - 1);
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
vislib::StringW vislib::sys::Path::GetApplicationPathW(void) {
#ifdef _WIN32
    vislib::StringW retval;
    const DWORD nSize = 0xFFFF;
    if (::GetModuleFileNameW(NULL, retval.AllocateBuffer(nSize), nSize)
            == ERROR_INSUFFICIENT_BUFFER) {
        retval.Clear();
    } else {
        if (::GetLastError() != ERROR_SUCCESS) {
            retval.Clear();
        }
    }
    return retval;
#else /* _WIN32 */
    return A2W(GetApplicationPathA());
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::GetCurrentDirectoryA
 */
vislib::StringA vislib::sys::Path::GetCurrentDirectoryA(void) {
#ifdef _WIN32
    DWORD bufferSize = ::GetCurrentDirectoryA(0, NULL);
    char *buffer = new char[bufferSize];

    if (::GetCurrentDirectoryA(bufferSize, buffer) == 0) {
        ARY_SAFE_DELETE(buffer);
        throw SystemException(__FILE__, __LINE__);
    }

    StringA retval(buffer);
    ARY_SAFE_DELETE(buffer);

#else /* _WIN32 */
    const SIZE_T BUFFER_GROW = 32;
    SIZE_T bufferSize = 256;
    char *buffer = new char[bufferSize];

    while (::getcwd(buffer, bufferSize) == NULL) {
        ARY_SAFE_DELETE(buffer);

        if (errno == ERANGE) {
            bufferSize += BUFFER_GROW;
            buffer = new char[bufferSize];
        } else {
            throw SystemException(errno, __FILE__, __LINE__);
        }
    }

    StringA retval(buffer);
    ARY_SAFE_DELETE(buffer);

#endif /* _WIN32 */

    if (!retval.EndsWith(SEPARATOR_A)) {
        retval += SEPARATOR_A;
    }

    return retval;
}


/*
 * vislib::sys::Path::GetCurrentDirectoryW
 */
vislib::StringW vislib::sys::Path::GetCurrentDirectoryW(void) {
#ifdef _WIN32
    DWORD bufferSize = ::GetCurrentDirectoryW(0, NULL);
    wchar_t *buffer = new wchar_t[bufferSize];

    if (::GetCurrentDirectoryW(bufferSize, buffer) == 0) {
        ARY_SAFE_DELETE(buffer);
        throw SystemException(__FILE__, __LINE__);
    }

    StringW retval(buffer);
    ARY_SAFE_DELETE(buffer);

    if (!retval.EndsWith(SEPARATOR_W)) {
        retval += SEPARATOR_W;
    }
    return retval;

#else /* _WIN32 */
    return StringW(GetCurrentDirectoryA());
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::GetDirectoryName
 */
vislib::StringA vislib::sys::Path::GetDirectoryName(const char *path) {
    StringA retval(path);
    StringA::Size end = retval.FindLast(SEPARATOR_A);

    if (end == StringA::INVALID_POS) {
        retval.Clear();
    } else {
        retval.Truncate(end);
    }

    return retval;
}


/*
 * vislib::sys::Path::GetDirectoryName
 */
vislib::StringW vislib::sys::Path::GetDirectoryName(const wchar_t *path) {
    StringW retval(path);
    StringW::Size end = retval.FindLast(SEPARATOR_W);

    if (end == StringW::INVALID_POS) {
        retval.Clear();
    } else {
        retval.Truncate(end);
    }

    return retval;
}


/*
 * vislib::sys::Path::GetTempDirectoryA
 */
vislib::StringA vislib::sys::Path::GetTempDirectoryA(void) {
#ifdef _WIN32
    char buffer[MAX_PATH + 1];
    buffer[MAX_PATH] = buffer[0] = 0;
    if (::GetTempPathA(MAX_PATH, buffer) > MAX_PATH) {
        throw SystemException(__FILE__, __LINE__);
    }
    return buffer;
//    return Environment::GetVariable("TEMP", false);
#else /* _WIN32 */
    return StringA("/tmp");
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::GetTempDirectoryW
 */
vislib::StringW vislib::sys::Path::GetTempDirectoryW(void) {
#ifdef _WIN32
    wchar_t buffer[MAX_PATH + 1];
    buffer[MAX_PATH] = buffer[0] = 0;
    if (::GetTempPathW(MAX_PATH, buffer) > MAX_PATH) {
        throw SystemException(__FILE__, __LINE__);
    }
    return buffer;
//    return Environment::GetVariable(L"TEMP", false);
#else /* _WIN32 */
    return StringW(L"/tmp");
#endif /* _WIN32 */
}



/*
 * vislib::sys::Path::GetUserHomeDirectoryA
 */
vislib::StringA vislib::sys::Path::GetUserHomeDirectoryA(void) {
#ifdef _WIN32
    StringA retval;

    if (FAILED(::SHGetFolderPathA(NULL, CSIDL_PERSONAL, NULL, 0, 
            retval.AllocateBuffer(MAX_PATH)))) {
        throw SystemException(ERROR_NOT_FOUND, __FILE__, __LINE__);
    }

#else /* _WIN32 */
    char *path = getenv("HOME"); // Crowbar

    if (path == NULL) {
        throw SystemException(ENOENT, __FILE__, __LINE__);
    }

    StringA retval(path);
#endif /* _WIN32 */

    if (!retval.EndsWith(SEPARATOR_A)) {
        retval += SEPARATOR_A;
    }

    return retval;
}


/*
 * vislib::sys::Path::GetUserHomeDirectoryW
 */
vislib::StringW vislib::sys::Path::GetUserHomeDirectoryW(void) {
#ifdef _WIN32
    StringW retval;

    if (FAILED(::SHGetFolderPathW(NULL, CSIDL_PERSONAL, NULL, 0, 
            retval.AllocateBuffer(MAX_PATH)))) {
        throw SystemException(ERROR_NOT_FOUND, __FILE__, __LINE__);
    }

    if (!retval.EndsWith(SEPARATOR_W)) {
        retval += SEPARATOR_W;
    }
    return retval;

#else /* _WIN32 */
    return StringW(GetUserHomeDirectoryA());

#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::IsRelative
 */
bool vislib::sys::Path::IsRelative(const StringA& path) {
#ifdef _WIN32
    return (::PathIsRelativeA(path.PeekBuffer()) != FALSE)
        || path.IsEmpty()
        || ((path.PeekBuffer()[0] == SEPARATOR_A) && (path.PeekBuffer()[1] != SEPARATOR_A));
#else /* _WIN32 */
    return !path.StartsWith(SEPARATOR_A);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::IsRelative
 */
bool vislib::sys::Path::IsRelative(const StringW& path) {
#ifdef _WIN32
    return (::PathIsRelativeW(path.PeekBuffer()) != FALSE)
        || path.IsEmpty()
        || ((path.PeekBuffer()[0] == SEPARATOR_W) && (path.PeekBuffer()[1] != SEPARATOR_W));
#else /* _WIN32 */
    return !path.StartsWith(SEPARATOR_W);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::MakeDirectory
 */
void vislib::sys::Path::MakeDirectory(const StringA& path) {
    Stack<StringA> missingParts;
    StringA firstBuilt;
    StringA curPath = Resolve(path);

    while (!File::Exists(curPath)) {
        StringA::Size pos = curPath.FindLast(SEPARATOR_A);
        if (pos != StringA::INVALID_POS) {
            missingParts.Push(curPath.Substring(pos + 1));
            if (missingParts.Peek()->IsEmpty()) {
                // Remove empty directories as the incremental directory 
                // creation later on will fail for these.
                missingParts.Pop();
            }
            curPath.Truncate(pos);

        } else {
            // Problem: No Separators left, but directory still does not exist.
#ifdef _WIN32
            throw vislib::sys::SystemException(ERROR_INVALID_NAME, __FILE__, __LINE__);
#else /* _WIN32 */
            throw vislib::sys::SystemException(EINVAL, __FILE__, __LINE__);
#endif /* _WIN32 */
        }
    }

    // curPath exists
    if (!File::IsDirectory(curPath)) {
        // the latest existing directory is not a directory (may be a file?)
#ifdef _WIN32
        throw vislib::sys::SystemException(ERROR_DIRECTORY, __FILE__, __LINE__);
#else /* _WIN32 */
        throw vislib::sys::SystemException(EEXIST, __FILE__, __LINE__);
#endif /* _WIN32 */
    }

    while (!missingParts.IsEmpty()) {
        curPath += SEPARATOR_A;
        curPath += missingParts.Pop();

#ifdef _WIN32
        if (CreateDirectoryA(curPath, NULL) != 0) {
#else /* _WIN32 */
        if (mkdir(curPath, S_IRWXG | S_IRWXO | S_IRWXU) == 0) { // TODO: Check
#endif /* _WIN32 */
            // success, so go on.
            if (firstBuilt.IsEmpty()) {
                firstBuilt = curPath;
            }

        } else {
            DWORD errorCode = GetLastError();

            try {
                // failure, so try to remove already created paths and throw exception.
                DeleteDirectory(firstBuilt, true);
            } catch(...) {
            }

            throw vislib::sys::SystemException(errorCode, __FILE__, __LINE__);
        }
    }
    // we are done!
}


/*
 * vislib::sys::Path::MakeDirectory
 */
void vislib::sys::Path::MakeDirectory(const StringW& path) {
#ifdef _WIN32
    Stack<StringW> missingParts;
    StringW firstBuilt;
    StringW curPath = Resolve(path);

    while (!File::Exists(curPath)) {
        StringW::Size pos = curPath.FindLast(SEPARATOR_W);
        if (pos != StringW::INVALID_POS) {
            missingParts.Push(curPath.Substring(pos + 1));
            if (missingParts.Peek()->IsEmpty()) {
                // Remove empty directories as the incremental directory 
                // creation later on will fail for these.
                missingParts.Pop();
            }
            curPath.Truncate(pos);

        } else {
            // Problem: No Separators left, but directory still does not exist.
            throw vislib::sys::SystemException(ERROR_INVALID_NAME, __FILE__, __LINE__);
        }
    }

    // curPath exists
    if (!File::IsDirectory(curPath)) {
        // the latest existing directory is not a directory (may be a file?)
        throw vislib::sys::SystemException(ERROR_DIRECTORY, __FILE__, __LINE__);
    }

    while (!missingParts.IsEmpty()) {
        curPath += SEPARATOR_W;
        curPath += missingParts.Pop();

        if (CreateDirectoryW(curPath, NULL) != 0) {
            // success, so go on.
            if (firstBuilt.IsEmpty()) {
                firstBuilt = curPath;
            }

        } else {
            DWORD errorCode = GetLastError();

            try {
                // failure, so try to remove already created paths and throw exception.
                DeleteDirectory(firstBuilt, true);
            } catch(...) {
            }

            throw vislib::sys::SystemException(errorCode, __FILE__, __LINE__);
        }
    }
    // we are done!

#else /* _WIN32 */
    // linux is stupid
    MakeDirectory(W2A(path));

#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::PurgeDirectory
 */
void vislib::sys::Path::PurgeDirectory(const StringA& path, bool recursive) {
    StringA fullpath = Resolve(path);
    if (!fullpath.EndsWith(SEPARATOR_A)) fullpath += SEPARATOR_A;
    DirectoryIteratorA iter(fullpath);

    while (iter.HasNext()) {
        DirectoryEntryA entry = iter.Next();
        if (entry.Type == DirectoryEntryA::FILE) {
            vislib::sys::File::Delete(fullpath + entry.Path);

        } else
        if (entry.Type == DirectoryEntryA::DIRECTORY) {
            if (recursive) {
                DeleteDirectory(fullpath + entry.Path, true);
            }

        } else {
            ASSERT(false); // DirectoryEntry is something unknown to this 
                           // implementation. Check DirectoryIterator for 
                           // changes.
        }
    }
}


/*
 * vislib::sys::Path::PurgeDirectory
 */
void vislib::sys::Path::PurgeDirectory(const StringW& path, bool recursive) {
    StringW fullpath = Resolve(path);
    if (!fullpath.EndsWith(SEPARATOR_W)) fullpath += SEPARATOR_W;
    DirectoryIteratorW iter(fullpath);

    while (iter.HasNext()) {
        DirectoryEntryW entry = iter.Next();
        if (entry.Type == DirectoryEntryW::FILE) {
            vislib::sys::File::Delete(fullpath + entry.Path);

        } else
        if (entry.Type == DirectoryEntryW::DIRECTORY) {
            if (recursive) {
                DeleteDirectory(fullpath + entry.Path, true);
            }

        } else {
            ASSERT(false); // DirectoryEntry is something unknown to this 
                           // implementation. Check DirectoryIterator for 
                           // changes.
        }
    }
}


/*
 * vislib::sys::Path::Resolve
 */
vislib::StringA vislib::sys::Path::Resolve(StringA path, StringA basepath) {
    // TODO: Windows shell API resolve does not work in the expected
    // way, so we use the same manual approach for Windows and Linux.

#ifdef _WIN32
    /* Replace unchefm‰ﬂige path separators. */
    basepath.Replace('/', SEPARATOR_A);
    path.Replace('/', SEPARATOR_A);
#endif /* _WIN32 */

    if (Path::IsRelative(basepath)) {
        basepath = Resolve(basepath);
    }

    if (path.IsEmpty()) {
        /* Path is empty, i. e. return current working directory. */
        return Path::Canonicalise(basepath);
    
    } else if (Path::IsAbsolute(path)) {
        /* Path is absolute, just return it. */
        return Path::Canonicalise(path);

    } else if ((path[0] == MYDOCUMENTS_MARKER_A) 
            && ((path.Length() == 1) || path[1] == SEPARATOR_A)) {
        /*
         * replace leading ~ with users home directory
         */
        path.Replace(MYDOCUMENTS_MARKER_A, Path::GetUserHomeDirectoryA(), 1);
        return Path::Canonicalise(path);

    } else if ((path[0] == SEPARATOR_A) && (path[1] != SEPARATOR_A)) {
        /*
         * Concatenate current drive and relative path, and canonicalise
         * the result.
         */
        return Path::Concatenate(basepath.Substring(0, 2), path, true);

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
vislib::StringW vislib::sys::Path::Resolve(StringW path, StringW basepath) {
    // TODO: Windows shell API resolve does not work in the expected
    // way, so we use the same manual approach for Windows and Linux.

#ifdef _WIN32
    /* Replace unchefm‰ﬂige path separators. */
    basepath.Replace(L'/', SEPARATOR_W);
    path.Replace(L'/', SEPARATOR_W);
#endif /* _WIN32 */

    if (Path::IsRelative(basepath)) {
        basepath = Resolve(basepath);
    }

    if (path.IsEmpty()) {
        /* Path is empty, i. e. return current working directory. */
        return Path::Canonicalise(basepath);

    } else if (Path::IsAbsolute(path)) {
        /* Path is absolute, just return it. */
        return Path::Canonicalise(path);

    } else if ((path[0] == MYDOCUMENTS_MARKER_W) 
            && ((path.Length() == 1) || path[1] == SEPARATOR_W)) {
        /*
         * replace leading ~ with users home directory
         */
        path.Replace(MYDOCUMENTS_MARKER_W, Path::GetUserHomeDirectoryW());
        return Path::Canonicalise(path);

    } else if ((path[0] == SEPARATOR_W) && (path[1] != SEPARATOR_W)) {
        /*
         * Concatenate current drive and relative path, and canonicalise
         * the result.
         */
        return Path::Concatenate(basepath.Substring(0, 2), path, true);

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
void vislib::sys::Path::SetCurrentDirectory(const StringA& path) {
#ifdef _WIN32
    if (::SetCurrentDirectoryA(path) != TRUE) {
        throw SystemException(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    if (::chdir(static_cast<const char *>(path)) == -1) {
        throw SystemException(__FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::SetCurrentDirectory
 */
void vislib::sys::Path::SetCurrentDirectory(const StringW& path) {
#ifdef _WIN32
    if (::SetCurrentDirectoryW(path) != TRUE) {
        throw SystemException(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    SetCurrentDirectory(static_cast<StringA>(path));
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
