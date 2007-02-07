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
#endif /* _WIN32 */

#include "vislib/assert.h"
#include "vislib/Console.h"
#include "vislib/error.h"
#include "vislib/memutils.h"
#include "vislib/SystemException.h"


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
    return (::PathIsRelativeA(path.PeekBuffer()) != FALSE);
#else /* _WIN32 */
    return !path.StartsWith(SEPARATOR_A);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::IsRelative
 */
bool vislib::sys::Path::IsRelative(const StringW& path) {
#ifdef _WIN32
    return (::PathIsRelativeW(path.PeekBuffer()) != FALSE);
#else /* _WIN32 */
    return !path.StartsWith(SEPARATOR_W);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::Resolve
 */
vislib::StringA vislib::sys::Path::Resolve(StringA path) {
    // TODO: Windows shell API resolve does not work in the expected
    // way, so we use the same manual approach for Windows and Linux.

#ifdef _WIN32
    /* Remove unchefm‰ﬂige path separators. */
    path.Replace('/', SEPARATOR_A);
#endif /* _WIN32 */

    if (path.IsEmpty()) {
        /* Path is empty, i. e. return current working directory. */
        return Path::GetCurrentDirectoryA();
    
    } else if (Path::IsAbsolute(path)) {
        /* Path is absolute, just return it. */
#ifdef _WIN32
        if ((path.Length() < 2) || ((path[1] != SEPARATOR_A) 
                && (path[1] != ':'))) {
            return Path::GetCurrentDirectoryA().Substring(0, 1) + ':' + path;
        } else {
            /* UNC path or begins with drive letter. */
            return path;
        }
#else /* _WIN32 */
        return path;
#endif /* _WIN32 */

    } else if ((path[0] == MYDOCUMENTS_MARKER_A) 
            && ((path.Length() == 1) || path[1] == SEPARATOR_A)) {
        path.Replace(MYDOCUMENTS_MARKER_A, Path::GetUserHomeDirectoryA());
        return Path::Canonicalise(path);

    } else {
        /*
         * Concatenate current directory and relative path, and canonicalise
         * the result.
         */

        return Path::Concatenate(Path::GetCurrentDirectoryA(), path, true);
    }
}


/*
 * vislib::sys::Path::Resolve
 */
vislib::StringW vislib::sys::Path::Resolve(StringW path) {
    // TODO: Windows shell API resolve does not work in the expected
    // way, so we use the same manual approach for Windows and Linux.

#ifdef _WIN32
    /* Remove unchefm‰ﬂige path separators. */
    path.Replace(L'/', SEPARATOR_W);
#endif /* _WIN32 */

    if (path.IsEmpty()) {
        /* Path is empty, i. e. return current working directory. */
        return Path::GetCurrentDirectoryW();

    } else if (Path::IsAbsolute(path)) {
        /* Path is absolute, just return it. */
#ifdef _WIN32
        if ((path.Length() < 2) || ((path[1] != SEPARATOR_W) 
                && (path[1] != L':'))) {
            return Path::GetCurrentDirectoryW().Substring(0, 1) + L':' + path;
        } else {
            /* UNC path or begins with drive letter. */
            return path;
        }
#else /* _WIN32 */
        return path;
#endif /* _WIN32 */

    } else if ((path[0] == MYDOCUMENTS_MARKER_W) 
            && ((path.Length() == 1) || path[1] == SEPARATOR_W)) {
        path.Replace(MYDOCUMENTS_MARKER_W, Path::GetUserHomeDirectoryW());
        return Path::Canonicalise(path);

    } else {
        /*
         * Concatenate current directory and relative path, and canonicalise
         * the result.
         */
        return Concatenate(Path::GetCurrentDirectoryW(), path, true);
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
