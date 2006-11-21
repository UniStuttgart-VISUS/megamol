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
#include <unistd.h>
#endif /* _WIN32 */

#include "vislib/assert.h"
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

    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATORSTR_A);
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATORSTR_A);

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
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATORSTR_A);
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATORSTR_A);

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

    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATORSTR_W);
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATORSTR_W);

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
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATORSTR_W);
    retval.Replace(DOUBLE_SEPARATOR.PeekBuffer(), SEPARATORSTR_W);

    return retval;
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
    return retval;

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
    return retval;

#endif /* _WIN32 */
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
    return retval;

#else /* _WIN32 */
    return StringW(GetCurrentDirectoryA());
#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::Resolve
 */
vislib::StringA vislib::sys::Path::Resolve(const StringA& path) {
#ifdef _WIN32
    return StringA(Resolve(static_cast<StringW>(path)));

#else /* _WIN32 */
    if (path.IsEmpty()) {
        /* Path is empty, i. e. return current working directory. */
        return GetCurrentDirectoryA();

    } else if (path[0] == SEPARATOR_A) {
        /* Path is absolute, just return it. */
        return path;

    } else {
        /*
         * Concatenate current directory and relative path, and canonicalise
         * the result.
         */
        return Canonicalise(GetCurrentDirectoryA() + SEPARATOR_A + path);
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::Path::Resolve
 */
vislib::StringW vislib::sys::Path::Resolve(const StringW& path) {
#ifdef _WIN32
    StringW retval;
    wchar_t *p = retval.AllocateBuffer(MAX_PATH);
    ::wcsncpy(p, path.PeekBuffer(), MAX_PATH); 

    if (::PathCanonicalizeW(p, path) != TRUE) {
        throw SystemException(__FILE__, __LINE__);
    }

    return retval;
#else /* _WIN32 */
    if (path.IsEmpty()) {
        /* Path is empty, i. e. return current working directory. */
        return GetCurrentDirectoryW();

    } else if (path[0] == SEPARATOR_W) {
        /* Path is absolute, just return it. */
        return path;

    } else {
        /*
         * Concatenate current directory and relative path, and canonicalise
         * the result.
         */
        return Canonicalise(GetCurrentDirectoryW() + SEPARATOR_W + path);
    }

#endif /* _WIN32 */
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
 * vislib::sys::Path::SEPARATORSTR_A
 */
const char vislib::sys::Path::SEPARATORSTR_A[] 
    = { SEPARATOR_A, static_cast<char>(0) };


/*
 * vislib::sys::Path::SEPARATORSTR_W
 */
const wchar_t vislib::sys::Path::SEPARATORSTR_W[] 
    = { SEPARATOR_W, static_cast<wchar_t>(0) };


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
