/*
 * CharTraits.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/CharTraits.h"

#include <cstdlib>
#include <wchar.h>

#include "the/stack_trace.h"


/*
 * vislib::CharTraitsA<char>::ParseBool
 */
bool vislib::CharTraits<char>::ParseBool(const Char *str) {
    if (str == NULL) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    }

    if (
#ifdef _WIN32
            (_stricmp("true", str) == 0) || (_stricmp("t", str) == 0) || 
            (_stricmp("yes", str) == 0) || (_stricmp("y", str) == 0) || 
            (_stricmp("on", str) == 0)
#else /* _WIN32 */
            (strcasecmp("true", str) == 0) || (strcasecmp("t", str) == 0) ||
            (strcasecmp("yes", str) == 0) || (strcasecmp("y", str) == 0) || 
            (strcasecmp("on", str) == 0)
#endif /* _WIN32 */
            ) {
        return true;
    }

    if (
#ifdef _WIN32
            (_stricmp("false", str) == 0) || (_stricmp("f", str) == 0) || 
            (_stricmp("no", str) == 0) || (_stricmp("n", str) == 0) || 
            (_stricmp("off", str) == 0)
#else /* _WIN32 */
            (strcasecmp("false", str) == 0) || (strcasecmp("f", str) == 0) || 
            (strcasecmp("no", str) == 0) || (strcasecmp("n", str) == 0) || 
            (strcasecmp("off", str) == 0)
#endif /* _WIN32 */
            ) {
        return false;
    }

    try {
        int i = ParseInt(str);
        return (i != 0);
    } catch (...) {
    }

    throw the::format_exception("Cannot convert String to Boolean", __FILE__, 
        __LINE__);
}


/*
 * vislib::CharTraits<char>::ParseDouble
 */
double vislib::CharTraits<char>::ParseDouble(const Char *str) {
    double retval;
    if (str == NULL) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    }
    
    if (
#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
            sscanf_s
#else  /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
            sscanf
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
            (str, "%lf", &retval) != 1) {
        throw the::format_exception("Cannot convert String to Double", __FILE__, 
            __LINE__);
    }

    return retval;
}


/*
 * vislib::CharTraits<char>::ParseInt
 */
int vislib::CharTraits<char>::ParseInt(const Char *str) {
    int retval;
    if (str == NULL) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    }

    if (
#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
            sscanf_s
#else  /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
            sscanf
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
            (str, "%d", &retval) != 1) {
        throw the::format_exception("Cannot convert String to Integer", __FILE__, 
            __LINE__);
    }
    
    return retval;
}


/*
 * vislib::CharTraits<char>::ParseInt64
 */
int64_t vislib::CharTraits<char>::ParseInt64(const Char *str) {
    THE_STACK_TRACE;
    if (str == NULL) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    }
    Char *end = NULL;
    int64_t retval = 0;

#ifdef _WIN32
    retval = ::_strtoi64(str, &end, 10);
#else /* _WIN32 */
    retval = static_cast<int64_t>(::strtoll(str, &end, 10));
#endif /* _WIN32 */

    if (str < end) {
        return retval;
    } else {
        throw the::format_exception("Cannot convert String to 64 bit integer",
            __FILE__, __LINE__);
    }
}


/*
 * vislib::CharTraits<char>::ParseUInt64
 */
uint64_t vislib::CharTraits<char>::ParseUInt64(const Char *str) {
    THE_STACK_TRACE;
    if (str == NULL) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    }
    Char *end = NULL;
    uint64_t retval = 0;

#ifdef _WIN32
    retval = ::_strtoui64(str, &end, 10);
#else /* _WIN32 */
    retval = static_cast<uint64_t>(::strtoull(str, &end, 10));
#endif /* _WIN32 */

    if (str < end) {
        return retval;
    } else {
        throw the::format_exception("Cannot convert String to 64 bit integer",
            __FILE__, __LINE__);
    }
}


/*
 * vislib::CharTraits<char>::Format
 */
vislib::CharTraits<char>::Size vislib::CharTraits<char>::Format(Char *dst,
        const Size cnt, const Char *fmt, va_list argptr) {
    int retval = -1;

#ifdef _WIN32
    if ((dst == NULL) || (cnt <= 0)) {
        /* Answer the prospective size of the string. */
        retval = ::_vscprintf(fmt, argptr);

    } else {
#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
        retval = ::_vsnprintf_s(dst, cnt, cnt, fmt, argptr);
#else /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
        retval = ::_vsnprintf(dst, cnt, fmt, argptr);
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
    } /* end if ((dst == NULL) || (cnt <= 0)) */

#else /* _WIN32 */
    retval = ::vsnprintf(dst, cnt, fmt, argptr);

    if ((dst != NULL) && (cnt > 0) && (retval > cnt - 1)) {
        retval = -1;
    }
#endif /* _WIN32 */ 

    /* Ensure string being terminated. */
    if ((dst != NULL) && (cnt > 0)) {
        dst[cnt - 1] = 0;
    }
    return static_cast<Size>(retval);
}


/*
 * vislib::CharTraits<char>::ToLower
 */
vislib::CharTraits<char>::Size vislib::CharTraits<char>::ToLower(
        Char *dst, const Size cnt, const Char *str) {
    // TODO: This implementation is a hack! Size might change if conversion
    // is performed correctly.
    THE_ASSERT(str != NULL);
    Size retval = static_cast<Size>(::strlen(str));

#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
    if (::strncpy_s(dst, cnt, str, _TRUNCATE) != 0) {
        retval = -1;
    } else if (::_strlwr_s(dst, cnt) != 0) {
        retval = -1;
    }

#else /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
    if (cnt >= retval + 1) {

#ifdef _WIN32
        ::strcpy(dst, str);
        ::_strlwr(dst);
#else /* _WIN32 */
        Char *d = dst;
        const Char *s = str;
        while ((*d++ = ToLower(*s++)) != static_cast<Char>(0));
#endif /* _WIN32 */
    } else {
        retval = -1;
    }
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */

    /* Ensure string being terminated. */
    if ((dst != NULL) && (cnt > 0)) {
        dst[cnt - 1] = 0;
    }
    return retval;
}


/*
 * vislib::CharTraits<char>::ToUpper
 */
vislib::CharTraitsA::Size vislib::CharTraits<char>::ToUpper(
        Char *dst, const Size cnt, const Char *str) {
    // TODO: This implementation is a hack! Size might change if conversion
    // is performed correctly.
    THE_ASSERT(str != NULL);
    Size retval = static_cast<Size>(::strlen(str));

#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
    if (::strncpy_s(dst, cnt, str, _TRUNCATE) != 0) {
        retval = -1;
    } else if (::_strupr_s(dst, cnt) != 0) {
        retval = -1;
    }

#else /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
    if (cnt >= retval + 1) {

#ifdef _WIN32
        ::strcpy(dst, str);
        ::_strupr(dst);
#else /* _WIN32 */
        Char *d = dst;
        const Char *s = str;
        while ((*d++ = ToUpper(*s++)) != static_cast<Char>(0));
#endif /* _WIN32 */
    } else {
        retval = -1;
    }
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */

    /* Ensure string being terminated. */
    if ((dst != NULL) && (cnt > 0)) {
        dst[cnt - 1] = 0;
    }
    return retval;
}


////////////////////////////////////////////////////////////////////////////////


/*
 * vislib::CharTraits<wchar_t>::ParseBool
 */
bool vislib::CharTraits<wchar_t>::ParseBool(const Char *str) {
    if (str == NULL) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    }

    if (
#ifdef _WIN32
            (_wcsicmp(L"true", str) == 0) || (_wcsicmp(L"t", str) == 0) || 
            (_wcsicmp(L"yes", str) == 0) || (_wcsicmp(L"y", str) == 0) || 
            (_wcsicmp(L"on", str) == 0)
#else /* _WIN32 */
            (wcscasecmp(L"true", str) == 0) || (wcscasecmp(L"t", str) == 0) || 
            (wcscasecmp(L"yes", str) == 0) || (wcscasecmp(L"y", str) == 0) || 
            (wcscasecmp(L"on", str) == 0)
#endif /* _WIN32 */
            ) {
        return true;
    }

    if (
#ifdef _WIN32
            (_wcsicmp(L"false", str) == 0) || (_wcsicmp(L"f", str) == 0) || 
            (_wcsicmp(L"no", str) == 0) || (_wcsicmp(L"n", str) == 0) || 
            (_wcsicmp(L"off", str) == 0)
#else /* _WIN32 */
            (wcscasecmp(L"false", str) == 0) || (wcscasecmp(L"f", str) == 0) || 
            (wcscasecmp(L"no", str) == 0) || (wcscasecmp(L"n", str) == 0) || 
            (wcscasecmp(L"off", str) == 0)
#endif /* _WIN32 */
            ) {
        return false;
        }

        try {
            int i = ParseInt(str);
            return (i != 0);
        } catch (...) {
        }

        throw the::format_exception("Cannot convert String to Boolean", __FILE__, 
            __LINE__);
    }


/*
 * vislib::CharTraits<wchar_t>::ParseDouble
 */
double vislib::CharTraits<wchar_t>::ParseDouble(const Char *str) {
    double retval;
    if (str == NULL) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    }

    if (
#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
            swscanf_s
#else  /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
            swscanf
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
            (str, L"%lf", &retval) != 1) {
        throw the::format_exception("Cannot convert String to Double", __FILE__, 
            __LINE__);
    }

    return retval;
}


/*
 * vislib::CharTraits<wchar_t>::ParseInt
 */
int vislib::CharTraits<wchar_t>::ParseInt(const Char *str) {
    int retval;
    if (str == NULL) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    }

    if (
#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
            swscanf_s
#else  /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
            swscanf
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
            (str, L"%d", &retval) != 1) {
        throw the::format_exception("Cannot convert String to Integer", __FILE__, 
            __LINE__);
    }
    
    return retval;
}


/*
 * vislib::CharTraits<char>::ParseInt64
 */
int64_t vislib::CharTraits<wchar_t>::ParseInt64(const Char *str) {
    THE_STACK_TRACE;
    if (str == NULL) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    }
    Char *end = NULL;
    int64_t retval = 0;

#ifdef _WIN32
    retval = ::_wcstoi64(str, &end, 10);
#else /* _WIN32 */
    retval = static_cast<int64_t>(::wcstoll(str, &end, 10));
#endif /* _WIN32 */

    if (str < end) {
        return retval;
    } else {
        throw the::format_exception("Cannot convert String to 64 bit integer",
            __FILE__, __LINE__);
    }
}


/*
 * vislib::CharTraits<char>::ParseUInt64
 */
uint64_t vislib::CharTraits<wchar_t>::ParseUInt64(const Char *str) {
    THE_STACK_TRACE;
    if (str == NULL) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    }
    Char *end = NULL;
    uint64_t retval = 0;

#ifdef _WIN32
    retval = ::_wcstoui64(str, &end, 10);
#else /* _WIN32 */
    retval = static_cast<uint64_t>(::wcstoull(str, &end, 10));
#endif /* _WIN32 */

    if (str < end) {
        return retval;
    } else {
        throw the::format_exception("Cannot convert String to 64 bit integer",
            __FILE__, __LINE__);
    }
}


/*
 * vislib::CharTraits<wchar_t>::Format
 */
vislib::CharTraits<wchar_t>::Size vislib::CharTraits<wchar_t>::Format(
        Char *dst, const Size cnt, const Char *fmt, va_list argptr) {
    int retval = -1;

#ifdef _WIN32
    if ((dst == NULL) || (cnt <= 0)) {
        /* Answer the prospective size of the string. */
        retval = ::_vscwprintf(fmt, argptr);

    } else {
#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
        retval = ::_vsnwprintf_s(dst, cnt, cnt, fmt, argptr);
#else /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
        retval = ::_vsnwprintf(dst, cnt, fmt, argptr);
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
    } /* end if ((dst == NULL) || (cnt <= 0)) */

#else /* _WIN32 */
    // Yes, you can trust your eyes: The char and wide char implementations
    // under Linux have a completely different semantics. vswprintf cannot 
    // be used for determining the required size as vsprintf can.
    size_t bufferSize, bufferGrow;
    Char *buffer = NULL;

    if ((dst == NULL) || (cnt <= 0)) {
        /* Just count. */
        bufferSize = static_cast<size_t>(1.1 * static_cast<float>(
            ::wcslen(fmt)) + 1);
        bufferGrow = static_cast<size_t>(0.5 * static_cast<float>(
            bufferSize));
        buffer = new Char[bufferSize];

        while ((retval = ::vswprintf(buffer, bufferSize, fmt, argptr)) 
                == -1) {
            the::safe_array_delete(buffer);
            bufferSize += bufferGrow;
            buffer = new Char[bufferSize];
        }

        retval = ::wcslen(buffer);
        the::safe_array_delete(buffer);
            
    } else {
        /* Format the string. */
        retval = ::vswprintf(dst, cnt, fmt, argptr);
    }
#endif /* _WIN32 */

    /* Ensure string being terminated. */
    if ((dst != NULL) && (cnt > 0)) {
        dst[cnt - 1] = 0;
    }
    return static_cast<Size>(retval);
}


/*
* vislib::CharTraits<wchar_t>::ToLower
 */
vislib::CharTraits<wchar_t>::Size vislib::CharTraits<wchar_t>::ToLower(
        Char *dst, const Size cnt, const Char *str) {
    // TODO: This implementation is a hack! Size might change if conversion
    // is performed correctly.
    THE_ASSERT(str != NULL);
    Size retval = static_cast<Size>(::wcslen(str));

#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
    if (::wcsncpy_s(dst, cnt, str, _TRUNCATE) != 0) {
        retval = -1;
    } else if (::_wcslwr_s(dst, cnt) != 0) {
        retval = -1;
    }

#else /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
    if (cnt >= retval + 1) {

#ifdef _WIN32
        ::wcscpy(dst, str);
        ::_wcslwr(dst);
#else /* _WIN32 */
        Char *d = dst;
        const Char *s = str;
        while ((*d++ = ToLower(*s++)) != static_cast<Char>(0));
#endif /* _WIN32 */
    } else {
        retval = -1;
    }
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */

    /* Ensure string being terminated. */
    if ((dst != NULL) && (cnt > 0)) {
        dst[cnt - 1] = 0;
    }
    return retval;
}


/*
 * vislib::CharTraits<wchar_t>::ToUpper
 */
vislib::CharTraits<wchar_t>::Size vislib::CharTraits<wchar_t>::ToUpper(
        Char *dst, const Size cnt, const Char *str) {
    // TODO: This implementation is a hack! Size might change if conversion
    // is performed correctly.
    THE_ASSERT(str != NULL);
    Size retval = static_cast<Size>(::wcslen(str));

#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
    if (::wcsncpy_s(dst, cnt, str, _TRUNCATE) != 0) {
        retval = -1;
    } else if (::_wcsupr_s(dst, cnt) != 0) {
        retval = -1;
    }

#else /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
    if (cnt >= retval + 1) {

#ifdef _WIN32
        ::wcscpy(dst, str);
        ::_wcsupr(dst);
#else /* _WIN32 */
        Char *d = dst;
        const Char *s = str;
        while ((*d++ = ToUpper(*s++)) != static_cast<Char>(0));
#endif /* _WIN32 */
    } else {
        retval = -1;
    }
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */

    /* Ensure string being terminated. */
    if ((dst != NULL) && (cnt > 0)) {
        dst[cnt - 1] = 0;
    }
    return retval;
}
