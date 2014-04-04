/*
 * Environment.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/Environment.h"

#include <cstdarg>

#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include <cstdlib>
#endif /* _WIn32 */

#include "the/assert.h"
#include "the/memory.h"
#include "the/text/string_converter.h"
#include "the/system/system_exception.h"
#include "the/trace.h"
#include "the/text/string_buffer.h"

#ifndef _WIN32
/** Gain access to the global environment data provided by the system. */
extern char **environ;
#endif /* !_WIN32 */


/*
 * vislib::sys::Environment::Snapshot::Snapshot
 */
vislib::sys::Environment::Snapshot::Snapshot(void) : data(NULL) {
}


/*
 * vislib::sys::Environment::Snapshot::Snapshot
 */
vislib::sys::Environment::Snapshot::Snapshot(const char *variable, ...) 
        : data(NULL) {
    va_list argptr;
    size_t dataSize = 0;
    const char *arg;
    
    if (variable != NULL) {
#ifdef _WIN32
        wchar_t *insPos = NULL;

        /* Determine the required buffer size. */
        dataSize = ::strlen(variable) + 2;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const char *)) != NULL) {
            dataSize += (::strlen(arg) + 1);
        }
        va_end(argptr);

        /* Allocate buffer. */
        insPos = this->data.allocate(dataSize);

        /* Copy the input. */
        dataSize = ::strlen(variable) + 1;
        ::memcpy(insPos, THE_A2W(variable), dataSize * sizeof(wchar_t));
        insPos += dataSize;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const char *)) != NULL) {
            dataSize = ::strlen(arg) + 1;
            ::memcpy(insPos, THE_A2W(arg), dataSize * sizeof(wchar_t));
            insPos += dataSize;
        }
        va_end(argptr);

        /* Insert terminating double zero. */
        *insPos = L'\0';

#else /* _WIN32 */
        char **insPos = NULL;

        /* Count parameters. */
        dataSize = 1;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const char *)) != NULL) {
            dataSize++;
        }
        va_end(argptr);

        /* Allocate parameter array. */
        this->data = insPos = new char *[dataSize];

        /* Allocate variable memory and copy data. */
        *insPos = new char[::strlen(variable) + 1];
        ::strcpy(*insPos, variable);
        insPos++;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const char *)) != NULL) {
            *insPos = new char[::strlen(arg) + 1];
            ::strcpy(*insPos, arg);
            insPos++;
        }
        va_end(argptr);

        /* Last array element must be a NULL pointer. */
        *insPos = NULL;

#endif /* _WIN32 */
    }
}


/*
 * vislib::sys::Environment::Snapshot::Snapshot
 */
vislib::sys::Environment::Snapshot::Snapshot(const wchar_t *variable, ...) 
        : data(NULL) {
    va_list argptr;
    size_t dataSize = 0;
    const wchar_t *arg;
    
    if (variable != NULL) {
#ifdef _WIN32
        wchar_t *insPos = NULL;

        /* Determine the required buffer size. */
        dataSize = ::wcslen(variable) + 2;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const wchar_t *)) != NULL) {
            dataSize += (::wcslen(arg) + 1);
        }
        va_end(argptr);

        /* Allocate buffer. */
        insPos = this->data.allocate(dataSize);

        /* Copy the input. */
        dataSize = ::wcslen(variable) + 1;
        ::memcpy(insPos, variable, dataSize * sizeof(wchar_t));
        insPos += dataSize;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const wchar_t *)) != NULL) {
            dataSize = ::wcslen(arg) + 1;
            ::memcpy(insPos, arg, dataSize * sizeof(wchar_t));
            insPos += dataSize;
        }
        va_end(argptr);

        /* Insert terminating double zero. */
        *insPos = L'\0';

#else /* _WIN32 */
        char **insPos = NULL;

        /* Count parameters. */
        dataSize = 1;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const wchar_t *)) != NULL) {
            dataSize++;
        }
        va_end(argptr);

        /* Allocate parameter array. */
        this->data = insPos = new char *[dataSize];

        /* Allocate variable memory and copy data. */
        *insPos = new char[::wcslen(variable) + 1];
        ::strcpy(*insPos, THE_W2A(variable));
        insPos++;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const wchar_t *)) != NULL) {
            *insPos = new char[::wcslen(arg) + 1];
            ::strcpy(*insPos, THE_W2A(arg));
            insPos++;
        }
        va_end(argptr);

        /* Last array element must be a NULL pointer. */
        *insPos = NULL;

#endif /* _WIN32 */
    }
}


/*
 * vislib::sys::Environment::Snapshot::Snapshot
 */
vislib::sys::Environment::Snapshot::Snapshot(const Snapshot& rhs) : data(NULL) {
#ifdef _WIN32
    this->data = rhs.data;
#else /* _WIN32 */
    this->assign(const_cast<const char **>(rhs.data));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::Snapshot::Snapshot
 */
vislib::sys::Environment::Snapshot::~Snapshot(void) {
    this->Clear();
}


/*
 * vislib::sys::Environment::Snapshot::Clear
 */
void vislib::sys::Environment::Snapshot::Clear(void) {
#ifdef _WIN32
    this->data.clear();

#else /* _WIN32 */
    if (this->data != NULL) {
        char **cursor = this->data;

        while (*cursor != NULL) {
            the::safe_array_delete(*cursor);
            cursor++;
        }
    }

    the::safe_array_delete(this->data);
    THE_ASSERT(this->data == NULL);
#endif /* !_WIN32 */
}


/*
 * vislib::sys::Environment::Snapshot::GetAt
 */
void vislib::sys::Environment::Snapshot::GetAt(const size_t idx,
        the::astring& outName, the::astring& outValue) {
#ifdef _WIN32
    the::wstring name, value;
    this->GetAt(idx, name, value);
    the::text::string_converter::convert(outName, name);
    the::text::string_converter::convert(outValue, value);
#else /* _WIN32 */
    size_t cntVariables = Snapshot::count(const_cast<const char **>(data));

    if (idx < cntVariables) {
        const char *tmp = this->data[idx];
        while (*tmp++ != '=');
        outName = the::astring(this->data[idx], tmp - this->data[idx] - 1);
        outValue = the::astring(tmp);
    } else {
        throw the::index_out_of_range_exception(static_cast<int>(idx), 0, cntVariables - 1,
            __FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::Snapshot::GetAt
 */
void vislib::sys::Environment::Snapshot::GetAt(const size_t idx, 
        the::wstring& outName, the::wstring& outValue)  {
#ifdef _WIN32
    the::wstring tmp = this->data[idx];
    auto splitPos = tmp.find(L'=');
    THE_ASSERT(splitPos != the::wstring::npos);
    outName = tmp.substr(0, splitPos);
    outValue = tmp.substr(splitPos + 1);
#else /* _WIN32 */
    the::astring name, value;
    this->GetAt(idx, name, value);
    the::text::string_converter::convert(outName, name);
    the::text::string_converter::convert(outValue, value);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::Snapshot::GetVariable
 */
the::astring vislib::sys::Environment::Snapshot::GetVariable(
        const char *name) const {
#ifdef _WIN32
    return the::text::string_converter::to_a(this->GetVariable(THE_A2W(name)));
#else /* _WIN32 */
    if ((name != NULL) && (*name != 0)) {
        const char *value = Snapshot::find(name, 
            const_cast<const char **>(this->data));
        if (value != NULL) {
            while (*value++ != '=');
            return the::astring(value);
        }
    }

    return the::astring();
#endif /* _WIN32 */
}

/*
 * vislib::sys::Environment::Snapshot::GetVariable
 */
the::wstring vislib::sys::Environment::Snapshot::GetVariable(
        const wchar_t *name) const {
#ifdef _WIN32
    if ((name != NULL) && (*name != 0)) {
        const wchar_t *value = Snapshot::find(name, this->data.data());
        if (value != NULL) {
            while (*value++ != L'=');
            return the::wstring(value);
        }
    }

    return the::wstring();
#else /* _WIN32 */
    return the::text::string_converter::to_w(this->GetVariable(THE_W2A(name)));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::Snapshot::IsSet
 */
bool vislib::sys::Environment::Snapshot::IsSet(const char *name) const {
#ifdef _WIN32
    return (Snapshot::find(THE_A2W(name), this->data.data()) != NULL);
#else /* _WIN32 */
    return (Snapshot::find(name, const_cast<const char **>(this->data)) 
        != NULL);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::Snapshot::IsSet
 */
bool vislib::sys::Environment::Snapshot::IsSet(const wchar_t *name) const {
#ifdef _WIN32
    return (Snapshot::find(name, this->data.data()) != NULL);
#else /* _WIN32 */
    return (Snapshot::find(THE_W2A(name), const_cast<const char **>(this->data)) 
        != NULL);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::Snapshot::Snapshot
 */
vislib::sys::Environment::Snapshot& 
vislib::sys::Environment::Snapshot::operator =(const Snapshot& rhs) {
    if (this != &rhs) {
#ifdef _WIN32
        this->data = rhs.data;
#else /* _WIN32 */
        this->assign(const_cast<const char **>(rhs.data));
#endif /* _WIN32 */
    }

    return *this;
}


#ifndef _WIN32
/*
 * vislib::sys::Environment::Snapshot::count
 */
size_t vislib::sys::Environment::Snapshot::count(const char **const data) {
    const char **cursor = data;

    if (cursor != NULL) {
        while ((*cursor) != NULL) {
            cursor++;
        }
    }

    return (cursor - data);
}
#endif /* !_WIN32 */


/*
 * vislib::sys::Environment::Snapshot::find
 */
#ifdef _WIN32
const wchar_t *vislib::sys::Environment::Snapshot::find(const wchar_t *name,
        const wchar_t *data) {
    const wchar_t *cursor = data;
    size_t cntName = ::wcslen(name);
    
    if (cursor != NULL) {
        while (*cursor != 0) {
            // Windows environment variables are case insensitive
            if ((::_wcsnicmp(name, cursor, cntName) == 0)
                    && (cursor[cntName] == L'=')) {
                /* Variable found. */
                return cursor;
            } else {
                /* Consume name-variable pair. */
                while (*cursor++ != 0);
            }

        }
    }
    /* Nothing found at this point. */

    return NULL;
}
#else /* _WIN32 */
const char *vislib::sys::Environment::Snapshot::find(const char *name, 
        const char **const data) {
    size_t cntVariables = Snapshot::count(data);
    size_t cntName = ::strlen(name);
    
    for (size_t i = 0; i < cntVariables; i++) {
        // Linux environment variables are case sensitive.
        if ((::strncmp(name, data[i], cntName) == 0) 
                && (data[i][cntName] == '=')) {
            /* Variable found. */
            return data[i];
        }
    }
    /* Nothing found at this point. */

    return NULL;
}
#endif /* _WIN32 */


#ifndef _WIN32
/*
 * vislib::sys::Environment::Snapshot::assign
 */
void vislib::sys::Environment::Snapshot::assign(const char **const data) {
    size_t dataSize = 0;

    this->Clear();
    THE_ASSERT(this->data == NULL);
    
    if (data != NULL) {
        /* Count parameters. */
        dataSize = Snapshot::count(data);

        /* Allocate parameter array. */
        this->data = new char *[dataSize + 1];

        /* Allocate variable memory and copy data. */
        for (size_t i = 0; i < dataSize; i++) {
            this->data[i] = new char[::strlen(data[i]) + 1];
            ::strcpy(this->data[i], data[i]);
        }

    this->data[dataSize] = NULL;
    }
}
#endif /* !_WIN32 */


/*
 * vislib::sys::Environment::CreateSnapshot
 */
vislib::sys::Environment::Snapshot 
vislib::sys::Environment::CreateSnapshot(void) {
    Snapshot retval;    

#ifdef _WIN32
    wchar_t *env = ::GetEnvironmentStringsW();
    if (env == NULL) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    retval.data = env;
    
    if (!::FreeEnvironmentStringsW(env)) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    retval.assign(const_cast<const char **>(::environ));
#endif /* _WIN32 */
    
    return retval;
}


/*
 * vislib::sys::Environment::GetVariable
 */
the::astring vislib::sys::Environment::GetVariable(const char *name, 
                                                      const bool isLenient) {
#ifdef _WIN32
    the::astring retval;
    unsigned int error = NO_ERROR;
    unsigned int size = ::GetEnvironmentVariableA(name, NULL, 0);
    
    if (size == 0) {
        if (!isLenient || ((error = ::GetLastError()) 
                != ERROR_ENVVAR_NOT_FOUND)) {
            throw the::system::system_exception(error, __FILE__, __LINE__);
        }
    }

    size = ::GetEnvironmentVariableA(name,
        the::text::string_buffer_allocate(retval, size + 1), size + 1);
    if (size == 0) {
        if (!isLenient || ((error = ::GetLastError()) 
                != ERROR_ENVVAR_NOT_FOUND)) {
            throw the::system::system_exception(error, __FILE__, __LINE__);
        } else {
            retval[0] = 0;
        }
    }

    return retval;

#else /* _WIN32 */
    return the::astring(::getenv(name));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::GetVariable
 */
the::wstring vislib::sys::Environment::GetVariable(const wchar_t *name,
                                                      const bool isLenient) {
#ifdef _WIN32
    the::wstring retval;
    unsigned int error = NO_ERROR;
    unsigned int size = ::GetEnvironmentVariableW(name, NULL, 0);
    
    if (size == 0) {
        if (!isLenient || ((error = ::GetLastError()) 
                != ERROR_ENVVAR_NOT_FOUND)) {
            throw the::system::system_exception(error, __FILE__, __LINE__);
        }
    }

    size = ::GetEnvironmentVariableW(name,
        the::text::string_buffer_allocate(retval, size + 1), size + 1);
    if (size == 0) {
        if (!isLenient || ((error = ::GetLastError()) 
                != ERROR_ENVVAR_NOT_FOUND)) {
            throw the::system::system_exception(error, __FILE__, __LINE__);
        } else {
            retval[0] = 0;
        }
    }

    return retval;
#else /* _WIN32 */
    return the::text::string_converter::to_w(Environment::GetVariable(THE_W2A(name)));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::IsSet
 */
bool vislib::sys::Environment::IsSet(const char *name) {
#ifdef _WIN32
    if (::GetEnvironmentVariableA(name, NULL, 0) == 0) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    return (::GetLastError() != ERROR_ENVVAR_NOT_FOUND);
#else /* _WIN32 */
    return (::getenv(name) != NULL);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::IsSet
 */
bool vislib::sys::Environment::IsSet(const wchar_t *name) {
#ifdef _WIN32
    if (::GetEnvironmentVariableW(name, NULL, 0) == 0) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    return (::GetLastError() != ERROR_ENVVAR_NOT_FOUND);
#else /* _WIN32 */
    return Environment::IsSet(THE_W2A(name));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::SetVariable
 */
void vislib::sys::Environment::SetVariable(const char *name, 
                                           const char *value) {
#ifdef _WIN32
    if (!::SetEnvironmentVariableA(name, value)) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    if (value != NULL) {
        if (::setenv(name, value, 1) == -1) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }
    } else {
        if (::unsetenv(name) == -1) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::SetVariable
 */
void vislib::sys::Environment::SetVariable(const wchar_t *name, 
                                           const wchar_t *value) {
#ifdef _WIN32
    if (!::SetEnvironmentVariableW(name, value)) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    Environment::SetVariable(THE_W2A(name), THE_W2A(value));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::~Environment
 */
vislib::sys::Environment::~Environment(void) {
    /* Nothing to do. */
}


/*
 * vislib::sys::Environment::Environment
 */
vislib::sys::Environment::Environment(void) {
    /* Nothing to do. */
}
