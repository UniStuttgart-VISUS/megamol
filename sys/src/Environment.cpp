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

#include "vislib/assert.h"
#include "vislib/memutils.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"

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
    SIZE_T dataSize = 0;
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
        insPos = this->data.AllocateBuffer(dataSize);

        /* Copy the input. */
        dataSize = ::strlen(variable) + 1;
        ::memcpy(insPos, A2W(variable), dataSize * sizeof(wchar_t));
        insPos += dataSize;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const char *)) != NULL) {
            dataSize = ::strlen(arg) + 1;
            ::memcpy(insPos, A2W(arg), dataSize * sizeof(wchar_t));
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
    SIZE_T dataSize = 0;
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
        insPos = this->data.AllocateBuffer(dataSize);

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
        ::strcpy(*insPos, W2A(variable));
        insPos++;

        va_start(argptr, variable);
        while ((arg = va_arg(argptr, const wchar_t *)) != NULL) {
            *insPos = new char[::wcslen(arg) + 1];
            ::strcpy(*insPos, W2A(arg));
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
    this->data.Clear();

#else /* _WIN32 */
    if (this->data != NULL) {
        char **cursor = this->data;

        while (*cursor != NULL) {
            ARY_SAFE_DELETE(*cursor);
            cursor++;
        }
    }

    ARY_SAFE_DELETE(this->data);
    ASSERT(this->data == NULL);
#endif /* !_WIN32 */
}


/*
 * vislib::sys::Environment::Snapshot::GetAt
 */
void vislib::sys::Environment::Snapshot::GetAt(const SIZE_T idx,
        StringA& outName, StringA& outValue) {
#ifdef _WIN32
    StringW name, value;
    this->GetAt(idx, name, value);
    outName = name;
    outValue = value;
#else /* _WIN32 */
    SIZE_T cntVariables = Snapshot::count(const_cast<const char **>(data));

    if (idx < cntVariables) {
        const char *tmp = this->data[idx];
        while (*tmp++ != '=');
        outName = StringA(this->data[idx], tmp - this->data[idx] - 1);
        outValue = StringA(tmp);
    } else {
        throw OutOfRangeException(static_cast<int>(idx), 0, cntVariables - 1,
            __FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::Snapshot::GetAt
 */
void vislib::sys::Environment::Snapshot::GetAt(const SIZE_T idx, 
        StringW& outName, StringW& outValue)  {
#ifdef _WIN32
    StringW tmp = this->data[idx];
    StringW::Size splitPos = tmp.Find(L'=');
    ASSERT(splitPos != StringW::INVALID_POS);
    outName = tmp.Substring(0, splitPos);
    outValue = tmp.Substring(splitPos + 1);
#else /* _WIN32 */
    StringA name, value;
    this->GetAt(idx, name, value);
    outName = name;
    outValue = value;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::Snapshot::GetVariable
 */
vislib::StringA vislib::sys::Environment::Snapshot::GetVariable(
        const char *name) const {
#ifdef _WIN32
    return StringA(this->GetVariable(A2W(name)));
#else /* _WIN32 */
    if ((name != NULL) && (*name != 0)) {
        const char *value = Snapshot::find(name, 
            const_cast<const char **>(this->data));
        if (value != NULL) {
            while (*value++ != '=');
            return StringA(value);
        }
    }

    return StringA();
#endif /* _WIN32 */
}

/*
 * vislib::sys::Environment::Snapshot::GetVariable
 */
vislib::StringW vislib::sys::Environment::Snapshot::GetVariable(
        const wchar_t *name) const {
#ifdef _WIN32
    if ((name != NULL) && (*name != 0)) {
        const wchar_t *value = Snapshot::find(name, this->data.PeekBuffer());
        if (value != NULL) {
            while (*value++ != L'=');
            return StringW(value);
        }
    }

    return StringW();
#else /* _WIN32 */
    return StringW(this->GetVariable(W2A(name)));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::Snapshot::IsSet
 */
bool vislib::sys::Environment::Snapshot::IsSet(const char *name) const {
#ifdef _WIN32
    return (Snapshot::find(A2W(name), this->data.PeekBuffer()) != NULL);
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
    return (Snapshot::find(name, this->data.PeekBuffer()) != NULL);
#else /* _WIN32 */
    return (Snapshot::find(W2A(name), const_cast<const char **>(this->data)) 
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
SIZE_T vislib::sys::Environment::Snapshot::count(const char **const data) {
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
    SIZE_T cntName = ::wcslen(name);
    
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
    SIZE_T cntVariables = Snapshot::count(data);
    SIZE_T cntName = ::strlen(name);
    
    for (SIZE_T i = 0; i < cntVariables; i++) {
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
    SIZE_T dataSize = 0;

    this->Clear();
    ASSERT(this->data == NULL);
    
    if (data != NULL) {
        /* Count parameters. */
        dataSize = Snapshot::count(data);

        /* Allocate parameter array. */
        this->data = new char *[dataSize + 1];

        /* Allocate variable memory and copy data. */
        for (SIZE_T i = 0; i < dataSize; i++) {
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
        throw SystemException(__FILE__, __LINE__);
    }

    retval.data = env;
    
    if (!::FreeEnvironmentStringsW(env)) {
        throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    retval.assign(const_cast<const char **>(::environ));
#endif /* _WIN32 */
    
    return retval;
}


/*
 * vislib::sys::Environment::GetVariable
 */
vislib::StringA vislib::sys::Environment::GetVariable(const char *name, 
                                                      const bool isLenient) {
#ifdef _WIN32
    vislib::StringA retval;
    DWORD error = NO_ERROR;
    DWORD size = ::GetEnvironmentVariableA(name, NULL, 0);
    
    if (size == 0) {
        if (!isLenient || ((error = ::GetLastError()) 
                != ERROR_ENVVAR_NOT_FOUND)) {
            throw SystemException(error, __FILE__, __LINE__);
        }
    }

    size = ::GetEnvironmentVariableA(name, retval.AllocateBuffer(size), 
        size + 1);
    if (size == 0) {
        if (!isLenient || ((error = ::GetLastError()) 
                != ERROR_ENVVAR_NOT_FOUND)) {
            throw SystemException(error, __FILE__, __LINE__);
        } else {
            retval[0] = 0;
        }
    }

    return retval;

#else /* _WIN32 */
    return vislib::StringA(::getenv(name));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::GetVariable
 */
vislib::StringW vislib::sys::Environment::GetVariable(const wchar_t *name,
                                                      const bool isLenient) {
#ifdef _WIN32
    vislib::StringW retval;
    DWORD error = NO_ERROR;
    DWORD size = ::GetEnvironmentVariableW(name, NULL, 0);
    
    if (size == 0) {
        if (!isLenient || ((error = ::GetLastError()) 
                != ERROR_ENVVAR_NOT_FOUND)) {
            throw SystemException(error, __FILE__, __LINE__);
        }
    }

    size = ::GetEnvironmentVariableW(name, retval.AllocateBuffer(size), 
        size + 1);
    if (size == 0) {
        if (!isLenient || ((error = ::GetLastError()) 
                != ERROR_ENVVAR_NOT_FOUND)) {
            throw SystemException(error, __FILE__, __LINE__);
        } else {
            retval[0] = 0;
        }
    }

    return retval;
#else /* _WIN32 */
    return vislib::StringW(Environment::GetVariable(W2A(name)));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::IsSet
 */
bool vislib::sys::Environment::IsSet(const char *name) {
#ifdef _WIN32
    if (::GetEnvironmentVariableA(name, NULL, 0) == 0) {
        throw SystemException(__FILE__, __LINE__);
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
        throw SystemException(__FILE__, __LINE__);
    }

    return (::GetLastError() != ERROR_ENVVAR_NOT_FOUND);
#else /* _WIN32 */
    return Environment::IsSet(W2A(name));
#endif /* _WIN32 */
}


/*
 * vislib::sys::Environment::SetVariable
 */
void vislib::sys::Environment::SetVariable(const char *name, 
                                           const char *value) {
#ifdef _WIN32
    if (!::SetEnvironmentVariableA(name, value)) {
        throw SystemException(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    if (value != NULL) {
        if (::setenv(name, value, 1) == -1) {
            throw SystemException(__FILE__, __LINE__);
        }
    } else {
        if (::unsetenv(name) == -1) {
            throw SystemException(__FILE__, __LINE__);
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
        throw SystemException(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    Environment::SetVariable(W2A(name), W2A(value));
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
