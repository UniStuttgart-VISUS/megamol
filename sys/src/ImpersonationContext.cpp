/*
 * ImpersonationContext.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/ImpersonationContext.h"

#include <stdexcept>

#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#include <shadow.h>
#endif /* !_WIN32 */

#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/memutils.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"



/*
 * vislib::sys::ImpersonationContext::ImpersonationContext
 */
vislib::sys::ImpersonationContext::ImpersonationContext(void) {
#ifdef _WIN32
    this->hToken = INVALID_HANDLE_VALUE;
#else /* _WIN32 */
    this->revertToUid = 0;
    this->revertToGid = 0;
#endif /* _WIN32 */
}


/*
 * vislib::sys::ImpersonationContext::~ImpersonationContext
 */
vislib::sys::ImpersonationContext::~ImpersonationContext(void) {
    this->revert(true);
}


/*
 * vislib::sys::ImpersonationContext::Impersonate
 */
void vislib::sys::ImpersonationContext::Impersonate(const char *username,
        const char *domain, const char *password) {

    /* Do not allow impersonation chains. */
    this->revert(false);

#ifdef _WIN32
    if (::LogonUserA(username, domain, password, LOGON32_LOGON_INTERACTIVE,
            LOGON32_PROVIDER_DEFAULT, &this->hToken) == FALSE) {
        throw SystemException(__FILE__, __LINE__);
    }
    
    if (::ImpersonateLoggedOnUser(this->hToken) == FALSE) {
        DWORD errorCode = ::GetLastError();
        ::CloseHandle(this->hToken);
        this->hToken = NULL;
        throw SystemException(errorCode, __FILE__, __LINE__);
    }

#else /* _WIN32 */
    struct passwd pw;               // Receives password data.
    struct passwd *ppw = NULL;      // Pointer to password data if found.
    struct spwd spw;                // Receives shadow password data.
    struct spwd *pspw = NULL;       // Pointer to shadow data if found.
    char *buf = NULL;               // Recevies strings of 'pw' and 'spw'.
    int bufLen = 0;                 // Size of 'buf' and in bytes.
    int errorCode = 0;              // Preserves last API error code.

    /* First, determine the user to revert to. */
    this->revertToUid = ::geteuid();
    this->revertToGid = ::getegid();

    /* Allocate buffer for password data. */
    if ((bufLen = ::sysconf(_SC_GETPW_R_SIZE_MAX)) == -1) {
        errorCode = ::GetLastError();
        VLTRACE(Trace::LEVEL_VL_ERROR, "::sysconf(_SC_GETPW_R_SIZE_MAX) "
            "failed.\n");
        throw SystemException(__FILE__, __LINE__);
    }
    buf = new char[bufLen];
    ASSERT(buf != NULL);
    /* From now on, 'buf' must be deallocated before leaving. */

    /* Get the UID and GID first (string fields will not be used!). */
    if (::getpwnam_r(username, &pw, buf, bufLen, &ppw) != 0) {
        errorCode = ::GetLastError();
        ARY_SAFE_DELETE(buf);
        VLTRACE(Trace::LEVEL_VL_ERROR, "::getpwnam_r failed.\n");
        throw SystemException(errorCode, __FILE__, __LINE__);
    }
    if (ppw == NULL) {
        /* User was not found. */
        ARY_SAFE_DELETE(buf);
        VLTRACE(Trace::LEVEL_VL_ERROR, "Cannot impersonate, because the "
            "user \"%s\" does not exist.\n", username);
        throw SystemException(ENOENT, __FILE__, __LINE__);
    }

    /* Get shadow password of target user (pw.passwd is not valid!). */
    if (::getspnam_r(username, &spw, buf, bufLen, &pspw) != 0) {
        errorCode = ::GetLastError();
        ARY_SAFE_DELETE(buf);
        VLTRACE(Trace::LEVEL_VL_ERROR, "::getspnam_r failed.\n");
        throw SystemException(errorCode, __FILE__, __LINE__);
    }
    ASSERT(pspw != NULL);
    if (pspw == NULL) {
        /* User was not found. */
        ARY_SAFE_DELETE(buf);
        VLTRACE(Trace::LEVEL_VL_ERROR, "Cannot impersonate, because the "
            "user \"%s\" does not exist.\n", username);
        throw SystemException(ENOENT, __FILE__, __LINE__);
    }

    /* Check login data using passwd we retrieved. */
    if (spw.sp_pwdp != NULL) {
        if ((password == NULL) || (::strcmp(spw.sp_pwdp, ::crypt(password,
                spw.sp_pwdp)) != 0)) {
            ARY_SAFE_DELETE(buf);
            VLTRACE(Trace::LEVEL_VL_ERROR, "Cannot impersonate, because the "
                "password is invalid: Expected \"%s\", but got \"%s\".\n",
                spw.sp_pwdp, ::crypt(password, spw.sp_pwdp));
            throw SystemException(EPERM, __FILE__, __LINE__);
        }
    }
    /* Authentication OK when here. */

    /* Switch effective user and group ID. */
    if (::seteuid(pw.pw_uid) != 0) {
        errorCode = ::GetLastError();
        ARY_SAFE_DELETE(buf);
        VLTRACE(Trace::LEVEL_VL_ERROR, "::seteuid failed.\n");
        throw SystemException(errorCode, __FILE__, __LINE__);
    }
    if (::setegid(pw.pw_gid) != 0) {
        errorCode = ::GetLastError();
        ::seteuid(this->revertToUid);
        ARY_SAFE_DELETE(buf);
        VLTRACE(Trace::LEVEL_VL_ERROR, "::setegid failed.\n");
        throw SystemException(errorCode, __FILE__, __LINE__);
    }

    ARY_SAFE_DELETE(buf);
#endif /* _WIN32 */
}


/*
 * vislib::sys::ImpersonationContext::Impersonate
 */
void vislib::sys::ImpersonationContext::Impersonate(const wchar_t *username, 
        const wchar_t *domain, const wchar_t *password) {
#ifdef _WIN32
    /* Do not allow impersonation chains. */
    this->revert(false);

    if (::LogonUserW(username, domain, password, LOGON32_LOGON_INTERACTIVE,
            LOGON32_PROVIDER_DEFAULT, &this->hToken) == FALSE) {
        throw SystemException(__FILE__, __LINE__);
    }

    if (::ImpersonateLoggedOnUser(this->hToken) == FALSE) {
        DWORD errorCode = ::GetLastError();
        ::CloseHandle(this->hToken);
        this->hToken = NULL;
        throw SystemException(errorCode, __FILE__, __LINE__);
    }

#else /* _WIN32 */
    return this->Impersonate(W2A(username), NULL, W2A(password));
#endif /* _WIN32 */
}


/*
 * vislib::sys::ImpersonationContext::ImpersonationContext
 */
vislib::sys::ImpersonationContext::ImpersonationContext(
        const ImpersonationContext& rhs) {
    throw UnsupportedOperationException("ImpersonationContext", __FILE__, 
        __LINE__);
}


/*
 * vislib::sys::ImpersonationContext::revert
 */
void vislib::sys::ImpersonationContext::revert(const bool isSilent) {
#ifdef _WIN32
    if (this->hToken != NULL) {
        if (!::RevertToSelf() && !isSilent) {
            throw SystemException(__FILE__, __LINE__);
        }

        ::CloseHandle(this->hToken);
        this->hToken = NULL;
    }

#else /* _WIN32 */
    if ((::seteuid(this->revertToUid) != 0) && !isSilent) {
        throw SystemException(__FILE__, __LINE__);
    }
    if ((::setegid(this->revertToGid) != 0) && !isSilent) {
        throw SystemException(__FILE__, __LINE__);
    }

    this->revertToUid = 0;
    this->revertToGid = 0;
#endif /* _WIN32 */
}


/*
 * vislib::sys::ImpersonationContext::operator =
 */
vislib::sys::ImpersonationContext& 
vislib::sys::ImpersonationContext::operator =(const ImpersonationContext& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
