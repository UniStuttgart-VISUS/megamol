/*
 * ImpersonationContext.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/ImpersonationContext.h"

#include <stdexcept>

#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/memutils.h"
#include "vislib/PAMException.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"



/*
 * vislib::sys::ImpersonationContext::ImpersonationContext
 */
vislib::sys::ImpersonationContext::ImpersonationContext(void) {
#ifdef _WIN32
    this->hToken = INVALID_HANDLE_VALUE;
#else /* _WIN32 */
    this->hPAM = NULL;
    this->origUID = 0;
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
#ifdef _WIN32
    this->revert(false);

    if (::LogonUserA(username, domain, password, LOGON32_LOGON_INTERACTIVE,
            LOGON32_PROVIDER_DEFAULT, &this->hToken) == FALSE) {
        throw SystemException(__FILE__, __LINE__);
    }
    
    if (::ImpersonateLoggedOnUser(this->hToken) == FALSE) {
        ::CloseHandle(this->hToken);
        this->hToken = NULL;
        throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    struct pam_conv conv;                   // PAM conversation context.
    int pamResult = PAM_SUCCESS;            // Result of last PAM operation.

    /* End open transaction, if there is a prior one. */
    this->revert(true);

    /* Start PAM transaction. */
    conv.conv = logonUserConversation;
    conv.appdata_ptr = const_cast<char *>(password);

    // Note: It is an evil cheat to pretend being "login" ...
    if ((pamResult = ::pam_start("login", username, &conv, &this->hPAM))
            != PAM_SUCCESS) {
        throw PAMException(hPAM, pamResult, __FILE__, __LINE__);
    }

    /* Try to authenticate the user. */
    if ((pamResult = ::pam_authenticate(this->hPAM, PAM_SILENT)) 
            != PAM_SUCCESS) {
        throw PAMException(this->hPAM, pamResult, __FILE__, __LINE__);
    }

    /* Check whether account is still active. */
    if ((pamResult = ::pam_acct_mgmt(this->hPAM, PAM_SILENT)) != PAM_SUCCESS) {
        throw PAMException(this->hPAM, pamResult, __FILE__, __LINE__);
    }

    /* Set credentials. */
    if (::seteuid(ImpersonationContext::resolveUID(username)) != 0) {
    //if (::setuid(ImpersonationContext::resolveUID(username)) != 0) {
        throw SystemException(__FILE__, __LINE__);
    }
    if ((pamResult = ::pam_setcred(this->hPAM, PAM_ESTABLISH_CRED)) 
            != PAM_SUCCESS) {
        throw PAMException(this->hPAM, pamResult, __FILE__, __LINE__);
    }

    /* Start session as logged on user. */
    if ((pamResult = ::pam_open_session(this->hPAM, PAM_SILENT)) 
            != PAM_SUCCESS) {
        throw PAMException(this->hPAM, pamResult, __FILE__, __LINE__);
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::ImpersonationContext::Impersonate
 */
void vislib::sys::ImpersonationContext::Impersonate(const wchar_t *username, 
        const wchar_t *domain, const wchar_t *password) {
#ifdef _WIN32
    if (::LogonUserW(username, domain, password, LOGON32_LOGON_INTERACTIVE,
            LOGON32_PROVIDER_DEFAULT, &this->hToken) == FALSE) {
        throw SystemException(__FILE__, __LINE__);
    }

    if (::ImpersonateLoggedOnUser(this->hToken) == FALSE) {
        ::CloseHandle(this->hToken);
        this->hToken = NULL;
        throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    return this->Impersonate(W2A(username), NULL, W2A(password));
#endif /* _WIN32 */
}


#ifndef _WIN32
/*
 * vislib::sys::ImpersonationContext::logonUserConversation
 */
int vislib::sys::ImpersonationContext::logonUserConversation(int cntMsg, 
        const struct pam_message **msg, struct pam_response **response, 
        void *userData) {
    struct pam_response *r = NULL;
    
    r = static_cast<pam_response *>(::malloc(cntMsg 
        * sizeof(struct pam_response)));
    if (r == NULL) {
        return PAM_BUF_ERR;
    }

    ::memset(r, 0, cntMsg * sizeof (struct pam_response));
    for (int i = 0; i < cntMsg; i++) {
        if (msg[i]->msg_style == PAM_PROMPT_ECHO_OFF) {
            r[i].resp = ::strdup(reinterpret_cast<char *>(userData));
            break;
        }
    }

    *response = r;
    // TODO: Search for some documentation.
    // PAM will release the memory using free(). I hope so, at least.
    return PAM_SUCCESS;
}


/*
 * vislib::sys::ImpersonationContext::resolveUID
 */
int vislib::sys::ImpersonationContext::resolveUID(const char *username) {
    FILE *fp = NULL;
    StringA idQuery;
    int retval = 0;

    // TODO: Use some shell abstraction class instead of popen.
    idQuery.Format("id -u %s", username);
    if ((fp = ::popen(idQuery.PeekBuffer(), "r")) == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

    if (::fscanf(fp, "%d", &retval) != 1) {
        ::pclose(fp);
        throw SystemException(__FILE__, __LINE__);
    }

    ::pclose(fp);
    return retval;
}
#endif /* _WIN32 */


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
    int pamResult = PAM_SUCCESS;            // Result of last PAM operation.

    if (this->hPAM != NULL) {
        /* Close our session. */
        if (((pamResult = ::pam_close_session(this->hPAM, PAM_SILENT)) 
                != PAM_SUCCESS) && !isSilent) {
            throw PAMException(this->hPAM, pamResult, __FILE__, __LINE__);
        }

        /* Restore original UID. */
        if ((::seteuid(this->origUID) != 0) && !isSilent) {
        //if ((::setuid(this->origUID) != 0) && !isSilent) {
            throw SystemException(__FILE__, __LINE__);
        }
        
        /* End PAM transaction. */
        ::pam_end(this->hPAM, PAM_SUCCESS);
        this->hPAM = NULL;
    }
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
