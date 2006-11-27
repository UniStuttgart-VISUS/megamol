/*
 * ImpersonationContext.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IMPERSONATIONCONTEXT_H_INCLUDED
#define VISLIB_IMPERSONATIONCONTEXT_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include <security/pam_appl.h>
#include <security/pam_misc.h>
#endif /* _WIN32 */


namespace vislib {
namespace sys {


    /**
     * This class allows a process to impersonate as a different user than the
     * one that created the process. The user's credentials are required to
     * impersonate.
     *
     * On Windows, the new user is logged on and then impersonated.
     *
     * On Linux, the user is logged on using PAM and the effective UID is 
     * changed to the one of the user, if logon succeeded. PAM itself obviousely
     * does not change the UID, if a new session is started.
     * Note, that only the effective UID is changed as otherwise, returning to
     * the root user is not possible.
     */
    class ImpersonationContext {

    public:

        /** Ctor. */
        ImpersonationContext(void);

        /** Dtor. */
        ~ImpersonationContext(void);

        /**
         * Try to logon the specified user and impersonate as this user.
         *
         * @param username The user name to logon.
         * @param domain   The domain to logon to. 
         *                 This value has no effect on Linux and can have any 
         *                 value,. On Windows, "." can be used for the local 
         *                 machine, or NULL, if 'username' has the format 
         *                 'user@domain'.
         * @param password The password of the user.
         *
         * @return true, if the user was impersonated, false, if logging on 
         *         failed.
         *
         * @throws SystemException If the impersonation failed.
         *                         On Windows, this exception is throws, if 
         *                         logging on or impersonation failed.
         *                         On Linux, this exception is only thrown, if 
         *                         the effective UID could not be changed.
         * @throws PAMException    Only on Linux, if the PAM module could not be
         *                         initialised.
         */
        void Impersonate(const char *username, const char *domain, 
            const char *password);

        /**
         * Try to logon the specified user and impersonate as this user.
         *
         * @param username The user name to logon.
         * @param domain   The domain to logon to. 
         *                 This value has no effect on Linux and can have any 
         *                 value,. On Windows, "." can be used for the local 
         *                 machine, or NULL, if 'username' has the format 
         *                 'user@domain'.
         * @param password The password of the user.
         *
         * @return true, if the user was impersonated, false, if logging on 
         *         failed.
         *
         * @throws SystemException If the impersonation failed.
         *                         On Windows, this exception is throws, if 
         *                         logging on or impersonation failed.
         *                         On Linux, this exception is only thrown, if 
         *                         the effective UID could not be changed.
         * @throws PAMException    Only on Linux, if the PAM module could not be
         */
        void Impersonate(const wchar_t *username, const wchar_t *domain,
            const wchar_t *password);

        /**
         * Revert to original user.
         *
         * @throws SystemException On Windows, if reverting to own user failed.
         *                         On Linux, if setting the prior UID failed.
         * @throws PAMException    Only on Linux, if ending the PAM session 
         *                         failed.
         */
        inline void Revert(void) {
            this->revert(false);
        }

    private:

#ifndef _WIN32
        /**
         * This is the PAM conversation callback that provides the user 
         * password, if requested.
         *
         * @param cntMsg
         * @param msg
         * @param response
         * @param userData
         *
         * @return
         */
        static int logonUserConversation(int cntMsg, 
            const struct pam_message **msg, struct pam_response **response,
            void *userData);

        /**
         * Answer the UID of the user with name 'username'.
         *
         * @param username A user name.
         *
         * @return The UID of the user on this system.
         *
         * @throws SystemException If resolving the user name failed.
         */
        static int resolveUID(const char *username);
#endif /* _WIN32 */

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        ImpersonationContext(const ImpersonationContext& rhs);

        /**
         * Revert to original user and clean up all open handles. If 'isSilent'
         * is true, no exceptions will be thrown.
         *
         * @param isSilent Supress exceptions in case of an error.
         *
         * @throws SystemException
         * @throws PAMException
         */
        void revert(const bool isSilent);

        /**
         * Assignemnt operator.
         *
         * @param rhs The right hand side operand.
         *
         * @throws IllegalParamException If (this != rhs).
         */
        ImpersonationContext& operator =(const ImpersonationContext& rhs);

#ifdef _WIN32
        /** The logon token of the user which for LogonUser was called. */
        HANDLE hToken;
#else /* _WIN32 */
        /** PAM handle. If this handle is NULL, no impersonation was done. */
        pam_handle_t *hPAM;

        /** UID to revert to. */
        int origUID;
#endif /* _WIN32 */
    };
    
} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_IMPERSONATIONCONTEXT_H_INCLUDED */

