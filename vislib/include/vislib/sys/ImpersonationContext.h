/*
 * ImpersonationContext.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IMPERSONATIONCONTEXT_H_INCLUDED
#define VISLIB_IMPERSONATIONCONTEXT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include <sys/types.h>
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
     * On Linux, the impersonation is done by chaning the effective user and 
     * group ID of the process. The caller must therefore run as root.
     * Link -lcrypt on Linux.
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
         */
        void Impersonate(const wchar_t *username, const wchar_t *domain,
            const wchar_t *password);

        /**
         * Revert to original user.
         *
         * @throws SystemException If reverting to own user failed.
         */
        inline void Revert(void) {
            this->revert(false);
        }

    private:

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
         * @throws SystemException If the operation failed.
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
        /** The UID of the group to revert to. */
        gid_t revertToGid;

        /** The UID of the user to revert to. */
        uid_t revertToUid;
#endif /* _WIN32 */
    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IMPERSONATIONCONTEXT_H_INCLUDED */
