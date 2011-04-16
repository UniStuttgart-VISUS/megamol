/*
 * DynamicLoadLibrary.h  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DYNAMICLINKLIBRARY_H_INCLUDED
#define VISLIB_DYNAMICLINKLIBRARY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifndef _WIN32
#include "vislib/memutils.h"
#endif /* _WIN32 */
#include "vislib/String.h"
#include "vislib/SystemException.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * DLLException is thrown if a DynamicLinkLibrary load operation fails.
     * 
     * Implementation note: This is a hack because Linux does not support any
     * machine-usable error code, not to mention an error code that is 
     * compatible with the system-defined error constants. The only error 
     * information we can get is a string. Therefore, DLLException is a normal
     * Exception on Linux.
     */
#ifdef _WIN32
    typedef SystemException DLLException;
#else /* _WIN32 */
    typedef Exception DLLException;
#endif /* _WIN32 */


    /**
     * This class wraps dynamic link libraries or shared objects.
     */
    class DynamicLinkLibrary {

    public:

        /** Ctor. */
        DynamicLinkLibrary(void);

        /**
         * Dtor.
         *
         * If the library is still open, it is freed.
         */
        ~DynamicLinkLibrary(void);

        /**
         * Frees the current library, if there is one. It is safe to call
         * the method, if no library has been loaded.
         *
         * @throws DLLException If a library was loaded and cannot be released.
         */
        void Free(void);

        /**
         * Answer a function pointer to the function named 'procName'.
         *
         * @param procName The name of the function to be seached.
         *
         * @return A pointer to the function. If the function was not found,
         *         NULL is returned.
         */
#ifdef _WIN32
        FARPROC GetProcAddress(const CHAR *procName) const;
#else /* _WIN32 */
        void *GetProcAddress(const CHAR *procName) const;
#endif /* _WIN32 */

        /**
         * Answer, whether a library is currently loaded.
         *
         * @return true, if a library is loaded, false otherwise.
         */
        bool IsLoaded(void) const {
            return (this->hModule != NULL);
        }

        /**
         * Answer the error message from the last call of 'Load'. If the last
         * call to 'Load' was sucessful, the returned string will be empty.
         *
         * @return A human-readable error message from the last call of 'Load'
         */
        inline const vislib::StringA& LastLoadErrorMessage(void) const {
            return this->loadErrorMsg;
        }

        /**
         * Loads the library designated by the path 'moduleName'.
         *
         * @param moduleName The name of the module to load. If the name is not
         *                   a fully qualified path, the module is searched for
         *                   in a platform dependent manner.
         * @param dontResolveReferences References from the module to other
         *                              dynamic modules should not be resolved
         *                              by loading the other dynamic modules.
         *                              (Ignored on Linux platforms)
         * @param alternateSearchPath If set to true an alternative search
         *                            pattern for dependencies is used.
         *                            Consult the os-dependent documentation
         *                            for further information.
         *
         * @return true in case of success, false, if the library could not be 
         *         loaded. If 'false' is returned, use 'LastLoadErrorMessage'
         *         to get a human-readable error message.
         *
         * @throws IllegalStateException If a library was already loaded and not
         *                               freed before this call to Load().
         */
        bool Load(const char *moduleName, bool dontResolveReferences = false,
            bool alternateSearchPath = false);

        /**
         * Loads the library designated by the path 'moduleName'.
         *
         * @param moduleName The name of the module to load. If the name is not
         *                   a fully qualified path, the module is searched for
         *                   in a platform dependent manner.
         * @param dontResolveReferences References from the module to other
         *                              dynamic modules should not be resolved
         *                              by loading the other dynamic modules.
         *                              (Ignored on Linux platforms)
         * @param alternateSearchPath If set to true an alternative search
         *                            pattern for dependencies is used.
         *                            Consult the os-dependent documentation
         *                            for further information.
         *
         * @return true in case of success, false, if the library could not be 
         *         loaded. If 'false' is returned, use 'LastLoadErrorMessage'
         *         to get a human-readable error message.
         *
         * @throws IllegalStateException If a library was already loaded and not
         *                               freed before this call to Load().
         */
        bool Load(const wchar_t *moduleName, bool dontResolveReferences = false,
            bool alternateSearchPath = false);

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        DynamicLinkLibrary(const DynamicLinkLibrary& rhs);

        /**
         * Forbidden assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        DynamicLinkLibrary& operator =(const DynamicLinkLibrary& rhs);

        /** The module handle. */
#ifdef _WIN32
        HMODULE hModule;
#else /* _WIN32 */
        void *hModule;
#endif /* _WIN32 */

        /** The error message describing the error during the last 'Load' */
        vislib::StringA loadErrorMsg;

    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DYNAMICLINKLIBRARY_H_INCLUDED */
