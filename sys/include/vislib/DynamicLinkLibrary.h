/*
 * DynamicLoadLibrary.h  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DYNAMICLINKLIBRARY_H_INCLUDED
#define VISLIB_DYNAMICLINKLIBRARY_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#ifndef _WIN32
#include "vislib/memutils.h"
#endif /* _WIN32 */

#include "vislib/tchar.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {

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
         * @throws SystemException If a library was loaded and cannot be
         *                         released.
         */
        void Free(void);

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
         * Loads the library designated by the path 'moduleName'.
         *
         * @return true in case of success, false, if the library could not be 
         *         loaded.
         *
         * @throws
         */
        // TODO: Behaviour: Throw exception? Do not free automatically? Might leave dangling references.
        bool Load(const TCHAR *moduleName);

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
    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_DYNAMICLINKLIBRARY_H_INCLUDED */
