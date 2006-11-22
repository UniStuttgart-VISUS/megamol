/*
 * Path.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PATH_H_INCLUDED
#define VISLIB_PATH_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/String.h"


namespace vislib {
namespace sys {


    /**
     * This class contains functionality for manipulation path names.
     */
    class Path {

    public:

        static StringA Canonicalise(const StringA& path);

        static StringW Canonicalise(const StringW& path);

        /**
         * Concatenate the paths 'lhs' and 'rhs'. The method ensures, that
         * at a path separator is inserted, if necessary. If 'lhs' has a path
         * separator at its end or 'rhs' has a path spearator at its begin, no
         * spearator is inserted. You can also set 'canonicalise' true in order
         * to remove unnecessary path separators already contained in 'lhs' or
         * 'rhs'.
         *
         * @param lhs          The left hand side path.
         * @param rhs          The right hand side path.
         * @param canonicalise Canonicalise the path before returning it.
         *
         * @return The concatenated path.
         */
        static StringA Concatenate(const StringA& lhs, const StringA& rhs,
            const bool canonicalise = false);

        /**
         * Concatenate the paths 'lhs' and 'rhs'. The method ensures, that
         * at a path separator is inserted, if necessary. If 'lhs' has a path
         * separator at its end or 'rhs' has a path spearator at its begin, no
         * spearator is inserted. You can also set 'canonicalise' true in order
         * to remove unnecessary path separators already contained in 'lhs' or
         * 'rhs'.
         *
         * @param lhs          The left hand side path.
         * @param rhs          The right hand side path.
         * @param canonicalise Canonicalise the path before returning it.
         *
         * @return The concatenated path.
         */
        static StringW Concatenate(const StringW& lhs, const StringW& rhs,
            const bool canonicalise = false);

        /**
         * Answer the current working directory.
         *
         * @return The current working directory.
         *
         * @throws SystemException If the directory cannot be retrieved
         * @throws std::bad_alloc If there is not enough memory for storing the
         *                        directory.
         */
        static StringA GetCurrentDirectoryA(void);

        /**
         * Answer the current working directory.
         *
         * @return The current working directory.
         *
         * @throws SystemException If the directory cannot be retrieved
         * @throws std::bad_alloc If there is not enough memory for storing the
         *                        directory.
         */
        static StringW GetCurrentDirectoryW(void);

        /**
         * Answer, whether 'path' is an absolute path.
         *
         * @param path The path to be tested.
         *
         * @return true, if 'path' is absolute, false otherwise.
         */
        inline static bool IsAbsolute(const StringA& path) {
            return !Path::IsRelative(path);
        }

        /**
         * Answer, whether 'path' is an absolute path.
         *
         * @param path The path to be tested.
         *
         * @return true, if 'path' is absolute, false otherwise.
         */
        inline static bool IsAbsolute(const StringW& path) {
            return !Path::IsRelative(path);
        }

        /**
         * Answer, whether 'path' is a relative path.
         *
         * @param path The path to be tested.
         *
         * @return true, if 'path' is relative, false otherwise.
         */
        static bool IsRelative(const StringA& path);

        /**
         * Answer, whether 'path' is a relative path.
         *
         * @param path The path to be tested.
         *
         * @return true, if 'path' is relative, false otherwise.
         */
        static bool IsRelative(const StringW& path);

        /**
         * Answer the absolute path of 'path'. 'path' can be absolute itself and
         * will not be altered in this case.
         *
         * @param path A path to a file or directory.
         *
         * @return The absolute path.
         */
        static StringA Resolve(const StringA& path);

        /**
         * Answer the absolute path of 'path'. 'path' can be absolute itself and
         * will not be altered in this case.
         *
         * @param path A path to a file or directory.
         *
         * @return The absolute path.
         */
        static StringW Resolve(const StringW& path);

        /**
         * Changes the current directory to be 'path'.
         *
         * @param path The path to the new current directory.
         *
         * @throws SystemException If setting a new current directory fails, 
         *                         e. g. 'path' does not exist.
         */
        static void SetCurrentDirectory(const StringA& path);

        /**
         * Changes the current directory to be 'path'.
         *
         * @param path The path to the new current directory.
         *
         * @throws SystemException If setting a new current directory fails, 
         *                         e. g. 'path' does not exist.
         */
        static void SetCurrentDirectory(const StringW& path);

        /** The ANSI path separator character. */
        static const char SEPARATOR_A;

        /** The Unicode path separator character. */
        static const wchar_t SEPARATOR_W;

        /** The ANSI path separator character as zero-terminated string. */
        static const char SEPARATORSTR_A[];

        /** The ANSI path separator character as zero-terminated string. */
        static const wchar_t SEPARATORSTR_W[];

        /** Dtor. */
        ~Path(void);

    private:

        /** Disallow instances. */
        Path(void);
    };
    
} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_PATH_H_INCLUDED */

