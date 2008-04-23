/*
 * Path.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PATH_H_INCLUDED
#define VISLIB_PATH_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/String.h"

#ifdef _MSC_VER
#pragma comment(lib, "shlwapi")
#endif /* _MSC_VER */

namespace vislib {
namespace sys {


    /**
     * This class contains functionality for manipulation path names.
     */
    class Path {

    public:

        /**
         * Canonicalises the path. 
         *
         * The method removes all relative references to the current path (".") 
         * and to the previous path part ("..") and all sequences of more than
         * one path separator except for the UNC marker at the begin of the 
         * path.
         *
         * @param path The path to canonicalise.
         *
         * @return The canonicalised path.
         */
        static StringA Canonicalise(const StringA& path);

        /**
         * Canonicalises the path. 
         *
         * The method removes all relative references to the current path (".") 
         * and to the previous path part ("..") and all sequences of more than
         * one path separator except for the UNC marker at the begin of the 
         * path.
         *
         * @param path The path to canonicalise.
         *
         * @return The canonicalised path.
         */
        static StringW Canonicalise(const StringW& path);

        /**
         * Compares two paths. Both paths are considdered equal if the 
         * canonicalised absolute paths are equal. Under windows the
         * comparision is done case insensitive. Under linux the comparision
         * is case sensitive.
         *
         * @param lhs The first path to be compared.
         * @param rhs The second path to be compared.
         *
         * @return True if both paths are considdered equal, false otherwise.
         */
        template<class T>
        static bool Compare(const String<T>& lhs, const String<T>& rhs);

        /**
         * Concatenate the paths 'lhs' and 'rhs'. The method ensures, that
         * at a path separator is inserted, if necessary. If 'lhs' has a path
         * separator at its end or 'rhs' has a path spearator at its begin, no
         * spearator is inserted. You can also set 'canonicalise' true in order
         * to remove unnecessary path separators already contained in 'lhs' or
         * 'rhs'. This is not equal 'Resolve(rhs, lhs)'!
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
         * 'rhs'. This is not equal 'Resolve(rhs, lhs)'!
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
         * Deletes a directory and optional all files and subdirectories.
         *
         * @param path      The path to the directory to be deleted.
         * @param recursive Flag wether or not to remove items recursively. If
         *                  true, all files and subdirectories will be also
         *                  removed. If false and the directory is not empty
         *                  the function will fail.
         *
         * @throws SystemException if an error occured.
         */
        static void DeleteDirectory(const StringA& path, bool recursive);

        /**
         * Deletes a directory and optional all files and subdirectories.
         *
         * @param path      The path to the directory to be deleted.
         * @param recursive Flag wether or not to remove items recursively. If
         *                  true, all files and subdirectories will be also
         *                  removed. If false and the directory is not empty
         *                  the function will fail.
         *
         * @throws SystemException if an error occured.
         */
        static void DeleteDirectory(const StringW& path, bool recursive);

        /**
         * Answer the current working directory.
         *
         * The returned string is guaranteed to end with a path separator.
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
         * The returned string is guaranteed to end with a path separator.
         *
         * @return The current working directory.
         *
         * @throws SystemException If the directory cannot be retrieved
         * @throws std::bad_alloc If there is not enough memory for storing the
         *                        directory.
         */
        static StringW GetCurrentDirectoryW(void);

        /**
         * Answer the home directory of the user. On windows the 'My Documents'
         * folder is returned.
         *
         * The returned string is guaranteed to end with a path separator.
         *
         * @return The users home directory.
         *
         * @throws SystemException If the directory cannot be retrieved
         */
        static StringA GetUserHomeDirectoryA(void);

        /**
         * Answer the home directory of the user. On windows the 'My Documents'
         * folder is returned.
         *
         * The returned string is guaranteed to end with a path separator.
         *
         * @return The users home directory.
         *
         * @throws SystemException If the directory cannot be retrieved
         */
        static StringW GetUserHomeDirectoryW(void);

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
         * Creates a directory and all intermediate directories on the path 
         * which currently do not exist. If the creating of on directory fails
         * all intermediate directories created by this methode call will be 
         * removed, if possible.
         *
         * @param path The path to the directory to be created. 
         *
         * @throws SystemException if an error occured.
         */
        static void MakeDirectory(const StringA& path);

        /**
         * Creates a directory and all intermediate directories on the path 
         * which currently do not exist. If the creating of on directory fails
         * all intermediate directories created by this methode call will be 
         * removed, if possible.
         *
         * @param path The path to the directory to be created. 
         *
         * @throws SystemException if an error occured.
         */
        static void MakeDirectory(const StringW& path);

        /**
         * Clears a directory by removing all files and optionally all 
         * subdirectories.
         *
         * @param path The director to purge.
         * @param recursive If true all subdirectories will be also removed.
         */
        static void PurgeDirectory(const StringA& path, bool recursive);

        /**
         * Clears a directory by removing all files and optionally all 
         * subdirectories.
         *
         * @param path The director to purge.
         * @param recursive If true all subdirectories will be also removed.
         */
        static void PurgeDirectory(const StringW& path, bool recursive);

        /**
         * Answer the absolute path of 'path'. 'path' can be absolute itself and
         * will only be canonicalised this case.
         *
         * If the path consists only of the the '~' character, it is expanded to
         * be the user home respectively the "My Documents" folder as returned
         * by GetUserHomeDirectory().
         * If the path begins with the sequence "~/", this part is expanded to
         * be the user home, too. 
         * Any other occurrence of the '~' character remains unchanged, i. e.
         * "./~" is assumed to reference a local directory "~".
         * 
         * On Windows, Resolve additionally changes every '/' in the path to the
         * Windows path separator '\'.
         *
         * @param path A path to a file or directory.
         *
         * @return The absolute path.
         */
        static inline StringA Resolve(StringA path) {
            return Resolve(path, GetCurrentDirectoryA());
        }

        /**
         * Answer the absolute path of 'path'. 'path' can be absolute itself and
         * will only be canonicalised this case.
         *
         * If the path consists only of the the '~' character, it is expanded to
         * be the user home respectively the "My Documents" folder as returned
         * by GetUserHomeDirectory().
         * If the path begins with the sequence "~/", this part is expanded to
         * be the user home, too. 
         * Any other occurrence of the '~' character remains unchanged, i. e.
         * "./~" is assumed to reference a local directory "~".
         * 
         * On Windows, Resolve additionally changes every '/' in the path to the
         * Windows path separator '\'.
         *
         * @param path A path to a file or directory.
         *
         * @return The absolute path.
         */
        static inline StringW Resolve(StringW path) {
            return Resolve(path, GetCurrentDirectoryW());
        }
        
        /**
         * Answer the absolute path of 'path'. 'path' can be absolute itself and
         * will only be canonicalised this case.
         *
         * If the path consists only of the the '~' character, it is expanded to
         * be the user home respectively the "My Documents" folder as returned
         * by GetUserHomeDirectory().
         * If the path begins with the sequence "~/", this part is expanded to
         * be the user home, too. 
         * Any other occurrence of the '~' character remains unchanged, i. e.
         * "./~" is assumed to reference a local directory "~".
         * 
         * On Windows, Resolve additionally changes every '/' in the path to the
         * Windows path separator '\'.
         *
         * @param path A path to a file or directory.
         * @param basePath The base path which will be used to resolve 'path'.
         *                 Must not be an unc path.
         *
         * @return The absolute path.
         */
        static StringA Resolve(StringA path, StringA basepath);

        /**
         * Answer the absolute path of 'path'. 'path' can be absolute itself and
         * will only be canonicalised this case.
         *
         * If the path consists only of the the '~' character, it is expanded to
         * be the user home respectively the "My Documents" folder as returned
         * by GetUserHomeDirectory().
         * If the path begins with the sequence "~/", this part is expanded to
         * be the user home, too. 
         * Any other occurrence of the '~' character remains unchanged, i. e.
         * "./~" is assumed to reference a local directory "~".
         * 
         * On Windows, Resolve additionally changes every '/' in the path to the
         * Windows path separator '\'.
         *
         * @param path A path to a file or directory.
         * @param basePath The base path which will be used to resolve 'path'.
         *                 Must not be an unc path.
         *
         * @return The absolute path.
         */
        static StringW Resolve(StringW path, StringW basepath);

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

        /** The home directory marker character. */
        static const char MYDOCUMENTS_MARKER_A;

        /** The home directory marker character. */
        static const char MYDOCUMENTS_MARKER_W;

        /** The ANSI path separator character. */
        static const char SEPARATOR_A;

        /** The Unicode path separator character. */
        static const wchar_t SEPARATOR_W;

        /** Dtor. */
        ~Path(void);

    private:

        /** Disallow instances. */
        Path(void);
    };


    /*
     * Path::Compare<T>
     */
    template<class T>
    bool Path::Compare(const String<T>& lhs, const String<T>& rhs) {
        String<T> rlhs = (Path::IsAbsolute(lhs)) ? Path::Canonicalise(lhs) : Path::Resolve(lhs);
        String<T> rrhs = (Path::IsAbsolute(rhs)) ? Path::Canonicalise(rhs) : Path::Resolve(rhs);
#ifdef _WIN32
        return rlhs.CompareInsensitive(rrhs);
#else /* _WIN32 */
        return rlhs.Compare(rrhs);
#endif /* _WIN32 */
    }
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_PATH_H_INCLUDED */
