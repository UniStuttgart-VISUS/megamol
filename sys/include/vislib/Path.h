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


#include "the/string.h"

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
        static the::astring Canonicalise(const the::astring& path);

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
        static the::wstring Canonicalise(const the::wstring& path);

        /**
         * Return a new path that has 'extension' as new extension.
         *
         * The 'extension' parameter can contain multiple periods and any valid 
         * path characters, and can be any length.
         *
         * The 'path' parameter can contain multiple periods. In this case, the
         * text after the last period will be replaced.
         *
         * If 'extension' is NULL or empty, the returned string contains the 
         * contents of path without the last extension.
         *
         * @param path      The path to be modified.
         * @param extension The new extension (with or without a leading 
         *                  period). Specifying NULL or an empty string will
         *                  remove an existing extension.
         *
         * @return The modified path.
         */
        static the::astring ChangeExtension(const char *path, const char *extension);

        /**
         * Return a new path that has 'extension' as new extension.
         *
         * The 'extension' parameter can contain multiple periods and any valid 
         * path characters, and can be any length.
         *
         * The 'path' parameter can contain multiple periods. In this case, the
         * text after the last period will be replaced.
         *
         * If 'extension' is NULL or empty, the returned string contains the 
         * contents of path without the last extension.
         *
         * @param path      The path to be modified.
         * @param extension The new extension (with or without a leading 
         *                  period). Specifying NULL or an empty string will
         *                  remove an existing extension.
         *
         * @return The modified path.
         */
        static the::wstring ChangeExtension(const wchar_t *path, 
            const wchar_t *extension);

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
        static bool Compare(const T& lhs, const T& rhs);

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
        static the::astring Concatenate(const the::astring& lhs, const the::astring& rhs,
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
        static the::wstring Concatenate(const the::wstring& lhs, const the::wstring& rhs,
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
         * @throws the::system::system_exception if an error occured.
         */
        static void DeleteDirectory(const the::astring& path, bool recursive);

        /**
         * Deletes a directory and optional all files and subdirectories.
         *
         * @param path      The path to the directory to be deleted.
         * @param recursive Flag wether or not to remove items recursively. If
         *                  true, all files and subdirectories will be also
         *                  removed. If false and the directory is not empty
         *                  the function will fail.
         *
         * @throws the::system::system_exception if an error occured.
         */
        static void DeleteDirectory(const the::wstring& path, bool recursive);

        /**
         * Answer the fully qualified path to a given executable. This method
         * searches for the requested executable in a operating system
         * dependent way, usually checking the current working directory and
         * a system environment search path.
         *
         * @param filename The file name of the executable to search for. Do
         *                 not specify any path information.
         *
         * @return The fully qualified path to the executable or an empty
         *         string if the executable was not found.
         */
        static the::astring FindExecutablePath(const the::astring& filename);

        /**
         * Answer the fully qualified path to a given executable. This method
         * searches for the requested executable in a operating system
         * dependent way, usually checking the current working directory and
         * a system environment search path.
         *
         * @param filename The file name of the executable to search for. Do
         *                 not specify any path information.
         *
         * @return The fully qualified path to the executable or an empty
         *         string if the executable was not found.
         */
        static the::wstring FindExecutablePath(const the::wstring& filename);

        /**
         * Answer the path to the currently running application including the
         * application name itself.
         *
         * @return The fully qualified path to the currently running
         *         application or an empty string in case of an error.
         */
        static the::astring GetApplicationPathA(void);

        /**
         * Answer the path to the currently running application including the
         * application name itself.
         *
         * @return The fully qualified path to the currently running
         *         application or an empty string in case of an error.
         */
        static the::wstring GetApplicationPathW(void);

        /**
         * Answer the current working directory.
         *
         * The returned string is guaranteed to end with a path separator.
         *
         * @return The current working directory.
         *
         * @throws the::system::system_exception If the directory cannot be retrieved
         * @throws std::bad_alloc If there is not enough memory for storing the
         *                        directory.
         */
        static the::astring GetCurrentDirectoryA(void);

        /**
         * Answer the current working directory.
         *
         * The returned string is guaranteed to end with a path separator.
         *
         * @return The current working directory.
         *
         * @throws the::system::system_exception If the directory cannot be retrieved
         * @throws std::bad_alloc If there is not enough memory for storing the
         *                        directory.
         */
        static the::wstring GetCurrentDirectoryW(void);

        /**
         * Answer the directory path containing the given path.
         *
         * If 'path' designates a file, the directory containing the file is 
         * returned.
         *
         * If 'path' designates a directory, the directory containing this 
         * directory is returned.
         *
         * @param path Path of a file or directory.
         */
        static the::astring GetDirectoryName(const char *path);

        /**
         * Answer the directory path containing the given path.
         *
         * If 'path' designates a file, the directory containing the file is 
         * returned.
         *
         * If 'path' designates a directory, the directory containing this 
         * directory is returned.
         *
         * @param path Path of a file or directory.
         */
        static the::wstring GetDirectoryName(const wchar_t *path);

        /**
         * Answer the current temp directory.
         *
         * Windows: On Windows, the %TEMP% environment variable is evaluated and
         * returned.
         *
         * Linux: The result is always "/tmp"
         *
         * @return The current temp directory.
         *
         * @throws the::system::system_exception if an error occured.
         */
        static the::astring GetTempDirectoryA(void);

        /**
         * Answer the current temp directory.
         *
         * Windows: On Windows, the %TEMP% environment variable is evaluated and
         * returned.
         *
         * Linux: The result is always "/tmp"
         *
         * @return The current temp directory.
         *
         * @throws the::system::system_exception if an error occured.
         */
        static the::wstring GetTempDirectoryW(void);

        /**
         * Answer the home directory of the user. On windows the 'My Documents'
         * folder is returned.
         *
         * The returned string is guaranteed to end with a path separator.
         *
         * @return The users home directory.
         *
         * @throws the::system::system_exception If the directory cannot be retrieved
         */
        static the::astring GetUserHomeDirectoryA(void);

        /**
         * Answer the home directory of the user. On windows the 'My Documents'
         * folder is returned.
         *
         * The returned string is guaranteed to end with a path separator.
         *
         * @return The users home directory.
         *
         * @throws the::system::system_exception If the directory cannot be retrieved
         */
        static the::wstring GetUserHomeDirectoryW(void);

        /**
         * Answer, whether 'path' is an absolute path.
         *
         * @param path The path to be tested.
         *
         * @return true, if 'path' is absolute, false otherwise.
         */
        inline static bool IsAbsolute(const the::astring& path) {
            return !Path::IsRelative(path);
        }

        /**
         * Answer, whether 'path' is an absolute path.
         *
         * @param path The path to be tested.
         *
         * @return true, if 'path' is absolute, false otherwise.
         */
        inline static bool IsAbsolute(const the::wstring& path) {
            return !Path::IsRelative(path);
        }

        /**
         * Answer, whether 'path' is a relative path.
         *
         * @param path The path to be tested.
         *
         * @return true, if 'path' is relative, false otherwise.
         */
        static bool IsRelative(const the::astring& path);

        /**
         * Answer, whether 'path' is a relative path.
         *
         * @param path The path to be tested.
         *
         * @return true, if 'path' is relative, false otherwise.
         */
        static bool IsRelative(const the::wstring& path);

        /**
         * Creates a directory and all intermediate directories on the path 
         * which currently do not exist. If the creating of on directory fails
         * all intermediate directories created by this methode call will be 
         * removed, if possible.
         *
         * @param path The path to the directory to be created. 
         *
         * @throws the::system::system_exception if an error occured.
         */
        static void MakeDirectory(const the::astring& path);

        /**
         * Creates a directory and all intermediate directories on the path 
         * which currently do not exist. If the creating of on directory fails
         * all intermediate directories created by this methode call will be 
         * removed, if possible.
         *
         * @param path The path to the directory to be created. 
         *
         * @throws the::system::system_exception if an error occured.
         */
        static void MakeDirectory(const the::wstring& path);

        /**
         * Clears a directory by removing all files and optionally all 
         * subdirectories.
         *
         * @param path The director to purge.
         * @param recursive If true all subdirectories will be also removed.
         */
        static void PurgeDirectory(const the::astring& path, bool recursive);

        /**
         * Clears a directory by removing all files and optionally all 
         * subdirectories.
         *
         * @param path The director to purge.
         * @param recursive If true all subdirectories will be also removed.
         */
        static void PurgeDirectory(const the::wstring& path, bool recursive);

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
        static inline the::astring Resolve(the::astring path) {
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
        static inline the::wstring Resolve(the::wstring path) {
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
        static the::astring Resolve(the::astring path, the::astring basepath);

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
        static the::wstring Resolve(the::wstring path, the::wstring basepath);

        /**
         * Changes the current directory to be 'path'.
         *
         * @param path The path to the new current directory.
         *
         * @throws the::system::system_exception If setting a new current directory fails, 
         *                         e. g. 'path' does not exist.
         */
        static void SetCurrentDirectory(const the::astring& path);

        /**
         * Changes the current directory to be 'path'.
         *
         * @param path The path to the new current directory.
         *
         * @throws the::system::system_exception If setting a new current directory fails, 
         *                         e. g. 'path' does not exist.
         */
        static void SetCurrentDirectory(const the::wstring& path);

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
    bool Path::Compare(const T& lhs, const T& rhs) {
        T rlhs = (Path::IsAbsolute(lhs)) ? Path::Canonicalise(lhs) : Path::Resolve(lhs);
        T rrhs = (Path::IsAbsolute(rhs)) ? Path::Canonicalise(rhs) : Path::Resolve(rhs);
#ifdef _WIN32
        return the::text::string_utility::compare(rlhs, rrhs, false) == 0;
#else /* _WIN32 */
        return the::text::string_utility::compare(rlhs, rrhs, true) == 0;
#endif /* _WIN32 */
    }
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_PATH_H_INCLUDED */
