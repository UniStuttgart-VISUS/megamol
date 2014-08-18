/*
 * sysfunctions.h
 *
 * Copyright (C) 2006-2011 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SYSFUNCTIONS_H_INCLUDED
#define VISLIB_SYSFUNCTIONS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <shlwapi.h>
#else /* _WIN32 */
#include <sys/types.h>
#include <stddef.h>
#endif /* _WIN32 */
#include <stdio.h>
#include <stdarg.h>

#include "vislib/CharTraits.h"
#include "vislib/File.h"
#include "vislib/sysfunctions.h"
#include "vislib/RawStorage.h"
#include "vislib/String.h"
#include "vislib/SystemException.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * Possible text file formats
     */
    enum TextFileFormat {
        TEXTFF_UNSPECIFIC,
        TEXTFF_ASCII,       // locale dependent
        TEXTFF_UNICODE,     // OS dependent
        TEXTFF_UTF8,
        TEXTFF_UTF16,
        TEXTFF_UTF16_BE,
        TEXTFF_UTF32,
        TEXTFF_UTF32_BE,
        TEXTFF_UTF7,
        TEXTFF_UTF1,
        TEXTFF_UTF_EBCDIC,
        TEXTFF_SCSU,
        TEXTFF_BOCU1,
        TEXTFF_GB18030
    };

    /**
     * Possible options for writing BOM
     */
    enum TextFileFormatBOM {
        TEXTFF_BOM_UNSPECIFIC,  // write BOM when writing suitable format
        TEXTFF_BOM_YES,         // always write BOM when possible
        TEXTFF_BOM_NO           // never write BOM
    };

    /**
     * powered by miniport.h:
     * Returns the containing struct for the address of a given field.
     *
     * @param address The address of the field.
     * @param type    The type of the containing struct.
     * @param field   The name of the field of address.
     *
     * @return The address of the containing struct.
     */
    #ifndef CONTAINING_RECORD
#ifdef _WIN32
    #define CONTAINING_RECORD(address, type, field) \
        ((type *)((PCHAR)(address) - (ULONG_PTR)(&((type *)0)->field)))
#else /* _WIN32 */
    #define CONTAINING_RECORD(address, type, field) \
        ((type *)((PCHAR)(address) - ((ULONG_PTR)(&((type *)4)->field) - 4)))
#endif /* _WIN32 */
    #endif /* CONTAINING_RECORD */
    #define CONTAINING_STRUCT(address, type, field) \
        CONTAINING_RECORD(address, type, field)

    /** Default maximal line size to be read by the read line functions. */
    const unsigned int defMaxLineSize = 1024;

    /**
     * Answer whether the file name 'filename' matches with the glob pattern
     * 'pattern'. Be aware that this function does not try to match directory
     * names, but only file names (handled as simple strings without any
     * additional information).
     *
     * Pattern syntax:
     *  ?       matches any one character
     *  *       matches any number (even zero) characters
     *  [...]   matches one character from the given list of characters. Note
     *          that this character list does not support ranges!
     *  \?      matches the ? character
     *  \*      matches the * character
     *  \[      matches the [ character
     *  \]      matches the ] character
     *  \\      matches the \ character
     *
     * @param filename The file name string to test. This name should not
     *                 contain any directory information.
     * @param pattern The glob pattern to test. This pattern should not
     *                contain any directory information.
     *
     * @return 'true' if the pattern matches the file name, 'false' otherwise.
     */
    template <class T>
    bool FilenameGlobMatch(const T* filename, const T* pattern) {
        SIZE_T fnl = vislib::CharTraits<T>::SafeStringLength(filename);
        SIZE_T fnp = 0;
        SIZE_T pl = vislib::CharTraits<T>::SafeStringLength(pattern);
        SIZE_T pp = 0;

        while ((fnp < fnl) && (pp < pl)) {
            switch (pattern[pp]) {
                case '?':
                    pp++;
                    fnp++;
                    break;
                case '*':
                    pp++;
                    if (pp == pl) {
                         // pattern always matches rest of the filename
                        return true;
                    }
                    // this is super slow and lazy, but works
                    for (SIZE_T skipSize = 0; skipSize < (fnl - fnp);
                            skipSize++) {
                        if (FilenameGlobMatch(filename + fnp + skipSize,
                                pattern + pp)) {
                            return true;
                        }
                    }
                    return false;
                case '[': {
                    SIZE_T pps = pp;
                    while ((pp < pl) && (pattern[pp] != ']')) pp++;
                    if (pp == pl) return false;
                    vislib::String<vislib::CharTraits<T> > matchGroup(
                        pattern + pps + 1, static_cast<typename vislib::String<
                            vislib::CharTraits<T> >::Size>(pp - (1 + pps)));
                    if (!matchGroup.Contains(filename[fnp])) return false;
                    fnp++;
                    pp++;
                } break;
                case '\\': pp++; // fall through
                default:
                    if (filename[fnp] != pattern[pp]) return false;
                    fnp++;
                    pp++;
                    break;
            }
        }

        // pattern matches only if pattern and filename are consumed at same speed.
        return (fnp == fnl) && (pp == pl);
    }


    /**
     * Load a resource from the specified module.
     *
     * @param out          A RawStorage that will receive the resource data. 
     * @param hModule      A handle to the module whose executable file contains 
     *                     the resource. If hModule is NULL, the resource is 
     *                     loaded from the module that was used to create the 
     *                     current process.
     * @param resourceID   The name of the resource. Alternately, rather than 
     *                     a pointer, this parameter can be MAKEINTRESOURCE(ID),
     *                     where ID is the integer identifier of the resource.
     * @param resourceType The resource type. 
     *
     * @return A RawStorage containing the raw resource data. This is the same
     *         object passed in as out.
     *
     * @throws SystemException If the resource lookup or loading the resource
     *                         failed.
     * @throws UnsupportedOperationException On Linux.
     */
    RawStorage& LoadResource(RawStorage& out,
#ifdef _WIN32
        HMODULE hModule, 
#else /* _WIN32 */
        void *hModule,
#endif /* _WIN32 */        
        const char *resourceID, const char *resourceType);

    /**
     * Load a resource from the specified module.
     *
     * @param out          A RawStorage that will receive the resource data. 
     * @param hModule      A handle to the module whose executable file contains 
     *                     the resource. If hModule is NULL, the resource is 
     *                     loaded from the module that was used to create the 
     *                     current process.
     * @param resourceID   The name of the resource. Alternately, rather than 
     *                     a pointer, this parameter can be MAKEINTRESOURCE(ID),
     *                     where ID is the integer identifier of the resource.
     * @param resourceType The resource type. 
     *
     * @return A RawStorage containing the raw resource data. This is the same
     *         object passed in as out.
     *
     * @throws SystemException If the resource lookup or loading the resource
     *                         failed.
     * @throws UnsupportedOperationException On Linux.
     */
    RawStorage& LoadResource(RawStorage& out,
#ifdef _WIN32
        HMODULE hModule, 
#else /* _WIN32 */
        void *hModule,
#endif /* _WIN32 */           
        const wchar_t *resourceID, const wchar_t *resourceType);


    /**
     * Reads ansi characters from the file until the end of file, a line break 
     * is reached, or size characters are read. The returned string does not
     * contain the line break if one had been read.
     * Remarks: The methode does not perform any buffering, so you might want 
     * to use a buffered file.
     *
     * @param input The input file.
     * @param size The maximum number of character to read.
     *
     * @return The string holding the line read.
     *
     * @throws IOException If the file cannot be read.
     * @throws std::bad_alloc If there is not enough memory to store the line.
     */
    StringA ReadLineFromFileA(File& input, unsigned int size = defMaxLineSize);

    /**
     * Reads unicode characters from the file until the end of file, a line 
     * break is reached, or size characters are read. The returned string does 
     * not contain the line break if one had been read.
     * Remarks: The methode does not perform any buffering, so you might want 
     * to use a buffered file. wchar_t characters are read, so keep in mind
     * that files will not be compatible between windows and linux because of
     * different values of sizeof(wchar_t).
     *
     * @param input The input file.
     * @param size The maximum number of character to read.
     *
     * @return The string holding the line read.
     *
     * @throws IOException If the file cannot be read.
     * @throws std::bad_alloc If there is not enough memory to store the line.
     */
    StringW ReadLineFromFileW(File& input, unsigned int size = defMaxLineSize);

#if defined(UNICODE) || defined(_UNICODE)
#define ReadLineFromFile ReadLineFromFileW
#else /* defined(UNICODE) || defined(_UNICODE) */
#define ReadLineFromFile ReadLineFromFileA
#endif /* defined(UNICODE) || defined(_UNICODE) */

    /**
     * Read the content of the file 'filename' into 'outSrc'. 'outSrc' is 
     * being erased by this operation.
     *
     * @param outStr      The string to receive the content
     * @param filename    The name of the file being read
     * @param format      The format of the text to read
     * @param forceFormat If true 'format' is used even if a BOM is found
     *
     * @return true, if the file could be read, false, if the file was not 
     *         found or could not be opened.
     *
     * @throws IOException If reading from the file failed.
     */
    template<class tp1, class tp2>
    bool ReadTextFile(String<tp1>& outStr, const tp2 *filename,
            TextFileFormat format = TEXTFF_UNSPECIFIC,
            bool forceFormat = false) {
        File file;
        bool retval = false;
        if (file.Open(filename, File::READ_ONLY, File::SHARE_READ,
                File::OPEN_ONLY)) {
            retval = ReadTextFile(outStr, file, format, forceFormat);
            file.Close();
        } else {
            // works because the last error still contains the correct value
            throw SystemException(__FILE__, __LINE__);
        }
        return retval;
    }

    /**
     * Read the content of the file 'filename' into 'outSrc'. 'outSrc' is 
     * being erased by this operation.
     *
     * @param outStr   The string to receive the content
     * @param filename The name of the file being read
     * @param format   The format of the text to read
     * @param forceFormat If true 'format' is used even if a BOM is found
     *
     * @return true, if the file could be read, false, if the file was not 
     *         found or could not be opened.
     *
     * @throws IOException If reading from the file failed.
     */
    template<class tp1, class tp2>
    bool ReadTextFile(String<tp1>& outStr, const String<tp2>& filename,
            TextFileFormat format = TEXTFF_UNSPECIFIC,
            bool forceFormat = false) {
        return ReadTextFile(outStr, filename.PeekBuffer(), format,
            forceFormat);
    }

    /**
     * Read the content of the file 'file' into 'outSrc'. 'outSrc' is being 
     * erased by this operation. 'file' will be read from the current position,
     * will be read until EoF, and will not be closed after operation.
     *
     * @param outStr The string to receive the content
     * @param file   The file object being read
     * @param format The format of the text to read
     * @param forceFormat If true 'format' is used even if a BOM is found
     *
     * @return true, if the file could be read, false, if the file was not 
     *         found or could not be opened.
     *
     * @throws IOException If reading from the file failed.
     */
    bool ReadTextFile(StringA& outStr, File& file,
        TextFileFormat format = TEXTFF_UNSPECIFIC, bool forceFormat = false);

    /**
     * Read the content of the file 'file' into 'outSrc'. 'outSrc' is being 
     * erased by this operation. 'file' will be read from the current position,
     * will be read until EoF, and will not be closed after operation.
     *
     * @param outStr The string to receive the content
     * @param file   The file object being read
     * @param format The format of the text to read
     * @param forceFormat If true 'format' is used even if a BOM is found
     *
     * @return true, if the file could be read, false, if the file was not 
     *         found or could not be opened.
     *
     * @throws IOException If reading from the file failed.
     */
    bool ReadTextFile(StringW& outStr, File& file,
        TextFileFormat format = TEXTFF_UNSPECIFIC, bool forceFormat = false);

    /**
     * Answer the number of milliseconds since midnight of the current day.
     *
     * @return milliseconds since midnight.
     */
    unsigned int GetTicksOfDay(void);

#ifdef _WIN32
    /**
     * Answer the version of a Windows system DLL.
     *
     * @param outVersion Receives the version of the specified module. The 
     *                   'cbSize' must have been set to the actual size of the
     *                   'outVersion' structure before calling the function.
     * @param moduleName The name of the system DLL to retrieve the version of.
     *
     * @return The return value of the DllGetVersion of 'moduleName', which is
     *         NOERROR in case of success or an appropriate error code 
     *         otherwise.
     *
     * @throws SystemException If the specified module could not be opened or if
     *                         it has no DllGetVersion function.
     */
    HRESULT GetDLLVersion(DLLVERSIONINFO& outVersion, const char *moduleName);

    /**
     * Answer the version of a Windows system DLL.
     *
     * @param outVersion Receives the version of the specified module. The 
     *                   'cbSize' must have been set to the actual size of the
     *                   'outVersion' structure before calling the function.
     * @param moduleName The name of the system DLL to retrieve the version of.
     *
     * @return The return value of the DllGetVersion of 'moduleName', which is
     *         NOERROR in case of success or an appropriate error code 
     *         otherwise.
     *
     * @throws SystemException If the specified module could not be opened or if
     *                         it has no DllGetVersion function.
     */
    HRESULT GetDLLVersion(DLLVERSIONINFO& outVersion, 
        const wchar_t * moduleName);
#endif /* _WIN32 */

    /**
     * Remove Windows kernel namespace prefixes "Global" and "Local" from 
     * 'name'. The function is case-insensitive.
     *
     * @param name A string that potentially begins with a Windows kernel
     *             namespace prefix.
     *
     * @return The name without kernel namespace prefix.
     */
    vislib::StringA RemoveKernelNamespace(const char *name);

    /**
     * Remove Windows kernel namespace prefixes "Global" and "Local" from 
     * 'name'. The function is case-insensitive.
     *
     * @param name A string that potentially begins with a Windows kernel
     *             namespace prefix.
     *
     * @return The name without kernel namespace prefix.
     */
    vislib::StringW RemoveKernelNamespace(const wchar_t *name);

    /**
     * Release the COM pointer 'ptr' and set it NULL if not yet NULL.
     *
     * @param ptr A pointer to a COM object (or any other object implementing
     *            reference counting via a Release() method.
     */
    template<class T> void SafeRelease(T*& ptr) {
        if (ptr != NULL) {
            ptr->Release();        
            ptr = NULL;    
        }
    }

    /**
     * Take a Windows IPC resource name and construct a POSIX name for Linux 
     * it. This involves removing a possible kernel namespace and prepending
     * a slash ('/').
     *
     * @param name A string that potentially begins with a Windows kernel
     *             namespace prefix.
     *
     * @return The name in POSIX-compatible format without kernel namespace.
     */
    vislib::StringA TranslateWinIpc2PosixName(const char *name);

    /**
     * Take a Windows IPC resource name and construct a POSIX name for Linux 
     * it. This involves removing a possible kernel namespace and prepending
     * a slash ('/').
     *
     * @param name A string that potentially begins with a Windows kernel
     *             namespace prefix.
     *
     * @return The name in POSIX-compatible format without kernel namespace.
     */
    vislib::StringW TranslateWinIpc2PosixName(const wchar_t *name);

#ifndef _WIN32
    /**
     * Convert a IPC resource name 'name', which might start with a Windows 
     * kernel namespace prefix, to a Linux System V IPC unique key.
     *
     * @param name The name of the resource.
     *
     * @return The System V unique key for the name.
     *
     * @throws SystemException If the key could not be created.
     */
    key_t TranslateIpcName(const char *name);
#endif /* !_WIN32 */

    /**
     * Writes a formatted string to a file. 'format' uses 'printf' syntax.
     * The template parameter should be automatically chosen to be 'char' or 
     * 'wchar_t'.
     *
     * @param out The file to which the formatted text line is written to.
     * @param format The text line format string, similar to 'printf'.
     *
     * @return 'true' on success, 'false' it not all the data has been
     *         written.
     *
     * @throws SystemException If there was an IO error.
     */
    template<class T>
    bool WriteFormattedLineToFile(File &out,
            const T *format, ...) {
        vislib::String<vislib::CharTraits<T> > tmp;
        va_list argptr;
        va_start(argptr, format);
        tmp.FormatVa(format, argptr);
        va_end(argptr);
        SIZE_T len = tmp.Length() * sizeof(T);
        return out.Write(tmp.PeekBuffer(), len) == len;
    }

    /**
     * Writes a string to a file. The template parameter should be
     * automatically chosen to be 'char' or 'wchar_t'.
     *
     * @param out The file to which the text line is written to.
     * @param text The text line to be written.
     *
     * @return 'true' on success, 'false' it not all the data has been
     *         written.
     *
     * @throws SystemException If there was an IO error.
     */
    template<class T>
    bool WriteLineToFile(File &out, const T *text) {
        SIZE_T len = vislib::CharTraits<T>::SafeStringLength(text)
            * sizeof(T);
        return out.Write(text, len) == len;
    }

    /**
     * Writes a text to a file. If the file exists and force is 'true' the
     * existing file is overwritten.
     *
     * @param filename The path to the file to be written.
     * @param text The text to be written.
     * @param force Flag whether or not to overwrite an existing file.
     * @param format The text file format to produce
     *
     * @return true if the data was written successfully, false otherwise.
     *
     * @throws SystemException in case of an error.
     */
    template<class tp1, class tp2>
    bool WriteTextFile(const String<tp1>& filename, const String<tp2>& text,
            bool force = false, TextFileFormat format = TEXTFF_UNSPECIFIC,
            TextFileFormatBOM bom = TEXTFF_BOM_UNSPECIFIC) {
        return WriteTextFile(filename.PeekBuffer(), text, force, format, bom);
    }

    /**
     * Writes a text to a file. If the file exists and force is 'true' the
     * existing file is overwritten.
     *
     * @param filename The path to the file to be written.
     * @param text The text to be written.
     * @param force Flag whether or not to overwrite an existing file.
     * @param format The text file format to produce
     *
     * @return true if the data was written successfully, false otherwise.
     *
     * @throws SystemException in case of an error.
     */
    template<class tp1, class tp2>
    bool WriteTextFile(const tp1 *filename, const String<tp2>& text,
            bool force = false, TextFileFormat format = TEXTFF_UNSPECIFIC,
            TextFileFormatBOM bom = TEXTFF_BOM_UNSPECIFIC) {
        bool retval = false;
        File file;
        if (file.Open(filename, File::WRITE_ONLY, File::SHARE_EXCLUSIVE,
                force ? File::CREATE_OVERWRITE : File::CREATE_ONLY)) {
            retval = WriteTextFile(file, text, format, bom);
            file.Close();
        } else {
            // works because the last error still contains the correct value
            throw SystemException(__FILE__, __LINE__);
        }
        return retval;
    }

    /**
     * Writes a text to a file.
     *
     * @param file The file stream to be written to
     * @param text The text to be written
     * @param format The text file format to produce
     *
     * @return true if the data was written successfully, false otherwise.
     *
     * @throws SystemException in case of an error.
     */
    bool WriteTextFile(File& file, const StringA& text,
        TextFileFormat format = TEXTFF_ASCII,
        TextFileFormatBOM bom = TEXTFF_BOM_UNSPECIFIC);

    /**
     * Writes a text to a file.
     *
     * @param file The file stream to be written to
     * @param text The text to be written
     * @param format The text file format to produce
     *
     * @return true if the data was written successfully, false otherwise.
     *
     * @throws SystemException in case of an error.
     */
    bool WriteTextFile(File& file, const StringW& text,
        TextFileFormat format = TEXTFF_UNICODE,
        TextFileFormatBOM bom = TEXTFF_BOM_UNSPECIFIC);

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SYSFUNCTIONS_H_INCLUDED */
