/*
 * sysfunctions.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
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
#endif /* _WIN32 */

#include "vislib/File.h"
#include "vislib/String.h"
#include "vislib/SystemException.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /** Default maximal line size to be read by the read line functions. */
    const unsigned int defMaxLineSize = 1024;

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
     * @param outStr   The string to receive the content.
     * @param filename The name of the file being read.
     *
     * @return true, if the file could be read, false, if the file was not 
     *         found or could not be opened.
     *
     * @throws IOException If reading from the file failed.
     */
    bool ReadTextFile(StringA& outStr, const char *filename);

    /**
     * Read the content of the file 'file' into 'outSrc'. 'outSrc' is being 
     * erased by this operation. 'file' will be read from the current position,
     * will be read until EoF, and will not be closed after operation.
     *
     * @param outStr The string to receive the content.
     * @param file   The file object being read.
     *
     * @return true, if the file could be read, false, if the file was not 
     *         found or could not be opened.
     *
     * @throws IOException If reading from the file failed.
     */
    bool ReadTextFile(StringA& outStr, File& file);

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
     * Writes a text to a file. If the file exists and force is 'true' the
     * existing file is overwritten.
     *
     * @param filename The path to the file to be written.
     * @param text The text to be written.
     * @param force Flag whether or not to overwrite an existing file.
     *
     * @return true if the data was written successfully, false otherwise.
     *
     * @throws SystemException in case of an error.
     */
    template<class tp1, class tp2>
    bool WriteTextFile(const String<tp1>& filename, const String<tp2>& text, 
            bool force = false) {
        bool retval = false;
        File file;
        if (file.Open(filename, File::WRITE_ONLY, File::SHARE_EXCLUSIVE,
                force ? File::CREATE_OVERWRITE : File::CREATE_ONLY)) {
            retval = WriteTextFile(file, text);
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
     * @param filename The file to be written.
     * @param text The text to be written.
     *
     * @return true if the data was written successfully, false otherwise.
     *
     * @throws SystemException in case of an error.
     */
    template<class tp>
    bool WriteTextFile(File& file, const String<tp>& text) {
        unsigned int len = text.Length() * sizeof(tp::Char);
        return (file.Write(text.PeekBuffer(), len) == len);
    }

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SYSFUNCTIONS_H_INCLUDED */
