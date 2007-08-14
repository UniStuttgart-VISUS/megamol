/*
 * sysfunctions.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SYSFUNCTIONS_H_INCLUDED
#define VISLIB_SYSFUNCTIONS_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <shlwapi.h>
#endif /* _WIN32 */

#include "vislib/File.h"
#include "vislib/String.h"
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

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SYSFUNCTIONS_H_INCLUDED */
