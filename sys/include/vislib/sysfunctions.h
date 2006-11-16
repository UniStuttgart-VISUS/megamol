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


#include "vislib/File.h"
#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {

    /** Default maximal line size to be read by the read line functions. */
    const unsigned int defMaxLineSize = 1024;

    /**
     * Answer the current working directory.
     *
     * @return The current working directory.
     *
     * @throws SystemException If the directory cannot be retrieved
     * @throws std::bad_alloc If there is not enough memory for storing the
     *                        directory.
     */
    StringA GetWorkingDirectoryA(void);

    /**
     * Answer the current working directory.
     *
     * @return The current working directory.
     *
     * @throws SystemException If the directory cannot be retrieved
     * @throws std::bad_alloc If there is not enough memory for storing the
     *                        directory.
     */
    StringW GetWorkingDirectoryW(void);

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
     * @throws IllegalParamException if input is NULL.
     * @throws IOException If the file cannot be read.
     * @throws std::bad_alloc If there is not enough memory to store the line.
     */
    StringA ReadLineFromFileA(File *input, unsigned int size = defMaxLineSize);

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
     * @throws IllegalParamException if input is NULL.
     * @throws IOException If the file cannot be read.
     * @throws std::bad_alloc If there is not enough memory to store the line.
     */
    StringW ReadLineFromFileW(File *input, unsigned int size = defMaxLineSize);

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_SYSFUNCTIONS_H_INCLUDED */

