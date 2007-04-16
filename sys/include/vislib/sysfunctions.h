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
     * Answer the number of milliseconds since midnight of the current day.
     *
     * @return milliseconds since midnight.
     */
    unsigned int GetTicksOfDay(void);

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SYSFUNCTIONS_H_INCLUDED */
