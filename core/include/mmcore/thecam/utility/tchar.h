/*
 * the/tchar.h
 *
 * Copyright (c) 2012, TheLib Team (http://www.thelib.org/license)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of TheLib, TheLib Team, nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THELIB TEAM AS IS AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THELIB TEAM BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 * tchar.h  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef THE_TCHAR_H_INCLUDED
#define THE_TCHAR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#ifdef THE_WINDOWS
#    include <tchar.h>
#else /* THE_WINDOWS */

#    if defined(UNICODE) || defined(_UNICODE)

#        include <wchar.h>

typedef wchar_t TCHAR;

#        define _T(x) L##x

#        define _tprintf wprintf
#        define _ftprintf fwprintf
#        define _vftprintf vfwprintf
#        define _tcslen wcslen
#        define _vsntprintf vswprintf
#        define _tcscpy wcscpy
#        define _tsystem wsystem
#        define _tcscmp wcscmp
#        define _tcsicmp wcscasecmp
//#define _ttoi megamol::core::thecam::text::char_utility::parse_int // unsure about that one

#    else /* defined(UNICODE) || defined(_UNICODE) */

typedef char TCHAR;

#        define _T(x) x

#        define _tprintf printf
#        define _ftprintf fprintf
#        define _vftprintf vfprintf
#        define _tcslen strlen
#        define _vsntprintf vsnprintf
#        define _tcscpy strcpy
#        define _tsystem system
#        define _tcscmp strcmp
#        define _tcsicmp strcasecmp
#        define _ttoi atoi

#    endif /* defined(UNICODE) || defined(_UNICODE) */

#endif /* THE_WINDOWS */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_TCHAR_H_INCLUDED */
