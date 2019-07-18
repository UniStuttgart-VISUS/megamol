/*
 * thecam/config/config.h
 *
 * Copyright (c) 2012-2014, TheLib Team (http://www.thelib.org/license)
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
#ifndef THE_CONFIG_H_INCLUDED
#define THE_CONFIG_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


/*
 * Select operating system
 */
#if defined(_WIN32)
/* this is Windows */
#    define THE_WINDOWS (1)
#elif (defined(linux) || defined(__linux) || defined(__linux__))
/* this is linux */
#    define THE_LINUX (1)
#endif /* defined(_WIN32) */


/* sanity checks for os settings */
#if defined(THE_WINDOWS) && defined(THE_LINUX)
#    error Your OS has dissociative identity disorder.
#endif

#if !defined(THE_WINDOWS) && !defined(THE_LINUX)
#    error You are using an unsupported operating system.
#endif


/*
 * Select processor word size
 */
#if defined(_WIN64) || defined(_M_X64) || defined(_M_IA64) || (defined(__LP64__) && (__LP64 == 1)) ||                  \
    (defined(_LP64) && (_LP64 == 1))
/* this is 64 bit os */
#    define THE_64BIT (1)

#elif defined(_M_IX86) || defined(THE_LINUX)
/* this is 32 bit os */
#    define THE_32BIT (1)

#endif

/* sanity checks of processor word size */
#if defined(THE_32BIT) && defined(THE_64BIT)
#    error Your OS has dissociative identity disorder concerning its size.
#endif

#if !defined(THE_32BIT) && !defined(THE_64BIT)
#    error Unable to detect processor word size.
#endif


/*
 * Combined macros
 */
#if defined(THE_WINDOWS) && (THE_32BIT)
#    define THE_WINDOWS32 (1)
#endif
#if defined(THE_WINDOWS) && (THE_64BIT)
#    define THE_WINDOWS64 (1)
#endif
#if defined(THE_LINUX) && (THE_32BIT)
#    define THE_LINUX32 (1)
#endif
#if defined(THE_LINUX) && (THE_64BIT)
#    define THE_LINUX64 (1)
#endif


/* Central (non-) debug version identification. */
#if (defined(DEBUG) || defined(_DEBUG))
#    define THE_DEBUG
#    if (defined(NDEBUG) || defined(_NDEBUG))
#        error "DEBUG and NDEBUG must not be defined at the same time!"
#    endif /* (defined(NDEBUG) || defined(_NDEBUG)) */

#else /* (defined(DEBUG) || defined(_DEBUG)) */
#    define THE_NDEBUG
#endif /* (defined(DEBUG) || defined(_DEBUG)) */


/* (Fully qualified) function name macro. */
#ifdef _MSC_VER
#    define THE_FUNCTION __FUNCTION__
#elif __GNUC__
#    define THE_FUNCTION __PRETTY_FUNCTION__
#else /* _MSC_VER */
#    error "THE_FUNCTION is not defined for this compiler!"
#endif /* _MSC_VER */


/* Ensure the latest Windows SDK version being defined. */
#ifdef THE_WINDOWS
#    include <SDKDDKVer.h>
#endif /* THE_WINDOWS */


/* Marco for compiler dependent pragmas (avoid warnings on different compilers) */
#ifdef _MSC_VER
#    define THE_MSVC_PRAGMA(A) __pragma(A)
#else /* _MSC_VER */
#    define THE_MSVC_PRAGMA(A)
#endif /* _MSC_VER */


/* Enable or disable certain features of TheLib. */
#include "mmcore/thecam/utility/features.h"

#if (defined(WITH_THE_DIRECTX9) || defined(WITH_THE_DIRECTX10) || defined(WITH_THE_DIRECTX11) ||                       \
     defined(WITH_THE_DIRECTX12))
#    define WITH_THE_DIRECTX
#endif /* (defined(WITH_THE_DIRECTX9) ... */

#if (defined(WITH_THE_DIRECTX10) || defined(WITH_THE_DIRECTX11) || defined(WITH_THE_DIRECTX12))
#    define WITH_THE_XMATH
#endif /* (defined(WITH_THE_DIRECTX9) ... */

#define WITH_THE_GLM

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_CONFIG_H_INCLUDED */
