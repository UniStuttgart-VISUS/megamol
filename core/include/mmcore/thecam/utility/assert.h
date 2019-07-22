/*
 * thecam/utility/assert.h
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
 * assert.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef THE_ASSERT_H_INCLUDED
#define THE_ASSERT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <cassert>


#ifdef THE_DEBUG
#    define THE_ASSERT(exp) assert(exp)
#else /* THE_DEBUG */
#    define THE_ASSERT(exp) ((void)0)
#endif /* THE_DEBUG */


#ifndef ASSERT
#    define ASSERT(exp) THE_ASSERT(exp)
#endif /* ASSERT */


// If DEBUG and NDEBUG are defined at the same time, the VERIFY macro goes mad!
#if (defined(DEBUG) || defined(_DEBUG)) && defined(NDEBUG)
#    error "{ DEBUG | _DEBUG } and NDEBUG must not be defined at the same time!"
#endif /* (defined(DEBUG) || defined(_DEBUG)) && defined(NDEBUG) */


#ifndef THE_VERIFY
#    if (defined(DEBUG) || defined(_DEBUG))
#        define THE_VERIFY(exp) THE_ASSERT(exp)
#    else
#        define THE_VERIFY(exp) (exp)
#    endif /* (defined(DEBUG) || defined(_DEBUG)) */
#endif     /* THE_VERIFY */


#ifndef VERIFY
#    define VERIFY(exp) THE_VERIFY(exp)
#endif /* VERIFY */


#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_ASSERT_H_INCLUDED */
