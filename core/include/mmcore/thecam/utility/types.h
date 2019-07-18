/*
 * thecam/utility/types.h
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
 * types.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef THE_TYPES_H_INCLUDED
#define THE_TYPES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#if defined(THE_WINDOWS)
#    include <windows.h>

#elif defined(THE_LINUX)

#    define __STDC_LIMIT_MACROS
#    include <cstddef>
#    include <inttypes.h>

typedef uint8_t byte;

#endif /* defined(THE_WINDOWS) */

#include <cstdint>


namespace megamol {
namespace core {
namespace thecam {
namespace utility {

/**
 * Dummy for providing overloads of parameterless methods that should
 * have a return value or not (similar hack like operator ++).
 */
typedef int discard_return_t;

/** Dummy for requesting uninitialised objects (eg megamol::core::thecam::math::vector). */
typedef int do_not_initialise_t;


/**
 * A constant that can be passed to some methods (eg megamol::core::thecam::ring_buffer::pop)
 * for omitting return values.
 */
extern const discard_return_t discard_return;

/**
 * A constant that can be passed to some constructors (eg of
 * megamol::core::thecam::math::vector) for omitting initialisation.
 */
extern const do_not_initialise_t do_not_initialise;

} /* end namespace utility */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */


#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_TYPES_H_INCLUDED */
