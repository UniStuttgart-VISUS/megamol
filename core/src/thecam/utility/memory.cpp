/*
 * the\memory.cpp
 *
 * Copyright (C) 2014 TheLib Team (http://www.thelib.org/license).
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

#include "mmcore/thecam/utility/memory.h"

#include <cstdlib>
#include <exception>

#include <stdexcept>
#include "mmcore/thecam/math/functions.h"


/*
 * megamol::core::thecam::utility::aligned_malloc
 */
void* megamol::core::thecam::utility::aligned_malloc(const size_t size, const size_t alignment) {
    void* retval = nullptr;

    /* Sanity checks. */
    if (!megamol::core::thecam::math::is_power_of_two(alignment)) {
        throw std::runtime_error("alignment");
    }

    /* Allocate the memory. */
#if defined(THE_WINDOWS)
    retval = ::_aligned_malloc(size, alignment);

#elif defined(THE_LINUX)
    if (::posix_memalign(&retval, alignment, size) != 0) {
        throw std::bad_alloc();
    }

#else /* defined(THE_WINDOWS) */
#    error "Implementation of megamol::core::thecam::aligned_malloc is missing!"

#endif /* defined(THE_WINDOWS) */

    /* Make it C++-style... */
    if (retval == nullptr) {
        throw std::bad_alloc();
    }

    return retval;
}
