/*
 * the/aligned_allocator.h
 *
 * Copyright (c) 2016, TheLib Team (http://www.thelib.org/license)
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

#ifndef THE_ALIGNED_ALLOCATOR_H_INCLUDED
#define THE_ALIGNED_ALLOCATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

#include "mmcore/thecam/utility/memory.h"
#include "mmcore/thecam/utility/not_copyable.h"


namespace megamol {
namespace core {
namespace thecam {
namespace utility {

/**
 * An STL-compatible allocator which performs aligned memory allocation.
 *
 * megamol::core::thecam::utility::aligned_allocator is stateless.
 *
 * See http://en.cppreference.com/w/cpp/concept/Allocator for general
 * information about the allocator concept in the STL and
 * http://blogs.msdn.com/b/vcblog/archive/2008/08/28/the-mallocator.aspx
 * for an allocator sample.
 *
 * @tparam T The type to be allocated.
 * @tparam A The alignment in bytes.
 */
template <class T, size_t A> class aligned_allocator {

public:
    /** A constant pointer to allocated objects. */
    typedef const T* const_pointer;

    /** The type to express differences between addresses. */
    typedef std::ptrdiff_t difference_type;

    /** A constant reference to allocated objects. */
    typedef const T& const_reference;

    /** A pointer to allocated objects. */
    typedef T* pointer;

    /** A reference to allocated objects. */
    typedef T& reference;

    /** The type to express memory sizes. */
    typedef std::size_t size_type;

    /** The type of allocated objects. */
    typedef T value_type;

    // The following must be the same for all allocators.
    template <class U> struct rebind { typedef aligned_allocator<U, A> other; };

    /**
     * Initialises a new instance.
     */
    aligned_allocator(void) {}

    /**
     * Clone 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    inline aligned_allocator(const aligned_allocator& rhs) {}

    /**
     * Rebind 'rhs' to Tp.
     */
    template <typename Tp> inline aligned_allocator(const aligned_allocator<Tp, A>& rhs) {}

    /**
     * Finalises the instance.
     */
    inline ~aligned_allocator(void) {}

    /**
     * Determine the actual address of 'obj', event if value_type has an
     * overloaded operator &().
     *
     * @param ptr An object of value_type to determine the address of.
     *
     * @return The address of 'obj'.
     */
    inline pointer address(reference obj) const { return std::addressof(obj); }

    /**
     * Determine the actual address of 'obj', event if value_type has an
     * overloaded operator &().
     *
     * @param ptr An object of value_type to determine the address of.
     *
     * @return The address of 'obj'.
     */
    inline const_pointer address(const_reference obj) const { return std::addressof(obj); }

    /**
     * Answer the alignment (in bytes) that the allocator enforces.
     *
     * @return The alignment of the allocator.
     */
    inline size_type alignment(void) const { return A; }

    /**
     * Allocates uninitialised, aligned memory of size
     * 'cnt' * sizeof(value_type).
     *
     * @param cnt  The number of object to allocate memory for.
     * @param hint A pointer to a nearby memory location, which is ignored
     *             by this allocator.
     *
     * @return A pointer to 'cnt' * sizeof(value_type) bytes aligned to a
     *         alignment() address.
     *
     * @throws std::bad_alloc    In case the allocation failed.
     * @throws std::length_error In case 'cnt' exceeds max_size().
     */
    pointer allocate(const size_type cnt, std::allocator<void>::const_pointer hint = nullptr) const {
        // According to the above-mentioned sample in MSDN, the behaviour of
        // allocate(0) is undefined, so we return nullptr.
        if (cnt == 0) {
            return nullptr;
        }

        // According to the MSDN sample, all allocators should contain an
        // integer overflow check. The Standardisation Committee recommends
        // that std::length_error be thrown in the case of integer overflow.
        if (cnt > this->max_size()) {
            throw std::length_error("Integer overflow in "
                                    "megamol::core::thecam::utility::aligned_allocator.");
        }

        // Allocate the memory. We do not need to check for nullptr, because
        // megamol::core::thecam::utility::aligned_malloc already raises std::bad_alloc in case it
        // fails.
        auto retval = aligned_malloc(cnt * sizeof(value_type), this->alignment());
        THE_ASSERT(retval != nullptr);

        return static_cast<pointer>(retval);
    }

    /**
     * Constructs and object of value_type in allocated, uninitialised
     * memory designated by 'ptr'.
     *
     * @param ptr   The pointer to the uninitialised storage.
     * @param value A value passed to the copy constructor of value_type.
     */
    inline void construct(pointer ptr, const_reference value) const {
        ::new (static_cast<void*>(ptr)) value_type(value);
    }

    /**
     * Constructs and object of value_type in allocated, uninitialised
     * memory designated by 'ptr'.
     *
     * @tparam Tp
     * @tparam P
     *
     * @param ptr    The pointer to the uninitialised storage.
     * @param params The parameter list passed to the constructor of Tp.
     */
    template <class Tp, class... P> inline void construct(Tp* ptr, P&&... params) const {
        ::new (static_cast<void*>(ptr)) Tp(std::forward<P>(params)...);
    }

    /**
     * Deallocates the memory designated by 'ptr'
     *
     * @param ptr A pointer obtained from allocate().
     * @param cnt The number of objects passed to allocate() when 'ptr' was
     *            obtained.
     */
    inline void deallocate(pointer ptr, const size_type cnt) const { aligned_free(static_cast<void*>(ptr)); }

    /**
     * Destructs an object of value_type located at 'ptr'.
     *
     * @param A pointer to the object to be destructed.
     */
    inline void destroy(pointer ptr) const { ptr->~T(); }

    /**
     * Destructs an object located at 'ptr' as if it were of value_type.
     *
     * @param A pointer to the object to be destructed.
     */
    template <class Tp> inline void destroy(Tp* ptr) const { this->destroy(reinterpret_cast<pointer>(ptr)); }

    /**
     * Returns the maximum theoretically possible value of elements that can
     * be requested from allocate().
     *
     * @return The maximum number of objects that can be theoretically
     *         allocated.
     */
    inline size_type max_size(void) const {
        // return (static_cast<size_type>(0) - static_cast<size_type>(1))
        //    / sizeof(value_type);
        return (std::numeric_limits<size_type>::max)() / sizeof(value_type);
    }

    /**
     * Test fore equality.
     *
     * The operator must only return true if storage allocated from this
     * allocator can be deallocated from 'rhs', and vice versa. As
     * megamol::core::thecam::utility::aligned_allocator is stateless, this is always true.
     *
     * @param rhs The right-hand side operand.
     *
     * @return true.
     */
    inline bool operator==(const aligned_allocator& rhs) const { return true; }

    /**
     * Test for inequality.
     *
     * @param rhs The right-hand side operand.
     *
     * @return false.
     */
    inline bool operator!=(const aligned_allocator& rhs) const { return !(*this == rhs); }
};

} /* end namespace utility*/
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_ALIGNED_ALLOCATOR_H_INCLUDED */
