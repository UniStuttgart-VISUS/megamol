/*
 * PoolAllocator.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_POOLALLOCATOR_H_INCLUDED
#define VISLIB_POOLALLOCATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/assert.h"
#include "vislib/CriticalSection.h"
#include "vislib/NullLockable.h"
#include "vislib/sysfunctions.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * This class creates typed memory for a single object of the template type.
     * It therefore cannot be used for allocating continuous arrays.
     *
     * The allocator uses the C++ allocation and deallocation mechanisms and 
     * therefore guarantees that the default ctor is called on the newly
     * allocated object and that the dtor is called before deallocating an
     * object.
     *
     * The objects handled by this allocator are held in pools, so that the
     * allocation operations are minimized.
     *
     * The template parameter L must be a lockable for thread synchronisation.
     * By default the NullLockable is used for instances. The default instance
     * used by the static members of this class however use the LS template
     * parameter which is a Lockable using a CriticalSection.
     *
     * The default allocation size is set to 100. It is not recommended to
     * use a PoolAllocator when an application only requires few objects of
     * that class.
     */
    template<class T, class L = NullLockable,
        class LS = CriticalSectionLockable> class PoolAllocator : public L {
    public:

        /**
         * Allocate an object of type T.
         *
         * @return A pointer to the newly allocated object.
         *
         * @throws std::bad_alloc If there was not enough memory to allocate the
         *                        object.
         */
        static inline T *Allocate(void) {
            return DefaultPool().Allocate();
        }

        /**
         * Deallocate 'ptr' and set it NULL.
         *
         * @param inOutPtr The pointer to be deallocated. The pointer will be 
         *                 set NULL before the method returns.
         */
        static inline void Deallocate(T *& inOutPtr) {
            deallocate(inOutPtr);
        }

        /**
         * Gets the instance of the default pool for this class.
         *
         * @return The instance of the default pool for this class.
         */
        static inline PoolAllocator& DefaultPool(void) {
            static PoolAllocator<T, LS, LS> defPool;
            return defPool;
        }

        /** The default allocation size for all pools */
        static const unsigned int DefaultAllocationSize;

        /** Ctor. */
        PoolAllocator(void);

        /** Dtor. */
        virtual ~PoolAllocator(void);

        /**
         * Allocate an object of type T.
         *
         * @return A pointer to the newly allocated object.
         *
         * @throws std::bad_alloc If there was not enough memory to allocate
         *                        the object.
         */
        T *AllocateObj(void);

        /**
         * Returns the allocation size in number of objects. The pool will
         * always grow by allocating memory pages of this size.
         *
         * @return The allocation size in number of objects.
         */
        unsigned int AllocationSize(void) const;

        /**
         * Cleans the pool object up. This method will deallocate memory pages
         * which do not contain objects currently in use. This method will not
         * compact the used objects and so the application might suffer from
         * scattering.
         */
        void Cleanup(void);

        /**
         * Deallocate 'ptr' and set it NULL.
         *
         * @param inOutPtr The pointer to be deallocated. The pointer will be 
         *                 set NULL before the method returns.
         */
        inline void DeallocateObj(T *& inOutPtr) {
            deallocate(inOutPtr);
        }

        /**
         * Sets the allocation size in number of objects. The pool will always
         * grow by allocating memory pages of this size.
         *
         * @param s The new allocation size in number of objects. Must not be
         *          Zero and small numbers are not recommended.
         */
        void SetAllocationSize(unsigned int s);

#if defined(DEBUG) || defined(_DEBUG)
        /**
         * Gets the three characteristic counts of this pool. THIS METHOD IS
         * ONLY AVAILABLE IN DEBUG BUILDS! This method is meant for internal
         * use only. Do not call!
         *
         * @param outPageCount  Receives the number of allocated memory pages.
         * @param outElCount    Receives the number of allocated elements.
         * @param outActElCount Receives the number of active elements.
         */
        inline void _GetCounts(UINT &outPageCount, UINT &outElCount,
                UINT &outActElCount) const {

            outPageCount = 0;
            outElCount = 0;
            outActElCount = 0;

            for (Page *p = this->firstPage; p; p = p->next) {
                outPageCount++;
                outElCount += p->cnt;
                Element *e = reinterpret_cast<Element*>(
                    reinterpret_cast<char*>(p) + sizeof(Page));
                for (unsigned int i = 0; i < p->cnt; i++) {
                    if (e[i].link == p) outActElCount++;
                }
            }

        }
#endif /* defined(DEBUG) || defined(_DEBUG) */

    private:

        /** private nested structure for objects in a memory page */
        typedef struct _Element_t {

            /** the link to the next free element or the page */
            void *link;

            /** The object stored within this element */
            T obj;

        } Element;

        /** private nested structure for a memory page */
        typedef struct _Page_t {

            /** The next page */
            struct _Page_t *next;

            /** The owner of the page */
            PoolAllocator *owner;

            /** The number of elements stored */
            unsigned int cnt;

        } Page;

        /**
         * Deallocate 'ptr' and set it NULL.
         *
         * @param inOutPtr The pointer to be deallocated. The pointer will be 
         *                 set NULL before the method returns.
         */
        static void deallocate(T *& inOutPtr);

        /**
         * Tests whether a page has active objects, or not.
         *
         * @param page The page to test.
         *
         * @return 'true' if the page has no active elements, or 'false' if
         *         the page has at least one active element.
         */
        static bool isPageClean(const Page *page);

        /** Forbidden copy ctor. */
        PoolAllocator(const PoolAllocator& src) {
            // intentionally empty
        }

        /** Forbidden assignment operator */
        PoolAllocator& operator=(const PoolAllocator& rhs) {
            // intentionally empty
            return *this;
        }

        /** The allocation size */
        unsigned int allocSize;

        /** The first page of the pool */
        Page *firstPage;

        /** The first unused element available in the pool */
        Element *firstUnused;

    };


    /*
     * PoolAllocator<T, L, LS>::DefaultAllocationSize
     */
    template<class T, class L, class LS>
    const unsigned int PoolAllocator<T, L, LS>::DefaultAllocationSize = 100;


    /*
     * PoolAllocator<T, L, LS>::PoolAllocator
     */
    template<class T, class L, class LS>
    PoolAllocator<T, L, LS>::PoolAllocator(void)
            : allocSize(DefaultAllocationSize), firstPage(NULL),
            firstUnused(NULL) {
        // intentionally empty
    }


    /*
     * PoolAllocator<T, L, LS>::~PoolAllocator
     */
    template<class T, class L, class LS>
    PoolAllocator<T, L, LS>::~PoolAllocator(void) {
        this->firstUnused = NULL;
        while (this->firstPage) {
            Page *nextPage = this->firstPage->next;

            this->firstPage->owner = NULL; // mark page as orphan
            this->firstPage->next = NULL;

            // keep unclean pages, because they will be deleted when the last
            // object is deallocated.
            if (isPageClean(this->firstPage)) {
                // all dtors for all elements have already been called.
                ::free(static_cast<void*>(this->firstPage));
            }

            this->firstPage = nextPage;
        }
    }


    /*
     * PoolAllocator<T, L, LS>::AllocateObj
     */
    template<class T, class L, class LS>
    T *PoolAllocator<T, L, LS>::AllocateObj(void) {
        Element *e = NULL;

        this->Lock();

        if (this->firstUnused) {
            e = this->firstUnused;
            this->firstUnused = static_cast<Element *>(e->link);
            e->link = NULL; // find the page ... :-(
            Page *p = this->firstPage;
            while (p) {
                SIZE_T offset = (reinterpret_cast<char*>(e)
                    - reinterpret_cast<char*>(p)) - sizeof(Page);
                if (offset <= p->cnt * sizeof(Element)) {
                    ASSERT((offset % sizeof(Element)) == 0);
                    e->link = static_cast<void*>(p);
                    break;
                }
                p = p->next;
            }

        } else {
            Page *newpage = static_cast<Page*>(
                ::malloc(sizeof(Page) + this->allocSize * sizeof(Element)));
            e = reinterpret_cast<Element*>(
                reinterpret_cast<char *>(newpage) + sizeof(Page));
            for (unsigned int i = 0; i < this->allocSize; i++) {
                e[i].link = static_cast<void*>(&e[i + 1]);
            }
            e[this->allocSize - 1].link = NULL; // this->firstUnused
            newpage->cnt = this->allocSize;
            newpage->owner = this;
            newpage->next = this->firstPage;
            this->firstPage = newpage;
            this->firstUnused = reinterpret_cast<Element*>(e->link);
            e->link = static_cast<void*>(this->firstPage);
        }

        this->Unlock();

        ASSERT(e != NULL);
        ASSERT(e->link != NULL);
        new (&e->obj) T; // ctor object
        return &e->obj;
    }


    /*
     * PoolAllocator<T, L, LS>::AllocationSize
     */
    template<class T, class L, class LS>
    unsigned int PoolAllocator<T, L, LS>::AllocationSize(void) const {
        // no need to synchronise this
        return this->allocSize;
    }


    /*
     * PoolAllocator<T, L, LS>::Cleanup
     */
    template<class T, class L, class LS>
    void PoolAllocator<T, L, LS>::Cleanup(void) {
        // Remove all pages without any active objects.
        this->Lock();
        Page *pages = this->firstPage;
        Page *nextPage;
        this->firstPage = NULL;
        bool eelDirty = false;

        // remove pages
        while (pages) {
            nextPage = pages->next;
            ASSERT(pages->owner == this);
            if (isPageClean(pages)) {
                // page not in use. => deallocate
                ::free(static_cast<void*>(pages));
                eelDirty = true;
            } else {
                // page in use, so keep it
                pages->next = this->firstPage;
                this->firstPage = pages;
            }
            pages = nextPage;
        }

        // reconstruct empty element list
        if (eelDirty) {
            this->firstUnused = NULL;
            for (pages = this->firstPage; pages; pages = pages->next) {
                Element *e = reinterpret_cast<Element *>(
                    reinterpret_cast<char*>(pages) + sizeof(Page));
                for (unsigned int i = 0; i < pages->cnt; i++) {
                    if (e[i].link != static_cast<void*>(pages)) {
                        e[i].link = this->firstUnused;
                        this->firstUnused = &e[i];
                    }
                }
            }
        }

        this->Unlock();
    }


    /*
     * PoolAllocator<T, L, LS>::SetAllocationSize
     */
    template<class T, class L, class LS>
    void PoolAllocator<T, L, LS>::SetAllocationSize(unsigned int s) {
        ASSERT(s > 0);
        // no need to synchronise this
        this->allocSize = s;
    }


    /*
     * PoolAllocator<T, L, LS>::deallocate
     */
    template<class T, class L, class LS>
    void PoolAllocator<T, L, LS>::deallocate(T *& inOutPtr) {
        if (inOutPtr == NULL) {
            return;
        }

        Element *e = CONTAINING_STRUCT(inOutPtr, Element, obj);
        Page *p = static_cast<Page *>(e->link);
        ASSERT(((reinterpret_cast<char*>(e) - reinterpret_cast<char*>(p))
            - sizeof(Page)) % sizeof(Element) == 0);
        ASSERT((((reinterpret_cast<char*>(e) - reinterpret_cast<char*>(p))
            - sizeof(Page)) / sizeof(Element)) <= p->cnt);

        if (p->owner != NULL) {
            // normal page
            p->owner->Lock();

            // Dtor.
            e->obj.~T();
            e->link = p->owner->firstUnused;
            p->owner->firstUnused = e;

            p->owner->Unlock();
        } else {
            // orphan page
            static CriticalSection cs; // avoid double freeing of the page

            cs.Lock();

            // Dtor.
            e->obj.~T();
            e->link = NULL; // We don't need a real empty element list here
                            // anymore, so just mark the element as unused.

            if (isPageClean(p)) {
                ::free(static_cast<void*>(p));
            }
            cs.Unlock();

        }

        inOutPtr = NULL;
    }


    /*
     * PoolAllocator<T, L, LS>::isPageClean
     */
    template<class T, class L, class LS>
    bool PoolAllocator<T, L, LS>::isPageClean(
            const typename PoolAllocator<T, L, LS>::Page *page) {
        const Element *e = reinterpret_cast<const Element *>(
            reinterpret_cast<const char*>(page) + sizeof(Page));
        for (unsigned int i = 0; i < page->cnt; i++) {
            if (e[i].link == static_cast<const void*>(page)) {
                // e[i] is an active object!
                return false;
            }
        }
        return true;
    }


} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_POOLALLOCATOR_H_INCLUDED */
