/*
 * SingleLinkedList.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SINGLELINKEDLIST_H_INCLUDED
#define VISLIB_SINGLELINKEDLIST_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/memutils.h"
#include "vislib/Iterator.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/NoSuchElementException.h"
#include "vislib/assert.h"
#include "vislib/OrderedCollection.h"


namespace vislib {


    /**
     * class of a single linked list of object from type T
     */
    template <class T> class SingleLinkedList : public OrderedCollection<T> {
    private:

        /** Type for storing items */
        typedef struct _SLLItem {
            T item;
            struct _SLLItem *next;
        } Item;

    public:

        /**
         * Nested implementation of an iterator
         */
        class Iterator: public vislib::Iterator<T> {
        public:
            friend class SingleLinkedList<T>;

            /** default ctor */
            Iterator(void);

            /** copy ctor for assignment */
            Iterator(const Iterator& rhs);

            /** Dtor. */
            virtual ~Iterator(void);

            /** Behaves like Iterator<T>::HasNext */
            virtual bool HasNext(void) const;

            /** 
             * Behaves like Iterator<T>::Next 
             *
             * @throw IllegalStateException if there is no next element
             */
            virtual T& Next(void);

            /** assignment operator */
            Iterator& operator=(const Iterator& rhs);

        private:

            /** Ctor. */
            Iterator(SingleLinkedList &parent); 

            /** pointer to the next element store */
            Item *next;

            /** pointer to the last element returned by 'Next' */
            Item *prev;

        };

        /** ctor */
        SingleLinkedList(void);

        /**
         * copy ctor 
         * the created list creates items identical to the items of rhs.
         *
         * @param rhs The linked list to copy from.
         */
        SingleLinkedList(const SingleLinkedList<T>& rhs);

        /** Dtor. */
        virtual ~SingleLinkedList(void);

        /** 
         * Appends an item to the end of the list. Runtime complexity: O(1)
         *
         * @param item The item to be added.
         */
        virtual inline void Add(const T& element) {
            this->Append(element);
        }

        /** 
         * Appends an item to the end of the list. Runtime complexity: O(1)
         *
         * @param item The item to be added.
         */
        virtual void Append(const T& item);

        /**
         * Clears the whole list.
         */
        virtual void Clear(void);

        /**
         * Checks whether an item is contained in the list.
         *
         * @param item The item.
         *
         * @return true if the item is contained in the list, false otherwise.
         */
        virtual bool Contains(const T& item) const;

        /**
         * Answer the number of items in the collection.
         *
         * @return Number of items in the collection.
         */
        virtual SIZE_T Count(void) const;

        /**
         * Answer a pointer to the first copy of 'element' in the collection. 
         * If no element equal to 'element' is found, a NULL pointer is 
         * returned.
         *
         * @param element The element to be tested.
         *
         * @return A pointer to the local copy of 'element' or NULL, if no such
         *         element is found.
         */
        virtual const T *Find(const T& element) const;

        /**
         * Answer a pointer to the first copy of 'element' in the collection. 
         * If no element equal to 'element' is found, a NULL pointer is 
         * returned.
         *
         * @param element The element to be tested.
         *
         * @return A pointer to the local copy of 'element' or NULL, if no such
         *         element is found.
         */
        virtual T *Find(const T& element);

        /**
         * Answer the first element in the collection. Runtime complexity: O(1)
         *
         * @return A reference to the first element.
         *
         * @throws NoSuchElementException, if the collection is empty.
         */
        virtual const T& First(void) const;

        /**
         * Answer the first element in the collection. Runtime complexity: O(1)
         *
         * @return A reference to the first element.
         *
         * @throws NoSuchElementException, if the collection is empty.
         */
        virtual T& First(void);

        /**
         * Answer whether there is no element in the collection. Runtime 
         * complexity: O(1)
         *
         * @return true, if the collection is empty, false otherwise.
         */
        virtual inline bool IsEmpty(void) const {
            return this->first == NULL;
        }

        /**
         * Answer the last element in the collection. Runtime complexity: O(1)
         *
         * @return A reference to the last element.
         *
         * @throws NoSuchElementException, if the collection is empty.
         */
        virtual const T& Last(void) const;

        /**
         * Answer the last element in the collection. Runtime complexity: O(1)
         *
         * @return A reference to the last element.
         *
         * @throws NoSuchElementException, if the collection is empty.
         */
        virtual T& Last(void);

        /**
         * Adds an item to the beginning of the list. Runtime complexity: O(1)
         *
         * @param item The item to be added.
         */
        virtual void Prepend(const T& item);

// The semantics of Remove will be changed in the future to remove the first 
// occurrence of an object in an OrderedCollection. Use RemoveAll to erase all 
// items. Remove will not be supported on unordered collections any more.
#ifdef _WIN32
__declspec(deprecated("Remove will change its semantics in future versions. Use RemoveAll instead."))
#endif
        inline void Remove(const T& item) {
#ifndef _WIN32
#warning "Remove will change its semantics in future versions. Use RemoveAll instead."
#endif 
            this->RemoveAll(item);
        }

        /**
         * Removes an item from the list.
         * This method removes all items from the list that are equal to the
         * provided item.
         *
         * @param item The item to be removed.
         */
        virtual void RemoveAll(const T& item);

        /**
         * Removes an item from the list.
         * This method removes the item the given iterator has returned the 
         * last time 'Next' was called from the list. Items considered equal 
         * to that item are not removed. 
         *
         * @param iter The iterator of the item to be removed.
         *
         * @throw IllegalParamException if the iterator has not returned any
         *        item of this list at the last 'Next' call.
         */
        virtual void Remove(Iterator& iter);

        /**
         * Remove the first element from the collection. If the collection
         * is empty, this method has no effect. Runtime complexity: O(1)
         */
        virtual void RemoveFirst(void);

        /**
         * Remove the last element from the collection. If the collection is
         * empty, this method has no effect. Runtime complexity: O(n)
         */
        virtual void RemoveLast(void);

        /**
         * Returns an Iterator to the list, pointing before the first element.
         *
         * @return An iterator to the list.
         */
        class Iterator GetIterator(void);

        /**
         * Assignment operator. This list removes all items and then creates 
         * new items identical to the items of rhs.
         *
         * @param rhs The linked list to copy from.
         *
         * @return Reference to this list.
         */
        SingleLinkedList<T>& operator=(const SingleLinkedList<T>& rhs);

        /**
         * Compare operator. Two single linked lists are equal if the elements
         * in both lists are equal and in same order. Runtime complexity: O(n)
         *
         * @param rhs The right hand side operand
         *
         * @return if the lists are considered equal
         */
        bool operator==(const SingleLinkedList<T>& rhs) const;


    private:

        /** anchor of the single linked list */
        Item *first;

        /** last element of the single linked list */
        Item *last;

    };


    /*
     * SingleLinkedList<T>::Iterator::Iterator
     */
    template<class T>
    SingleLinkedList<T>::Iterator::Iterator(void) : next(NULL), prev(NULL) {
    }


    /*
     * SingleLinkedList<T>::Iterator::Iterator
     */
    template<class T>
    SingleLinkedList<T>::Iterator::Iterator(const typename SingleLinkedList<T>::Iterator& rhs) 
        : next(rhs.next), prev(rhs.prev) {
    }


    /*
     * SingleLinkedList<T>::Iterator::~Iterator
     */
    template<class T>
    SingleLinkedList<T>::Iterator::~Iterator(void) {
    }


    /*
     * SingleLinkedList<T>::Iterator::HasNext
     */
    template<class T>
    bool SingleLinkedList<T>::Iterator::HasNext(void) const {
        return (this->next != NULL);
    }


    /*
     * SingleLinkedList<T>::Iterator::HasNext
     */
    template<class T>
    T& SingleLinkedList<T>::Iterator::Next(void) {
        this->prev = this->next;
        if (!this->next) {
            throw IllegalStateException("No next element.", __FILE__, __LINE__);
        }
        this->next = this->next->next;
        return this->prev->item;
    }


    /*
     * SingleLinkedList<T>::Iterator::operator=
     */
    template<class T>
    typename SingleLinkedList<T>::Iterator& 
        SingleLinkedList<T>::Iterator::operator=(
            const typename SingleLinkedList<T>::Iterator& rhs) {
        this->next = rhs.next;
        this->prev = rhs.prev;
        return *this;
    }


    /*
     * SingleLinkedList<T>::Iterator::Iterator
     */
    template<class T>
    SingleLinkedList<T>::Iterator::Iterator(SingleLinkedList<T> &parent) 
        : next(parent.first), prev(NULL) {
    }


    /*
     * SingleLinkedList<T>::SingleLinkedList
     */
    template<class T>
    SingleLinkedList<T>::SingleLinkedList(void) : OrderedCollection<T>(), first(NULL), last(NULL) {
    }


    /*
     * SingleLinkedList<T>::SingleLinkedList
     */
    template<class T>
    SingleLinkedList<T>::SingleLinkedList(const SingleLinkedList<T>& rhs)
        : OrderedCollection<T>(), first(NULL), last(NULL) {
        *this = rhs;
    }


    /*
     * SingleLinkedList<T>::~SingleLinkedList
     */
    template<class T>
    SingleLinkedList<T>::~SingleLinkedList(void) {
        this->Clear();
    }


    /*
     * SingleLinkedList<T>::Append
     */
    template<class T>
    void SingleLinkedList<T>::Append(const T& item) {
        if (this->last) {
            this->last->next = new Item;
            this->last = this->last->next;
        } else {
            this->first = this->last = new Item;
        }
        this->last->next = NULL;
        this->last->item = item;
    }


    /*
     * SingleLinkedList<T>::Clear
     */
    template<class T>
    void SingleLinkedList<T>::Clear(void) {
        while (first) {
            last = first->next;
            delete first;
            first = last;
        }
    }


    /*
     * SingleLinkedList<T>::Contains
     */
    template<class T>
    bool SingleLinkedList<T>::Contains(const T& item) const {
        const Item *i = this->first;
        while(i) {
            if (i->item == item) return true;
            i = i->next;
        }
        return false;
    }


    /*
     * SingleLinkedList<T>::Count 
     */
    template<class T>
    SIZE_T SingleLinkedList<T>::Count(void) const {
        unsigned int c = 0;
        Item *i = this->first;
        while (i) {
            c++;
            i = i->next;
        }
        return c;
    }


    /*
     * SingleLinkedList<T>::Find
     */
    template<class T>
    const T *SingleLinkedList<T>::Find(const T& element) const {
        const Item *i = this->first;
        while(i) {
            if (i->item == element) return &i->item;
            i = i->next;
        }
        return NULL;
    }


    /*
     * SingleLinkedList<T>::Find
     */
    template<class T> T *SingleLinkedList<T>::Find(const T& element) {
        Item *i = this->first;
        while(i) {
            if (i->item == element) return &i->item;
            i = i->next;
        }
        return NULL;
    }


    /*
     * SingleLinkedList<T>::First
     */
    template<class T> const T& SingleLinkedList<T>::First(void) const {
        if (this->first == NULL) {
            throw vislib::NoSuchElementException("List is empty", __FILE__, __LINE__);
        }
        return this->first->item;
    }


    /*
     * SingleLinkedList<T>::First
     */
    template<class T> T& SingleLinkedList<T>::First(void) {
        if (this->first == NULL) {
            throw vislib::NoSuchElementException("List is empty", __FILE__, __LINE__);
        }
        return this->first->item;
    }


    /*
     * SingleLinkedList<T>::Last
     */
    template<class T> const T& SingleLinkedList<T>::Last(void) const {
        if (this->last == NULL) {
            throw vislib::NoSuchElementException("List is empty", __FILE__, __LINE__);
        }
        return this->last->item;
    }


    /*
     * SingleLinkedList<T>::Last
     */
    template<class T> T& SingleLinkedList<T>::Last(void) {
        if (this->last == NULL) {
            throw vislib::NoSuchElementException("List is empty", __FILE__, __LINE__);
        }
        return this->last->item;
    }


    /*
     * SingleLinkedList<T>::Prepend
     */
    template<class T>
    void SingleLinkedList<T>::Prepend(const T& item) {
        Item *i = new Item;
        i->next = this->first;
        this->first = i;
        if (!this->last) {
            this->last = this->first;
        }
        i->item = item;
    }


    /*
     * SingleLinkedList<T>::RemoveAll
     */
    template<class T>
    void SingleLinkedList<T>::RemoveAll(const T& item) {
        Item *i = this->first, *j = NULL;
        while(i) {
            if (i->item == item) {
                if (j) {
                    j->next = i->next;
                    if (this->last == i) {
                        this->last = j;
                    }
                    delete i;
                    i = j->next;
                } else {
                    ASSERT(this->first == i);
                    this->first = i->next;
                    if (this->last == i) {
                        this->last = i->next;
                    }
                    delete i;
                    i = this->first;
                }
            } else {
                j = i;
                i = i->next;
            }
        }
    }


    /*
     * SingleLinkedList<T>::Remove
     */
    template<class T>
    void SingleLinkedList<T>::Remove(
            typename SingleLinkedList<T>::Iterator& iter) {
        Item *i = this->first;
        if (iter.prev == NULL) {
            throw IllegalParamException("Invalid Iterator state", 
                __FILE__, __LINE__);
        }

        if (this->first == iter.prev) {
            this->first = this->first->next;
            if (!this->first) this->last = NULL;
            delete i;
        } else {
            while ((i->next) && (i->next != iter.prev)) {
                i = i->next;
            }
            if (i->next == iter.prev) {
                i->next = iter.next;
                delete iter.prev;
            } else {
                throw IllegalParamException("Invalid Iterator", 
                    __FILE__, __LINE__);
            }
        }
        iter.prev = NULL;
    }


    /*
     * SingleLinkedList<T>::RemoveFirst
     */
    template<class T> void SingleLinkedList<T>::RemoveFirst(void) {
        if (this->first) {
            Item *i = this->first;
            this->first = this->first->next;
            if (!this->first) this->last = NULL;
            delete i;
        }
    }


    /*
     * SingleLinkedList<T>::RemoveLast
     */
    template<class T> void SingleLinkedList<T>::RemoveLast(void) {
        if (this->last) {
            Item *newlast = NULL;
            if (this->first != this->last) {
                newlast = this->first;
                while (newlast->next != this->last) {
                    newlast = newlast->next;
                    ASSERT(newlast);
                }
            } else {
                this->first = NULL;
            }
            delete this->last;
            this->last = newlast;
        }
    }


    /*
     * SingleLinkedList<T>::GetIterator
     */
    template<class T>
    typename SingleLinkedList<T>::Iterator SingleLinkedList<T>::GetIterator(void) {
        return Iterator(*this);
    }


    /*
     * SingleLinkedList<T>::operator=
     */
    template<class T>
    SingleLinkedList<T>& SingleLinkedList<T>::operator=(const SingleLinkedList<T>& rhs) {
        if (this == &rhs) {
            return *this;
        }

        // might be implemented more intelligent reusing the item object 
        //  already present in this
        this->Clear();
        Iterator it = const_cast<SingleLinkedList<T>&>(rhs).GetIterator();
        while(it.HasNext()) {
            this->Append(it.Next());
        }

        return *this;
    }


    /*
     * SingleLinkedList<T>::operator==
     */
    template<class T>
    bool SingleLinkedList<T>::operator==(const SingleLinkedList<T>& rhs) const {
        const Item *i = this->first;
        const Item *j = rhs.first;

        while (i) {
            if ((!j) || (!(i->item == j->item))) return false;
            i = i->next;
            j = j->next;
        }

        return (j == NULL);
    }


} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SINGLELINKEDLIST_H_INCLUDED */
