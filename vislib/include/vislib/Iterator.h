/*
 * Iterator.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib {


/**
 * Base class for container iterators
 */
template<class T>
class Iterator {
public:
    /** The type returned by the iterator */
    typedef T Type;

    /** Dtor. */
    virtual ~Iterator();

    /**
     * Answer whether there is a next element to iterator to.
     *
     * @return true if there is a next element, false otherwise.
     */
    virtual bool HasNext() const = 0;

    /**
     * Iterates to the next element and returns this element.
     *
     * @return The next element, which becomes the current element after
     *         calling this methode.
     */
    virtual T& Next() = 0;
};

/*
 * Iterator::~Iterator
 */
template<class T>
Iterator<T>::~Iterator() {}

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
