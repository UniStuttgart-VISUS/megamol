/*
 * functioncast.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FUNCTIONCAST_H_INCLUDED
#define VISLIB_FUNCTIONCAST_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


/**
 * The 'function_cast' is used to cast between function pointers and void 
 * pointers. If a non-void, non-function pointer is used the behaviour is
 * undefined. Be sure not to cast away any specifications of calling 
 * conventions or this might result in stack corruption.
 */

#ifdef _WIN32

#define function_cast reinterpret_cast

#else /* _WIN32 */

/**
 * Utility template for implementing the cast from void pointer to an 
 * arbitrary function pointer. Template parameter 'T' is the function pointer
 * type.
 */
template<typename T> class function_cast {
public:

    /**
     * Ctor.
     *
     * @param input The function pointer
     */
    explicit function_cast(T input) {
        value.typePtr = input;
    }

    /**
     * Ctor.
     *
     * @param input The void pointer.
     */
    explicit function_cast(void * input) {
        value.voidPtr = input;
    }

    /**
     * Answers the void pointer of this cast.
     *
     * @return The void pointer of this cast.
     */
    void * VoidPtr(void) {
        return value.voidPtr;
    }

    /**
     * Answers the function pointer of this cast.
     *
     * @return The function pointer of this cast.
     */
    operator T(void) {
        return value.typePtr;
    }

private:

    /** The nested helper type for this cast. */
    typedef union {
        void * voidPtr;
        T typePtr;
    } castCrowbar;

    /** The pointer of this cast. */
    castCrowbar value;

};


/**
 * Template specialisation for casts from function pointers to void pointers.
 */
template<> class function_cast<void*> {
public:

    /**
     * Ctor.
     * The template parameter 'Tp' is the function pointer type.
     *
     * @param input The function pointer.
     */
    template<typename Tp> explicit function_cast(Tp input) {
        function_cast<Tp> helper(input);
        this->ptr = helper.VoidPtr();
    }

    /**
     * Answers the void pointer of this cast.
     *
     * @return The void pointer of this cast.
     */
    operator void*(void) {
        return this->ptr;
    }

private:

    /** The pointer of this cast */
    void *ptr;

};

#endif /* _WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FUNCTIONCAST_H_INCLUDED */
