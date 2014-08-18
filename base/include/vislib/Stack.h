/*
 * Stack.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_STACK_H_INCLUDED
#define VISLIB_STACK_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/memutils.h"
#include "vislib/NoSuchElementException.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * This classimplements a stack. The stack is implemented as a linked list.
     */
    template<class T> class Stack {

    public:

        /**
         * Create an empty stack.
         */
        Stack(void);

        /**
         * Copy ctor.
         * Makes a deep copy of 'src'
         *
         * @param src The object to clone from.
         */
        Stack(const Stack& src);

        /**
         * Create a new stack initially containing 'element'. This has the same
         * effect as creating an empty stack and calling Push(element) 
         * afterwards.
         *
         * @param element.
         */
        explicit Stack(const T& element);

        /** Dtor. */
        ~Stack(void);

        /**
         * Erase all elements from the stack.
         */
        void Clear(void);

        /**
         * Answer the elements in the stack. 
         *
         * This method has linear runtime complexity.
         *
         * @return The number of elements in the stack.
         */
        SIZE_T Count(void) const;

        /**
         * Answer whether the stack is empty.
         *
         * @return true, if the stack is empty, false otherwise.
         */
        inline bool IsEmpty(void) const {
            return (this->top == NULL);
        }

        /**
         * Answer a pointer to the topmost element on the stack.
         *
         * @return A pointer to the element on top of the stack.
         *
         * @throws NoSuchElementException If the stack is empty.
         */
        T *Peek(void) const;

        /**
         * Remove the topmost element from the stack and return it.
         *
         * @return The element on top of the stack.
         *
         * @throws NoSuchElementException If the stack is empty.
         */
        T Pop(void);

        /**
         * Push 'element' to top of the stack.
         *
         * @param element The element to be pushed on the stack.
         */
        void Push(const T& element);

        /**
         * Remove the topmost element from the stack, if there is any.
         */
        void RemoveTop(void);

        /**
         * Answer the topmost element on the stack without removing it.
         *
         * @return The element on top of the stack.
         *
         * @throws NoSuchElementException If the stack is empty.
         */
        const T& Top(void) const;

        /**
         * Assignment operator.
         * Makes a deep copy from 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return A reference to 'this' object.
         */
        Stack& operator=(const Stack& rhs);

    private:

        /** The storage for the stack elements. */
        typedef struct Element_t {
            T element;
            struct Element_t *next;
        } Element;

        /** The topmost element on the stack. */
        Element *top;

    };


    /*
     * vislib::Stack<T>::Stack
     */
    template<class T>
    Stack<T>::Stack(void) : top(NULL) {
    }


    /*
     * vislib::Stack<T>::Stack
     */
    template<class T>
    Stack<T>::Stack(const Stack& src) : top(NULL) {
        *this = src;
    }


    /*
     * vislib::Stack<T>::Stack
     */
    template<class T>
    Stack<T>::Stack(const T& element) {
        this->top = new Element;
        this->top->element = element;
        this->top->next = NULL;
    }


    /*
     * vislib::Stack<T>::~Stack
     */
    template<class T>
    Stack<T>::~Stack(void) {
        this->Clear();
    }


    /*
     * vislib::Stack<T>::Clear
     */
    template<class T>
    void Stack<T>::Clear(void) {
        Element *cursor = this->top;
        Element *tmp = NULL;

        while (cursor != NULL) {
            tmp = cursor;
            cursor = cursor->next;
            SAFE_DELETE(tmp);
        }

        this->top = NULL;
    }


    /*
     * vislib::Stack<T>::Count
     */
    template<class T>
    SIZE_T Stack<T>::Count(void) const {
        Element *cursor = this->top;
        SIZE_T retval = 0;

        while (cursor != NULL) {
            retval++;
            cursor = cursor->next;
        }

        return retval;
    }


    /*
     * vislib::Stack<T>::Push
     */
    template<class T>
    void Stack<T>::Push(const T& element) {
        Element *e = new Element;
        e->element = element;
        e->next = this->top;

        this->top = e;
    }


    /*
     * vislib::Stack<T>::Peek
     */
    template<class T>
    T *Stack<T>::Peek(void) const {
        if (this->top != NULL) {
            return &(this->top->element);
        } else {
            throw NoSuchElementException("Empty stack.", __FILE__, __LINE__);
        }
    }


    /*
     * vislib::Stack<T>::Pop
     */
    template<class T>
    T Stack<T>::Pop(void) {
        if (this->top != NULL) {
            T retval = this->top->element;
            this->RemoveTop();
            return retval;
        } else {
            throw NoSuchElementException("Empty stack.", __FILE__, __LINE__);
        }
    }


    /*
     * vislib::Stack<T>::RemoveTop
     */
    template<class T>
    void Stack<T>::RemoveTop(void) {
        Element *tmp = this->top;

        if (tmp != NULL) {
            this->top = tmp->next;
            SAFE_DELETE(tmp);
        }
    }


    /*
     * vislib::Stack<T>::Top
     */
    template<class T>
    const T& Stack<T>::Top(void) const {
        if (this->top != NULL) {
            return this->top->element;
        } else {
            throw NoSuchElementException("Empty stack.", __FILE__, __LINE__);
        }
    }

    /*
     * vislib::Stack<T>::operator=
     */
    template<class T>
    Stack<T>& Stack<T>::operator=(const Stack<T> &rhs) {
        if (this != &rhs) {
            this->Clear();

            if (rhs.top != NULL) {
                Element *ed = this->top = new Element(), *es = rhs.top;
                ed->element = es->element;
                while (es->next) {
                    ed->next = new Element();
                    ed = ed->next;
                    es = es->next;
                    ed->element = es->element;
                }
                ed->next = NULL;
            }
        }
        return *this;
    }

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_STACK_H_INCLUDED */
