/*
 * Pair.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PAIR_H_INCLUDED
#define VISLIB_PAIR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib {


    /**
     * A pair of two object paired together.
     */
    template<class T, class U> class Pair {

    public:

        /** 
         * Create a new pair. This is only supported, if classes T and U have
         * an apropriate default constructor.
         */
        inline Pair(void) {}

        /**
         * Create and initialise a new pair.
         *
         * @param first  The first element.
         * @param second The second element.
         */
        inline Pair(const T& first, const U& second);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Pair(const Pair& rhs);

        /** Dtor. */
        ~Pair(void);

        /**
         * Provides direct access to the first element.
         *
         * @return The first element.
         */
        inline const T& First(void) const {
            return this->first;
        }

        /**
         * Provides direct access to the first element.
         *
         * @return The first element.
         */
        inline T& First(void) {
            return this->first;
        }

        /**
         * Provides direct access to the first element.
         *
         * @return The first element.
         */
        inline const T& GetFirst(void) const {
            return this->first;
        }

        /**
         * Provides direct access to the second element.
         *
         * @return The second element.
         */
        inline const U& GetSecond(void) const {
            return this->second;
        }

        /**
         * Answer the first element.
         *
         * @return The first element.
         */
        inline const T& Key(void) const {
            return this->first;
        }

        /**
         * Provides direct access to the second element.
         *
         * @return The second element.
         */
        inline const U& Second(void) const {
            return this->second;
        }

        /**
         * Provides direct access to the second element.
         *
         * @return The second element.
         */
        inline U& Second(void) {
            return this->second;
        }

        /**
         * Set the first element.
         *
         * @param first The new value.
         */
        inline void SetFirst(const T& first) {
            this->first = first;
        }
        
        /**
         * Set the first element.
         *
         * @param first The new value.
         */
        inline void SetSecond(const U& second) {
            this->second = second;
        }

        /**
         * Get the second element.
         *
         * @return The second element.
         */
        inline const U& Value(void) const {
            return this->second;
        }

        /**
         * Test for equality. Two pair are considered equal, if their
         * elements are equal. Apropriate operators are required for T and U
         * in order for this operator to work.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if the pairs are equal, false otherwise.
         */
        bool operator ==(const Pair& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if the pairs are not equal, false otherwise.
         */
        inline bool operator !=(const Pair& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        Pair& operator =(const Pair& rhs);

    private:

        /** The first member. */
        T first;

        /** The second member. */
        U second;

    };


    /*
     * vislib::Pair<T, U>::Pair
     */
    template<class T, class U> Pair<T, U>::Pair(const T& first, const U& second)
            : first(first), second(second) {
    }

    /*
     * vislib::Pair<T, U>::~Pair
     */
    template<class T, class U> Pair<T, U>::Pair(const Pair& rhs) 
            : first(rhs.first), second(rhs.second) {
    }

    /*
     * vislib::Pair<T, U>::~Pair
     */
    template<class T, class U> Pair<T, U>::~Pair(void) {
    }


    /*
     * vislib::Pair<T, U>::operator ==
     */
    template<class T, class U> 
    bool Pair<T, U>::operator ==(const Pair& rhs) const {
        return ((this->first == rhs.first) && (this->second == rhs.second));
    }


    /*
     * Pair<T, U>::operator =
     */
    template<class T, class U> 
    Pair<T, U>& Pair<T, U>::operator =(const Pair& rhs) {
        if (this != &rhs) {
            this->first = rhs.first;
            this->second = rhs.second;
        }

        return *this;
    }
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_PAIR_H_INCLUDED */
