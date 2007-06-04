/*
 * StringTokeniser.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_STRINGTOKENISER_H_INCLUDED
#define VISLIB_STRINGTOKENISER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/Iterator.h"


namespace vislib {


    /**
     * Tokeniser class splitting a string into substrings based on a delimitor 
     * string. The template parameter C must be the CharTraits class of the 
     * corresponding string type.
     */
    template<class C> class StringTokeniser: public Iterator<const String<C> > {
    public:

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param delimiter The delimiting string. Must not be Empty.
         */
        StringTokeniser(const String<C>& input, const String<C>& delimiter);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param delimiter The delimiting string. Must not be NULL.
         */
        StringTokeniser(const String<C>& input, const typename C::Char *delimiter);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param delimiter The delimiting character.
         */
        StringTokeniser(const String<C>& input, const typename C::Char delimiter);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param delimiter The delimiting string. Must not be Empty.
         */
        StringTokeniser(const typename C::Char *input, const String<C>& delimiter);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param delimiter The delimiting string. Must not be NULL.
         */
        StringTokeniser(const typename C::Char *input, const typename C::Char *delimiter);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param delimiter The delimiting character.
         */
        StringTokeniser(const typename C::Char *input, const typename C::Char delimiter);

        /** Dtor. */
        virtual ~StringTokeniser(void);

        /**
         * Answers the delimiter string.
         *
         * @return The delimiter string.
         */
        inline const String<C>& Delimiter(void) const {
            return this->delimiter;
        }

        /**
         * Answer whether there is a next element to iterator to.
         *
         * @return true if there is a next element, false otherwise.
         */
        virtual bool HasNext(void) const;

        /**
         * Answer the input string.
         *
         * @return The input string.
         */
        inline const String<C>& InputString(void) const {
            return this->input;
        }

        /**
         * Iterates to the next element and returns this element. The content
         * of the return value is undefined if HasNext has returned false.
         *
         * @return The next element, which becomes the current element after
         *         calling this methode.
         */
        virtual const String<C>& Next(void);

        /** 
         * Resets the StringTokeniser. The iterating methods "HasNext" and 
         * "Next" will behave exactly like directly after the construction of
         * the object.
         */
        void Reset(void);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised.
         *
         * @param input The new input string.
         */
        void Set(const String<C>& input);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised.
         *
         * @param input The new input string.
         */
        void Set(const typename C::Char* input);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new delimitor string.
         *
         * @param input The new input string.
         * @param delimitor The new delimitor string. Must not be Empty.
         */
        void Set(const String<C>& input, const String<C>& delimitor);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new delimitor string.
         *
         * @param input The new input string.
         * @param delimitor The new delimitor string. Must not be NULL.
         */
        void Set(const String<C>& input, const typename C::Char* delimitor);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new delimitor string.
         *
         * @param input The new input string.
         * @param delimitor The new delimitor string.
         */
        void Set(const String<C>& input, const typename C::Char delimitor);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new delimitor string.
         *
         * @param input The new input string.
         * @param delimitor The new delimitor string. Must not be Empty.
         */
        void Set(const typename C::Char* input, const String<C>& delimitor);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new delimitor string.
         *
         * @param input The new input string.
         * @param delimitor The new delimitor string. Must not be NULL.
         */
        void Set(const typename C::Char* input, const typename C::Char* delimitor);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new delimitor string.
         *
         * @param input The new input string.
         * @param delimitor The new delimitor string.
         */
        void Set(const typename C::Char* input, const typename C::Char delimitor);

    private:

        /** Prepares the next token */
        void prepareNextToken(void);

        /** the input string */
        String<C> input;

        /** the delimiter string */
        String<C> delimiter;

        /** the search position inside input */
        typename String<C>::Size inputPos;

        /** the return element for next */
        String<C> next;

    };


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const String<C>& input, const String<C>& delimiter)
            : input(input), delimiter(delimiter), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const String<C>& input, const typename C::Char *delimiter)
            : input(input), delimiter(delimiter), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const String<C>& input, const typename C::Char delimiter)
            : input(input), delimiter(delimiter, 1), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const typename C::Char *input, const String<C>& delimiter)
            : input(input), delimiter(delimiter), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const typename C::Char *input, const typename C::Char *delimiter)
            : input(input), delimiter(delimiter), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const typename C::Char *input, const typename C::Char delimiter)
            : input(input), delimiter(delimiter, 1), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::~StringTokeniser
     */
    template<class C> StringTokeniser<C>::~StringTokeniser(void) {
    }

    
    /*
     * StringTokeniser<C>::HasNext
     */
    template<class C> bool StringTokeniser<C>::HasNext(void) const {
        return this->inputPos != String<C>::INVALID_POS; 
    }

    
    /*
     * StringTokeniser<C>::Next
     */
    template<class C> const String<C>& StringTokeniser<C>::Next(void) {
        this->prepareNextToken();
        return this->next;
    }


    /*
     * StringTokeniser<C>::Reset
     */
    template<class C> void StringTokeniser<C>::Reset(void) {
        ASSERT(delimiter.Length() > 0);
        this->inputPos = 0;
        this->prepareNextToken();
        this->inputPos = 0;
    }


    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(const String<C>& input) {
        this->input = input;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(const typename C::Char* input) {
        this->input = input;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(const String<C>& input, const String<C>& delimitor) {
        this->input = input;
        this->delimitor = delimitor;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(const String<C>& input, const typename C::Char* delimitor) {
        this->input = input;
        this->delimitor = delimitor;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(const String<C>& input, const typename C::Char delimitor) {
        this->input = input;
        this->delimitor = String<C>(delimitor, 1);
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(const typename C::Char* input, const String<C>& delimitor) {
        this->input = input;
        this->delimitor = delimitor;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(const typename C::Char* input, const typename C::Char* delimitor) {
        this->input = input;
        this->delimitor = delimitor;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(const typename C::Char* input, const typename C::Char delimitor) {
        this->input = input;
        this->delimitor = String<C>(delimitor, 1);
        this->Reset();
    }


    /*
     * StringTokeniser<C>::prepareNextToken
     */
    template<class C> void StringTokeniser<C>::prepareNextToken(void) {
        typename String<C>::Size pos = this->input.Find(this->delimiter, this->inputPos);
        if (pos != String<C>::INVALID_POS) {
            this->next = this->input.Substring(this->inputPos, pos - this->inputPos);
            this->inputPos = pos + this->delimiter.Length();
        } else {
            this->next = this->input.Substring(this->inputPos);
            this->inputPos = String<C>::INVALID_POS;
        }
    }


    /** Template instantiation for ANSI strings. */
    typedef StringTokeniser<CharTraitsA> StringTokeniserA;

    /** Template instantiation for wide strings. */
    typedef StringTokeniser<CharTraitsW> StringTokeniserW;

    /** Template instantiation for TCHARs. */
    typedef StringTokeniser<TCharTraits> TStringTokeniser;

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_STRINGTOKENISER_H_INCLUDED */

