/*
 * StringTokeniser.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_STRINGTOKENISER_H_INCLUDED
#define VISLIB_STRINGTOKENISER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Array.h"
#include "the/assert.h"
#include "the/string.h"
#include "vislib/Iterator.h"
#include "the/no_such_element_exception.h"


namespace vislib {


    /**
     * Tokeniser class splitting a string into substrings based on a separator
     * string. The template parameter C must be the CharTraits class of the 
     * corresponding string type.
     */
    template<class C>
    class StringTokeniser : public Iterator<const C > {
    public:

        /**
         * Splits a string into an array of substrings based of a given
         * separator.
         *
         * @param input The input string to be separated.
         * @param separator The separator.
         * @param removeEmpty If 'true' empty elements will be removed from
         *                    the result array before it is returned.
         *
         * @return An array of the separated substrings.
         */
        static Array<C > Split(const C& input,
            const C& separator, bool removeEmpty = false);

        /**
         * Splits a string into an array of substrings based of a given
         * separator.
         *
         * @param input The input string to be separated.
         * @param separator The separator.
         * @param removeEmpty If 'true' empty elements will be removed from
         *                    the result array before it is returned.
         *
         * @return An array of the separated substrings.
         */
        static Array<C > Split(const C& input,
            const typename C::value_type *separator, bool removeEmpty = false);

        /**
         * Splits a string into an array of substrings based of a given
         * separator.
         *
         * @param input The input string to be separated.
         * @param separator The separator.
         * @param removeEmpty If 'true' empty elements will be removed from
         *                    the result array before it is returned.
         *
         * @return An array of the separated substrings.
         */
        static Array<C > Split(const C& input,
            const typename C::value_type separator, bool removeEmpty = false);

        /**
         * Splits a string into an array of substrings based of a given
         * separator.
         *
         * @param input The input string to be separated.
         * @param separator The separator.
         * @param removeEmpty If 'true' empty elements will be removed from
         *                    the result array before it is returned.
         *
         * @return An array of the separated substrings.
         */
        static Array<C > Split(const typename C::value_type *input,
            const C& separator, bool removeEmpty = false);

        /**
         * Splits a string into an array of substrings based of a given
         * separator.
         *
         * @param input The input string to be separated.
         * @param separator The separator.
         * @param removeEmpty If 'true' empty elements will be removed from
         *                    the result array before it is returned.
         *
         * @return An array of the separated substrings.
         */
        static Array<C > Split(const typename C::value_type *input,
            const typename C::value_type *separator, bool removeEmpty = false);

        /**
         * Splits a string into an array of substrings based of a given
         * separator.
         *
         * @param input The input string to be separated.
         * @param separator The separator.
         * @param removeEmpty If 'true' empty elements will be removed from
         *                    the result array before it is returned.
         *
         * @return An array of the separated substrings.
         */
        static Array<C > Split(const typename C::value_type *input,
            const typename C::value_type separator, bool removeEmpty = false);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating string. Must not be Empty.
         */
        StringTokeniser(const C& input, const C& separator);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating string. Must not be NULL.
         */
        StringTokeniser(const C& input, const typename C::value_type *separator);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating character.
         */
        StringTokeniser(const C& input, const typename C::value_type separator);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating string. Must not be Empty.
         */
        StringTokeniser(const typename C::value_type *input, const C& separator);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating string. Must not be NULL.
         */
        StringTokeniser(const typename C::value_type *input, const typename C::value_type *separator);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating character.
         */
        StringTokeniser(const typename C::value_type *input, const typename C::value_type separator);

        /** Dtor. */
        virtual ~StringTokeniser(void);

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
        inline const C& InputString(void) const {
            return this->input;
        }

        /**
         * Iterates to the next element and returns this element. The content
         * of the return value is undefined if HasNext has returned false.
         *
         * @return The next element, which becomes the current element after
         *         calling this methode.
         */
        virtual const C& Next(void);

        /**
         * Iterates to the next nonempty element and returns this element. If
         * all remaining elements are empty an exception is thrown.
         *
         * @return The next non empty element if there is one, which becomes
         *         the current element after calling this method.
         *
         * @throws no_such_element_exception If all remaining elements are empty.
         */
        inline const C& NextNonEmpty(void) {
            while(this->HasNext()) {
                const C& str = this->Next();
                if (!str.empty()) {
                    return str;
                }
            }
            throw the::no_such_element_exception("End of Token Sequence",
                __FILE__, __LINE__);
        }

        /**
         * Iterates to the next nonempty element and returns this element. If
         * all remaining elements are empty 'outValid' is set to 'false' and
         * the value of the returned String is undefined.
         *
         * @param outValid Variable to receive the info whether or not the
         *                 returned element is valid or if the end of the
         *                 sequence has been reached.
         *
         * @return The next non empty element if there is one, which becomes
         *         the current element after calling this method.
         */
        inline const C& NextNonEmpty(bool& outValid) {
            outValid = false;
            while(this->HasNext()) {
                const C& str = this->Next();
                if (!str.empty()) {
                    outValid = true;
                    return str;
                }
            }
            static C empty; // I really do not like this!
            return empty;
        }

        /** 
         * Resets the StringTokeniser. The iterating methods "HasNext" and 
         * "Next" will behave exactly like directly after the construction of
         * the object.
         */
        void Reset(void);

        /**
         * Answers the separator string.
         *
         * @return The separator string.
         */
        inline const C& Separator(void) const {
            return this->separator;
        }

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised.
         *
         * @param input The new input string.
         */
        void Set(const C& input);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised.
         *
         * @param input The new input string.
         */
        void Set(const typename C::value_type* input);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string. Must not be Empty.
         */
        void Set(const C& input, const C& separator);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string. Must not be NULL.
         */
        void Set(const C& input, const typename C::value_type* separator);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string.
         */
        void Set(const C& input, const typename C::value_type separator);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string. Must not be Empty.
         */
        void Set(const typename C::value_type* input, const C& separator);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string. Must not be NULL.
         */
        void Set(const typename C::value_type* input,
            const typename C::value_type* separator);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string.
         */
        void Set(const typename C::value_type* input,
            const typename C::value_type separator);

    private:

        /**
         * Implementation of split.
         *
         * @param input The input string.
         * @param separator The separator string.
         * @param removeEmpty The remove empty flag.
         * @param outArray The array receiving the generated strings.
         */
        static void split(const typename C::value_type* input,
            const typename C::value_type* separator, bool removeEmpty,
            Array<C >& outArray);

        /** Prepares the next token */
        void prepareNextToken(void);

        /** the input string */
        C input;

        /** the separator string */
        C separator;

        /** the search position inside input */
        typename C::size_type inputPos;

        /** the return element for next */
        C next;

    };


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<C > StringTokeniser<C>::Split(
            const C& input, const C& separator,
            bool removeEmpty) {
        Array<C > retval;
        StringTokeniser<C>::split(input.c_str(), separator.c_str(),
            removeEmpty, retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<C > StringTokeniser<C>::Split(
            const C& input, const typename C::value_type *separator,
            bool removeEmpty) {
        Array<C > retval;
        StringTokeniser<C>::split(input.c_str(), separator,
            removeEmpty, retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<C > StringTokeniser<C>::Split(
            const C& input, const typename C::value_type separator,
            bool removeEmpty) {
        typename C::value_type sep[2] = { separator, 0 };
        Array<C > retval;
        StringTokeniser<C>::split(input.c_str(), sep, removeEmpty, 
            retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<C > StringTokeniser<C>::Split(
            const typename C::value_type *input, const C& separator,
            bool removeEmpty) {
        Array<C > retval;
        StringTokeniser<C>::split(input, separator.c_str(), 
            removeEmpty, retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<C > StringTokeniser<C>::Split(
            const typename C::value_type *input, const typename C::value_type *separator,
            bool removeEmpty) {
        Array<C > retval;
        StringTokeniser<C>::split(input, separator, removeEmpty, retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<C > StringTokeniser<C>::Split(
            const typename C::value_type *input, const typename C::value_type separator,
            bool removeEmpty) {
        typename C::value_type sep[2] = { separator, 0 };
        Array<C > retval;
        StringTokeniser<C>::split(input, sep, removeEmpty, retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const C& input, const C& separator)
            : input(input), separator(separator), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const C& input, const typename C::value_type *separator)
            : input(input), separator(separator), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const C& input, const typename C::value_type separator)
            : input(input), separator(separator, 1), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const typename C::value_type *input, const C& separator)
            : input(input), separator(separator), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const typename C::value_type *input, const typename C::value_type *separator)
            : input(input), separator(separator), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const typename C::value_type *input, const typename C::value_type separator)
            : input(input), separator(separator, 1), inputPos(0) {
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
        return this->inputPos != C::npos; 
    }

    
    /*
     * StringTokeniser<C>::Next
     */
    template<class C> const C& StringTokeniser<C>::Next(void) {
        this->prepareNextToken();
        return this->next;
    }


    /*
     * StringTokeniser<C>::Reset
     */
    template<class C> void StringTokeniser<C>::Reset(void) {
        THE_ASSERT(separator.size() > 0);
        this->inputPos = 0;
        this->prepareNextToken();
        this->inputPos = 0;
    }


    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const C& input) {
        this->input = input;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const typename C::value_type* input) {
        this->input = input;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const C& input, const C& separator) {
        this->input = input;
        this->separator = separator;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const C& input, const typename C::value_type* separator) {
        this->input = input;
        this->separator = separator;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const C& input, const typename C::value_type separator) {
        this->input = input;
        this->separator = C(separator, 1);
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const typename C::value_type* input, const C& separator) {
        this->input = input;
        this->separator = separator;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const typename C::value_type* input, const typename C::value_type* separator) {
        this->input = input;
        this->separator = separator;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const typename C::value_type* input, const typename C::value_type separator) {
        this->input = input;
        this->separator = C(separator, 1);
        this->Reset();
    }


    /*
     * StringTokeniser<C>::split
     */
    template<class C> void StringTokeniser<C>::split(
            const typename C::value_type* input, const typename C::value_type* separator,
            bool removeEmpty, Array<C >& outArray) {
        outArray.Clear();
        StringTokeniser<C> tokeniser(input, separator);
        while (tokeniser.HasNext()) {
            const C& el = tokeniser.Next();
            if (!removeEmpty || !el.empty()) {
                outArray.Append(el);
            }
        }
    }


    /*
     * StringTokeniser<C>::prepareNextToken
     */
    template<class C> void StringTokeniser<C>::prepareNextToken(void) {
        typename C::size_type pos 
            = this->input.find(this->separator, this->inputPos);
        if (pos != C::npos) {
            this->next 
                = this->input.substr(this->inputPos, pos - this->inputPos);
            this->inputPos = pos + this->separator.size();
        } else {
            this->next = this->input.substr(this->inputPos);
            this->inputPos = C::npos;
        }
    }


    /** Template instantiation for ANSI strings. */
    typedef StringTokeniser<the::astring> StringTokeniserA;

    /** Template instantiation for wide strings. */
    typedef StringTokeniser<the::wstring> StringTokeniserW;

    /** Template instantiation for TCHARs. */
    typedef StringTokeniser<the::tstring> TStringTokeniser;

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_STRINGTOKENISER_H_INCLUDED */

