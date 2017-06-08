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
#include "vislib/assert.h"
#include "vislib/String.h"
#include "vislib/Iterator.h"
#include "vislib/NoSuchElementException.h"


namespace vislib {


    /**
     * Tokeniser class splitting a string into substrings based on a separator
     * string. The template parameter C must be the CharTraits class of the 
     * corresponding string type.
     */
    template<class C>
    class StringTokeniser : public Iterator<const String<C> > {
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
        static Array<String<C> > Split(const String<C>& input,
            const String<C>& separator, bool removeEmpty = false);

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
        static Array<String<C> > Split(const String<C>& input,
            const typename C::Char *separator, bool removeEmpty = false);

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
        static Array<String<C> > Split(const String<C>& input,
            const typename C::Char separator, bool removeEmpty = false);

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
        static Array<String<C> > Split(const typename C::Char *input,
            const String<C>& separator, bool removeEmpty = false);

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
        static Array<String<C> > Split(const typename C::Char *input,
            const typename C::Char *separator, bool removeEmpty = false);

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
        static Array<String<C> > Split(const typename C::Char *input,
            const typename C::Char separator, bool removeEmpty = false);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating string. Must not be Empty.
         */
        StringTokeniser(const String<C>& input, const String<C>& separator);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating string. Must not be NULL.
         */
        StringTokeniser(const String<C>& input, const typename C::Char *separator);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating character.
         */
        StringTokeniser(const String<C>& input, const typename C::Char separator);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating string. Must not be Empty.
         */
        StringTokeniser(const typename C::Char *input, const String<C>& separator);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating string. Must not be NULL.
         */
        StringTokeniser(const typename C::Char *input, const typename C::Char *separator);

        /** 
         * Ctor. 
         *
         * @param input The input string to be tokenised.
         * @param separator The separating character.
         */
        StringTokeniser(const typename C::Char *input, const typename C::Char separator);

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
         * Iterates to the next nonempty element and returns this element. If
         * all remaining elements are empty an exception is thrown.
         *
         * @return The next non empty element if there is one, which becomes
         *         the current element after calling this method.
         *
         * @throws NoSuchElementException If all remaining elements are empty.
         */
        inline const String<C>& NextNonEmpty(void) {
            while(this->HasNext()) {
                const String<C>& str = this->Next();
                if (!str.IsEmpty()) {
                    return str;
                }
            }
            throw NoSuchElementException("End of Token Sequence",
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
        inline const String<C>& NextNonEmpty(bool& outValid) {
            outValid = false;
            while(this->HasNext()) {
                const String<C>& str = this->Next();
                if (!str.IsEmpty()) {
                    outValid = true;
                    return str;
                }
            }
            static String<C> empty; // I really do not like this!
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
        inline const String<C>& Separator(void) const {
            return this->separator;
        }

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
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string. Must not be Empty.
         */
        void Set(const String<C>& input, const String<C>& separator);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string. Must not be NULL.
         */
        void Set(const String<C>& input, const typename C::Char* separator);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string.
         */
        void Set(const String<C>& input, const typename C::Char separator);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string. Must not be Empty.
         */
        void Set(const typename C::Char* input, const String<C>& separator);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string. Must not be NULL.
         */
        void Set(const typename C::Char* input,
            const typename C::Char* separator);

        /**
         * Resets the StringTokeniser and sets a new input string to be
         * tokenised and a new separator string.
         *
         * @param input The new input string.
         * @param separator The new separator string.
         */
        void Set(const typename C::Char* input,
            const typename C::Char separator);

    private:

        /**
         * Implementation of split.
         *
         * @param input The input string.
         * @param separator The separator string.
         * @param removeEmpty The remove empty flag.
         * @param outArray The array receiving the generated strings.
         */
        static void split(const typename C::Char* input,
            const typename C::Char* separator, bool removeEmpty,
            Array<String<C> >& outArray);

        /** Prepares the next token */
        void prepareNextToken(void);

        /** the input string */
        String<C> input;

        /** the separator string */
        String<C> separator;

        /** the search position inside input */
        typename String<C>::Size inputPos;

        /** the return element for next */
        String<C> next;

    };


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<String<C> > StringTokeniser<C>::Split(
            const String<C>& input, const String<C>& separator,
            bool removeEmpty) {
        Array<String<C> > retval;
        StringTokeniser<C>::split(input.PeekBuffer(), separator.PeekBuffer(),
            removeEmpty, retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<String<C> > StringTokeniser<C>::Split(
            const String<C>& input, const typename C::Char *separator,
            bool removeEmpty) {
        Array<String<C> > retval;
        StringTokeniser<C>::split(input.PeekBuffer(), separator,
            removeEmpty, retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<String<C> > StringTokeniser<C>::Split(
            const String<C>& input, const typename C::Char separator,
            bool removeEmpty) {
        typename C::Char sep[2] = { separator, 0 };
        Array<String<C> > retval;
        StringTokeniser<C>::split(input.PeekBuffer(), sep, removeEmpty, 
            retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<String<C> > StringTokeniser<C>::Split(
            const typename C::Char *input, const String<C>& separator,
            bool removeEmpty) {
        Array<String<C> > retval;
        StringTokeniser<C>::split(input, separator.PeekBuffer(), 
            removeEmpty, retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<String<C> > StringTokeniser<C>::Split(
            const typename C::Char *input, const typename C::Char *separator,
            bool removeEmpty) {
        Array<String<C> > retval;
        StringTokeniser<C>::split(input, separator, removeEmpty, retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::Split
     */
    template<class C> Array<String<C> > StringTokeniser<C>::Split(
            const typename C::Char *input, const typename C::Char separator,
            bool removeEmpty) {
        typename C::Char sep[2] = { separator, 0 };
        Array<String<C> > retval;
        StringTokeniser<C>::split(input, sep, removeEmpty, retval);
        return retval;
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const String<C>& input, const String<C>& separator)
            : input(input), separator(separator), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const String<C>& input, const typename C::Char *separator)
            : input(input), separator(separator), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const String<C>& input, const typename C::Char separator)
            : input(input), separator(separator, 1), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const typename C::Char *input, const String<C>& separator)
            : input(input), separator(separator), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const typename C::Char *input, const typename C::Char *separator)
            : input(input), separator(separator), inputPos(0) {
        this->Reset();
    }


    /*
     * StringTokeniser<C>::StringTokeniser
     */
    template<class C> StringTokeniser<C>::StringTokeniser(
            const typename C::Char *input, const typename C::Char separator)
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
        ASSERT(separator.Length() > 0);
        this->inputPos = 0;
        this->prepareNextToken();
        this->inputPos = 0;
    }


    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const String<C>& input) {
        this->input = input;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const typename C::Char* input) {
        this->input = input;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const String<C>& input, const String<C>& separator) {
        this->input = input;
        this->separator = separator;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const String<C>& input, const typename C::Char* separator) {
        this->input = input;
        this->separator = separator;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const String<C>& input, const typename C::Char separator) {
        this->input = input;
        this->separator = String<C>(separator, 1);
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const typename C::Char* input, const String<C>& separator) {
        this->input = input;
        this->separator = separator;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const typename C::Char* input, const typename C::Char* separator) {
        this->input = input;
        this->separator = separator;
        this->Reset();
    }
    

    /*
     * StringTokeniser<C>::Set
     */
    template<class C> void StringTokeniser<C>::Set(
            const typename C::Char* input, const typename C::Char separator) {
        this->input = input;
        this->separator = String<C>(separator, 1);
        this->Reset();
    }


    /*
     * StringTokeniser<C>::split
     */
    template<class C> void StringTokeniser<C>::split(
            const typename C::Char* input, const typename C::Char* separator,
            bool removeEmpty, Array<String<C> >& outArray) {
        outArray.Clear();
        StringTokeniser<C> tokeniser(input, separator);
        while (tokeniser.HasNext()) {
            const String<C>& el = tokeniser.Next();
            if (!removeEmpty || !el.IsEmpty()) {
                outArray.Append(el);
            }
        }
    }


    /*
     * StringTokeniser<C>::prepareNextToken
     */
    template<class C> void StringTokeniser<C>::prepareNextToken(void) {
        typename String<C>::Size pos 
            = this->input.Find(this->separator, this->inputPos);
        if (pos != String<C>::INVALID_POS) {
            this->next 
                = this->input.Substring(this->inputPos, pos - this->inputPos);
            this->inputPos = pos + this->separator.Length();
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

