/*
 * RegExCharTraits.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_REGEXCHARTRAITS_H_INCLUDED
#define VISLIB_REGEXCHARTRAITS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CharTraits.h"


namespace vislib {


    class RegExCharTraitsA : public CharTraitsA {

    protected:

        /** Superclass typedef. */
        typedef CharTraitsA Super;

        /** 
         * Array of abbreviations. The first character in each string is the 
         * abbreviation name, the rest is the regular expression that the
         * abbreviation stands for. The last entry in the array is a guard 
         * NULL.
         */
        static const Char *ABBREVIATIONS[];

        /** String containing all characters that must be escaped. */
        static const Char *ESCAPED_CHARS;

        static const Char TOKEN_BACKSLASH = '\\';
        static const Char TOKEN_BRACE_CLOSE = '}';
        static const Char TOKEN_BRACE_OPEN = '{';
        static const Char TOKEN_BRACKET_CLOSE = ']';
        static const Char TOKEN_BRACKET_OPEN = '[';
        static const Char TOKEN_CARET = '^';
        static const Char TOKEN_COMMA = ',';
        static const Char TOKEN_DOLLAR = '$';
        static const Char TOKEN_DOT = '.';
        static const Char TOKEN_MINUS = '-';
        static const Char TOKEN_PAREN_CLOSE = ')';
        static const Char TOKEN_PAREN_OPEN = '(';
        static const Char TOKEN_PLUS = '+';
        static const Char TOKEN_QUESTION = '?';
        static const Char TOKEN_STAR = '*';
        static const Char TOKEN_VERTICAL_BAR = '|';

        /** Forbidden Ctor. */
        RegExCharTraitsA(void);

        /* Declare our friends. */
        template<class T> friend class RegEx;

    }; /* end class RECharTraitsA */


    class RegExCharTraitsW : public CharTraitsW {

    protected:

        /** Superclass typedef. */
        typedef CharTraitsW Super;

        /** 
         * Array of abbreviations. The first character in each string is the
         * abbreviation name, the rest is the regular expression that the
         * abbreviation stands for. The last entry in the array is a guard 
         * NULL.
         */
        static const Char *ABBREVIATIONS[];

        /** String containing all characters that must be escaped. */
        static const Char *ESCAPED_CHARS;

        static const Char TOKEN_BACKSLASH = L'\\';
        static const Char TOKEN_BRACE_CLOSE = L'}';
        static const Char TOKEN_BRACE_OPEN = L'{';
        static const Char TOKEN_BRACKET_CLOSE = L']';
        static const Char TOKEN_BRACKET_OPEN = L'[';
        static const Char TOKEN_CARET = L'^';
        static const Char TOKEN_COMMA = L',';
        static const Char TOKEN_DOLLAR = L'$';
        static const Char TOKEN_DOT = L'.';
        static const Char TOKEN_MINUS = L'-';
        static const Char TOKEN_PAREN_CLOSE = L')';
        static const Char TOKEN_PAREN_OPEN = L'(';
        static const Char TOKEN_PLUS = L'+';
        static const Char TOKEN_QUESTION = L'?';
        static const Char TOKEN_STAR = L'*';
        static const Char TOKEN_VERTICAL_BAR = L'|';

        /** Forbidden Ctor. */
        RegExCharTraitsW(void);

        /* Declare our friends. */
        template<class T> friend class RegEx;

    }; /* end class RECharTraitsW */


    /* Typedef for TCHAR CharTraits. */
#if defined(UNICODE) || defined(_UNICODE)
    typedef RegExCharTraitsW TRegExCharTraits;
#else /* defined(UNICODE) || defined(_UNICODE) */
    typedef RegExCharTraitsA TRegExCharTraits;
#endif /* defined(UNICODE) || defined(_UNICODE) */

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_REGEXCHARTRAITS_H_INCLUDED */

