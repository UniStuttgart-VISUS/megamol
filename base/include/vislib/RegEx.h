/*
 * RegEx.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_REGEX_H_INCLUDED
#define VISLIB_REGEX_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/assert.h"
#include "vislib/memutils.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/String.h"
#include "vislib/Trace.h"
#include "vislib/types.h"


#define REGEX_TRACE_LEVEL (vislib::Trace::LEVEL_INFO + 1000)


namespace vislib {

    class RECharTraitsA : public CharTraitsA {
    
    protected:

        static const char *ABBREVIATIONS[];

        static const char TOKEN_BACKSLASH;
        static const char TOKEN_BRACE_CLOSE;
        static const char TOKEN_BRACE_OPEN;
        static const char TOKEN_BRACKET_CLOSE;
        static const char TOKEN_BRACKET_OPEN;
        static const char TOKEN_CARET;
        static const char TOKEN_COMMA;
        static const char TOKEN_DOLLAR;
        static const char TOKEN_DOT;
        static const char TOKEN_PAREN_CLOSE;
        static const char TOKEN_PAREN_OPEN;
        static const char TOKEN_PLUS;
        static const char TOKEN_QUESTION;
        static const char TOKEN_STAR;
        static const char TOKEN_VERTICAL_BAR;
        

        /* Declare our friends. */
        template<class T> friend class RegEx;

    }; /* end class RECharTraitsA */

    const char *RECharTraitsA::ABBREVIATIONS[] =  {
		"a([a-zA-Z0-9])",                   // Alpha numeric
		"b([ \\t])",		                // Whitespace
		"c([a-zA-Z])",	                    // Characters
		"d([0-9])",		                    // Digits
		"h([0-9a-fA-F])",	                // Hex digit
		"n(\r|(\r?\n))",	                // Newline
		//"q(\"[^\"]*\")|(\'[^\']*\')",	    // Quoted string
		//"w([a-zA-Z]+)",	                    // Simple word
		"z([0-9]+)",		                // Integer
		NULL
    };

    const char RECharTraitsA::TOKEN_BACKSLASH = '\\';
    const char RECharTraitsA::TOKEN_BRACE_CLOSE = '}';
    const char RECharTraitsA::TOKEN_BRACE_OPEN = '{';
    const char RECharTraitsA::TOKEN_BRACKET_CLOSE = ']';
    const char RECharTraitsA::TOKEN_BRACKET_OPEN = '[';
    const char RECharTraitsA::TOKEN_CARET = '^';
    const char RECharTraitsA::TOKEN_COMMA = ',';
    const char RECharTraitsA::TOKEN_DOLLAR = '$';
    const char RECharTraitsA::TOKEN_DOT = '.';
    const char RECharTraitsA::TOKEN_PAREN_CLOSE = ')';
    const char RECharTraitsA::TOKEN_PAREN_OPEN = '(';
    const char RECharTraitsA::TOKEN_PLUS = '+';
    const char RECharTraitsA::TOKEN_QUESTION = '?';
    const char RECharTraitsA::TOKEN_STAR = '*';
    const char RECharTraitsA::TOKEN_VERTICAL_BAR = '|'; 



    /**
     * This class implements a regular expression. It must be instantiated with
     * one of the RECharTrait classes.
     */
    template<class T> class RegEx {

    public:

        /** Define a local name for the character type. */
        typedef typename T::Char Char;

        /** Define a local name for the string size. */
        typedef typename T::Size Size;

        /**
         *
         */
        class MatchContext {

        public:

            MatchContext(void);

            ~MatchContext(void);

            void Clear(void);

            inline SIZE_T Count(void) const {
                return this->cntMatches;
            }

            /**
             *
             * @throws OutOfRangeException If 'idx' is not within 
             *                             [0, this->Count()].
             */
            void GetMatch(UINT& outBegin, UINT& outEnd, const UINT idx) const;

        private:

            /** 
             * A single match. A match is defined by the index of its first
             * character and the index of the character AFTER its last 
             * character.
             */
            typedef struct Match_t {

                Match_t(const UINT begin, const UINT end, Match_t *next = NULL) 
                    : begin(begin), end(end), next(next) {}

                /** Index of first character of the match. */
                UINT begin;

                /** Index after last character of the match. */
                UINT end;

                /** Next match in the list. */
                struct Match_t *next;
            } Match;

            /**
             * Add a new match to the context.
             *
             * @param begin Begin of the match.
             * @param end   End of the match.
             *
             * @throws std::bad_alloc If there is not enough memory for storing
             *                        the additional match.
             */
            void add(const UINT begin, const UINT end);

            /** The number of entries in the 'matches' list. */
            SIZE_T cntMatches;

            /** The list of matches. */
            Match *matches;

            friend class RegEx;
        }; /* end class MatchContext */


        /** Possible types of parse errors. */
        enum ParseErrorType {
            PARSE_ERROR_UNEXPECTED = 0,
            PARSE_ERROR_EMPTY_EXPRESSION,
            PARSE_ERROR_BRACE_CLOSE_EXPECTED,
            PARSE_ERROR_BRACKET_CLOSE_EXPECTED,
            PARSE_ERROR_PAREN_CLOSE_EXPECTED,
            PARSE_ERROR_INTEGER_EXPECTED,
            PARSE_ERROR_UNKNOWN_ESCAPE
        };

        /** This structure is used to report parse errors. */
        typedef struct ParseError_t {
            const Char *position;           // The erroneous substring.
            ParseErrorType type;            // The type of error.
        } ParseError;

        /** Ctor. */
        RegEx(void);

        /** Dtor. */
        ~RegEx(void);

        /**
         * Match 'str' against this regular expression. The regular epxression
         * must have been compiled using Parse() before.
         *
         * @param str        The string to be matched.
         * @param outMatches If not NULL, all match groups in the expression 
         *                   will be stored into this match context.
         *
         * @return true, if 'str' matches the regular expression, false 
         *         otherwise.
         */
        bool Match(const Char *str, MatchContext *outMatches = NULL) const;

        /**
         * Parse the regular expression 'expr', and prepare the object for
         * matching strings against this epxression.
         *
         * @param expr A string containing the regular expression to be parsed.
         *
         * @return PARSE_OK in case of success, other values for indicating an
         *         error.
         */
        bool Parse(const Char *expr, ParseError *outError = NULL);

    private:

        /** Parse abbreviation. */
        bool parseAbbreviation(Char *& inOutStr);

        /** Parse a wildcard sequence. */
        bool parseAny(Char *& inOutStr);

        /** Parse a escape sequence. */
        bool parseEscape(Char *& inOutStr);

        /** Parse a literal. */
        bool parseLiteral(Char *& inOutStr);
        
        /** Parse a top-level regular expression. */
        bool parseRegEx(Char *& inOutStr);

        /** Parse repeat between 'minRep' and 'maxRep'. */
        bool parseRepeat(Char *& inOutStr, const UINT minRep, 
            const UINT maxRep);

        /** Parse a repeat expression in braces. */
        bool parseRepeatEx(Char *& inOutStr);
        
        /** Parse a plus (non-empty closure). */
        bool parsePlus(Char *& inOutStr);

        /** Parse a option (one or no occurrence). */
        bool parseQuestion(Char *& inOutStr);

        /** Parse a character set. */
        bool parseSet(Char *& inOutStr);

        /** Parse a star (Kleene closure). */
        bool parseStar(Char *& inOutStr);

        /**
         * Report an error to the ParseError structure designated by 
         * 'this->error', if this pointer is not NULL.
         *
         * @param type     The type of error that occurred.
         * @param position Pointer to the substring where the erroneous part
         *                 begins.
         */
        void raiseError(const ParseErrorType type, const Char *position);

        /** Provides access to the error for the raiseError method. */
        ParseError *error;
    };


    /*
     * vislib::RegEx<T>::MatchContext::MatchContext
     */
    template<class T> RegEx<T>::MatchContext::MatchContext(void) 
            : cntMatches(0), matches(NULL) {
    }


    /*
     * vislib::RegEx<T>::MatchContext::~MatchContext
     */
    template<class T> RegEx<T>::MatchContext::~MatchContext(void) {
        this->Clear();
    }


    /*
     * vislib::RegEx<T>::MatchContext::Clear
     */
    template<class T> void RegEx<T>::MatchContext::Clear(void) {
        Match *cursor = this->matches;
        Match *tmp = NULL;

        while (cursor != NULL) {
            tmp = cursor;
            cursor = cursor->next;
            delete tmp;
        }

        this->cntMatches = 0;
        this->matches = NULL;
    }


    /*
     * vislib::RegEx<T>::MatchContext::GetMatch
     */
    template<class T> 
    void RegEx<T>::MatchContext::GetMatch(UINT& outBegin, UINT& outEnd, 
            const UINT idx) const {
        ASSERT(idx >= 0);
        ASSERT(idx < this->cntMatches);

        Match *cursor = this->matches;

        for (UINT i = 0; (i < this->cntMatches - idx) && (cursor != NULL); i++) {
            cursor = cursor->next;
        }

        if (cursor != NULL) {
            outBegin = cursor->begin;
            outEnd = cursor->end;
        } else {
            throw OutOfRangeException(idx, 0, this->cntMatches, __FILE__, 
                __LINE__);
        }

    }


    /*
     * vislib::RegEx<T>::MatchContext::add
     */
    template<class T> void RegEx<T>::MatchContext::add(const UINT begin, 
                                                       const UINT end) {
        ASSERT(end > begin);
        this->matches = new Match(begin, end, this->matches);
    }


    /*
     * vislib::RegEx<T>::~RegEx
     */
    template<class T> RegEx<T>::RegEx(void) : error(NULL) {
    }


    /*
     * vislib::RegEx<T>::~RegEx
     */
    template<class T> RegEx<T>::~RegEx(void) {
    }


    /*
     * vislib::RegEx<T>::Match
     */
    template<class T> 
    bool RegEx<T>::Match(const Char *str, MatchContext *outMatches) const {
        // TODO: Implementation missing.
        return false;
    }


    /*
     * vislib::RegEx<T>::Parse
     */
    template<class T> 
    bool RegEx<T>::Parse(const Char *expr, ParseError *outError) {
        const Char *s = expr;
        bool retval = true;
        this->error = outError;

        /* Disallow empty expressions and NULL pointers. */
        if ((s == NULL) || (s[0] == 0)) {
            this->raiseError(PARSE_ERROR_EMPTY_EXPRESSION, s);
            return false;
        }

        /* Process the input. */
        while ((s != 0) && (retval = this->parseRegEx(s)));

        return retval;
    }


    /*
     * vislib::RegEx<T>::parseAbbreviation
     */
    template<class T> bool RegEx<T>::parseAbbreviation(Char *& inOutStr) {
        ASSERT(inOutStr != NULL);
        ASSERT(*inOutStr == T::TOKEN_BACKSLASH);
        TRACE(REGEX_TRACE_LEVEL, "RegEx<T>::parseAbbreviation\n");

        const Char **current = T::ABBREVIATIONS;

        inOutStr++;                         // Consume backslash.

        while (*current != NULL) {
            if (*inOutStr == **current) {
                inOutStr++;                 
                // TODO: Call appropriate method and return result.
                return false;
            } else {
                current++;
            }
        }

        this->raiseError(PARSE_ERROR_UNKNOWN_ESCAPE, inOutStr);
        return false;
    }


    /*
     * vislib::RegEx<T>::parseAny
     */
    template<class T> bool RegEx<T>::parseAny(Char *& inOutStr) {
        // TODO: Implementation missing.
        return false;
    }

    /*
     * vislib::RegEx<T>::parseEscape
     */
    template<class T> bool RegEx<T>::parseEscape(Char *& inOutStr) {
        ASSERT(inOutStr != NULL);
        ASSERT(*inOutStr == T::TOKEN_BACKSLASH);
        TRACE(REGEX_TRACE_LEVEL, "RegEx<T>::parseEscape\n");

        /* Check for abbreviations first. */
        if (this->parseAbbreviation(inOutStr)) {
            return true;
        }


        // TODO: Implementation missing.
        return false;
    }


    /*
     * vislib::RegEx<T>::parseLiteral
     */
    template<class T> bool RegEx<T>::parseLiteral(Char *& inOutStr) {
        // TODO: Implementation missing.
        return false;
    }

    
    /*
     * vislib::RegEx<T>::parseRegEx
     */
    template<class T> bool RegEx<T>::parseRegEx(Char *& inOutStr) {
        ASSERT(inOutStr != 0);
        
        switch (*inOutStr) {

            case T::TOKEN_CARET:        // Match begin of line.
                // push begin
                break;

            case T::TOKEN_DOLLAR:       // Match end of line.
                // push end
                break;

            case T::TOKEN_BRACE_OPEN:   // Begin of repeat expression.
                return this->parseRepeatEx(inOutStr);

            case T::TOKEN_STAR:         // Kleene closure.
                return this->parseRepeat(inOutStr, 0, -1);  // TODO

            case T::TOKEN_PLUS:         // Non-empty closure.
                return this->parseRepeat(inOutStr, 1, -1); // TODO

            case T::TOKEN_QUESTION:     // Optional sequence.
                return this->parseRepeat(inOutStr, 0, 1);

            case T::TOKEN_BACKSLASH:    // Escaped text.
                return this->parseEscape(inOutStr);

            case T::TOKEN_DOT:          // Wildcard character.
                return this->parseAny(inOutStr);

            case T::TOKEN_BRACKET_OPEN: // Begin of character set.
                return this->parseSet(inOutStr);

            case T::TOKEN_PAREN_OPEN:   // Begin of grouping.
                // Push match group
                break;

            default:                    // Must be normal literal.
                return this->parseLiteral(inOutStr);
        }
    }


    /*
     * vislib::RegEx<T>::parseRepeat
     */
    template<class T> 
    bool RegEx<T>::parseRepeat(Char *& inOutStr, const UINT minRep, const UINT maxRep) {
        // TODO: Implementation missing.
        return false;
    }

    
    /*
     * vislib::RegEx<T>::parseRepeatEx
     */
    template<class T> bool RegEx<T>::parseRepeatEx(Char *& inOutStr) {
        ASSERT(inOutStr != NULL);
        ASSERT(*inOutStr == T::TOKEN_BRACE_OPEN);
        TRACE(REGEX_TRACE_LEVEL, "RegEx<T>::parseRepeatEx\n");

        Char *begin = NULL;                 // Begin of a number to parse. 
        UINT minRep = 0;                    // Minimum repeat value.
        UINT maxRep = 0;                    // Maximum repeat value.
        
        inOutStr++;                         // Consume opening brace.
    
        /* Recognise and convert first number. */
        while (T::IsDigit(*inOutStr)) {
            minRep *= 10;
            minRep += static_cast<UINT>(*inOutStr);
            inOutStr++;
        }

        /* Check whether at least one digit was consumed. */
        if (inOutStr == begin) {
            this->raiseError(PARSE_ERROR_INTEGER_EXPECTED, begin);
            return false;
        }

        /* Check for end of expression or begin of maximum repeat. */
        switch (*inOutStr) {

            case T::TOKEN_COMMA:
                inOutStr++;                 // Consume comma and proceed.
                break;

            case T::TOKEN_BRACE_CLOSE:
                inOutStr++;                 // Consume brace and return.
                maxRep = minRep;
                return this->parseRepeat(inOutStr, minRep, maxRep);

            default:
                this->raiseError(PARSE_ERROR_BRACE_CLOSE_EXPECTED, inOutStr);
                return false;
        }

        /* Recognise and convert first number. */
        begin = inOutStr;
        while (T::IsDigit(*inOutStr)) {
            maxRep *= 10;
            maxRep += static_cast<UINT>(*inOutStr);
            inOutStr++;
        }

        /* Check whether at least one digit was consumed. */
        if (inOutStr == begin) {
            this->raiseError(PARSE_ERROR_INTEGER_EXPECTED, begin);
            return false;
        }
        
        /* Check for closing brace. */
        if (inOutStr == T::TOKEN_BRACE_CLOSE) {
            inOutStr++;
            return this->parseRepeat(inOutStr, minRep, maxRep);

        } else {
            this->raiseError(PARSE_ERROR_BRACE_CLOSE_EXPECTED, inOutStr);
            return false;
        }
    }


    /*
     * vislib::RegEx<T>::parseSet
     */
    template<class T> bool RegEx<T>::parseSet(Char *& inOutStr) {
        ASSERT(inOutStr != NULL);
        ASSERT(*inOutStr == T::TOKEN_BRACKET_OPEN);

        inOutStr++;

        while ((inOutStr != 0) && (inOutStr != T::TOKEN_BRACKET_CLOSE)) {
            // TODO: Implementation missing.
        }

        if (inOutStr == NULL) {
            this->raiseError(PARSE_ERROR_BRACKET_CLOSE_EXPECTED, inOutStr - 1);
            return false;
        } else {
            return true;
        }
    }


    /*
     * vislib::RegEx<T>::parseStar
     */
    template<class T> bool RegEx<T>::parseStar(Char *& inOutStr) {
        // TODO: Implementation missing.
        return false;
    }
    

    /*
     * vislib::RegEx<T>::raiseError
     */
    template<class T>
    void RegEx<T>::raiseError(const ParseErrorType type, const Char *position) {
        if (this->error != NULL) {
            this->error->type = type;
            this->error->position = position;
        }
    }

    /** Template instantiation for ANSI strings. */
    typedef RegEx<RECharTraitsA> RegExA;

    ///** Template instantiation for wide strings. */
    //typedef RegEx<CharTraitsW> StringW;

    ///** Template instantiation for TCHARs. */
    //typedef String<TCharTraits> TString;
} /* end namespace vislib */

#endif /* VISLIB_REGEX_H_INCLUDED */
