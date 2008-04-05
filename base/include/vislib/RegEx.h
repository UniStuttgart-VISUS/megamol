/*
 * RegEx.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2007 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_REGEX_H_INCLUDED
#define VISLIB_REGEX_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/assert.h"
#include "vislib/Exception.h"
#include "vislib/PtrArray.h"
#include "vislib/RegExCharTraits.h"
#include "vislib/String.h"
#include "vislib/types.h"


//
// RegEx IS WORK IN PROGRESS AN CANNOT YET BE USED!
//


namespace vislib {

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
        class ParseException : public vislib::Exception {

        public:

            /** Possible types of parse errors. */
            enum ErrorType {
                UNEXPECTED = 0,
                EMPTY_EXPRESSION,
                BRACE_CLOSE_EXPECTED,
                BRACKET_CLOSE_EXPECTED,
                PAREN_CLOSE_EXPECTED,
                INTEGER_EXPECTED,
                UNKNOWN_ESCAPE
            };

            ParseException(const ErrorType errorType, const Char *vicinity,
                const char *file, const int line);

            ParseException(const ParseException& rhs);

            virtual ~ParseException(void);

            inline ErrorType GetErrorType(void) const {
                return this->errorType;
            }

            inline const String<typename T::Super>& GetVicinity(void) const {
                return this->vicinity;
            }

            ParseException& operator =(const ParseException& rhs);

        private:

            /** The type of parse error to signal. */
            ErrorType errorType;

            /** The string in whose vicinity the error occurred. */
            String<typename T::Super> vicinity;
        }; /* end class ParseException */

    protected:

        /**
         * This is the abstract superclass for nodes in the RE automaton. It
         * implements the functionality of a sequence by providing a pointer
         * to the next node.
         */
        class Expression {

        public:

            /** Dtor. */
            virtual ~Expression(void);

            /**
             * Consumes a part of the input string.
             *
             * @param input
             *
             * @return
             */
            virtual const Char *Consume(const Char *input) = 0;

            /**
             * Dump the expression into the specified stream and prepent 
             * 'indent' tabs before each line.
             *
             * This method is intended solely for debugging purposes.
             *
             * @parm stream The stream to dump the expression to.
             */
            virtual void Dump(FILE *stream, const UINT indent = 0) = 0;

            /**
             * Gets the next expression in the RE sequence.
             *
             * @return The next expression in the sequence or NULL if this is
             *         the last node.
             */
            inline const Expression *GetNext(void) const {
                return this->next;
            }

            /**
             * Gets the next expression in the RE sequence.
             *
             * @return The next expression in the sequence or NULL if this is
             *         the last node.
             */
            inline Expression *GetNext(void) {
                return this->next;
            }

            /**
             * Set the next expression in the sequence. 
             *
             * The current next pointer will be overwritten by the operation 
             * and must therefore be NULL. THIS METHOD IS SOME KIND OF UNSAFE, 
             * but this is no major problem as the class is not visible outside.
             *
             * @param next The new next node.
             */
            inline void SetNext(Expression *next) {
                ASSERT((this->next == NULL) || (next == NULL));
                this->next = next;
            }

        protected:

            /**
             * Create a new instance followed by the specified next node.
             *
             * @param next The next node. Defaults to NULL.
             */
            Expression(Expression *next = NULL);

            /** The next expression in the automaton. */
            Expression *next;

        }; /* end class Expression */


        /**
         * This abstract expression is used as base for recursive 
         * expressions that need additional sub-expressions like a choice
         * or the Kleene star.
         */
        class RecursiveExpression : public Expression {

        public:

            /** Dtor. */
            virtual ~RecursiveExpression(void);

            /**
             * Append 'child' to the collection of children. If 'child' is 
             * NULL, nothing happens. 
             *
             * 'child' MUST HAVE BEEN ALLOCATED ON THE HEAP USING new! The 
             * object takes ownership of 'child'.
             *
             * @param child The child to be added. This object must have been
             *              allocated on the heap using new.
             */
            void AppendChild(Expression *child);

            /**
             * Answer the number of child nodes the expression has.
             *
             * @return The number of children.
             */
            inline SIZE_T CountChildren(void) const {
                return this->children.Count();
            }

            /**
             * Dump the expression into the specified stream and prepent 
             * 'indent' tabs before each line.
             *
             * @parm stream The stream to dump the expression to.
             */
            virtual void Dump(FILE *stream, const UINT indent = 0);

            /**
             * Get the 'which'th child node.
             *
             * @param which The index of the child node to retrieve. This must
             *              must be within [0, this->CountChildren()[.
             *
             * @return A pointer to the child node. This pointer is guaranteed
             *         to be non-NULL as long as the collection of children has
             *         not been manipulated with a const_cast or any other dirty
             *         hack. The object remains owner of the returned object.
             *
             * @throws OutOfRangeException If 'which' is not a valid child 
             *                             index.
             */
            inline const Expression *GetChild(const SIZE_T which) const {
                return this->children[which];
            }

            /**
             * Get the 'which'th child node.
             *
             * @param which The index of the child node to retrieve. This must
             *              must be within [0, this->CountChildren()[.
             *
             * @return A pointer to the child node. This pointer is guaranteed
             *         to be non-NULL as long as the collection of children has
             *         not been manipulated with a const_cast or any other dirty
             *         hack. The object remains owner of the returned object.
             *
             * @throws OutOfRangeException If 'which' is not a valid child 
             *                             index.
             */
            inline Expression *GetChild(const SIZE_T which) {
                return this->children[which];
            }

            /**
             * Answer the children collection.
             *
             * You should never const_cast this collection and manipulate it
             * directly.
             *
             * @return A reference to the collection of children.
             */
            inline const PtrArray<Expression>& GetChildren(void) const {
                return this->children;
            }

        protected:

            /**
             * Create a new instance followed by the specified next node.
             *
             * 'firstChild' is added as the first child node. THIS OBJECT MUST
             * HAVE BEEN ALLOCATED ON THE HEAP USING new! The object takes 
             * ownership of 'firstChild'.
             *
             * @param firstChild The first child node. This object must have 
             *                   been allocated on the heap using new. Defaults
             *                   to NULL. The object takes ownership of the
             *                   parameter.
             * @param next       The next node. Defaults to NULL.
             */
            RecursiveExpression(Expression *firstChild = NULL, 
                Expression *next = NULL);

            /** The next expression in the automaton. */
            PtrArray<Expression> children;

        }; /* end class RecursiveExpression */


        /**
         * This expression matches any character.
         */
        class AnyExpression : public Expression {

        public:

            AnyExpression(void);

            virtual ~AnyExpression(void);

            virtual const Char *Consume(const Char *input);

        }; /* class AnyExpression */


        /**
         * Implements the choice (OR operator).
         */
        class ChoiceExpression: public RecursiveExpression {

            /**
             * Create a new choice expression that matches 'expr0' or 'expr1'.
             *
             * 'expr0' and 'expr1' MUST HAVE BEEN ALLOCATED ON THE HEAP USING 
             * new! The object takes ownership of 'expr0' and 'expr1'.
             *
             * 'expr0' and 'expr1' MUST NOT BE NULL!
             *
             * @param expr0 The left-hand-side operand of OR. This must not be
             *              NULL. The object takes ownership of the parameter.
             * @param expr1 The right-hand-side operand of OR. This must not be
             *              NULL. The object takes ownership of the parameter.
             */
            ChoiceExpression(Expression *expr0, Expression *expr1);

            /** Dtor. */
            virtual ~ChoiceExpression(void);

            virtual const Char *Consume(const Char *input);

            /**
             * Dump the expression into the specified stream and prepent 
             * 'indent' tabs before each line.
             *
             * @parm stream The stream to dump the expression to.
             */
            virtual void Dump(FILE *stream, const UINT indent = 0);

            /**
             * Get the left-hand-side expression (child 0).
             *
             * @return The left child.
             */
            inline const Expression *GetLeftChild(void) const {
                ASSERT(this->CountChildren() == 2);
                return this->GetChild(0);
            }

            /**
             * Get the left-hand-side expression (child 0).
             *
             * @return The left child.
             */
            inline Expression *GetLeftChild(void) {
                ASSERT(this->CountChildren() == 2);
                return this->GetChild(0);
            }

            /**
             * Get the right-hand-side expression (child 1).
             *
             * @return The right child.
             */
            inline const Expression *GetRightChild(void) const {
                ASSERT(this->CountChildren() == 2);
                return this->GetChild(1);
            }

            /**
             * Get the right-hand-side expression (child 1).
             *
             * @return The right child.
             */
            inline Expression *GetRightChild(void) {
                ASSERT(this->CountChildren() == 2);
                return this->GetChild(1);
            }

        }; /* end KleeneExpression */


        /**
         * This class implements an automaton for the Kleene star operator, 
         * the plus (at least one) operator and the question mark (zero or one)
         * operator. It can also be used for specific repeat counts in braces.
         */
        class RepeatExpression : public RecursiveExpression {

        public:

            /** Symbolic constant for specifying unbounded repeat. */
            static const int UNBOUNDED;

            /**
             * Create a new repeat expression that matches 'expr'.
             *
             * 'expr' MUST HAVE BEEN ALLOCATED ON THE HEAP USING new! The object
             * takes ownership of 'expr'.
             *
             * 'expr' MUST NOT BE NULL!
             *
             * The default values for 'minOccur' and 'maxOccur' are 0 and 
             * UNBOUNDED, which produces a repeat node for the Kleene star
             * operator.
             *
             * @param expr     The sub-expression to match multiple times with 
             *                 the Kleene star. This must have been allocated 
             *                 using new. This must not be NULL. The object 
             *                 takes ownership of 'expr'.
             * @param minOccur The minimum occurrence of 'expr' for the repeat
             *                 expression to succeed. Defaults to 0.
             * @param maxOccur The maximum occurrence of 'expr' for the repeat
             *                 expression to succeed. Defaults to UNBOUNDED.
             */
            RepeatExpression(Expression *expr, const int minOccur = 0,
                const int maxOccur = UNBOUNDED);

            /** Dtor. */
            virtual ~RepeatExpression(void);

            virtual const Char *Consume(const Char *input);

            /**
             * Dump the expression into the specified stream and prepent 
             * 'indent' tabs before each line.
             *
             * @parm stream The stream to dump the expression to.
             */
            virtual void Dump(FILE *stream, const UINT indent = 0);

            /**
             * Answer the maximum occurrence to be matched.
             *
             * @return The maximum occurence to be matched.
             */
            inline int GetMaxOccur(void) const {
                return this->maxOccur;
            }

            /**
             * Answer the minimum occurrence that must be found.
             *
             * @return The minimum occurence that must be found.
             */
            inline int GetMinOccur(void) const {
                return this->minOccur;
            }

        private:

            /** The maximum occurrence count. Unbounded if negative. */
            int maxOccur;

            /** The minimum occurrence count. Unbounded if negative. */
            int minOccur;

        }; /* end KleeneExpression */


        /**
         * This expression matches a literal string.
         */
        class LiteralExpression : public Expression {

        public:

            LiteralExpression(const Char *lit);

            LiteralExpression(const Char lit);

            inline void Append(const Char *lit) {
                this->lit += lit;
            }

            inline void Append(const Char lit) {
                this->lit += lit;
            }

            inline void Append(const LiteralExpression *expr) {
                this->lit += expr->lit;
            }

            virtual ~LiteralExpression(void);

            virtual const Char *Consume(const Char *input);

            /**
             * Dump the expression into the specified stream and prepent 
             * 'indent' tabs before each line.
             *
             * @parm stream The stream to dump the expression to.
             */
            virtual void Dump(FILE *stream, const UINT indent = 0);

            /** The literal expression to match. */
            vislib::String<T> lit;

        }; /* class LiteralExpression */


//
//        /**
//         *
//         */
//        class MatchContext {
//
//        public:
//
//            MatchContext(void);
//
//            ~MatchContext(void);
//
//            void Clear(void);
//
//            inline SIZE_T Count(void) const {
//                return this->cntMatches;
//            }
//
//            /**
//             *
//             * @throws OutOfRangeException If 'idx' is not within 
//             *                             [0, this->Count()].
//             */
//            void GetMatch(UINT& outBegin, UINT& outEnd, const UINT idx) const;
//
//        private:
//
//            /** 
//             * A single match. A match is defined by the index of its first
//             * character and the index of the character AFTER its last 
//             * character.
//             */
//            typedef struct Match_t {
//
//                Match_t(const UINT begin, const UINT end, Match_t *next = NULL) 
//                    : begin(begin), end(end), next(next) {}
//
//                /** Index of first character of the match. */
//                UINT begin;
//
//                /** Index after last character of the match. */
//                UINT end;
//
//                /** Next match in the list. */
//                struct Match_t *next;
//            } Match;
//
//            /**
//             * Add a new match to the context.
//             *
//             * @param begin Begin of the match.
//             * @param end   End of the match.
//             *
//             * @throws std::bad_alloc If there is not enough memory for storing
//             *                        the additional match.
//             */
//            void add(const UINT begin, const UINT end);
//
//            /** The number of entries in the 'matches' list. */
//            SIZE_T cntMatches;
//
//            /** The list of matches. */
//            Match *matches;
//
//            friend class RegEx;
//        }; /* end class MatchContext */
//

        public:
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
        //bool Match(const Char *str, MatchContext *outMatches = NULL) const;

        /**
         * Parse the regular expression 'expr', and prepare the object for
         * matching strings against this epxression.
         *
         * @param expr A string containing the regular expression to be parsed.
         *
         * @throws ParseException In case 'expr' has a syntax error.
         */
        void Parse(const Char *expr);

    private:

        /** Parse abbreviation. */
        Expression *parseAbbreviation(const Char *& inOutStr);

        /** Parse a wildcard sequence. */
        Expression *parseAny(const Char *& inOutStr);

        /** Parse a escape sequence. */
        Expression *parseEscape(const Char *& inOutStr);

        /** Parse a literal. */
        Expression *parseLiteral(const Char *& inOutStr);

        /** Parse a top-level regular expression. */
        Expression *parseRegEx(const Char *& inOutStr);

        /** Parse repeat between 'minRep' and 'maxRep'. */
        Expression *parseRepeat(const Char *& inOutStr, const int minRep,
            const int maxRep);

        /** Parse a repeat expression in braces. */
        Expression *parseRepeatEx(const Char *& inOutStr);

        /** Parse a plus (non-empty closure). */
        Expression *parsePlus(const Char *& inOutStr);

        /** Parse a option (one or no occurrence). */
        Expression *parseQuestion(const Char *& inOutStr);

        /** Parse a character set. */
        Expression *parseSet(const Char *& inOutStr);

        /** Parse a star (Kleene closure). */
        Expression *parseStar(const Char *& inOutStr);

        /** The root node of the automaton. */
        Expression *start;
    };
//
//
//    /*
//     * vislib::RegEx<T>::MatchContext::MatchContext
//     */
//    template<class T> RegEx<T>::MatchContext::MatchContext(void) 
//            : cntMatches(0), matches(NULL) {
//    }
//
//
//    /*
//     * vislib::RegEx<T>::MatchContext::~MatchContext
//     */
//    template<class T> RegEx<T>::MatchContext::~MatchContext(void) {
//        this->Clear();
//    }
//
//
//    /*
//     * vislib::RegEx<T>::MatchContext::Clear
//     */
//    template<class T> void RegEx<T>::MatchContext::Clear(void) {
//        Match *cursor = this->matches;
//        Match *tmp = NULL;
//
//        while (cursor != NULL) {
//            tmp = cursor;
//            cursor = cursor->next;
//            delete tmp;
//        }
//
//        this->cntMatches = 0;
//        this->matches = NULL;
//    }
//
//
//    /*
//     * vislib::RegEx<T>::MatchContext::GetMatch
//     */
//    template<class T> 
//    void RegEx<T>::MatchContext::GetMatch(UINT& outBegin, UINT& outEnd, 
//            const UINT idx) const {
//        ASSERT(idx >= 0);
//        ASSERT(idx < this->cntMatches);
//
//        Match *cursor = this->matches;
//
//        for (UINT i = 0; (i < this->cntMatches - idx) && (cursor != NULL); i++) {
//            cursor = cursor->next;
//        }
//
//        if (cursor != NULL) {
//            outBegin = cursor->begin;
//            outEnd = cursor->end;
//        } else {
//            throw OutOfRangeException(idx, 0, this->cntMatches, __FILE__, 
//                __LINE__);
//        }
//
//    }
//
//
//    /*
//     * vislib::RegEx<T>::MatchContext::add
//     */
//    template<class T> void RegEx<T>::MatchContext::add(const UINT begin, 
//                                                       const UINT end) {
//        ASSERT(end > begin);
//        this->matches = new Match(begin, end, this->matches);


    /** Template instantiation for ANSI strings. */
    typedef RegEx<RegExCharTraitsA> RegExA;

    /** Template instantiation for wide strings. */
    typedef RegEx<RegExCharTraitsA> RegExW;

    /** Template instantiation for TCHARs. */
    typedef RegEx<TRegExCharTraits> TRegEx;
} /* end namespace vislib */

#include "RegEx.inl"

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_REGEX_H_INCLUDED */
