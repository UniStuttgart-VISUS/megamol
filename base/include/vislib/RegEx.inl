/*
 * RegEx.inl
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "RegEx.h"

#include "vislib/memutils.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/StringConverter.h"
#include "vislib/Trace.h"

#include "vislib/MissingImplementationException.h"


#define VISLIB_REGEX_TRACE_LEVEL (vislib::Trace::LEVEL_VL_INFO + 1000)


////////////////////////////////////////////////////////////////////////////////
// Nested class ParseException

/*
 * vislib::RegEx::ParseException::ParseException
 */
template<class T>
vislib::RegEx<T>::ParseException::ParseException(const ErrorType errorType,
        const Char *vicinity, const char *file, const int line) 
        : Exception(__FILE__, __LINE__), errorType(errorType), 
        vicinity(vicinity) {
    StringA msg;
    StringA n(this->vicinity);

    switch (this->errorType) {
        case EMPTY_EXPRESSION:
            msg.Format("Empty expression at \"%s\" was not expected.", 
                n.PeekBuffer());
            break;

        case BRACE_CLOSE_EXPECTED:
            msg.Format("Closing brace expected at \"%s\".", n.PeekBuffer());
            break;

        case BRACKET_CLOSE_EXPECTED:
            msg.Format("Closing bracket expected at \"%s\".", n.PeekBuffer());
            break;

        case PAREN_CLOSE_EXPECTED:
            msg.Format("Closing parenthesis expected at \"%s\".", 
                n.PeekBuffer());
            break;

        case INTEGER_EXPECTED:
            msg.Format("Integer literal expected at \"%s\".", n.PeekBuffer());
            break;

        case UNKNOWN_ESCAPE:
            msg.Format("Unknown escape sequence found at \"%s\".", 
                n.PeekBuffer());
            break;

        default:
            msg.Format("Unexpected parse error at \"%s\".", n.PeekBuffer());
            break;
    }

    this->setMsg(msg.PeekBuffer());
}


/*
 * vislib::RegEx<T>::ParseException::ParseException
 */
template<class T>
vislib::RegEx<T>::ParseException::ParseException(const ParseException& rhs) 
        : Exception(rhs), errorType(rhs.errorType), vicinity(rhs.vicinity) {
    // Nothing else to do.
}


/*
 * vislib::RegEx<T>::ParseException::~ParseException
 */
template<class T> vislib::RegEx<T>::ParseException::~ParseException(void) {
    // Nothing to do.
}


/*
 * vislib::RegEx<T>::ParseException::operator =
 */
template<class T> typename vislib::RegEx<T>::ParseException& 
vislib::RegEx<T>::ParseException::operator =(const ParseException& rhs) {
    if (this != &rhs) {
        Exception::operator =(rhs);
        this->errorType = rhs.errorType;
        this->vicinity = rhs.vicinity;
    }

    return *this;
}

// End class ParseException
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Nested class Expression

/*
 * vislib::RegEx<T>::Expression::~Expression
 */
template<class T> vislib::RegEx<T>::Expression::~Expression(void) {
    SAFE_DELETE(this->next);
}


/*
 * vislib::RegEx<T>::Expression::Expression
 */
template<class T> vislib::RegEx<T>::Expression::Expression(Expression *next) 
        : next(next) {
    // Nothing else to do.
}

// End class Expression
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Nested class RecursiveExpression

/*
 * vislib::RegEx<T>::RecursiveExpression::~RecursiveExpression
 */
template<class T>
vislib::RegEx<T>::RecursiveExpression::~RecursiveExpression(void) {
    // Nothing to do.
}


/*
 * vislib::RegEx<T>::RecursiveExpression::AppendChild
 */
template<class T> 
void vislib::RegEx<T>::RecursiveExpression::AppendChild(Expression *child) {
    if (child != NULL) {
        this->children.Add(child);
    }
}


/*
 * vislib::RegEx<T>::RecursiveExpression::Dump
 */
template<class T> void vislib::RegEx<T>::RecursiveExpression::Dump(
        FILE *stream, const UINT indent) {
    ASSERT(stream != NULL);

    for (SIZE_T i = 0; i < this->children.Count(); i++) {
        this->children[i]->Dump(stream, indent + 1);
    }
}


/*
 * vislib::RegEx<T>::RecursiveExpression::RecursiveExpression
 */
template<class T> 
vislib::RegEx<T>::RecursiveExpression::RecursiveExpression(
        Expression *firstChild, Expression *next) : Expression(next) {
    this->AppendChild(firstChild);
}

// End class Expression
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Nested class AnyExpression

/*
 * vislib::RegEx<T>::AnyExpression::AnyExpression
 */
template<class T>
vislib::RegEx<T>::AnyExpression::AnyExpression(void) : Expression() {
    // Nothing else to do.
}


/*
 * vislib::RegEx<T>::AnyExpression::~AnyExpression
 */
template<class T>
vislib::RegEx<T>::AnyExpression::~AnyExpression(void) {
    // Nothing to do.
}


/*
 * vislib::RegEx<T>::AnyExpression::Consume
 */
template<class T> typename const vislib::RegEx<T>::Char *
vislib::RegEx<T>::AnyExpression::Consume(const Char *input) {
    TRACE(VISLIB_REGEX_TRACE_LEVEL, "Consuming AnyExpression on \"%s\" ...\n", 
        StringA(input).PeekBuffer());
    return (input + 1);
}

// End class AnyExpression
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Nested class ChoiceExpression

/*
 * vislib::RegEx<T>::ChoiceExpression::ChoiceExpression
 */
template<class T> vislib::RegEx<T>::ChoiceExpression::ChoiceExpression(
        Expression *expr0, Expression *expr1) : RecursiveExpression() {
    ASSERT(this->CountChilren() == 0);
    ASSERT(expr0 != NULL);
    ASSERT(expr1 != NULL);

    this->AppendChild(expr0);
    this->AppendChild(expr1);
    this->children.Trim();
}


/*
 * vislib::RegEx<T>::ChoiceExpression::~ChoiceExpression
 */
template<class T> vislib::RegEx<T>::ChoiceExpression::~ChoiceExpression(void) {
    // Nothing to do.
}


/*
 * vislib::RegEx<T>::ChoiceExpression::Consume
 */
template<class T> typename const vislib::RegEx<T>::Char *
vislib::RegEx<T>::ChoiceExpression::Consume(const Char *input) {
    ASSERT(this->CountChildren() == 2);
    const Char *retval = NULL;

    TRACE(VISLIB_REGEX_TRACE_LEVEL, "Consuming ChoiceExpression on "
        "\"%s\" ...\n", StringA(input).PeekBuffer());

    if ((retval = this->GetLeftChild()->Consume(input)) == NULL) {
        retval = this->GetRightChild()->Consume(input);
    }

    return retval;
}


/*
 * vislib::RegEx<T>::ChoiceExpression::Dump
 */
template<class T> 
void vislib::RegEx<T>::ChoiceExpression::Dump(FILE *stream, const UINT indent) {
    ASSERT(stream != NULL);

    ::fprintf(stream, "%ChoiceExpression (OR):\n",
        StringA('\t', indent).PeekBuffer());
    RecursiveExpression::Dump(stream, indent);
}

// End class ChoiceExpression
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Nested class RepeatExpression


/*
 * vislib::RegEx<T>::RepeatExpression::UNBOUNDED
 */
template<class T> const int vislib::RegEx<T>::RepeatExpression::UNBOUNDED = -1;


/*
 * vislib::RegEx<T>::RepeatExpression::RepeatExpression
 */
template<class T>
vislib::RegEx<T>::RepeatExpression::RepeatExpression(Expression *expr, 
            const int minOccur, const int maxOccur)
        : RecursiveExpression(expr), maxOccur(maxOccur), minOccur(minOccur) {
    ASSERT(this->CountChilren() == 1);
    ASSERT(this->chilren[0] != NULL);

    this->children.Trim();  // A RepeatExpression has exactly one child.

    /* Sanity checks. */
    if (this->minOccur < 0) {
        TRACE(VISLIB_REGEX_TRACE_LEVEL, "Insane 'minOccur' (%d) fixed.\n", 
            this->minOccur);
        this->maxOccur = 0;
    }
    if ((this->maxOccur >= 0) && (this->maxOccur < this->minOccur)) {
        TRACE(VISLIB_REGEX_TRACE_LEVEL, "Insane 'maxOccur' (%d) fixed.\n",
            this->maxOccur);
        this->maxOccur = this->minOccur;
    }
}


/*
 * vislib::RegEx<T>::RepeatExpression::~RepeatExpression
 */
template<class T>
vislib::RegEx<T>::RepeatExpression::~RepeatExpression(void) {
    // Nothing to do.
}


/*
 * vislib::RegEx<T>::RepeatExpression::Consume
 */
template<class T> typename const vislib::RegEx<T>::Char *
vislib::RegEx<T>::RepeatExpression::Consume(const Char *input) {
    ASSERT(this->CountChildren() == 1);
    int cntMatches = 0;         // The number of subsequent matches found.
    const Char *in = input;     // Input for next recursive/sequence expression.
    const Char *next = NULL;    // Result of recursive call (== next input).

    TRACE(VISLIB_REGEX_TRACE_LEVEL, "Consuming RepeatExpression on "
        "\"%s\" ...\n", StringA(input).PeekBuffer());

    /* Match as long as possible. */
    ASSERT(cntMatches == 0);
    while ((next = this->GetChild(0)->Consume(in)) != NULL) {
        cntMatches++;
        in = next;
    }
    TRACE(VISLIB_REGEX_TRACE_LEVEL, "RepeatExpression matched %d times.\n",
        cntMatches);

    /* Signal failure if match count was not matched. */
    if ((cntMatches < this->minOccur) || (cntMatches > this->maxOccur)) {
        // TODO: Ich bin mir nicht sicher ob das bzgl. maxOccur die richtige Semantik implementiert.
        TRACE(VISLIB_REGEX_TRACE_LEVEL, "Repeat count is not within "
            "[%d, %d].\n", this->minOccur, this->maxOccur);
        in = NULL;
    }

    return in;
}


/*
 * vislib::RegEx<T>::RepeatExpression::Dump
 */
template<class T> 
void vislib::RegEx<T>::RepeatExpression::Dump(FILE *stream, const UINT indent) {
    ASSERT(stream != NULL);

    ::fprintf(stream, "%sRepeatExpression: [%d, %d]\n",
        StringA('\t', indent).PeekBuffer(), this->minOccur, this->maxOccur);
    RecursiveExpression::Dump(stream, indent);
}


// End class RepeatExpression
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Nested class LiteralExpression


/*
 * vislib::RegEx<T>::LiteralExpression::LiteralExpression
 */
template<class T>
vislib::RegEx<T>::LiteralExpression::LiteralExpression(const Char *lit)
        : Expression(), lit(lit) {
    // Nothing else to do.
}


/*
 * vislib::RegEx<T>::LiteralExpression::LiteralExpression
 */
template<class T>
vislib::RegEx<T>::LiteralExpression::LiteralExpression(const Char lit) 
        : Expression(), lit(lit, 1) {
    // Nothing else to do.
}


/*
 * vislib::RegEx<T>::LiteralExpression::~LiteralExpression
 */
template<class T>
vislib::RegEx<T>::LiteralExpression::~LiteralExpression(void) {
    // Nothing to do.
}


/*
 * vislib::RegEx<T>::LiteralExpression::Consume
 */
template<class T> typename const vislib::RegEx<T>::Char *
vislib::RegEx<T>::LiteralExpression::Consume(const Char *input) {
    const Char *litChar = this->lit.PeekBuffer();
    Size litLen = this->lit.Length();

    TRACE(VISLIB_REGEX_TRACE_LEVEL, "Consuming LiteralExpression \"%s\" on "
        "\"%s\" ...\n", StringA(this->lit).PeekBuffer(), 
        StringA(input).PeekBuffer());

    Size i = 0;
    if (input != NULL) {
        while ((i < litLen) && (input[i] != 0) && (litChar[i] == input[i])) {
            i++;
        }
    }

    return ((i == litLen) ? input + (i - 1) : NULL);
}


/*
 * vislib::RegEx<T>::LiteralExpression::Dump
 */
template<class T> void vislib::RegEx<T>::LiteralExpression::Dump(FILE *stream,
        const UINT indent) {
    ASSERT(stream != NULL);

    ::fprintf(stream, "%sLiteralExpression: \"%s\"\n",
        StringA('\t', indent).PeekBuffer(),
        StringA(this->lit).PeekBuffer());
}

// End class LiteralExpression
////////////////////////////////////////////////////////////////////////////////

/*
 * vislib::RegEx<T>::~RegEx
 */
template<class T> vislib::RegEx<T>::RegEx(void) : start(NULL) {
}


/*
 * vislib::RegEx<T>::~RegEx
 */
template<class T> vislib::RegEx<T>::~RegEx(void) {
    SAFE_DELETE(this->start);
}

//
///*
// * vislib::RegEx<T>::Match
// */
//template<class T> 
//bool vislib::RegEx<T>::Match(const Char *str, MatchContext *outMatches) const {
//    // TODO: Implementation missing.
//    return false;
//}


/*
 * vislib::RegEx<T>::Parse
 */
template<class T> void vislib::RegEx<T>::Parse(const Char *expr) {
    const Char *s = expr;

    /* Delete possible old expression. */
    SAFE_DELETE(this->start);

    /* Disallow empty expressions and NULL pointers. */
    if ((s == NULL) || (s[0] == 0)) {
        throw ParseException(ParseException::EMPTY_EXPRESSION, expr, __FILE__,
            __LINE__);
    }

    /* Process the input. */
    while (*s != 0) {
        this->start = this->parseRegEx(s);
    }

    if (this->start != NULL) {
        this->start->Dump(stdout);
    }
}


/*
 * vislib::RegEx<T>::parseAbbreviation
 */
template<class T>
typename vislib::RegEx<T>::Expression *vislib::RegEx<T>::parseAbbreviation(
        const Char *& inOutStr) {
    ASSERT(inOutStr != NULL);
    ASSERT(*inOutStr == T::TOKEN_BACKSLASH);
    TRACE(VISLIB_REGEX_TRACE_LEVEL, "RegEx<T>::parseAbbreviation\n");

    /* Consume backslash, we are optimistic. */
    inOutStr++;

    for (int i = 0; T::ABBREVIATIONS[i] != NULL; i++) {
        if (T::ABBREVIATIONS[i][0] == *inOutStr) {
            const Char *abbr = T::ABBREVIATIONS[i] + 1;
            TRACE(VISLIB_REGEX_TRACE_LEVEL, "Replace abbreviation '\\%s' "
                "with expression \"%s\".\n", StringA(inOutStr, 1).PeekBuffer(),
                StringA(abbr).PeekBuffer());
            inOutStr++;
            return this->parseRegEx(abbr);
        }
    }
    /* Nothing found here, that is an error. */

    inOutStr--;
    throw ParseException(ParseException::UNKNOWN_ESCAPE, inOutStr,
        __FILE__, __LINE__);
}


/*
 * vislib::RegEx<T>::parseAny
 */
template<class T> 
typename vislib::RegEx<T>::Expression *vislib::RegEx<T>::parseAny(
        const Char *& inOutStr) {
    throw MissingImplementationException("RegEx<T>::parseAny", 
        __FILE__, __LINE__);
}


/*
 * vislib::RegEx<T>::parseEscape
 */
template<class T>
typename vislib::RegEx<T>::Expression *vislib::RegEx<T>::parseEscape(
        const Char *& inOutStr) {
    ASSERT(inOutStr != NULL);
    ASSERT(*inOutStr == T::TOKEN_BACKSLASH);
    TRACE(VISLIB_REGEX_TRACE_LEVEL, "RegEx<T>::parseEscape\n");

    /* Check for abbreviation first. */
    try {
        return this->parseAbbreviation(inOutStr);
    } catch (ParseException) {
        TRACE(VISLIB_REGEX_TRACE_LEVEL, "Parsing escape sequence '%s' as "
            "abbreviation failed.\n", StringA(inOutStr, 2).PeekBuffer());
    }

    /* Consume backslash as erroneous abbreviation parse should have reset. */
    ASSERT(*inOutStr == T::TOKEN_BACKSLASH);
    inOutStr++;

    Size cntEscaped = T::SafeStringLength(T::ESCAPED_CHARS);
    for (Size i = 0; i < cntEscaped; i++) {
        if (T::ESCAPED_CHARS[i] == *inOutStr) {
            TRACE(VISLIB_REGEX_TRACE_LEVEL, "Insert literal for escape "
                "sequence '\\%s'.\n", StringA(inOutStr, 1).PeekBuffer());
            LiteralExpression *retval = new LiteralExpression(*inOutStr);
            inOutStr++;
            return retval;
        }
    }
    /* Nothing found here, that is an error. */

    inOutStr--;
    throw ParseException(ParseException::UNKNOWN_ESCAPE, inOutStr,
        __FILE__, __LINE__);
}


/*
 * vislib::RegEx<T>::parseLiteral
 */
template<class T> 
typename vislib::RegEx<T>::Expression *vislib::RegEx<T>::parseLiteral(
        const Char *& inOutStr) {
    ASSERT(inOutStr != NULL);

    throw MissingImplementationException("RegEx<T>::parseLiteral", 
        __FILE__, __LINE__);
}


/*
 * vislib::RegEx<T>::parseRegEx
 */
template<class T> 
typename vislib::RegEx<T>::Expression *vislib::RegEx<T>::parseRegEx(
        const Char *& inOutStr) {
    ASSERT(inOutStr != 0);
    
    switch (*inOutStr) {

        case T::TOKEN_CARET:        // Match begin of line.
            // push begin
            throw MissingImplementationException("RegEx<T>::parseRegEx",
                __FILE__, __LINE__);
            break;

        case T::TOKEN_DOLLAR:       // Match end of line.
            // push end
            throw MissingImplementationException("RegEx<T>::parseRegEx",
                __FILE__, __LINE__);
            break;

        case T::TOKEN_BRACE_OPEN:   // Begin of repeat expression.
            return this->parseRepeatEx(inOutStr);

        case T::TOKEN_STAR:         // Kleene closure.
            return this->parseRepeat(inOutStr, 0, RepeatExpression::UNBOUNDED);  // TODO

        case T::TOKEN_PLUS:         // Non-empty closure.
            return this->parseRepeat(inOutStr, 1, RepeatExpression::UNBOUNDED); // TODO

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
            throw MissingImplementationException("RegEx<T>::parseRegEx",
                __FILE__, __LINE__);
            break;

        default:                    // Must be normal literal.
            return this->parseLiteral(inOutStr);
    }
}


/*
 * vislib::RegEx<T>::parseRepeat
 */
template<class T> 
typename vislib::RegEx<T>::Expression *vislib::RegEx<T>::parseRepeat(
        const Char *& inOutStr, const int minRep, const int maxRep) {
    ASSERT(inOutStr != NULL);
    throw MissingImplementationException("RegEx<T>::parseRepeat",
        __FILE__, __LINE__);
    return false;
}


/*
 * vislib::RegEx<T>::parseRepeatEx
 */
template<class T>
typename vislib::RegEx<T>::Expression *vislib::RegEx<T>::parseRepeatEx(
        const Char *& inOutStr) {
    ASSERT(inOutStr != NULL);
    ASSERT(*inOutStr == T::TOKEN_BRACE_OPEN);
    TRACE(VISLIB_REGEX_TRACE_LEVEL, "RegEx<T>::parseRepeatEx\n");

    const Char *begin = NULL;           // Begin of a number to parse. 
    int minRep = 0;                     // Minimum repeat value.
    int maxRep = 0;                     // Maximum repeat value.

    inOutStr++;                         // Consume opening brace.

    /* Recognise and convert first number. */
    begin = inOutStr;
    while (T::IsDigit(*inOutStr)) {
        inOutStr++;
    }
    try {
        minRep = T::ParseInt(String<T>(begin, inOutStr - begin));
        ASSERT(inOutStr != begin);
    } catch (...) {
        throw ParseException(ParseException::INTEGER_EXPECTED, begin, __FILE__,
            __LINE__);
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
            throw ParseException(ParseException::BRACE_CLOSE_EXPECTED, inOutStr,
                __FILE__, __LINE__);
    }

    /* Recognise and convert first number. */
    begin = inOutStr;
    while (T::IsDigit(*inOutStr)) {
        inOutStr++;
    }
    try {
        maxRep = T::ParseInt(String<T>(begin, inOutStr - begin));
        ASSERT(inOutStr != begin);
    } catch (...) {
        throw ParseException(ParseException::INTEGER_EXPECTED, begin, __FILE__,
            __LINE__);
    }
    
    /* Check for closing brace. */
    if (*inOutStr == T::TOKEN_BRACE_CLOSE) {
        inOutStr++;
        return this->parseRepeat(inOutStr, minRep, maxRep);

    } else {
        throw ParseException(ParseException::BRACE_CLOSE_EXPECTED, inOutStr,
            __FILE__, __LINE__);
    }
}


/*
 * vislib::RegEx<T>::parseSet
 */
template<class T> 
typename vislib::RegEx<T>::Expression *vislib::RegEx<T>::parseSet(
        const Char *& inOutStr) {
    ASSERT(inOutStr != NULL);
    ASSERT(*inOutStr == T::TOKEN_BRACKET_OPEN);

    /* Consume opening bracket. */
    inOutStr++;

    throw MissingImplementationException("RegEx<T>::parseSet",
        __FILE__, __LINE__);

    while ((*inOutStr != 0) && (*inOutStr != T::TOKEN_BRACKET_CLOSE)) {

        switch (*inOutStr) {

            case T::TOKEN_MINUS: 
                // TODO
                break;

            case T::TOKEN_BACKSLASH:
                // TODO
                //this->parseEscape(inOutStr);
                break;

            default:
                // TODO
                break;
        }
        
        inOutStr++;
        // TODO: Implementation missing.
    }

    if (inOutStr == NULL) {
        throw ParseException(ParseException::BRACKET_CLOSE_EXPECTED, 
            inOutStr - 1, __FILE__, __LINE__);
    }
}


/*
 * vislib::RegEx<T>::parseStar
 */
template<class T> 
typename vislib::RegEx<T>::Expression *vislib::RegEx<T>::parseStar(
        const Char *& inOutStr) {
    ASSERT(*inOutStr == T::TOKEN_STAR);
    // TODO: Implementation missing.
    throw MissingImplementationException("RegEx<T>::parseStar",
        __FILE__, __LINE__);
}
