/*
 * ConditionalParser.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/xml/ConditionalParser.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/SystemInformation.h"
#include "stdafx.h"
#include "vislib/CharTraits.h"
#include "vislib/String.h"
#include "vislib/Trace.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/assert.h"

//#define DO_SOME_BULB_TEXT 1


/*
 * using xml implementation namespaces
 */
using namespace megamol::core::utility::xml;


/*
 * ConditionalParser::ConditionalParser
 */
ConditionalParser::ConditionalParser(void) : XmlParser(), ifCheckerVer(VERSION_1_0), conditions() {}


/*
 * ConditionalParser::~ConditionalParser
 */
ConditionalParser::~ConditionalParser(void) {}


/*
 * ConditionalParser::setConditionalParserVersion
 */
void ConditionalParser::setConditionalParserVersion(int ver) {
    if (ver == 0) {
        if (this->ifCheckerVer == VERSION_1_0)
            return;
        this->ifCheckerVer = VERSION_1_0;

    } else if (ver == 1) {
        if (this->ifCheckerVer == VERSION_1_1)
            return;
        this->ifCheckerVer = VERSION_1_1;
        this->conditions.Clear();

    } else {
        // wtf?
        ASSERT(false);
    }
}


/*
 * ConditionalParser::StartTag
 */
bool ConditionalParser::StartTag(unsigned int num, unsigned int level, const XML_Char* name, const XML_Char** attrib,
    XmlReader::ParserState state, XmlReader::ParserState& outChildState, XmlReader::ParserState& outEndTagState,
    XmlReader::ParserState& outPostEndTagState) {

    if (XmlParser::StartTag(num, level, name, attrib, state, outChildState, outEndTagState, outPostEndTagState)) {
        return true; // handled by base class
    }

    switch (this->ifCheckerVer) {
    case VERSION_1_0:
        if ((level > 0) && MMXML_STRING("IF").Equals(name, false)) {
            if (!this->evaluateIf(attrib)) {
                outChildState = XmlReader::STATE_IGNORE_SUBTREE;
            }
            return true;
        }
        break;
    case VERSION_1_1:
        if (level > 0) {
            if (MMXML_STRING("if").Equals(name)) {
                const XML_Char* cond = NULL;

                for (int i = 0; attrib[i]; i += 2) {
                    if (MMXML_STRING("cond").Equals(attrib[i])) {
                        cond = attrib[i + 1];
                    } else {
                        this->WarnUnexpectedAttribut(name, attrib[i]);
                    }
                }

                if (cond == NULL) {
                    this->Error("Tag \"if\" without \"cond\" always results to false.");
                    outChildState = XmlReader::STATE_IGNORE_SUBTREE;
                } else {
                    if (!this->evaluateCondition(cond)) {
                        outChildState = XmlReader::STATE_IGNORE_SUBTREE;
                    }
                }
                return true;
            }

            if (MMXML_STRING("cond").Equals(name)) {
                const XML_Char* name = NULL;
                const XML_Char* value = NULL;

                for (int i = 0; attrib[i]; i += 2) {
                    if (MMXML_STRING("name").Equals(attrib[i])) {
                        name = attrib[i + 1];
                    } else if (MMXML_STRING("value").Equals(attrib[i])) {
                        value = attrib[i + 1];
                    } else {
                        this->WarnUnexpectedAttribut(name, attrib[i]);
                    }
                }

                if (name == NULL) {
                    this->Error("Tag \"cond\" without \"name\" ignored.");
                } else if (value == NULL) {
                    this->Error("Tag \"cond\" without \"value\" ignored.");
                } else {
                    this->conditions[vislib::StringW(name)] = this->evaluateCondition(value);
                }
                outChildState = XmlReader::STATE_IGNORE_SUBTREE;
                return true;
            }
        }
        break;

    default: // internal error
        ASSERT(false);
        break;
    }

    return false; // unhandled.
}


/*
 * ConditionalParser::EndTag
 */
bool ConditionalParser::EndTag(unsigned int num, unsigned int level, const XML_Char* name, XmlReader::ParserState state,
    XmlReader::ParserState& outPostEndTagState) {
    if (XmlParser::EndTag(num, level, name, state, outPostEndTagState)) {
        return true; // handled by base class
    }

    switch (this->ifCheckerVer) {
    case VERSION_1_0:
        if ((level > 0) && MMXML_STRING("IF").Equals(name, false)) {
            return true; // if tag
        }
        break;
    case VERSION_1_1:
        if ((level > 0) && (MMXML_STRING("if").Equals(name) || MMXML_STRING("cond").Equals(name))) {
            return true;
        }
        break;
    default: // internal error
        ASSERT(false);
        break;
    }

    return false; // unhandled.
}


/*
 * ConditionalParser::evaluateIf
 */
bool ConditionalParser::evaluateIf(const XML_Char** attrib) const {
    bool oneTrue = false;
    bool allTrue = true;

    for (unsigned int i = 0; attrib[i]; i += 2) {
        if (MMXML_STRING("bitwidth").Equals(attrib[i])) {
            int bitwidth;
            try {
                bitwidth = vislib::CharTraits<XML_Char>::ParseInt(attrib[i + 1]);
            } catch (...) { // format exception
                this->Error("Cannot evaluate \"bitwidth\". Test failed.");
                bitwidth = 0;
            }

            if (bitwidth == int(vislib::sys::SystemInformation::SelfWordSize())) {
                oneTrue = true;
            } else {
                allTrue = false;
            }

        } else if (MMXML_STRING("computer").Equals(attrib[i])) {

            if (MMXML_STRING(vislib::sys::SystemInformation::ComputerNameW()).Equals(attrib[i + 1], false)) {
                oneTrue = true;
            } else {
                allTrue = false;
            }

        } else if (MMXML_STRING("debug").Equals(attrib[i])) {

            bool debug = false;
            try {
                debug = vislib::CharTraits<XML_Char>::ParseBool(attrib[i + 1]);
#ifndef _DEBUG
                debug = !debug;
#endif /* !_DEBUG */
                if (debug) {
                    oneTrue = true;
                } else {
                    allTrue = false;
                }
            } catch (...) {
                this->Error("Cannot evaluate \"debug\". Test failed.");
                allTrue = false;
            }

        } else if (MMXML_STRING("os").Equals(attrib[i])) {

            switch (vislib::sys::SystemInformation::SystemType()) {
            case vislib::sys::SystemInformation::OSTYPE_WINDOWS:
                if (MMXML_STRING("windows").Equals(attrib[i + 1], false)) {
                    oneTrue = true;
                } else {
                    allTrue = false;
                }
                break;
            case vislib::sys::SystemInformation::OSTYPE_LINUX:
                if (MMXML_STRING("linux").Equals(attrib[i + 1], false)) {
                    oneTrue = true;
                } else {
                    allTrue = false;
                }
                break;
            default: // unable to detect local system.
                this->Error("Cannot evaluate \"os\". Test failed.");
                allTrue = false;
                break;
            }

        } else {
            vislib::StringA msg;
            msg.Format("Unknown test \"%s\" ignored.", vislib::StringA(attrib[i]).PeekBuffer());
            this->Error(msg.PeekBuffer());
        }
    }

    return oneTrue;
}


/*
 * ConditionalParser::evaluateCondition
 */
bool ConditionalParser::evaluateCondition(const XML_Char* expression) const {
    vislib::StringA exp(expression);
#ifdef DO_SOME_BULB_TEXT
    VLTRACE(VISLIB_TRCELVL_INFO, "Evaluating expression \"%s\"\n", exp.PeekBuffer());
#endif /* DO_SOME_BULB_TEXT */

    // sift expression
    exp.Append(" ");
    int state = 0;
    int charcode = 0;
    int start = 0;
    int len = exp.Length();
    vislib::Array<vislib::StringA> tokenz;
    bool error;

    for (int pos = 0; pos < len; pos++) {
        error = false;

        if (vislib::CharTraitsA::IsSpace(exp[pos])) {
            charcode = 0;
        } else if (exp[pos] == '(') {
            charcode = 1;
        } else if (exp[pos] == ')') {
            charcode = 2;
        } else if (exp[pos] == '!') {
            charcode = 3;
        } else if (exp[pos] == '=') {
            charcode = 4;
        } else if (exp[pos] == '&') {
            charcode = 6;
        } else if (exp[pos] == '|') {
            charcode = 7;
        } else {
            charcode = 5;
        }

        switch (state) {
        case 0: // b4
            switch (charcode) {
            case 0:
                break; // nothing to do
            case 1:
                tokenz.Add("(");
                break;
            case 2:
                tokenz.Add(")");
                break;
            case 3:
                state = 3;
                break;
            case 4:
                state = 2;
                break;
            case 5:
                start = pos;
                state = 1;
                break;
            case 6:
                state = 4;
                break;
            case 7:
                state = 5;
                break;
            }
            break;
        case 1: // id
            if (charcode == 5)
                break;
            tokenz.Add(exp.Substring(start, pos - start));
            switch (charcode) {
            case 0:
                state = 0;
                break;
            case 1:
                error = true;
                break;
            case 2:
                tokenz.Add(")");
                state = 0;
                break;
            case 3:
                state = 3;
                break;
            case 4:
                state = 2;
                break;
            case 6:
                state = 4;
                break;
            case 7:
                state = 5;
                break;
            }
            break;
        case 2: // 1=
            if (charcode == 4) {
                tokenz.Add("==");
                state = 0;
            } else {
                error = true;
            }
            break;
        case 3: // !
            switch (charcode) {
            case 0:
                tokenz.Add("!");
                state = 0;
                break;
            case 1:
                tokenz.Add("!");
                tokenz.Add("(");
                state = 0;
                break;
            case 2:
                error = true;
                break;
            case 3:
                tokenz.Add("!");
                break;
            case 4:
                tokenz.Add("!=");
                state = 0;
                break;
            case 5:
                tokenz.Add("!");
                start = pos;
                state = 1;
                break;
            case 6:
                error = true;
                break;
            case 7:
                error = true;
                break;
            }
            break;
        case 4: // 1&
            if (charcode == 6) {
                tokenz.Add("&&");
                state = 0;
            } else {
                error = true;
            }
            break;
        case 5: // 1|
            if (charcode == 7) {
                tokenz.Add("||");
                state = 0;
            } else {
                error = true;
            }
            break;
        }

        if (error) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                "Syntax error while evaluating expression \"%s\" on position \"%d\"."
                "Setting result to 'false'.\n",
                exp.PeekBuffer(), pos);
            return false;
        }
    }

    // removing double negations
    for (SIZE_T i = tokenz.Count() - 1; i > 0; i--) {
        if (tokenz[i].Equals("!") && tokenz[i - 1].Equals("!")) {
            i--;
            tokenz.RemoveAt(i);
            tokenz.RemoveAt(i);
        }
    }

    bool rv = this->evaluateCondition(
        exp, const_cast<vislib::StringA*>(tokenz.PeekElements()), static_cast<unsigned int>(tokenz.Count()));

#ifdef DO_SOME_BULB_TEXT
    VLTRACE(VISLIB_TRCELVL_INFO, "Evaluated expression \"%s\" to %s\n", exp.PeekBuffer(), rv ? "true" : "false");
#endif /* DO_SOME_BULB_TEXT */

    return rv;
}


/*
 * ConditionalParser::evaluateCondition
 */
bool ConditionalParser::evaluateCondition(
    const vislib::StringA& expression, vislib::StringA* tokenz, unsigned int tokenCount) const {
    using megamol::core::utility::log::Log;

    if (tokenCount == 0) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Error in expression \"%s\": internal error (empty token list)\n",
            expression.PeekBuffer());
        return false;
    }

    // evaluate identifiers
    if (tokenCount == 1) {
        if (tokenz[0].Equals("true", false)) {
            return true;
        } else if (tokenz[0].Equals("false", false)) {
            return false;
        } else if (tokenz[0].Equals("debug", false)) {
#if defined(DEBUG) || defined(_DEBUG)
            return true;
#else  /* defined(DEBUG) || defined(_DEBUG) */
            return false;
#endif /* defined(DEBUG) || defined(_DEBUG) */
        } else {
            const bool* v = this->conditions.FindValue(tokenz[0]);
            if (v != NULL) {
                return *v;
            } else {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Error in expression \"%s\": unknown identifier \"%s\"\n",
                    expression.PeekBuffer(), tokenz[0].PeekBuffer());
                return false;
            }
        }
    }

    // solve negations
    if (tokenz[0].Equals("!")) {
        if (tokenz[1].Equals("(")) {
            int cnt = 1;
            for (int i = 2; i < static_cast<int>(tokenCount); i++) {
                if (tokenz[i].Equals("("))
                    cnt++;
                if (tokenz[i].Equals(")")) {
                    cnt--;
                    if (cnt == 0) {
                        tokenz[i] = this->evaluateCondition(expression, tokenz + 2, i - 2) ? "false" : "true";
                        return this->evaluateCondition(expression, tokenz + i, tokenCount - i);
                    }
                }
            }

        } else {
            tokenz[1] = this->evaluateCondition(expression, tokenz + 1, 1) ? "false" : "true";
            return this->evaluateCondition(expression, tokenz + 1, tokenCount - 1);
        }
    }

    // solve subexpressions in brackets
#ifdef DO_SOME_BULB_TEXT
    vislib::StringA blub;
#endif /* DO_SOME_BULB_TEXT */
    int obc = 1, cbc = 1;

    while (obc > 0) {
#ifdef DO_SOME_BULB_TEXT
        blub.Clear();
#endif /* DO_SOME_BULB_TEXT */
        obc = cbc = 0;
        for (unsigned int i = 0; i < tokenCount; i++) {
            if (tokenz[i].Equals("("))
                obc++;
            if (tokenz[i].Equals(")"))
                cbc++;
#ifdef DO_SOME_BULB_TEXT
            blub.Append(" ");
            blub.Append(tokenz[i]);
#endif /* DO_SOME_BULB_TEXT */
        }

        if (obc != cbc) {
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Error in expression \"%s\": brackets mismatch\n", expression.PeekBuffer());
            return false;
        }
#ifdef DO_SOME_BULB_TEXT
        VLTRACE(VISLIB_TRCELVL_INFO, "\tEvaluating:%s\n", blub.PeekBuffer());
#endif /* DO_SOME_BULB_TEXT */
        if (obc > 0) {
            int start = -1;
            for (unsigned int i = 0; i < tokenCount; i++) {
                if (tokenz[i].Equals("(")) {
                    obc--;
                    if (obc == 0) {
                        start = i;
                    }
                }
                if (tokenz[i].Equals(")")) {
                    if (obc == 0) {
                        ASSERT(start >= 0);

                        tokenz[start].Clear();
                        tokenz[i].Clear();
                        ++start;
                        if ((i - start) > 0) {
                            bool v = this->evaluateCondition(expression, tokenz + start, i - start);
                            tokenz[start] = v ? "true" : "false";
                            for (++start; start < static_cast<int>(i); ++start) {
                                tokenz[start].Clear();
                            }
                        }

                        start = 0;
                        for (i = 0; i < tokenCount; i++) {
                            if (tokenz[i].IsEmpty())
                                continue;
                            tokenz[start] = tokenz[i];
                            start++;
                        }
                        tokenCount = start;

                        break;
                    }
                }
            }
            obc = 1;
        }
    }

    // solve basic expressions
    if (tokenCount < 3) {
        return this->evaluateCondition(expression, tokenz, tokenCount);
    } else if (tokenCount > 3) {
        vislib::StringA seq;
        seq.Format("%d:", tokenCount);
        for (unsigned int i = 0; i < tokenCount; i++) {
            seq.Append(" ");
            seq.Append(tokenz[i]);
        }
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Error in expression \"%s\": internal error (basic sequence of wrong length: %s)\n",
            expression.PeekBuffer(), seq.PeekBuffer());
        return false;
    }

    if (tokenz[1].Equals("==")) {
        if (tokenz[0].Equals(tokenz[2], false)) {
            return true;
        }
        if (tokenz[0].Equals("bitwidth", false)) {
            try {
                return vislib::CharTraitsA::ParseInt(tokenz[2]) == int(vislib::sys::SystemInformation::SelfWordSize());
            } catch (...) { return false; }
        }
        if (tokenz[2].Equals("bitwidth", false)) {
            try {
                return vislib::CharTraitsA::ParseInt(tokenz[0]) == int(vislib::sys::SystemInformation::SelfWordSize());
            } catch (...) { return false; }
        }
        if (tokenz[0].Equals("computer", false)) {
            return tokenz[2].Equals(vislib::sys::SystemInformation::ComputerNameA(), false);
        }
        if (tokenz[2].Equals("computer", false)) {
            return tokenz[0].Equals(vislib::sys::SystemInformation::ComputerNameA(), false);
        }
        if (tokenz[0].Equals("os", false)) {
            switch (vislib::sys::SystemInformation::SystemType()) {
            case vislib::sys::SystemInformation::OSTYPE_WINDOWS:
                return tokenz[2].Equals("windows", false);
            case vislib::sys::SystemInformation::OSTYPE_LINUX:
                return tokenz[2].Equals("linux", false);
            default: // unable to detect local system.
                return !tokenz[2].Equals("windows", false) && !tokenz[2].Equals("linux", false);
            }
        }
        if (tokenz[2].Equals("os", false)) {
            switch (vislib::sys::SystemInformation::SystemType()) {
            case vislib::sys::SystemInformation::OSTYPE_WINDOWS:
                return tokenz[0].Equals("windows", false);
            case vislib::sys::SystemInformation::OSTYPE_LINUX:
                return tokenz[0].Equals("linux", false);
            default: // unable to detect local system.
                return !tokenz[0].Equals("windows", false) && !tokenz[0].Equals("linux", false);
            }
        }

        return this->evaluateCondition(expression, tokenz, 1) == this->evaluateCondition(expression, tokenz + 2, 1);

    } else if (tokenz[1].Equals("!=")) {
        tokenz[1] = "==";
        return !this->evaluateCondition(expression, tokenz, 3);
    } else if (tokenz[1].Equals("&&")) {
        return this->evaluateCondition(expression, tokenz, 1) && this->evaluateCondition(expression, tokenz + 2, 1);
    } else if (tokenz[1].Equals("||")) {
        return this->evaluateCondition(expression, tokenz, 1) || this->evaluateCondition(expression, tokenz + 2, 1);
    } else {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Error in expression \"%s\": invalid binary operator %s\n",
            expression.PeekBuffer(), tokenz[1].PeekBuffer());
        return false;
    }

    // we should never ever reach this point
    ASSERT(false);
    return false;
}
