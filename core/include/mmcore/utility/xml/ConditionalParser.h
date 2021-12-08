/*
 * ConditionalParser.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CONDITIONALPARSER_INCLUDED
#define MEGAMOLCORE_CONDITIONALPARSER_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/utility/xml/XmlParser.h"
#include "mmcore/utility/xml/XmlReader.h"
#include "vislib/Array.h"
#include "vislib/Map.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace utility {
namespace xml {

/**
 * Base class for MegaMol conditional xml parsers.
 * Theses parsers can evaluate '<if>' tags.
 */
class ConditionalParser : public XmlParser {
public:
    /** Ctor. */
    ConditionalParser(void);

    /** Dtor. */
    virtual ~ConditionalParser(void);

protected:
    /**
     * Sets the version for the conditional parser
     *
     * @param ver The version
     */
    void setConditionalParserVersion(int ver);

    /**
     * Callback method for starting xml tags.
     * This implementation handles the '<if>' tags.
     *
     * @param num The number of the current xml tag.
     * @param level The level of the current xml tag.
     * @param name The name of the current xml tag. This string should be
     *             considdered utf8.
     * @param attrib The attributes of the current xml tag. These strings
     *               should be considdered utf8.
     * @param state The current state of the reader.
     * @param outChildState The ParserState used by the reader for the
     *                      first child of the current xml tag. The default
     *                      value is the current state of the reader.
     * @param outEndTagState The ParserState to be set when the end tag
     *                       callback is called for the end tag of this
     *                       start tag. The default value is the current
     *                       state of the reader.
     * @param outPostEndTagState The ParserState to be set after the end
     *                           tag callback has been returned. This value
     *                           may be altered in the end tag callback
     *                           method. The default value is the current
     *                           state of the reader.
     *
     * @return 'true' if the tag has been handled by this implementation.
     *         'false' if the tag was not handled.
     */
    virtual bool StartTag(unsigned int num, unsigned int level, const XML_Char* name, const XML_Char** attrib,
        XmlReader::ParserState state, XmlReader::ParserState& outChildState, XmlReader::ParserState& outEndTagState,
        XmlReader::ParserState& outPostEndTagState);

    /**
     * Callback method for the ending xml tags.
     * This implementation handles the '<if>' tags.
     *
     * @param num The number of the current xml tag (this is the same
     *            number used when 'startTag' was called).
     * @param level The level of the current xml tag.
     * @param name The name of the current xml tag. This string should be
     *             considdered utf8.
     * @param state The current state of the reader.
     * @param outPostEndTagState The state to be set after this callback
     *                           returned. The default value is the value
     *                           set by 'startTag'.
     *
     * @return 'true' if the tag has been handled by this implementation.
     *         'false' if the tag was not handled.
     */
    virtual bool EndTag(unsigned int num, unsigned int level, const XML_Char* name, XmlReader::ParserState state,
        XmlReader::ParserState& outPostEndTagState);

private:
    /** possible values for the ifChecker */
    enum IfCheckerVersion { VERSION_1_0, VERSION_1_1 };

    /**
     * Evaluates the attributes of an '<if>'-tag. Valid tests are
     * 'bitwidth', 'computer', 'debug', and 'os'. Multiple
     * tests in one tag will be or-combined.
     *
     * @param attrib The attribute array of the '<if>'-tag.
     *
     * @return The value evaluated from the tests.
     */
    bool evaluateIf(const XML_Char** attrib) const;

    /**
     * Evaluates a condition expression.
     *
     * @param expression The expression to be evaluated
     *
     * @return The result of the expression (false on error)
     */
    bool evaluateCondition(const XML_Char* expression) const;

    /**
     * Evaluates a condition expression.
     *
     * @param expression The expression to be evaluated
     * @param tokenz The sifted tokens
     * @param tokenCount The number of tokens in the array
     *
     * @return The result of the expression (false on error)
     */
    bool evaluateCondition(const vislib::StringA& expression, vislib::StringA* tokenz, unsigned int tokenCount) const;

    /** The if checker version */
    IfCheckerVersion ifCheckerVer;

    /** The evaluated conditions */
    vislib::Map<vislib::StringA, bool> conditions;
};

} /* end namespace xml */
} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CONDITIONALPARSER_INCLUDED */
