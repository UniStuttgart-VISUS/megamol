/*
 * XmlParser.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_XMLPARSER_INCLUDED
#define MEGAMOLCORE_XMLPARSER_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/utility/xml/XmlReader.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/Stack.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace utility {
namespace xml {

/**
 * Base class for MegaMol xml parsers.
 */
class XmlParser {
public:
    /** friend declaration of XmlReader for calling the callbacks */
    friend class XmlReader;

    /** Ctor. */
    XmlParser(void);

    /** Dtor. */
    virtual ~XmlParser(void);

    /**
     * Answers whether there are error messages present. This method is
     * typically called directly after 'Parse' returned.
     *
     * @return 'true' if there are error messages.
     */
    inline bool ErrorMessagesPresent(void) const {
        return this->errorMsgsPresent;
    }

    /**
     * Returns the messages produced while parsing. This method is
     * typically called directly after 'Parse' returned.
     *
     * @return An iterator over all messages.
     */
    vislib::SingleLinkedList<vislib::StringA>::Iterator Messages(void) const {
        return this->msgs.GetIterator();
    }

    /**
     * Answer whether or not messages are present. This method is typically
     * called directly after 'Parse' returned. Messages can be warnings and
     * errors.
     *
     * @return 'true' if there are messages.
     */
    inline bool MessagesPresent(void) const {
        return this->msgs.Count() > 0;
    }

    /**
     * Parses the xml file opened in 'reader'.
     *
     * @param reader The XmlReader holding the xml file.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool Parse(XmlReader& reader);

protected:
    /**
     * Callback method for character data.
     * This implementation ignores all data.
     *
     * @param level The level of the current xml tag.
     * @param text Pointer to the character data. Note that this is not
     *             zero-terminated.
     * @param len Length of the character data.
     * @param state The current state of the reader.
     */
    virtual void CharacterData(unsigned int level, const XML_Char* text, int len, XmlReader::ParserState state);

    /**
     * Callback method for comment data.
     * This implementation ignores all data.
     *
     * @param level The level of the current xml tag.
     * @param text Pointer to the character data.
     * @param state The current state of the reader.
     */
    virtual void Comment(unsigned int level, const XML_Char* text, XmlReader::ParserState state);

    /**
     * Is called when the parsing of the xml file is completed
     */
    virtual void Completed(void);

    /**
     * Checks whether the base tag of the file specified by the reader is
     * compatible with this parser.
     *
     * @param reader The current xml reader.
     *
     * @return 'true' if the file of the reader is compatible with this
     *         parser, 'false' otherwise.
     */
    virtual bool CheckBaseTag(const XmlReader& reader);

    /**
     * Logs an error with the given message 'msg' and the position within
     * the xml file. The parsing process is continued.
     *
     * @param msg The error message.
     */
    void Error(const char* msg) const;

    /**
     * Logs an error with the given message 'msg' and the position within
     * the xml file. The parsing process is also terminated.
     *
     * @param msg The error message.
     */
    void FatalError(const char* msg) const;

    /**
     * Answer the reader currently used. It is only safe to call this
     * method from within the callback methods.
     *
     * @return The reader currently used.
     */
    inline const XmlReader& Reader(void) const {
        ASSERT(this->reader != NULL);
        return *this->reader;
    }

    /**
     * Callback method for starting xml tags.
     * This implementation only handles the root tag.
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
     * This implementation only handles the root tag.
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

    /**
     * Logs a warning with the given message 'msg' and the position within
     * the xml file. The parsing process is continued.
     *
     * @param msg The warning message.
     */
    void Warning(const char* msg) const;

    /**
     * Logs a warning about an unexpected attribute ignored.
     *
     * @param tag The tag of the attribute.
     * @param attr The unexpected attribute.
     */
    void WarnUnexpectedAttribut(const XML_Char* tag, const XML_Char* attr) const;

private:
    /**
     * Callback method for starting xml tags.
     *
     * @param name The name of the current xml tag. This string should be
     *             considdered utf8.
     * @param attrib The attributes of the current xml tag. These strings
     *               should be considdered utf8.
     * @param outEndTagState The ParserState to be set when the end tag
     *                       callback is called for the end tag of this
     *                       start tag. The default value is the current
     *                       state of the reader.
     * @param outPostEndTagState The ParserState to be set after the end
     *                           tag callback has been returned. This value
     *                           may be altered in the end tag callback
     *                           method. The default value is the current
     *                           state of the reader.
     */
    void startXmlTag(const XML_Char* name, const XML_Char** attrib, XmlReader::ParserState& outEndTagState,
        XmlReader::ParserState& outPostEndTagState);

    /**
     * Callback method for the ending xml tags.
     *
     * @param name The name of the current xml tag. This string should be
     *             considdered utf8.
     * @param outPostEndTagState The state to be set after this callback
     *                           returned. The default value is the value
     *                           set by 'startTag'.
     */
    void endXmlTag(const XML_Char* name, XmlReader::ParserState& outPostEndTagState);

    /** pointer to the currently used XmlReader object */
    XmlReader* reader;

    /** The level of the current xml tag */
    unsigned int level;

    /** The number of the current xml tag */
    unsigned int number;

    /** The number stack of the end xml tags */
    vislib::Stack<unsigned int> endNumbers;

    /**
     * Flag indicating if the messages of the current parsing contain
     * errors.
     */
    mutable bool errorMsgsPresent;

    /** The messages of the current parsing */
    mutable vislib::SingleLinkedList<vislib::StringA> msgs;
};

} /* end namespace xml */
} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_XMLPARSER_INCLUDED */
