/*
 * XmlParser.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/utility/xml/XmlParser.h"


/*
 * using xml implementation namespaces
 */
using namespace megamol::core::utility::xml;


/*
 * XmlParser::XmlParser
 */
XmlParser::XmlParser(void) : reader(NULL), level(0), number(0), endNumbers(),
        errorMsgsPresent(false), msgs() {
}


/*
 * XmlParser::~XmlParser
 */
XmlParser::~XmlParser(void) {
}


/*
 * XmlParser::Parse
 */
bool XmlParser::Parse(XmlReader& reader) {
    bool retval = false;
    this->reader = &reader;

    try {

        if (!CheckBaseTag(reader)) {
            throw vislib::Exception("XML-file not compatible with this parser",
                __FILE__, __LINE__);
        }

        this->level = 0;
        this->number = 0;
        this->endNumbers.Clear();
        this->errorMsgsPresent = false;
        this->msgs.Clear();

        retval = reader.parse(this);

        if (this->errorMsgsPresent) {
            retval = false;
        }

    } catch(...) {
        this->reader = NULL;
        throw;
    }
    this->reader = NULL;

    return retval;
}


/*
 * XmlParser::CharacterData
 */
void XmlParser::CharacterData(unsigned int level, const XML_Char *text, int len,
        XmlReader::ParserState state) {
    // intentionally empty
}


/*
 * XmlParser::Comment
 */
void XmlParser::Comment(unsigned int level, const XML_Char *text,
        XmlReader::ParserState state) {
    // intentionally empty
}


/*
 * XmlParser::Completed
 */
void XmlParser::Completed(void) {
    // intentionally empty
}


/*
 * XmlParser::CheckBaseTag
 */
bool XmlParser::CheckBaseTag(const XmlReader& reader) {
    return true;
}


/*
 * XmlParser::Error
 */
void XmlParser::Error(const char *msg) const {
    this->errorMsgsPresent = true;
    vislib::StringA fullMsg;
    vislib::StringA pos = (this->reader) ? this->reader->xmlPosition() : vislib::StringA();

    if (pos.IsEmpty()) {
        fullMsg = "Error: ";
        fullMsg.Append(msg);
    } else {
        fullMsg.Format("Error (%s): %s", pos.PeekBuffer(), msg);
    }

    this->msgs.Add(fullMsg);
}


/*
 * XmlParser::FatalError
 */
void XmlParser::FatalError(const char *msg) const {
    this->Error(msg);
    if (this->reader) {
        this->reader->StopParsing();
    }
}


/*
 * XmlParser::StartTag
 */
bool XmlParser::StartTag(unsigned int num, unsigned int level,
        const XML_Char * name, const XML_Char ** attrib,
        XmlReader::ParserState state,
        XmlReader::ParserState& outChildState,
        XmlReader::ParserState& outEndTagState,
        XmlReader::ParserState& outPostEndTagState) {

    //for (unsigned int i = 0; i < level; i++) printf("  ");
    //printf("<%s %u>\n", name, num);

    //if (vislib::StringA("If").Equals(name, false)) {
    //    outChildState = XmlReader::STATE_IGNORE_SUBTREE;
    //}

    return (level == 0);
}


/*
 * XmlParser::EndTag
 */
bool XmlParser::EndTag(unsigned int num, unsigned int level,
        const XML_Char * name, XmlReader::ParserState state,
        XmlReader::ParserState& outPostEndTagState) {

    //for (unsigned int i = 0; i < level; i++) printf("  ");
    //printf("</%s %u>\n", name, num);

    return (level == 0);
}


/*
 * XmlParser::Warning
 */
void XmlParser::Warning(const char *msg) const {
    vislib::StringA fullMsg;
    vislib::StringA pos = (this->reader) ? this->reader->xmlPosition() : vislib::StringA();

    if (pos.IsEmpty()) {
        fullMsg = "Warning: ";
        fullMsg.Append(msg);
    } else {
        fullMsg.Format("Warning (%s): %s", pos.PeekBuffer(), msg);
    }

    this->msgs.Add(fullMsg);
}


/*
 * XmlParser::WarnUnexpectedAttribut
 */
void XmlParser::WarnUnexpectedAttribut(const XML_Char *tag, const XML_Char *attr) const {
    vislib::StringA msg;
    msg.Format("Unexpected attribute \"%s\" in tag \"%s\"",
        vislib::StringA(attr).PeekBuffer(),
        vislib::StringA(tag).PeekBuffer());
    this->Warning(msg);
}


/*
 * XmlParser::startXmlTag
 */
void XmlParser::startXmlTag(const XML_Char *name, const XML_Char **attrib,
        XmlReader::ParserState& outEndTagState,
        XmlReader::ParserState& outPostEndTagState) {
    bool handled = this->StartTag(this->number, this->level++, name, attrib, 
        this->reader->readerState, this->reader->readerState, outEndTagState, 
        outPostEndTagState);
    if (!handled) {

        vislib::StringA msg;
        msg.Format("Unexpected tag \"%s\" ignored",
            vislib::StringA(name).PeekBuffer());
        this->Warning(msg);

        this->reader->readerState = XmlReader::STATE_IGNORE_SUBTREE;
        outEndTagState = XmlReader::STATE_IGNORE_SUBTREE;
        // do not change 'outPostEndTagState'
    }
    this->endNumbers.Push(this->number++);
}


/*
 * XmlParser::endXmlTag
 */
void XmlParser::endXmlTag(const XML_Char *name,
        XmlReader::ParserState& outPostEndTagState) {
    this->EndTag(this->endNumbers.Pop(), --this->level, name, 
        this->reader->readerState, outPostEndTagState);
}
