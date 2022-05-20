/*
 * XmlReader.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "mmcore/utility/xml/XmlParser.h"
#include "mmcore/utility/xml/XmlReader.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/Exception.h"
#include "vislib/IllegalParamException.h"

#include <new>

/*
 * using xml implementation namespaces
 */
using namespace megamol::core::utility::xml;

/*
 * XmlReader::STATE_NULL
 */
const XmlReader::ParserState XmlReader::STATE_NULL = 0;

/*
 * XmlReader::STATE_ERROR
 */
const XmlReader::ParserState XmlReader::STATE_ERROR = XmlReader::STATE_NULL + 1;


/*
 * XmlReader::STATE_INITIALISE
 */
const XmlReader::ParserState XmlReader::STATE_INITIALISE = XmlReader::STATE_ERROR + 1;


/*
 * XmlReader::STATE_IGNORE_SUBTREE
 */
const XmlReader::ParserState XmlReader::STATE_IGNORE_SUBTREE = XmlReader::STATE_INITIALISE + 1;


/*
 * XmlReader::STATE_USER
 */
const XmlReader::ParserState XmlReader::STATE_USER = XmlReader::STATE_IGNORE_SUBTREE + 1;


/*
 * XmlReader::XmlReader
 */
XmlReader::XmlReader(void)
        : inputFile(NULL)
        , readerState(0)
        , endTagStates()
        , internalParser(NULL)
        , externalParser(NULL)
        , baseTag()
        , baseTagAttribs() {}


/*
 * XmlReader::~XmlReader
 */
XmlReader::~XmlReader(void) {
    this->CloseFile();
    SAFE_DELETE(this->inputFile);
    this->internalParser = NULL; // DO NOT DELETE
    this->externalParser = NULL; // DO NOT DELETE
}


/*
 * XmlReader::CloseFile
 */
void XmlReader::CloseFile(void) {
    if (this->inputFile != NULL) {
        this->inputFile->Close();
    }
    this->baseTag.Clear();
    this->baseTagAttribs.Clear();
}


/*
 * XmlReader::SetFile
 */
bool XmlReader::SetFile(vislib::sys::File* file) {
    if ((file == NULL) || !file->IsOpen()) {
        throw vislib::IllegalParamException("file", __FILE__, __LINE__);
    }

    if (this->inputFile != NULL) {
        this->CloseFile();
        delete this->inputFile;
    }
    this->inputFile = file;

    this->readerState = STATE_INITIALISE;
    try {
        this->baseTag.Clear();

        if (!this->parseXml()) {
            this->readerState = STATE_ERROR;
            return false;
        }

        this->readerState = STATE_NULL;

        return !this->baseTag.IsEmpty();

    } catch (...) {
        this->inputFile = NULL;
        throw;
    }
}


/*
 * XmlReader::XmlLine
 */
SIZE_T XmlReader::XmlLine(void) const {
    if (this->internalParser == NULL) {
        return 0;
    }
    XML_Parser& parser = *static_cast<XML_Parser*>(this->internalParser);
    return static_cast<SIZE_T>(XML_GetCurrentLineNumber(parser));
}


/*
 * XmlReader::StopParsing
 */
void XmlReader::StopParsing(void) {
    if (this->internalParser != NULL) {
        XML_Parser& parser = *static_cast<XML_Parser*>(this->internalParser);
        XML_StopParser(parser, false);
    }
}


/*
 * XmlReader::xmlStartTag
 */
void XMLCALL XmlReader::xmlStartTag(void* data, const XML_Char* tag, const XML_Char** attr) {
    static_cast<megamol::core::utility::xml::XmlReader*>(data)->startTag(tag, attr);
}


/*
 * XmlReader::xmlEndTag
 */
void XMLCALL XmlReader::xmlEndTag(void* data, const XML_Char* tag) {
    static_cast<megamol::core::utility::xml::XmlReader*>(data)->endTag(tag);
}


/*
 * XmlReader::xmlCharData
 */
void XMLCALL XmlReader::xmlCharData(void* data, const XML_Char* text, int len) {
    static_cast<megamol::core::utility::xml::XmlReader*>(data)->charData(text, len);
}


/*
 * XmlReader::xmlComment
 */
void XMLCALL XmlReader::xmlComment(void* data, const XML_Char* text) {
    static_cast<megamol::core::utility::xml::XmlReader*>(data)->comment(text);
}


/*
 * XmlReader::charData
 */
void XmlReader::charData(const XML_Char* text, int len) {
    ASSERT(this->internalParser != NULL);
    if (this->externalParser) {
        this->externalParser->CharacterData(this->externalParser->level, text, len, this->readerState);
    }
}


/*
 * XmlReader::comment
 */
void XmlReader::comment(const XML_Char* text) {
    ASSERT(this->internalParser != NULL);
    if (this->externalParser) {
        this->externalParser->Comment(this->externalParser->level, text, this->readerState);
    }
}


/*
 * XmlReader::endTag
 */
void XmlReader::endTag(const XML_Char* tag) {
    ASSERT(this->internalParser != NULL);

    ParserState postEnd = this->endTagStates.Pop();
    this->readerState = this->endTagStates.Pop();

    switch (this->readerState) {
    case STATE_INITIALISE:
        // No break!
    case STATE_ERROR:
        // TODO: Error handling
        this->StopParsing();
        break;
    case STATE_IGNORE_SUBTREE:
        // intentionally empty
        break;
    default:
        if (this->externalParser) {
            this->externalParser->endXmlTag(tag, postEnd);
        }
        break;
    }

    this->readerState = postEnd;
}


/*
 * XmlReader::parse
 */
bool XmlReader::parse(XmlParser* parser) {
    bool retval = false;

    if ((this->inputFile == NULL) || !this->inputFile->IsOpen()) {
        return false;
    }

    this->inputFile->SeekToBegin();

    this->externalParser = parser;

    try {

        this->readerState = STATE_NULL;

        if (!(retval = this->parseXml())) {
            this->readerState = STATE_ERROR;
            return false;
        }

    } catch (...) {
        this->readerState = STATE_ERROR;
        this->externalParser = NULL;
        throw;
    }
    this->readerState = STATE_NULL;
    this->externalParser = NULL;

    return retval;
}


/*
 * XmlReader::parseXml
 */
bool XmlReader::parseXml(void) {
    const unsigned int bufferSize = 1024; // magic number
    char buffer[bufferSize];
    vislib::sys::File::FileSize read;
    bool eof = false;
    XML_Parser parser = XML_ParserCreate(NULL);

    if (parser == NULL) {
        throw std::bad_alloc();
    }
    // store parser for internal calls
    this->internalParser = static_cast<void*>(&parser);

    try {

        XML_SetUserData(parser, static_cast<void*>(this));
        XML_SetElementHandler(parser, xmlStartTag, xmlEndTag);
        XML_SetCharacterDataHandler(parser, xmlCharData);
        XML_SetCommentHandler(parser, xmlComment);

        while (!eof) {
            read = this->inputFile->Read(buffer, bufferSize);
            eof = this->inputFile->IsEOF();
            if (XML_Parse(parser, buffer, int(read), eof) == XML_STATUS_ERROR) {

                // parser error
                XML_Error errorCode = XML_GetErrorCode(parser);

                if (errorCode == XML_ERROR_ABORTED)
                    break;

                int lineNumber = XML_GetCurrentLineNumber(parser);
                int column = XML_GetCurrentColumnNumber(parser);

                const XML_Char* errorString = XML_ErrorString(errorCode);
                vislib::StringA msg;
                msg.Format("Xml-file parse error %d[Ln:%d;Col:%d]: %s", int(errorCode), lineNumber, column,
                    vislib::StringA(errorString).PeekBuffer());
                throw vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
            }
        }

        if (this->externalParser != NULL) {
            this->externalParser->Completed();
        }

    } catch (...) {
        XML_ParserFree(parser);
        this->internalParser = NULL;
        throw;
    }

    XML_ParserFree(parser);
    this->internalParser = NULL;

    return true;
}


/*
 * XmlReader::startTag
 */
void XmlReader::startTag(const XML_Char* tag, const XML_Char** attributes) {
    ASSERT(this->internalParser != NULL);
    //    XML_Parser &parser = *static_cast<XML_Parser*>(this->internalParser);

    ParserState preEnd = this->readerState, postEnd = this->readerState;

    switch (this->readerState) {
    case STATE_ERROR:
        // TODO: Error handling.
        this->StopParsing();
        return;
    case STATE_INITIALISE:
        this->baseTag = tag;
        for (int i = 0; attributes[i]; i += 2) {
            this->baseTagAttribs.Add(XmlAttribute(attributes[i], attributes[i + 1]));
        }
        this->baseTagAttribs.Trim();
        this->StopParsing();
        return;
    case STATE_IGNORE_SUBTREE:
        // intentionally empty
        break;
    default:
        if (this->externalParser != NULL) {
            this->externalParser->startXmlTag(tag, attributes, preEnd, postEnd);
        }
        break;
    }

    this->endTagStates.Push(preEnd);
    this->endTagStates.Push(postEnd);
}


/*
 * XmlReader::xmlPosition
 */
vislib::StringA XmlReader::xmlPosition(void) const {
    vislib::StringA pos;
    if (this->internalParser == NULL) {
        return pos;
    }

    XML_Parser& parser = *static_cast<XML_Parser*>(this->internalParser);

    XML_Error errorCode = XML_GetErrorCode(parser);
    int lineNumber = XML_GetCurrentLineNumber(parser);
    int column = XML_GetCurrentColumnNumber(parser);

    pos.Format("Line: %d; Column: %d", lineNumber, column);

    if (errorCode != XML_ERROR_NONE) {
        vislib::StringA tmp;
        tmp.Format("Error: %d; ", int(errorCode));
        pos.Prepend(tmp);
    }

    return pos;
}
