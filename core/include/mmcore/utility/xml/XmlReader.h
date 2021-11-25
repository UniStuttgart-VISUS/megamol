/*
 * XmlReader.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_XMLREADER_H_INCLUDED
#define MEGAMOLCORE_XMLREADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <vislib/sys/Path.h>

#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/Stack.h"
#include "vislib/String.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/File.h"

#define XML_STATIC // We're linking expat as static library
#include <expat.h>

typedef vislib::String<vislib::CharTraits<XML_Char>> MMXML_STRING;

namespace megamol {
namespace core {
namespace utility {
namespace xml {

/** forward declaration of XmlParser */
class XmlParser;

/**
 * Class of xml file readers.
 */
class XmlReader {
private:
    /** The alias for the file type to use */
    typedef vislib::sys::FastFile FileType;

public:
    /**
     * friend declaration of abstract xml parser of calling the parse
     * method
     */
    friend class XmlParser;

    /** The type of states of the parser */
    typedef unsigned int ParserState;

    /**
     * Type of xml attributes. The key and the value strings must be
     * considdered utf8.
     */
    typedef vislib::Pair<MMXML_STRING, MMXML_STRING> XmlAttribute;

    /** Parser state indicating an unknown state. */
    static const ParserState STATE_NULL;

    /** Parser state indicating a fatal error. */
    static const ParserState STATE_ERROR;

    /** Parser state indicating the file initialisation */
    static const ParserState STATE_INITIALISE;

    /** Parser state indicating the state for ignoring whole subtrees */
    static const ParserState STATE_IGNORE_SUBTREE;

    /**
     * Parser state indicating a user state. All user states must at equal
     * or larger in value as this state.
     */
    static const ParserState STATE_USER;

    /** Ctor. */
    XmlReader(void);

    /** Dtor. */
    ~XmlReader(void);

    /**
     * Answer the base tag of the xml file. The returned string should be
     * considdered utf8.
     *
     * @return The name of the base tag of the xml file.
     */
    inline const MMXML_STRING& BaseTag(void) const {
        return this->baseTag;
    }

    /**
     * Answer the base tag attributes of the xml file. All strings should
     * be considdered utf8.
     *
     * @return All attributes of the base tag of the xml file.
     */
    inline const vislib::Array<XmlAttribute>& BaseTagAttributes(void) const {
        return this->baseTagAttribs;
    }

    /**
     * Closes the input file
     */
    void CloseFile(void);

    /**
     * Answers whether or not the input file is open.
     *
     * @return 'true' if the input file is open, 'false' otherwise.
     */
    inline bool IsFileOpen(void) const {
        return (this->inputFile == NULL) ? false : this->inputFile->IsOpen();
    }

    /**
     * Opens an file.
     *
     * @param filename The path to the file to open.
     *
     * @return 'true' on success, 'false' on failure.
     */
    inline bool OpenFile(const vislib::StringA& filename) {
        return this->OpenFile(filename.PeekBuffer());
    }

    /**
     * Opens an file.
     *
     * @param filename The path to the file to open.
     *
     * @return 'true' on success, 'false' on failure.
     */
    inline bool OpenFile(const vislib::StringW& filename) {
        return this->OpenFile(filename.PeekBuffer());
    }

    /**
     * Opens an file.
     *
     * @param filename The path to the file to open.
     *
     * @return 'true' on success, 'false' on failure.
     */
    inline bool OpenFile(const char* filename) {
        vislib::sys::File* f = new vislib::sys::File();
        if (!f->Open(
                filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
            f->Close();
            delete f;
            return false;
        }
        try {
            this->inputPath = vislib::sys::Path::GetDirectoryName(filename);
            return this->SetFile(f);
        } catch (...) {
            delete f;
            throw;
        }
    }

    /**
     * Opens an file.
     *
     * @param filename The path to the file to open.
     *
     * @return 'true' on success, 'false' on failure.
     */
    inline bool OpenFile(const wchar_t* filename) {
        vislib::sys::File* f = new vislib::sys::File();
        if (!f->Open(
                filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
            f->Close();
            delete f;
            return false;
        }
        try {
            this->inputPath = vislib::sys::Path::GetDirectoryName(filename);
            return this->SetFile(f);
        } catch (...) {
            delete f;
            throw;
        }
    }

    /**
     * Sets the input file to the given file object. This object takes
     * ownership of the file object (the file will be closed and the object
     * will be deleted by this object).
     *
     * This method also parses the main tag of the xml file. If this fails,
     * 'false' is returned but the object is destroid!
     *
     * The file object will always be destroid if the function fails with
     * the return value 'false'. If the function fails with an exception,
     * the file object will not be destroyed and the ownership of the it is
     * returned to the callee.
     *
     * @param file Pointer to the new input file object.
     *             Must not be 'NULL'.
     *
     * @return 'true' if the inputfile was successfully set to 'file',
     *         'false' otherwise.
     *
     * @throws vislib::IllegalParamException if 'file' is 'NULL' or not
     *         opened.
     */
    bool SetFile(vislib::sys::File* file);

    vislib::StringA GetPath() const {
        return this->inputPath;
    }

    /**
     * Answer the line within the xml file, if possible.
     *
     * @return The line within the xml file, if possible, otherwise 0.
     */
    SIZE_T XmlLine(void) const;

protected:
    /**
     * Stops a running parser, if there is one. The method should only be
     * called from within the tag callback methods.
     */
    void StopParsing(void);

private:
    /** declaration of xml parser callbacks */
    static void XMLCALL xmlStartTag(void*, const XML_Char*, const XML_Char**);
    static void XMLCALL xmlEndTag(void*, const XML_Char*);
    static void XMLCALL xmlCharData(void*, const XML_Char*, int);
    static void XMLCALL xmlComment(void*, const XML_Char*);

    /**
     * Callback handler for the character data.
     *
     * @param text Pointer to the character data. Note that this is not
     *             zero-terminated.
     * @param len Length of the character data.
     */
    void charData(const XML_Char* text, int len);

    /**
     * Callback handler for comment data.
     *
     * @param text Pointer to the zero-terminated comment string.
     */
    void comment(const XML_Char* text);

    /**
     * Callback handler for the ending xml tags.
     *
     * @param tag The name of the tag.
     */
    void endTag(const XML_Char* tag);

    /**
     * Parses the xml structure.
     *
     * @return 'true' on success, 'false' on any error.
     */
    bool parse(XmlParser* parser);

    /**
     * Parses the xml structure. The behaviour depends on the value of
     * 'readerState'.
     *
     * @return 'true' on success, 'false' on any error.
     */
    bool parseXml(void);

    /**
     * Callback handler for starting xml tags
     *
     * @param tag The name of the tag.
     * @param attributes The attributes of the tag.
     */
    void startTag(const XML_Char* tag, const XML_Char** attributes);

    /**
     * Answers the position within the xml file if possible.
     *
     * @return The position within the xml file, or an empty string.
     */
    vislib::StringA xmlPosition(void) const;

    /** The input file */
    vislib::sys::File* inputFile;

    vislib::StringA inputPath;

    /** The state of the reader state machine */
    unsigned int readerState;

    /** The stack of reader states of the end tags */
    vislib::Stack<unsigned int> endTagStates;

    /** Pointer to the internal expat parser */
    void* internalParser;

    /** Pointer to the external parser calling this */
    XmlParser* externalParser;

    /** The name string of the base tag */
    MMXML_STRING baseTag;

    /** The list of attributes of the base tag */
    vislib::Array<XmlAttribute> baseTagAttribs;
};

} /* end namespace xml */
} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_XMLREADER_H_INCLUDED */
