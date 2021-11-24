/*
 * ProjectParser.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PROJECTPARSER_INCLUDED
#define MEGAMOLCORE_PROJECTPARSER_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CoreInstance.h"
#include "mmcore/JobDescription.h"
#include "mmcore/ViewDescription.h"
#include "mmcore/utility/xml/ConditionalParser.h"
#include "mmcore/utility/xml/XmlReader.h"
#include "vislib/Exception.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/types.h"
#include <memory>

namespace megamol {
namespace core {
namespace utility {


/**
 * Class for the MegaMol project xml parser.
 */
class ProjectParser : public xml::ConditionalParser {
public:
    /**
     * Ctor.
     *
     * @param core The core instance to add the project elements to
     */
    ProjectParser(CoreInstance* coreInst);

    /** Dtor. */
    virtual ~ProjectParser(void);

    /**
     * Gets the list of view descriptions parsed from the project
     *
     * @return The list of view descriptions parsed from the project
     */
    inline std::shared_ptr<ViewDescription> PopViewDescription(void) {
        if (this->viewDescs.IsEmpty())
            return NULL;
        std::shared_ptr<ViewDescription> rv = this->viewDescs.First();
        this->viewDescs.RemoveFirst();
        return rv;
    }

    /**
     * Gets the list of job descriptions parsed from the project
     *
     * @return The list of job descriptions parsed from the project
     */
    inline std::shared_ptr<JobDescription> PopJobDescription(void) {
        if (this->jobDescs.IsEmpty())
            return NULL;
        std::shared_ptr<JobDescription> rv = this->jobDescs.First();
        this->jobDescs.RemoveFirst();
        return rv;
    }

protected:
    /**
     * Checks whether the xml file is a 'project', '1.0' MegaMol xml file.
     *
     * @param reader The current xml reader.
     *
     * @return 'true' if the file of the reader is compatible with this
     *         parser, 'false' otherwise.
     */
    virtual bool CheckBaseTag(const xml::XmlReader& reader);

    /**
     * Callback method for starting xml tags.
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
        xml::XmlReader::ParserState state, xml::XmlReader::ParserState& outChildState,
        xml::XmlReader::ParserState& outEndTagState, xml::XmlReader::ParserState& outPostEndTagState);

    /**
     * Callback method for the ending xml tags.
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
    virtual bool EndTag(unsigned int num, unsigned int level, const XML_Char* name, xml::XmlReader::ParserState state,
        xml::XmlReader::ParserState& outPostEndTagState);

private:
    CoreInstance* core;

    /** The view description which is currently created by the parser */
    std::shared_ptr<ViewDescription> vd;

    /** The view description which is currently created by the parser */
    std::shared_ptr<JobDescription> jd;

    /** The name of the currently created module */
    vislib::StringA modName;

    /** The view descriptions parsed from the project file */
    vislib::SingleLinkedList<std::shared_ptr<ViewDescription>> viewDescs;

    /** The job descriptions parsed from the project file */
    vislib::SingleLinkedList<std::shared_ptr<JobDescription>> jobDescs;
};


} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PROJECTPARSER_INCLUDED */
