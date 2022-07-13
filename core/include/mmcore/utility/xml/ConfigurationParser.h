/*
 * ConfigurationParser.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CONFIGURATIONPARSER_INCLUDED
#define MEGAMOLCORE_CONFIGURATIONPARSER_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/utility/Configuration.h"
#include "mmcore/utility/xml/ConditionalParser.h"
#include "mmcore/utility/xml/XmlReader.h"
#include "vislib/Exception.h"
#include "vislib/VersionNumber.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace utility {
namespace xml {

/**
 * Class for the MegaMol configuration xml parser.
 */
class ConfigurationParser : public ConditionalParser {
public:
    /**
     * Helper class for redirected configuration files.
     */
    class RedirectedConfigurationException : public vislib::Exception {
    public:
        /**
         * ctor
         *
         * @param path The redirected location of the configuration file.
         */
        RedirectedConfigurationException(const wchar_t* path);

        /** copy ctor */
        RedirectedConfigurationException(const RedirectedConfigurationException& rhs);

        /** dtor */
        virtual ~RedirectedConfigurationException(void);

        /** assignment operator */
        virtual RedirectedConfigurationException& operator=(const RedirectedConfigurationException& rhs);

        /**
         * Answer the path of the redirected configuration file.
         *
         * @return The path of the redirected configuration file.
         */
        inline const wchar_t* GetRedirectedConfiguration(void) const {
            return this->GetMsgW();
        }
    };

    /** Ctor. */
    ConfigurationParser(Configuration& config);

    /** Dtor. */
    virtual ~ConfigurationParser(void);

protected:
    /**
     * Checks whether the xml file is a 'config', '1.0' MegaMol xml file.
     *
     * @param reader The current xml reader.
     *
     * @return 'true' if the file of the reader is compatible with this
     *         parser, 'false' otherwise.
     */
    virtual bool CheckBaseTag(const XmlReader& reader);

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
        XmlReader::ParserState state, XmlReader::ParserState& outChildState, XmlReader::ParserState& outEndTagState,
        XmlReader::ParserState& outPostEndTagState);

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
    virtual bool EndTag(unsigned int num, unsigned int level, const XML_Char* name, XmlReader::ParserState state,
        XmlReader::ParserState& outPostEndTagState);

    /**
     * Is called when the parsing of the xml file is completed
     */
    virtual void Completed(void);

private:
    /**
     * Parses an attribute value string to a log level or echo level. This
     * also logs a warning the the parsing was unsuccessfull.
     *
     * @param attr The attribute value string.
     * @param def The default value which will be returned if the string
     *            could not be parsed successfully.
     *
     * @return The level value.
     */
    megamol::core::utility::log::Log::log_level parseLevelAttribute(
        const XML_Char* attr, megamol::core::utility::log::Log::log_level def);

    /** to configuration to be loaded */
    Configuration& config;

    /** The active instance */
    Configuration::InstanceRequest activeInstanceRequest;

    /** The version number of the xml file */
    vislib::VersionNumber xmlVersion;

    /** The legacy base directory */
    vislib::StringW legacyBaseDir;

    /** The legacy shader directory */
    vislib::StringW legacyShaderDir;
};

} /* end namespace xml */
} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CONFIGURATIONPARSER_INCLUDED */
