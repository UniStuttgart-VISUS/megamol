/*
 * ConfigurationParser.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/xml/ConfigurationParser.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/VersionNumber.h"
#include "vislib/sys/Path.h"


/*
 * using xml implementation namespaces
 */
using namespace megamol::core::utility::xml;


/*
 * ConfigurationParser::RedirectedConfigurationException
 *     ::RedirectedConfigurationException
 */
ConfigurationParser::RedirectedConfigurationException ::RedirectedConfigurationException(const wchar_t* path)
        : vislib::Exception(path, "", 0) {}


/*
 * ConfigurationParser::RedirectedConfigurationException
 *     ::RedirectedConfigurationException
 */
ConfigurationParser::RedirectedConfigurationException ::RedirectedConfigurationException(
    const RedirectedConfigurationException& rhs)
        : vislib::Exception(rhs) {}


/*
 * ConfigurationParser::RedirectedConfigurationException
 *     ::~RedirectedConfigurationException
 */
ConfigurationParser::RedirectedConfigurationException ::~RedirectedConfigurationException(void) {}


/*
 * ConfigurationParser::RedirectedConfigurationException::operator=
 */
ConfigurationParser::RedirectedConfigurationException& ConfigurationParser::RedirectedConfigurationException::operator=(
    const RedirectedConfigurationException& rhs) {
    vislib::Exception::operator=(rhs);
    return *this;
}


/*
 * ConfigurationParser::ConfigurationParser
 */
ConfigurationParser::ConfigurationParser(megamol::core::utility::Configuration& config)
        : ConditionalParser()
        , config(config)
        , activeInstanceRequest()
        , xmlVersion()
        , legacyBaseDir()
        , legacyShaderDir() {}


/*
 * ConfigurationParser::~ConfigurationParser
 */
ConfigurationParser::~ConfigurationParser(void) {}


/*
 * ConfigurationParser::CheckBaseTag
 */
bool ConfigurationParser::CheckBaseTag(const XmlReader& reader) {
    if (!reader.BaseTag().Equals(MMXML_STRING("MegaMol"), false)) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(1, "Config file does not specify <MegaMol> as base tag");
        return false;
    }

    bool typeValid = false;
    bool versionValid = false;

    const vislib::Array<XmlReader::XmlAttribute>& attrib = reader.BaseTagAttributes();
    for (unsigned int i = 0; i < attrib.Count(); i++) {
        const XmlReader::XmlAttribute& attr = attrib[i];

        if (attr.Key().Equals(MMXML_STRING("type"), false)) {
            if (attr.Value().Equals(MMXML_STRING("config"))) {
                typeValid = true;
            }

        } else if (attr.Key().Equals(MMXML_STRING("version"), false)) {
            if (this->xmlVersion.Parse(attr.Value()) >= 1) {
                if (this->xmlVersion < vislib::VersionNumber(1, 0)) {
                    versionValid = false; // pre 1.0 does not exist!
                } else if (this->xmlVersion < vislib::VersionNumber(1, 1)) {
                    versionValid = true; // 1.0.x.x
                    this->setConditionalParserVersion(0);
                    this->legacyBaseDir = vislib::sys::Path::GetCurrentDirectoryW();
                    this->legacyShaderDir.Clear();
                } else if (this->xmlVersion < vislib::VersionNumber(1, 2)) {
                    versionValid = true; // 1.1.x.x
                    this->setConditionalParserVersion(1);
                    this->legacyBaseDir = vislib::sys::Path::GetCurrentDirectoryW();
                    this->legacyShaderDir.Clear();
                } else if (this->xmlVersion < vislib::VersionNumber(1, 3)) {
                    versionValid = true; // 1.2.x.x
                    this->setConditionalParserVersion(1);
                    this->config.shaderDirs.Clear();
                    this->legacyBaseDir.Clear();
                    this->legacyShaderDir.Clear();
                } else {
                    versionValid = false; // >= 1.2 does not exist yet!
                }
            }
        } else {
            this->Warning(vislib::StringA("Ignoring unexpected base tag attribute ") + vislib::StringA(attr.Key()));
        }
    }

    if (!typeValid) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(1, "base tag attribute \"type\" not present or invalid.");
        return false;
    }
    if (!versionValid) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            1, "base tag attribute \"version\" not present or invalid.");
        return false;
    }
    if (this->xmlVersion < vislib::VersionNumber(1, 2)) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_WARN,
            "config file version %s marked deprecated. Upgrade to config file version 1.2",
            this->xmlVersion.ToStringA().PeekBuffer());
    }

    return typeValid && versionValid;
}


/*
 * ConfigurationParser::StartTag
 */
bool ConfigurationParser::StartTag(unsigned int num, unsigned int level, const XML_Char* name, const XML_Char** attrib,
    XmlReader::ParserState state, XmlReader::ParserState& outChildState, XmlReader::ParserState& outEndTagState,
    XmlReader::ParserState& outPostEndTagState) {

    if (ConditionalParser::StartTag(
            num, level, name, attrib, state, outChildState, outEndTagState, outPostEndTagState)) {
        return true; // handled by base class
    }

    if (state == XmlReader::STATE_USER + 1) {
        if (MMXML_STRING("param").Equals(name)) {
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
                this->Error("Tag \"param\" without \"name\" ignored.");
            } else if (value == NULL) {
                this->Error("Tag \"param\" without \"value\" ignored.");
            } else {
                this->activeInstanceRequest.AddParamValue(vislib::StringA(name), vislib::TString(value));
            }
            return true;
        }

        return false;
    }

    if (MMXML_STRING("redirect").Equals(name, false)) {
        const XML_Char* red = NULL;
        vislib::StringW redirection;

        // check attributs
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("target").Equals(attrib[i])) {
                red = attrib[i + 1]; // redirect

            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }

        if (red) {
            redirection = vislib::StringW(red);
            //if (!vislib::UTF8Encoder::Decode(redirection, red)) {
            //    redirection = A2W(red);
            //}
            throw RedirectedConfigurationException(redirection.PeekBuffer());

        } else {
            this->FatalError("Redirection tag without \"target\" attribute ignored.");
        }
        return true;
    }

    if (this->xmlVersion < vislib::VersionNumber(1, 2)) {
        if (MMXML_STRING("directory").Equals(name, false)) {

            unsigned int dirname = 0;
            const XML_Char* path = NULL;
            const char* pathA = NULL;
            for (int i = 0; attrib[i]; i += 2) {
                if (MMXML_STRING("name").Equals(attrib[i])) {
                    if (MMXML_STRING("base").Equals(attrib[i + 1], false)) {
                        dirname = 2;
                    } else if (MMXML_STRING("application").Equals(attrib[i + 1], false)) {
                        dirname = 3;
                    } else if (MMXML_STRING("shader").Equals(attrib[i + 1], false)) {
                        dirname = 4;
                    } else {
                        dirname = 1;
                    }
                } else if (MMXML_STRING("path").Equals(attrib[i])) {
                    path = attrib[i + 1];
                } else {
                    this->WarnUnexpectedAttribut(name, attrib[i]);
                }
            }

            if (dirname == 0) {
                this->Error("Tag \"directory\" without \"name\" ignored.");
            } else if (dirname == 1) {
                this->Error("Tag \"directory\" with unknown \"name\" ignored.");
            } else if (path == NULL) {
                this->Error("Tag \"directory\" without \"path\" ignored.");
            } else {
                vislib::StringW pathW(path);
                //if (!vislib::UTF8Encoder::Decode(pathW, path)) {
                //    pathW = A2W(path);
                //}
                switch (dirname) {
                case 2:
                    this->legacyBaseDir = pathW;
                    pathA = "base";
                    break;
                case 3:
                    this->config.appDir = pathW;
                    pathA = "application";
                    break;
                case 4:
                    this->legacyShaderDir = pathW;
                    pathA = "shader";
                    break;
                default:
                    this->FatalError("Internal Error while parsing directory tag.");
                    break;
                }
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 50,
                    "Directory \"%s\" set to \"%s\"", pathA, W2A(pathW));
            }
            return true;
        }

    } else if (this->xmlVersion < vislib::VersionNumber(1, 3)) {
        if (MMXML_STRING("appdir").Equals(name, false)) {
            const XML_Char* path = NULL;
            for (int i = 0; attrib[i]; i += 2) {
                if (MMXML_STRING("path").Equals(attrib[i])) {
                    path = attrib[i + 1];
                } else {
                    this->WarnUnexpectedAttribut(name, attrib[i]);
                }
            }
            if (path != NULL) {
                this->config.appDir = vislib::StringW(path);
            } else {
                this->Warning("\"appdir\" tag without \"path\" attribute ignored.\n");
            }
            return true;
        }
        if (MMXML_STRING("shaderdir").Equals(name, false)) {
            const XML_Char* path = NULL;
            for (int i = 0; attrib[i]; i += 2) {
                if (MMXML_STRING("path").Equals(attrib[i])) {
                    path = attrib[i + 1];
                } else {
                    this->WarnUnexpectedAttribut(name, attrib[i]);
                }
            }
            if (path != NULL) {
                this->config.AddShaderDirectory(path);
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 50,
                    "Path \"%s\" added as shader search path.", vislib::StringA(path).PeekBuffer());

            } else {
                this->Warning("\"shaderdir\" tag without \"path\" attribute ignored.\n");
            }
            return true;
        }
        if (MMXML_STRING("resourcedir").Equals(name, false)) {
            const XML_Char* path = NULL;
            for (int i = 0; attrib[i]; i += 2) {
                if (MMXML_STRING("path").Equals(attrib[i])) {
                    path = attrib[i + 1];
                } else {
                    this->WarnUnexpectedAttribut(name, attrib[i]);
                }
            }
            if (path != NULL) {
                this->config.AddResourceDirectory(path);
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 50,
                    "Path \"%s\" added as resource search path.", vislib::StringA(path).PeekBuffer());

            } else {
                this->Warning("\"resourcedir\" tag without \"path\" attribute ignored.\n");
            }
            return true;
        }
    }


    if (MMXML_STRING("plugin").Equals(name, false)) {
        const XML_Char* path = NULL;
        const XML_Char* name = NULL;
        bool inc = true;
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("path").Equals(attrib[i])) {
                path = attrib[i + 1];
            } else if (MMXML_STRING("name").Equals(attrib[i])) {
                name = attrib[i + 1];
            } else if (MMXML_STRING("action").Equals(attrib[i])) {
                if (MMXML_STRING("include").Equals(attrib[i + 1])) {
                    inc = true;
                } else if (MMXML_STRING("exclude").Equals(attrib[i + 1])) {
                    inc = false;
                } else {
                    try {
                        inc = vislib::CharTraits<XML_Char>::ParseBool(attrib[i + 1]);
                    } catch (...) {
                        vislib::StringA a(attrib[i + 1]);
                        vislib::StringA b;
                        b.Format("\"action\" attribute with illegal value \"%s\" interpreted as \"include\".",
                            a.PeekBuffer());
                        this->Warning(b);
                    }
                }
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }

        if ((path != NULL) && (name != NULL)) {
            config.AddPluginLoadInfo(vislib::TString(path), vislib::TString(name), inc);
        } else {
            this->Warning("\"plugin\" tag without \"name\" and \"path\" attribute ignored.\n");
        }
        return true;
    }

    if (MMXML_STRING("log").Equals(name, false)) {
        megamol::core::utility::Configuration::LogSettings* logSettings /* = NULL;
         logSettings */
            = new struct megamol::core::utility::Configuration::LogSettings();
        logSettings->logFileNameValid = false;
        logSettings->logLevelValid = false;
        logSettings->echoLevelValid = false;
        logSettings->logFileNameValue.clear();
        logSettings->logLevelValue = megamol::core::utility::log::Log::log_level::error;
        logSettings->echoLevelValue = megamol::core::utility::log::Log::log_level::error;

        // check attributs
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("file").Equals(attrib[i])) {
                if (!megamol::core::utility::Configuration::logFilenameLocked) {
                    // only set if there is a slight chance that we are
                    // allowed to set the value
                    logSettings->logFileNameValue = vislib::sys::Path::Resolve(attrib[i + 1]);
                    logSettings->logFileNameValid = true;
                    //if (!vislib::UTF8Encoder::Decode(
                    //        logSettings->logFileNameValue, attrib[i + 1])) {
                    //    logSettings->logFileNameValue = A2W(attrib[i + 1]);
                    //} else {
                    //    logSettings->logFileNameValid = true;
                    //    logSettings->logFileNameValue =
                    //        vislib::sys::Path::Resolve(
                    //        logSettings->logFileNameValue);
                    //}
                }
            } else if (MMXML_STRING("level").Equals(attrib[i])) {
                if (!megamol::core::utility::Configuration::logLevelLocked) {
                    // only parse if there is a slight chance that we are
                    // allowed to set the value
                    logSettings->logLevelValid = true;
                    logSettings->logLevelValue =
                        this->parseLevelAttribute(attrib[i + 1], megamol::core::utility::log::Log::log_level::error);
                }

            } else if (MMXML_STRING("echolevel").Equals(attrib[i])) {
                if (!megamol::core::utility::Configuration::
                        logEchoLevelLocked) { // only parse if there is a slight chance that we are
                    // allowed to set the value
                    logSettings->echoLevelValid = true;
                    logSettings->echoLevelValue =
                        this->parseLevelAttribute(attrib[i + 1], megamol::core::utility::log::Log::log_level::error);
                }
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }

        if (logSettings) {
            if ((!megamol::core::utility::Configuration::logEchoLevelLocked) && (logSettings->echoLevelValid)) {
                megamol::core::utility::log::Log::DefaultLog.SetEchoLevel(logSettings->echoLevelValue);
                //this->config.instanceLog->EchoOfflineMessages(true);
            }
            if ((!megamol::core::utility::Configuration::logLevelLocked) && (logSettings->logLevelValid)) {
                megamol::core::utility::log::Log::DefaultLog.SetLevel(logSettings->logLevelValue);
            }
            if ((!megamol::core::utility::Configuration::logFilenameLocked) && (logSettings->logFileNameValid)) {
                megamol::core::utility::log::Log::DefaultLog.AddFileTarget(
                    logSettings->logFileNameValue.c_str(), false);
            }
        }
        delete logSettings;

        return true;
    }

    if (MMXML_STRING("set").Equals(name, false)) {
        // general tag to set a configuration value
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
            this->Error("Tag \"set\" without \"name\" ignored.");
        } else if (value == NULL) {
            this->Error("Tag \"set\" without \"value\" ignored.");
        } else {
            // valid set tag
            this->config.setConfigValue(name, value);
        }
        return true;
    }

    if (MMXML_STRING("instance").Equals(name)) {
        // general tag to set a configuration value
        const XML_Char* name = NULL;
        const XML_Char* id = NULL;

        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("name").Equals(attrib[i])) {
                name = attrib[i + 1];
            } else if (MMXML_STRING("id").Equals(attrib[i])) {
                id = attrib[i + 1];
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }

        if (name == NULL) {
            this->Error("Tag \"instance\" without \"name\" ignored.");
            outChildState = XmlReader::STATE_IGNORE_SUBTREE;
        } else if (id == NULL) {
            this->Error("Tag \"instance\" without \"id\" ignored.");
            outChildState = XmlReader::STATE_IGNORE_SUBTREE;
        } else {
            // valid set tag
            this->activeInstanceRequest.SetIdentifier(vislib::StringA(id));
            this->activeInstanceRequest.SetDescription(vislib::StringA(name));
            this->activeInstanceRequest.ClearParamValues();
            outChildState = XmlReader::STATE_USER + 1;
        }

        return true;
    }

    return false; // unhandled.
}


/*
 * ConfigurationParser::EndTag
 */
bool ConfigurationParser::EndTag(unsigned int num, unsigned int level, const XML_Char* name,
    XmlReader::ParserState state, XmlReader::ParserState& outPostEndTagState) {
    if (ConditionalParser::EndTag(num, level, name, state, outPostEndTagState)) {
        return true; // handled by base class
    }

    if (MMXML_STRING("redirect").Equals(name, false) || MMXML_STRING("plugin").Equals(name, false) ||
        MMXML_STRING("log").Equals(name, false) || MMXML_STRING("set").Equals(name, false) ||
        MMXML_STRING("param").Equals(name)) {
        return true;
    }
    if (this->xmlVersion < vislib::VersionNumber(1, 2)) {
        if (MMXML_STRING("directory").Equals(name, false)) {
            return true;
        }
    } else if (this->xmlVersion < vislib::VersionNumber(1, 3)) {
        if (MMXML_STRING("appdir").Equals(name, false) || MMXML_STRING("shaderdir").Equals(name, false) ||
            MMXML_STRING("resourcedir").Equals(name, false)) {
            return true;
        }
    }

    if (MMXML_STRING("instance").Equals(name)) {
        if ((!this->activeInstanceRequest.Identifier().IsEmpty()) &&
            (!this->activeInstanceRequest.Description().IsEmpty())) {
            this->config.AddInstantiationRequest(this->activeInstanceRequest);
            this->activeInstanceRequest.SetIdentifier(vislib::StringA::EMPTY);
            this->activeInstanceRequest.SetDescription(vislib::StringA::EMPTY);
            this->activeInstanceRequest.ClearParamValues();
        }
        return true;
    }

    return false; // unhandled.
}


/*
 * ConfigurationParser::Completed
 */
void ConfigurationParser::Completed(void) {

    if (this->xmlVersion < vislib::VersionNumber(1, 2)) {
        // legacy config file was parsed!

        // make app path absolute
        if (vislib::sys::Path::IsRelative(this->config.appDir)) {
            this->config.appDir = vislib::sys::Path::Resolve(this->config.appDir, this->legacyBaseDir);
        }

        // make plugin paths absolute
        vislib::SingleLinkedList<Configuration::PluginLoadInfo>::Iterator iter =
            this->config.pluginLoadInfos.GetIterator();
        while (iter.HasNext()) {
            Configuration::PluginLoadInfo& info = iter.Next();
            if (vislib::sys::Path::IsRelative(info.directory)) {
                info.directory = vislib::sys::Path::Resolve(info.directory, W2T(this->legacyBaseDir));
            }
        }

        // add shader paths
        this->config.shaderDirs.Clear();
        this->config.AddShaderDirectory(vislib::sys::Path::Resolve(this->legacyShaderDir, this->legacyBaseDir));

    } else if (this->xmlVersion < vislib::VersionNumber(1, 3)) {

        // make app path absolute
        if (vislib::sys::Path::IsRelative(this->config.appDir)) {
            this->config.appDir = vislib::sys::Path::Resolve(
                this->config.appDir, vislib::sys::Path::GetDirectoryName(this->config.cfgFileLocations.First()));

            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "AppDir resolved to \"%s\"", vislib::StringA(this->config.appDir).PeekBuffer());
        }

        // make plugin paths absolute
        vislib::SingleLinkedList<Configuration::PluginLoadInfo>::Iterator iter =
            this->config.pluginLoadInfos.GetIterator();
        while (iter.HasNext()) {
            Configuration::PluginLoadInfo& info = iter.Next();
            if (vislib::sys::Path::IsRelative(info.directory)) {
                info.directory = vislib::sys::Path::Resolve(info.directory, W2T(this->config.appDir));
            }
        }

        // make shader paths absolute
        if (this->config.shaderDirs.Count() == 0) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                megamol::core::utility::log::Log::LEVEL_INFO, "No shader directories configured");

        } else
            for (SIZE_T i = 0; i < this->config.shaderDirs.Count(); i++) {
                if (vislib::sys::Path::IsRelative(this->config.shaderDirs[i])) {
                    this->config.shaderDirs[i] =
                        vislib::sys::Path::Resolve(this->config.shaderDirs[i], this->config.appDir);
                    if (!vislib::sys::File::IsDirectory(this->config.shaderDirs[i])) {
                        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                            megamol::core::utility::log::Log::LEVEL_WARN,
                            "Configured shader directory \"%s\" does not exist",
                            vislib::StringA(this->config.shaderDirs[i]).PeekBuffer());
                    }
                }
            }

        // make resource paths absolute
        if (this->config.resourceDirs.Count() == 0) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                megamol::core::utility::log::Log::LEVEL_INFO, "No resource directories configured");

        } else
            for (SIZE_T i = 0; i < this->config.resourceDirs.Count(); i++) {
                if (vislib::sys::Path::IsRelative(this->config.resourceDirs[i])) {
                    this->config.resourceDirs[i] =
                        vislib::sys::Path::Resolve(this->config.resourceDirs[i], this->config.appDir);
                    if (!vislib::sys::File::IsDirectory(this->config.resourceDirs[i])) {
                        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                            megamol::core::utility::log::Log::LEVEL_WARN,
                            "Configured resource directory \"%s\" does not exist",
                            vislib::StringA(this->config.resourceDirs[i]).PeekBuffer());
                    }
                }
            }
    }
}


/*
 * ConfigurationParser::parseLevelAttribute
 */
megamol::core::utility::log::Log::log_level ConfigurationParser::parseLevelAttribute(
    const XML_Char* attr, megamol::core::utility::log::Log::log_level def) {
    megamol::core::utility::log::Log::log_level retval = def;
    if (MMXML_STRING("error").Equals(attr, false)) {
        retval = megamol::core::utility::log::Log::log_level::error;
    } else if (MMXML_STRING("warn").Equals(attr, false)) {
        retval = megamol::core::utility::log::Log::log_level::warn;
    } else if (MMXML_STRING("warning").Equals(attr, false)) {
        retval = megamol::core::utility::log::Log::log_level::warn;
    } else if (MMXML_STRING("info").Equals(attr, false)) {
        retval = megamol::core::utility::log::Log::log_level::info;
    } else if (MMXML_STRING("none").Equals(attr, false)) {
        retval = megamol::core::utility::log::Log::log_level::none;
    } else if (MMXML_STRING("null").Equals(attr, false)) {
        retval = megamol::core::utility::log::Log::log_level::none;
    } else if (MMXML_STRING("zero").Equals(attr, false)) {
        retval = megamol::core::utility::log::Log::log_level::none;
    } else if (MMXML_STRING("all").Equals(attr, false)) {
        retval = megamol::core::utility::log::Log::log_level::all;
    } else if (MMXML_STRING("*").Equals(attr, false)) {
        retval = megamol::core::utility::log::Log::log_level::all;
    } /*else {
        try {
            retval = vislib::CharTraits<XML_Char>::ParseInt(attr);
        } catch (...) { retval = def; }
    }*/
    return retval;
}
