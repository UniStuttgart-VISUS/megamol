/*
 * Configuration.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/utility/Configuration.h"
#include "mmcore/utility/xml/ConfigurationParser.h"
#include "mmcore/utility/xml/XmlReader.h"

#include "mmcore/api/MegaMolCore.h"

#include <cstdlib>
#include <new>
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/Trace.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/Console.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/NoSuchElementException.h"
#include "vislib/sys/DirectoryIterator.h"
#include "vislib/sys/sysfunctions.h"

#ifndef _WIN32
#include <sys/types.h>
#include <unistd.h>
#endif


/*****************************************************************************/
#if /* region ConfigValueName implementation */ 1

/*
 * megamol::core::utility::Configuration::ConfigValueName::ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::ConfigValueName(void)
        : name() {
    // Intentionally empty
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::ConfigValueName(
        const megamol::core::utility::Configuration::ConfigValueName& src) 
        : name(src.name) {
    // Intentionally empty
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::ConfigValueName(
        const char* name) : name(name) {
    // Intentionally empty
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::ConfigValueName(
        const wchar_t* name) : name(name) {
    // Intentionally empty
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::ConfigValueName(
        const vislib::StringA& name) : name(name) {
    // Intentionally empty
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::ConfigValueName(
        const vislib::StringW& name) : name(name) {
    // Intentionally empty
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::~ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::~ConfigValueName(
        void) {
    // Intentionally empty
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator=
 */
megamol::core::utility::Configuration::ConfigValueName&
megamol::core::utility::Configuration::ConfigValueName::operator=(
        const megamol::core::utility::Configuration::ConfigValueName& src) {
    this->name = src.name;
    return *this;
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator=
 */
megamol::core::utility::Configuration::ConfigValueName&
megamol::core::utility::Configuration::ConfigValueName::operator=(
        const char* name) {
    this->name = name;
    return *this;
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator=
 */
megamol::core::utility::Configuration::ConfigValueName&
megamol::core::utility::Configuration::ConfigValueName::operator=(
        const wchar_t* name) {
    this->name = name;
    return *this;
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator=
 */
megamol::core::utility::Configuration::ConfigValueName& 
megamol::core::utility::Configuration::ConfigValueName::operator=(
        const vislib::StringA& name) {
    this->name = name;
    return *this;
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator=
 */
megamol::core::utility::Configuration::ConfigValueName&
megamol::core::utility::Configuration::ConfigValueName::operator=(
        const vislib::StringW& name) {
    this->name = name;
    return *this;
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator==
 */
bool megamol::core::utility::Configuration::ConfigValueName::operator==(
        const megamol::core::utility::Configuration::ConfigValueName& src)
        const {
    return this->name.Equals(src.name, false);
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator==
 */
bool megamol::core::utility::Configuration::ConfigValueName::operator==(
        const char* name) const {
    return this->name.Equals(vislib::StringW(name), false);
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator==
 */
bool megamol::core::utility::Configuration::ConfigValueName::operator==(
        const wchar_t* name) const {
    return this->name.Equals(name, false);
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator==
 */
bool megamol::core::utility::Configuration::ConfigValueName::operator==(
        const vislib::StringA& name) const {
    return this->name.Equals(vislib::StringW(name), false);
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator==
 */
bool megamol::core::utility::Configuration::ConfigValueName::operator==(
        const vislib::StringW& name) const {
    return this->name.Equals(name, false);
}

#endif /* region ConfigValueName implementation */

/*****************************************************************************/


/*
 * megamol::core::utility::Configuration::InstanceRequest::InstanceRequest
 */
megamol::core::utility::Configuration::InstanceRequest::InstanceRequest(void)
        : ParamValueSetRequest(), id(), descname() {
    // intentionally empty
}


/*
 * megamol::core::utility::Configuration::InstanceRequest::InstanceRequest
 */
megamol::core::utility::Configuration::InstanceRequest::InstanceRequest(
        const megamol::core::utility::Configuration::InstanceRequest& src)
        : ParamValueSetRequest(), id(), descname() {
    *this = src;
}


/*
 * megamol::core::utility::Configuration::InstanceRequest::~InstanceRequest
 */
megamol::core::utility::Configuration::InstanceRequest::~InstanceRequest(void) {
    // intentionally empty
}


/*
 * megamol::core::utility::Configuration::InstanceRequest::operator=
 */
megamol::core::utility::Configuration::InstanceRequest&
megamol::core::utility::Configuration::InstanceRequest::operator=(
        const megamol::core::utility::Configuration::InstanceRequest& rhs) {
    if (&rhs == this) return *this;
    ParamValueSetRequest::operator =(rhs);
    this->id = rhs.id;
    this->descname = rhs.descname;
    return *this;
}


/*
 * megamol::core::utility::Configuration::InstanceRequest::operator==
 */
bool megamol::core::utility::Configuration::InstanceRequest::operator==(
        const megamol::core::utility::Configuration::InstanceRequest& rhs)
        const {
    return ParamValueSetRequest::operator==(rhs)
        && (this->id == rhs.id)
        && (this->descname == rhs.descname);
}

/*****************************************************************************/


/*
 * megamol::core::utility::Configuration::logLevelLocked
 */
bool megamol::core::utility::Configuration::logLevelLocked = false;


/*
 * megamol::core::utility::Configuration::logEchoLevelLocked
 */
bool megamol::core::utility::Configuration::logEchoLevelLocked = false;


/*
 * megamol::core::utility::Configuration::logFilenameLocked
 */
bool megamol::core::utility::Configuration::logFilenameLocked = false;


/*
 * megamol::core::utility::Configuration::Configuration(void)
 */
megamol::core::utility::Configuration::Configuration(void) 
        : cfgFileName(), criticalParserError(false), appDir(),
        shaderDirs(), resourceDirs(), configValues(), instanceRequests(),
        pluginLoadInfos() {
    this->setDefaultValues();
}


/*
 * megamol::core::utility::Configuration::Configuration(void)
 */
megamol::core::utility::Configuration::Configuration(
        const Configuration& rhs) {
    throw ::vislib::UnsupportedOperationException("Configuration",
        __FILE__, __LINE__);
}


/*
 * megamol::core::utility::Configuration::operator=
 */
megamol::core::utility::Configuration&
megamol::core::utility::Configuration::operator=(const Configuration& rhs) {
    if (this != &rhs) {
        throw ::vislib::UnsupportedOperationException(
            "Configuration assignment", __FILE__, __LINE__);
    }
    return *this;
}


/*
 * megamol::core::utility::Configuration::~Configuration(void)
 */
megamol::core::utility::Configuration::~Configuration(void) {
}


/*
 * megamol::core::utility::Configuration::LoadConfig
 */
void megamol::core::utility::Configuration::LoadConfig(void) {
    this->criticalParserError = false;
    this->cfgFileLocations.Clear();

    // Search 1: Environment Variable 'MEGAMOLCONFIG'
#ifdef _WIN32
#pragma warning(disable: 4996)  
#endif /* _WIN32 */
    char *envVal = ::getenv("MEGAMOLCONFIG");
#ifdef _WIN32
#pragma warning(default: 4996)  
#endif /* _WIN32 */

    if (envVal != NULL) {
        vislib::StringW searchName = vislib::sys::Path::Resolve(A2W(envVal));

        if (vislib::sys::File::Exists(searchName)) {
            if (vislib::sys::File::IsFile(searchName)) {
                this->loadConfigFromFile(searchName);
                return;
            } else if (vislib::sys::File::IsDirectory(searchName)) {
                if (this->searchConfigFile(searchName)) return;
            } else {
                // log error
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR,
                    "Configuration specified by MEGAMOLCONFIG seams to be no "
                    "File or Directory.");
            }
        } else {
            // log error
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, 
                "Configuration specified by MEGAMOLCONFIG does not exist.");
        }
    }

    vislib::StringW dir;

    // Search 2: Binary Module Directory
#ifdef _WIN32
    const DWORD nSize = 0xFFFF;
    if (::GetModuleFileNameW(NULL, dir.AllocateBuffer(nSize), nSize) == ERROR_INSUFFICIENT_BUFFER) {
        dir.Clear();
    } else {
        if (::GetLastError() == ERROR_SUCCESS) {
            dir = vislib::sys::Path::GetDirectoryName(dir);
        } else {
            dir.Clear();
        }
    }
#else
    // This is the best I got for now. Requires '/proc'
    vislib::StringA pid;
    pid.Format("/proc/%d/exe", getpid());
    vislib::StringA path;
    const SIZE_T bufSize = 0xFFFF;
    char *buf = path.AllocateBuffer(bufSize);
    ssize_t size = readlink(pid.PeekBuffer(), buf, bufSize - 1);
    if (size >= 0) {
        buf[size] = 0;
        dir = buf;
        dir = vislib::sys::Path::GetDirectoryName(dir);
    } else {
        dir.Clear();
    }
#endif
    if (!dir.IsEmpty()) {
        if (!dir.EndsWith(vislib::sys::Path::SEPARATOR_W)) {
            dir += vislib::sys::Path::SEPARATOR_W;
        }
        if (this->searchConfigFile(dir)) return;
    }

    // Search 3: User Home
    dir = vislib::sys::Path::GetUserHomeDirectoryW();
    if (!dir.EndsWith(vislib::sys::Path::SEPARATOR_W)) {
        dir += vislib::sys::Path::SEPARATOR_W;
    }
    if (this->searchConfigFile(dir)) return;

    // Search 4: User Home Subdirectory '.megamol'
    dir += L".megamol";
    if (this->searchConfigFile(dir)) return;

    // Search 5: Current Directory
    if (this->searchConfigFile(vislib::sys::Path::GetCurrentDirectoryW())) {
        return;
    }

    // no configuration file was found
    // log warning
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, 
        "No configuration file was found. Using default values.");
    this->setDefaultValues();

}


/*
 * megamol::core::utility::Configuration::LoadConfig
 */
void megamol::core::utility::Configuration::LoadConfig(
        const vislib::StringW& searchName) {
    this->criticalParserError = false;
    this->cfgFileLocations.Clear();

    vislib::StringW sName = searchName.IsEmpty() 
        ? vislib::StringW() : vislib::sys::Path::Resolve(searchName);

    if (!sName.IsEmpty()) {
        if (vislib::sys::File::Exists(sName)) {
            if (vislib::sys::File::IsFile(sName)) {
                this->loadConfigFromFile(sName);
                return;
            } else if (vislib::sys::File::IsDirectory(sName)) {
                if (this->searchConfigFile(sName)) return;
            } else {
                // log error
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR,
                    "Configuration %s seems to be no File or Directory.",
                    W2A(sName));
            }
        } else {
            // log error
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR,
                "Configuration %s does not exist.", W2A(sName));
        }
    }

    // if not found continue with default search order.
    this->LoadConfig();
}


/*
 * megamol::core::utility::Configuration::searchConfigFile
 */
bool megamol::core::utility::Configuration::searchConfigFile(
        const vislib::StringW& path) {
    if (!vislib::sys::File::IsDirectory(path)) {
        return false; // omitt not existing directories
    }

    static const wchar_t *filenames[] = {L"megamolconfig.lua", L"megamolconfig.xml", 
        L"megamol.cfg", L".megamolconfig.xml", L".megamol.cfg"};
    static const unsigned int filenameCount 
        = sizeof(filenames) / sizeof(char*);
    vislib::StringW filename;
    vislib::StringW dir = vislib::sys::Path::Resolve(path);
    if (!dir.EndsWith(vislib::sys::Path::SEPARATOR_W)) dir 
        += vislib::sys::Path::SEPARATOR_W;

    for (unsigned int i = 0; i < filenameCount; i++) {
        filename = dir;
        filename += filenames[i];

        if (vislib::sys::File::Exists(filename) 
                && vislib::sys::File::IsFile(filename)) {
            this->loadConfigFromFile(filename);
            return true;
        }
    }

    return false;
}


/*
 * megamol::core::utility::Configuration::loadConfigFromFile
 */
void megamol::core::utility::Configuration::loadConfigFromFile(
        const vislib::StringW& filename) {
    vislib::StringW redirect;
    vislib::StringW file = vislib::sys::Path::Resolve(filename);

    // check 
    vislib::SingleLinkedList<vislib::StringW>::Iterator it 
        = this->cfgFileLocations.GetIterator();
    while (it.HasNext()) {
        vislib::StringW& i = it.Next();
        if (file.Equals(i, false)) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, 
                "Cyclic configuration redirection detected. Aborting.");
            this->criticalParserError = true;
            return;
        }
    }

    this->cfgFileLocations.Append(file);

    if (file.EndsWith(L".lua")) {

        LuaState lua(this);
        int ok;
        std::string res;
        ok = lua.RunFile(file.PeekBuffer(), res);
        if (ok) {
            //vislib::sys::Log::DefaultLog.WriteInfo("Lua execution is OK and returned '%s'", res.c_str());
        } else {
            vislib::sys::Log::DefaultLog.WriteError("Lua execution is NOT OK and returned '%s'", res.c_str());
        }
        // realize configuration values
        this->cfgFileName = filename;

        if (vislib::sys::Path::IsRelative(this->appDir)) {
            this->appDir = vislib::sys::Path::Resolve(this->appDir);
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_INFO + 50,
                "Directory \"application\" resolved to \"%s\"",
                W2A(this->appDir));
        } else {
            this->appDir = vislib::sys::Path::Canonicalise(this->appDir);
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_INFO + 150,
                "Directory \"application\" is \"%s\"", W2A(this->appDir));
        }

        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "Configuration sucessfully loaded from \"%s\"",
            W2A(this->cfgFileName));

    } else {
        // XML-based config file
        // TODO: deprecate
        try {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_INFO + 10,
                "Parsing configuration file \"%s\"", W2A(filename));

            megamol::core::utility::xml::XmlReader reader;
            reader.OpenFile(filename);
            megamol::core::utility::xml::ConfigurationParser parser(*this);

            if (!parser.Parse(reader)) {
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR,
                    "Unable to parse config file \"%s\"\n", W2A(filename));
                this->criticalParserError = true;
            }

            if (parser.MessagesPresent()) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                    "Parser Messages:");
                vislib::SingleLinkedList<vislib::StringA>::Iterator msgs
                    = parser.Messages();
                while (msgs.HasNext()) {
                    vislib::sys::Log::DefaultLog.WriteMsg(
                        vislib::sys::Log::LEVEL_WARN, "    %s",
                        msgs.Next().PeekBuffer());
                }
            }

            if (this->criticalParserError) {
                return;
            }

            // realize configuration values
            this->cfgFileName = filename;

            if (vislib::sys::Path::IsRelative(this->appDir)) {
                this->appDir = vislib::sys::Path::Resolve(this->appDir);
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_INFO + 50,
                    "Directory \"application\" resolved to \"%s\"",
                    W2A(this->appDir));
            } else {
                this->appDir = vislib::sys::Path::Canonicalise(this->appDir);
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_INFO + 150,
                    "Directory \"application\" is \"%s\"", W2A(this->appDir));
            }

            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "Configuration sucessfully loaded from \"%s\"",
                W2A(this->cfgFileName));

        } catch (megamol::core::utility::xml::ConfigurationParser
            ::RedirectedConfigurationException rde) {
        // log info
            redirect = vislib::sys::Path::Resolve(
                rde.GetRedirectedConfiguration(),
                vislib::sys::Path::GetDirectoryName(file));
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_INFO + 10,
                "Configuration redirected to \"%s\"", W2A(redirect));

        } catch (std::bad_alloc ba) {
            // log error
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Memory allocation error while parsing xml configuration file");
            this->criticalParserError = true;
        } catch (vislib::Exception e) {
            // log error
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Exception while parsing xml configuration file: %s", e.GetMsgA());
            this->criticalParserError = true;
        } catch (...) {
            // log error
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Generic Error while parsing xml configuration file");
            this->criticalParserError = true;
        }

        if (!redirect.IsEmpty()) {
            // make sure we start in a clean state
            this->setDefaultValues();

            if (vislib::sys::File::Exists(redirect)) {
                if (vislib::sys::File::IsFile(redirect)) {
                    this->loadConfigFromFile(redirect);
                } else if (vislib::sys::File::IsDirectory(redirect)) {
                    if (!this->searchConfigFile(redirect)) {
                        // log error
                        vislib::sys::Log::DefaultLog.WriteMsg(
                            vislib::sys::Log::LEVEL_ERROR,
                            "No Configuration file found at redirected location.");
                        this->criticalParserError = true;
                    }
                } else {
                    // log error
                    vislib::sys::Log::DefaultLog.WriteMsg(
                        vislib::sys::Log::LEVEL_ERROR,
                        "Redirected Configuration seams to be no File or "
                        "Directory.");
                    this->criticalParserError = true;
                }
            } else {
                // log error
                vislib::sys::Log::DefaultLog.WriteMsg(
                    vislib::sys::Log::LEVEL_ERROR,
                    "Redirected Configuration not found.");
                this->criticalParserError = true;
            }
        }
    }
}


/*
 * megamol::core::utility::Configuration::setDefaultValues
 */
void megamol::core::utility::Configuration::setDefaultValues(void) {
    cfgFileName.Clear();
    this->appDir = vislib::sys::Path::Resolve(
        vislib::sys::Path::GetCurrentDirectoryW());
    this->shaderDirs.Add(this->appDir);
    this->resourceDirs.Add(this->appDir);
    this->configValues.Clear();

    // set default value for new configuration values here

}


/*
 * megamol::core::utility::Configuration::GetValue
 */
const void * megamol::core::utility::Configuration::GetValue(
        mmcConfigID id, const char *name, mmcValueType *outType) const {
    switch (id) {
        case MMC_CFGID_APPLICATION_DIR:
            if (outType != NULL) { *outType = MMC_TYPE_WSTR; }
            return this->appDir.PeekBuffer();
        case MMC_CFGID_CONFIG_FILE:
            if (outType != NULL) { *outType = MMC_TYPE_WSTR; }
            return this->cfgFileName.PeekBuffer();
        case MMC_CFGID_VARIABLE:
            if (this->IsConfigValueSet(name)) {
                const vislib::StringW& rv = this->ConfigValue(name);
                if (outType != NULL) { *outType = MMC_TYPE_WSTR; }
                return rv.PeekBuffer();
            }
            if (outType != NULL) { *outType = MMC_TYPE_VOIDP; }
            return NULL;
        default:
            if (outType != NULL) { *outType = MMC_TYPE_VOIDP; }
            return NULL;
    }

    if (outType != NULL) { *outType = MMC_TYPE_VOIDP; }
    return NULL;
}


/*
 * megamol::core::utility::Configuration::GetValue
 */
const void * megamol::core::utility::Configuration::GetValue(
        mmcConfigID id, const wchar_t *name, mmcValueType *outType) const {
    switch (id) {
        case MMC_CFGID_APPLICATION_DIR:
            if (outType != NULL) { *outType = MMC_TYPE_WSTR; }
            return this->appDir.PeekBuffer();
        case MMC_CFGID_CONFIG_FILE:
            if (outType != NULL) { *outType = MMC_TYPE_WSTR; }
            return this->cfgFileName.PeekBuffer();
        case MMC_CFGID_VARIABLE:
            if (this->IsConfigValueSet(name)) {
                const vislib::StringW& rv = this->ConfigValue(name);
                if (outType != NULL) { *outType = MMC_TYPE_WSTR; }
                return rv.PeekBuffer();
            }
            if (outType != NULL) { *outType = MMC_TYPE_VOIDP; }
            return NULL;
        default:
            if (outType != NULL) { *outType = MMC_TYPE_VOIDP; }
            return NULL;
    }

    if (outType != NULL) { *outType = MMC_TYPE_VOIDP; }
    return NULL;
}


/*
 * megamol::core::utility::Configuration::setConfigValue
 */
void megamol::core::utility::Configuration::setConfigValue(
        const wchar_t *name, const wchar_t *value) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 50, 
        "Configuration value \"%s\" set to \"%s\".\n", W2A(name), W2A(value));
    this->configValues[name] = value;
}


/*
 * megamol::core::utility::Configuration::IsConfigValueSet
 */
bool megamol::core::utility::Configuration::IsConfigValueSet(
        const char* name) const {
    const vislib::StringW *value = this->configValues.FindValue(
        ConfigValueName(name));
    return (value != NULL);
}


/*
 * megamol::core::utility::Configuration::IsConfigValueSet
 */
bool megamol::core::utility::Configuration::IsConfigValueSet(
        const vislib::StringA& name) const {
    const vislib::StringW *value = this->configValues.FindValue(
        ConfigValueName(name));
    return (value != NULL);
}


/*
 * megamol::core::utility::Configuration::IsConfigValueSet
 */
bool megamol::core::utility::Configuration::IsConfigValueSet(
        const wchar_t* name) const {
    const vislib::StringW *value = this->configValues.FindValue(
        ConfigValueName(name));
    return (value != NULL);
}


/*
 * megamol::core::utility::Configuration::IsConfigValueSet
 */
bool megamol::core::utility::Configuration::IsConfigValueSet(
        const vislib::StringW& name) const {
    const vislib::StringW *value = this->configValues.FindValue(
        ConfigValueName(name));
    return (value != NULL);
}


/*
 * megamol::core::utility::Configuration::ConfigValue
 */
const vislib::StringW& megamol::core::utility::Configuration::ConfigValue(
        const char* name) const {
    const vislib::StringW* value = this->configValues.FindValue(
        ConfigValueName(name));
    if (value == NULL) {
        throw vislib::NoSuchElementException("Configuration value not found",
            __FILE__, __LINE__);
    }
    return *value;
}


/*
 * megamol::core::utility::Configuration::ConfigValue
 */
const vislib::StringW& megamol::core::utility::Configuration::ConfigValue(
        const vislib::StringA& name) const {
    const vislib::StringW* value = this->configValues.FindValue(
        ConfigValueName(name));
    if (value == NULL) {
        throw vislib::NoSuchElementException("Configuration value not found", 
            __FILE__, __LINE__);
    }
    return *value;
}


/*
 * megamol::core::utility::Configuration::ConfigValue
 */
const vislib::StringW& megamol::core::utility::Configuration::ConfigValue(
        const wchar_t* name) const {
    const vislib::StringW* value = this->configValues.FindValue(
        ConfigValueName(name));
    if (value == NULL) {
        throw vislib::NoSuchElementException("Configuration value not found",
            __FILE__, __LINE__);
    }
    return *value;
}


/*
 * megamol::core::utility::Configuration::ConfigValue
 */
const vislib::StringW& megamol::core::utility::Configuration::ConfigValue(
        const vislib::StringW& name) const {
    const vislib::StringW* value = this->configValues.FindValue(
        ConfigValueName(name));
    if (value == NULL) {
        throw vislib::NoSuchElementException("Configuration value not found",
            __FILE__, __LINE__);
    }
    return *value;
}


/*
 * megamol::core::utility::Configuration::ListPluginsToLoad
 */
void megamol::core::utility::Configuration::ListPluginsToLoad(
        vislib::SingleLinkedList<vislib::TString> &plugins) {
    plugins.Clear();

    vislib::SingleLinkedList<PluginLoadInfo>::Iterator i = this->pluginLoadInfos.GetIterator();
    while (i.HasNext()) {
        PluginLoadInfo &pli = i.Next();
        vislib::TString dir = vislib::sys::Path::Resolve(pli.directory, vislib::TString(this->appDir));
        vislib::sys::TDirectoryIterator diri(dir);
        while (diri.HasNext()) {
            vislib::sys::TDirectoryEntry& e = diri.Next();
            if (e.Type == vislib::sys::TDirectoryEntry::DIRECTORY) continue;
            if (!vislib::sys::FilenameGlobMatch<TCHAR>(e.Path, pli.name)) continue;
            vislib::TString name = vislib::sys::Path::Resolve(e.Path, dir);

            if (pli.inc) {
                if (!plugins.Contains(name)) {
                    plugins.Add(name);
                }
            } else {
                plugins.RemoveAll(name);
            }
        }
    }

}


/*
 * megamol::core::utility::Configuration::AddResourceDirectory
 */
void megamol::core::utility::Configuration::AddResourceDirectory(const char *dir) {
    this->resourceDirs.Add(vislib::StringW(dir));
}


/*
 * megamol::core::utility::Configuration::AddResourceDirectory
 */
void megamol::core::utility::Configuration::AddResourceDirectory(const wchar_t *dir) {
    this->resourceDirs.Add(vislib::StringW(dir));
}


/*
 * megamol::core::utility::Configuration::AddShaderDirectory
 */
void megamol::core::utility::Configuration::AddShaderDirectory(const char *dir) {
    this->shaderDirs.Add(vislib::StringW(dir));
}


/*
 * megamol::core::utility::Configuration::AddShaderDirectory
 */
void megamol::core::utility::Configuration::AddShaderDirectory(const wchar_t *dir) {
    this->shaderDirs.Add(vislib::StringW(dir));
}
