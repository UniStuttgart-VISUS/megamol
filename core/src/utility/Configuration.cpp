/*
 * Configuration.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/Configuration.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/NoSuchElementException.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/Trace.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/sys/DirectoryIterator.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/sysfunctions.h"
#include <cstdlib>
#include <new>

#ifndef _WIN32
#include <sys/types.h>
#include <unistd.h>
#endif


/*****************************************************************************/
#if /* region ConfigValueName implementation */ 1

/*
 * megamol::core::utility::Configuration::ConfigValueName::ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::ConfigValueName(void) : name() {
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
megamol::core::utility::Configuration::ConfigValueName::ConfigValueName(const char* name) : name(name) {
    // Intentionally empty
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::ConfigValueName(const wchar_t* name) : name(name) {
    // Intentionally empty
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::ConfigValueName(const vislib::StringA& name) : name(name) {
    // Intentionally empty
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::ConfigValueName(const vislib::StringW& name) : name(name) {
    // Intentionally empty
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::~ConfigValueName
 */
megamol::core::utility::Configuration::ConfigValueName::~ConfigValueName(void) {
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
megamol::core::utility::Configuration::ConfigValueName::operator=(const char* name) {
    this->name = name;
    return *this;
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator=
 */
megamol::core::utility::Configuration::ConfigValueName&
megamol::core::utility::Configuration::ConfigValueName::operator=(const wchar_t* name) {
    this->name = name;
    return *this;
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator=
 */
megamol::core::utility::Configuration::ConfigValueName&
megamol::core::utility::Configuration::ConfigValueName::operator=(const vislib::StringA& name) {
    this->name = name;
    return *this;
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator=
 */
megamol::core::utility::Configuration::ConfigValueName&
megamol::core::utility::Configuration::ConfigValueName::operator=(const vislib::StringW& name) {
    this->name = name;
    return *this;
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator==
 */
bool megamol::core::utility::Configuration::ConfigValueName::operator==(
    const megamol::core::utility::Configuration::ConfigValueName& src) const {
    return this->name.Equals(src.name, false);
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator==
 */
bool megamol::core::utility::Configuration::ConfigValueName::operator==(const char* name) const {
    return this->name.Equals(vislib::StringW(name), false);
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator==
 */
bool megamol::core::utility::Configuration::ConfigValueName::operator==(const wchar_t* name) const {
    return this->name.Equals(name, false);
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator==
 */
bool megamol::core::utility::Configuration::ConfigValueName::operator==(const vislib::StringA& name) const {
    return this->name.Equals(vislib::StringW(name), false);
}


/*
 * megamol::core::utility::Configuration::ConfigValueName::operator==
 */
bool megamol::core::utility::Configuration::ConfigValueName::operator==(const vislib::StringW& name) const {
    return this->name.Equals(name, false);
}

#endif /* region ConfigValueName implementation */

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
        : cfgFileName()
        , criticalParserError(false)
        , appDir()
        , shaderDirs()
        , resourceDirs()
        , configValues()
        , pluginLoadInfos() {
    this->setDefaultValues();
}


/*
 * megamol::core::utility::Configuration::Configuration(void)
 */
megamol::core::utility::Configuration::Configuration(const Configuration& rhs) {
    throw ::vislib::UnsupportedOperationException("Configuration", __FILE__, __LINE__);
}


/*
 * megamol::core::utility::Configuration::operator=
 */
megamol::core::utility::Configuration& megamol::core::utility::Configuration::operator=(const Configuration& rhs) {
    if (this != &rhs) {
        throw ::vislib::UnsupportedOperationException("Configuration assignment", __FILE__, __LINE__);
    }
    return *this;
}


/*
 * megamol::core::utility::Configuration::~Configuration(void)
 */
megamol::core::utility::Configuration::~Configuration(void) {}


/*
 * megamol::core::utility::Configuration::setDefaultValues
 */
void megamol::core::utility::Configuration::setDefaultValues(void) {
    cfgFileName.Clear();
    this->appDir = vislib::sys::Path::Resolve(vislib::sys::Path::GetCurrentDirectoryW());
    this->shaderDirs.Add(this->appDir);
    this->resourceDirs.Add(this->appDir);
    this->configValues.Clear();

    // set default value for new configuration values here
}


/*
 * megamol::core::utility::Configuration::GetValue
 */
const void* megamol::core::utility::Configuration::GetValue(
    mmcConfigID id, const char* name, mmcValueType* outType) const {
    switch (id) {
    case MMC_CFGID_APPLICATION_DIR:
        if (outType != NULL) {
            *outType = MMC_TYPE_WSTR;
        }
        return this->appDir.PeekBuffer();
    case MMC_CFGID_CONFIG_FILE:
        if (outType != NULL) {
            *outType = MMC_TYPE_WSTR;
        }
        return this->cfgFileName.PeekBuffer();
    case MMC_CFGID_VARIABLE:
        if (this->IsConfigValueSet(name)) {
            const vislib::StringW& rv = this->ConfigValue(name);
            if (outType != NULL) {
                *outType = MMC_TYPE_WSTR;
            }
            return rv.PeekBuffer();
        }
        if (outType != NULL) {
            *outType = MMC_TYPE_VOIDP;
        }
        return NULL;
    default:
        if (outType != NULL) {
            *outType = MMC_TYPE_VOIDP;
        }
        return NULL;
    }

    if (outType != NULL) {
        *outType = MMC_TYPE_VOIDP;
    }
    return NULL;
}


/*
 * megamol::core::utility::Configuration::GetValue
 */
const void* megamol::core::utility::Configuration::GetValue(
    mmcConfigID id, const wchar_t* name, mmcValueType* outType) const {
    switch (id) {
    case MMC_CFGID_APPLICATION_DIR:
        if (outType != NULL) {
            *outType = MMC_TYPE_WSTR;
        }
        return this->appDir.PeekBuffer();
    case MMC_CFGID_CONFIG_FILE:
        if (outType != NULL) {
            *outType = MMC_TYPE_WSTR;
        }
        return this->cfgFileName.PeekBuffer();
    case MMC_CFGID_VARIABLE:
        if (this->IsConfigValueSet(name)) {
            const vislib::StringW& rv = this->ConfigValue(name);
            if (outType != NULL) {
                *outType = MMC_TYPE_WSTR;
            }
            return rv.PeekBuffer();
        }
        if (outType != NULL) {
            *outType = MMC_TYPE_VOIDP;
        }
        return NULL;
    default:
        if (outType != NULL) {
            *outType = MMC_TYPE_VOIDP;
        }
        return NULL;
    }

    if (outType != NULL) {
        *outType = MMC_TYPE_VOIDP;
    }
    return NULL;
}


/*
 * megamol::core::utility::Configuration::setConfigValue
 */
void megamol::core::utility::Configuration::setConfigValue(const wchar_t* name, const wchar_t* value) {
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "Configuration value \"%s\" set to \"%s\".\n", W2A(name), W2A(value));
    this->configValues[name] = value;
}


/*
 * megamol::core::utility::Configuration::IsConfigValueSet
 */
bool megamol::core::utility::Configuration::IsConfigValueSet(const char* name) const {
    const vislib::StringW* value = this->configValues.FindValue(ConfigValueName(name));
    return (value != NULL);
}


/*
 * megamol::core::utility::Configuration::IsConfigValueSet
 */
bool megamol::core::utility::Configuration::IsConfigValueSet(const vislib::StringA& name) const {
    const vislib::StringW* value = this->configValues.FindValue(ConfigValueName(name));
    return (value != NULL);
}


/*
 * megamol::core::utility::Configuration::IsConfigValueSet
 */
bool megamol::core::utility::Configuration::IsConfigValueSet(const wchar_t* name) const {
    const vislib::StringW* value = this->configValues.FindValue(ConfigValueName(name));
    return (value != NULL);
}


/*
 * megamol::core::utility::Configuration::IsConfigValueSet
 */
bool megamol::core::utility::Configuration::IsConfigValueSet(const vislib::StringW& name) const {
    const vislib::StringW* value = this->configValues.FindValue(ConfigValueName(name));
    return (value != NULL);
}


/*
 * megamol::core::utility::Configuration::ConfigValue
 */
const vislib::StringW& megamol::core::utility::Configuration::ConfigValue(const char* name) const {
    const vislib::StringW* value = this->configValues.FindValue(ConfigValueName(name));
    if (value == NULL) {
        throw vislib::NoSuchElementException("Configuration value not found", __FILE__, __LINE__);
    }
    return *value;
}


/*
 * megamol::core::utility::Configuration::ConfigValue
 */
const vislib::StringW& megamol::core::utility::Configuration::ConfigValue(const vislib::StringA& name) const {
    const vislib::StringW* value = this->configValues.FindValue(ConfigValueName(name));
    if (value == NULL) {
        throw vislib::NoSuchElementException("Configuration value not found", __FILE__, __LINE__);
    }
    return *value;
}


/*
 * megamol::core::utility::Configuration::ConfigValue
 */
const vislib::StringW& megamol::core::utility::Configuration::ConfigValue(const wchar_t* name) const {
    const vislib::StringW* value = this->configValues.FindValue(ConfigValueName(name));
    if (value == NULL) {
        throw vislib::NoSuchElementException("Configuration value not found", __FILE__, __LINE__);
    }
    return *value;
}


/*
 * megamol::core::utility::Configuration::ConfigValue
 */
const vislib::StringW& megamol::core::utility::Configuration::ConfigValue(const vislib::StringW& name) const {
    const vislib::StringW* value = this->configValues.FindValue(ConfigValueName(name));
    if (value == NULL) {
        throw vislib::NoSuchElementException("Configuration value not found", __FILE__, __LINE__);
    }
    return *value;
}


/*
 * megamol::core::utility::Configuration::ListPluginsToLoad
 * TODO: old plugin system, remove?
 */
void megamol::core::utility::Configuration::ListPluginsToLoad(vislib::SingleLinkedList<vislib::TString>& plugins) {
    plugins.Clear();

    vislib::SingleLinkedList<PluginLoadInfo>::Iterator i = this->pluginLoadInfos.GetIterator();
    while (i.HasNext()) {
        PluginLoadInfo& pli = i.Next();
        vislib::TString dir = vislib::sys::Path::Resolve(pli.directory, vislib::TString(this->appDir));
        vislib::sys::TDirectoryIterator diri(dir);
        while (diri.HasNext()) {
            vislib::sys::TDirectoryEntry& e = diri.Next();
            if (e.Type == vislib::sys::TDirectoryEntry::DIRECTORY)
                continue;
            if (!vislib::sys::FilenameGlobMatch<TCHAR>(e.Path, pli.name))
                continue;
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
void megamol::core::utility::Configuration::AddResourceDirectory(const char* dir) {
    this->resourceDirs.Add(vislib::StringW(dir));
}


/*
 * megamol::core::utility::Configuration::AddResourceDirectory
 */
void megamol::core::utility::Configuration::AddResourceDirectory(const wchar_t* dir) {
    this->resourceDirs.Add(vislib::StringW(dir));
}


/*
 * megamol::core::utility::Configuration::AddShaderDirectory
 */
void megamol::core::utility::Configuration::AddShaderDirectory(const char* dir) {
    this->shaderDirs.Add(vislib::StringW(dir));
}


/*
 * megamol::core::utility::Configuration::AddShaderDirectory
 */
void megamol::core::utility::Configuration::AddShaderDirectory(const wchar_t* dir) {
    this->shaderDirs.Add(vislib::StringW(dir));
}
