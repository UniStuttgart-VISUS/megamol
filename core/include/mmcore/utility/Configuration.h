/*
 * Configuration.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CONFIGURATION_H_INCLUDED
#define MEGAMOLCORE_CONFIGURATION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/LuaState.h"
#include "mmcore/ParamValueSetRequest.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/Map.h"
#include "vislib/Pair.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"

#include <string>

/** Possible values for CONFIGURATION ID */
// TODO Moved here from deleted mmcore/api/MegaMolCore.h
typedef enum _mmcConfigID : int {
    MMC_CFGID_INVALID, // an invalid object!
    MMC_CFGID_APPLICATION_DIR,
    MMC_CFGID_CONFIG_FILE,
    MMC_CFGID_VARIABLE // a configured variable set-tag
} mmcConfigID;

/** Possible value types. */
// TODO Moved here from deleted mmcore/api/MegaMolCore.h
typedef enum _mmcValueTypeEnum : int {
    MMC_TYPE_INT32,  // 32 bit signed integer.(Pointer to!)
    MMC_TYPE_UINT32, // 32 bit unsigned integer.(Pointer to!)
    MMC_TYPE_INT64,  // 64 bit signed integer.(Pointer to!)
    MMC_TYPE_UINT64, // 64 bit unsigned integer.(Pointer to!)
    MMC_TYPE_BYTE,   // 8 bit unsigned integer.(Pointer to!)
    MMC_TYPE_BOOL,   // bool (platform specific integer size) (Pointer to!)
    MMC_TYPE_FLOAT,  // 32 bit float (Pointer to!)
    MMC_TYPE_CSTR,   // Ansi string (Pointer or Array of ansi characters).
    MMC_TYPE_WSTR,   // Unicode string (Pointer or Array of wide characters).
#if defined(UNICODE) || defined(_UNICODE)
#define MMC_TYPE_TSTR MMC_TYPE_WSTR
#else /* defined(UNICODE) || defined(_UNICODE) */
#define MMC_TYPE_TSTR MMC_TYPE_CSTR
#endif             /* defined(UNICODE) || defined(_UNICODE) */
    MMC_TYPE_VOIDP // Manuel type convertion. Use with care!
} mmcValueType;

namespace megamol {
namespace core {

/** forward declaration of the instance class */
class CoreInstance;

namespace utility {
namespace xml {

} /* end namespace xml */

/**
 * Class hold the data of the MegaMol xml configuration file
 */
class Configuration {
public:
    /** only Entry may create Configuration objects */
    friend class megamol::core::CoreInstance;

    /** LuaState is the new configuration, so it may set values */
    friend class megamol::core::LuaState;


    /**
     * Class holding an instance request for a view or a job.
     */
    class InstanceRequest : public ParamValueSetRequest {
    public:
        /**
         * Ctor.
         */
        InstanceRequest(void);

        /**
         * Copy ctor.
         *
         * @param src The object to clone from
         */
        InstanceRequest(const InstanceRequest& src);

        /**
         * Dtor.
         */
        virtual ~InstanceRequest(void);

        /**
         * Answer the description for the instance to be instantiated.
         *
         * @return The description
         */
        inline const vislib::StringA& Description(void) const {
            return this->descname;
        }

        /**
         * Answer the identifier for the instance to be instantiated.
         *
         * @return The identifier
         */
        inline const vislib::StringA& Identifier(void) const {
            return this->id;
        }

        /**
         * Sets the description for the instance to be instantiated.
         *
         * @param desc The new description
         */
        inline void SetDescription(const vislib::StringA& desc) {
            this->descname = desc;
        }

        /**
         * Sets the identifier for the instance to be instantiated.
         *
         * @param id The new identifier
         */
        inline void SetIdentifier(const vislib::StringA& id) {
            this->id = id;
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to 'this'
         */
        InstanceRequest& operator=(const InstanceRequest& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if 'this' is equal to 'rhs'
         */
        bool operator==(const InstanceRequest& rhs) const;

    private:
        /** The instance identifier */
        vislib::StringA id;

        /** The name of the description */
        vislib::StringA descname;
    };

    /** dtor */
    virtual ~Configuration(void);

    /**
     * Answer whether there was a critical error during the last load of a
     * configuration file.
     *
     * @return true if there was a critical error, false otherwise.
     */
    inline bool CriticalErrorOccurred(void) {
        return this->criticalParserError;
    }

    /**
     * Answer a configuration value. The memory the returned pointer points
     * to is only guaranteed to be valid as long as this configuration
     * object exists and until 'GetValue' is called the next time.
     *
     * @param id The id of the value to be returned.
     * @param name The name of the value to be returned. The effect of this
     *             parameter depends on the value of 'id'.
     * @param outType A variable to receive the type of the returned value.
     *
     * @return The requested value.
     */
    const void* GetValue(mmcConfigID id, const char* name, mmcValueType* outType) const;


    /**
     * Answer a configuration value. The memory the returned pointer points
     * to is only guaranteed to be valid as long as this configuration
     * object exists and until 'GetValue' is called the next time.
     *
     * @param id The id of the value to be returned.
     * @param name The name of the value to be returned. The effect of this
     *             parameter depends on the value of 'id'.
     * @param outType A variable to receive the type of the returned value.
     *
     * @return The requested value.
     */
    const void* GetValue(mmcConfigID id, const wchar_t* name, mmcValueType* outType) const;

    /**
     * Sets a configuration value.
     *
     * @param id Must be MMC_CFGID_VARIABLE
     * @param name The name of the variable to be set
     * @param val The value of the variable to be set
     *
     * @return True on success
     */
    template<class T>
    inline bool SetValue(mmcConfigID id, const T* name, const T* val) {
        if (id != MMC_CFGID_VARIABLE)
            return false;
        this->setConfigValue(name, val);
        return true;
    }

    /**
     * Answers wether a configuration value with the specified name is set.
     * The name is case insensitive.
     *
     * @param name The name of the configuration value to test.
     *
     * @return 'true' if the configuration value is set, 'false' otherwise.
     */
    bool IsConfigValueSet(const char* name) const;

    /**
     * Answers wether a configuration value with the specified name is set.
     * The name is case insensitive.
     *
     * @param name The name of the configuration value to test.
     *
     * @return 'true' if the configuration value is set, 'false' otherwise.
     */
    bool IsConfigValueSet(const vislib::StringA& name) const;

    /**
     * Answers wether a configuration value with the specified name is set.
     * The name is case insensitive.
     *
     * @param name The name of the configuration value to test.
     *
     * @return 'true' if the configuration value is set, 'false' otherwise.
     */
    bool IsConfigValueSet(const wchar_t* name) const;

    /**
     * Answers wether a configuration value with the specified name is set.
     * The name is case insensitive.
     *
     * @param name The name of the configuration value to test.
     *
     * @return 'true' if the configuration value is set, 'false' otherwise.
     */
    bool IsConfigValueSet(const vislib::StringW& name) const;

    /**
     * Answers a configuration value. The name is case insensitive.
     *
     * @param name The name of the configuration value to be returned.
     *
     * @return The value of the configuration.
     *
     * @throw vislib::NoSuchElementException if there is no configuration
     *        value with this name.
     */
    const vislib::StringW& ConfigValue(const char* name) const;

    /**
     * Answers a configuration value. The name is case insensitive.
     *
     * @param name The name of the configuration value to be returned.
     *
     * @return The value of the configuration.
     *
     * @throw vislib::NoSuchElementException if there is no configuration
     *        value with this name.
     */
    const vislib::StringW& ConfigValue(const vislib::StringA& name) const;

    /**
     * Answers a configuration value. The name is case insensitive.
     *
     * @param name The name of the configuration value to be returned.
     *
     * @return The value of the configuration.
     *
     * @throw vislib::NoSuchElementException if there is no configuration
     *        value with this name.
     */
    const vislib::StringW& ConfigValue(const wchar_t* name) const;

    /**
     * Answers a configuration value. The name is case insensitive.
     *
     * @param name The name of the configuration value to be returned.
     *
     * @return The value of the configuration.
     *
     * @throw vislib::NoSuchElementException if there is no configuration
     *        value with this name.
     */
    const vislib::StringW& ConfigValue(const vislib::StringW& name) const;

    /**
     * Adds an instantiation request.
     *
     * @param name The name of the job/view to be instantiated.
     * @param id The name of the instance to be created.
     */
    inline void AddInstantiationRequest(const InstanceRequest& req) {
        this->instanceRequests.Add(req);
    }

    /**
     * Answers whether there are pending instantiation requests.
     *
     * @return 'true' if there are pending requests
     */
    inline bool HasInstantiationRequests(void) const {
        return !this->instanceRequests.IsEmpty();
    }

    /**
     * Answers the next instantiation request.
     *
     * @return The next instantiation request.
     */
    inline InstanceRequest GetNextInstantiationRequest(void) {
        InstanceRequest rv = this->instanceRequests.First();
        this->instanceRequests.RemoveFirst();
        return rv;
    }

    /**
     * Adds information on which plugins to load.
     *
     * @param dir The directory used for the search
     * @param name The name pattern for the plug-in search
     * @param inc 'true' if found plugins should be load, 'false' if they
     *            should not be loaded.
     */
    inline void AddPluginLoadInfo(const vislib::TString& dir, const vislib::TString& name, bool inc) {
        PluginLoadInfo pli;
        this->pluginLoadInfos.Add(pli);
        this->pluginLoadInfos.Last().directory = dir;
        this->pluginLoadInfos.Last().name = name;
        this->pluginLoadInfos.Last().inc = inc;
    }

    /**
     * Lists all plug-ins to be loaded.
     *
     * @param plugins The list receiving the full paths of all plug-ins to
     *                be loaded.
     */
    void ListPluginsToLoad(vislib::SingleLinkedList<vislib::TString>& plugins);

    /**
     * Sets 'dir' as application directory
     *
     * @param dir The path to be set
     */
    void SetApplicationDirectory(const char* dir) {
        this->appDir = vislib::StringW{dir};
    }

    /**
     * Adds 'dir' as search path for resource files
     *
     * @param dir The resource path to be added
     */
    void AddResourceDirectory(const char* dir);

    /**
     * Adds 'dir' as search path for resource files
     *
     * @param dir The resource path to be added
     */
    void AddResourceDirectory(const wchar_t* dir);

    /**
     * Answer the array of resource search directories
     *
     * @return The array of resource search directories
     */
    inline const vislib::Array<vislib::StringW>& ResourceDirectories(void) const {
        return this->resourceDirs;
    }

    /**
     * Adds 'dir' as search path for shader files
     *
     * @param dir The shader path to be added
     */
    void AddShaderDirectory(const char* dir);

    /**
     * Adds 'dir' as search path for shader files
     *
     * @param dir The shader path to be added
     */
    void AddShaderDirectory(const wchar_t* dir);

    /**
     * Answer the array of shader search directories
     *
     * @return The array of shader search directories
     */
    inline const vislib::Array<vislib::StringW>& ShaderDirectories(void) const {
        return this->shaderDirs;
    }

private:
    /** ctor */
    Configuration(void);

    /** forbidden copy ctor */
    Configuration(const Configuration& rhs);

    /** forbidden assignment operator */
    Configuration& operator=(const Configuration& rhs);

    /** Sets all values to default. */
    void setDefaultValues(void);

    /** helper struct for setting log file options */
    struct LogSettings {

        /** validity flag of the log file name */
        bool logFileNameValid : 1;

        /** validity flag of the log level */
        bool logLevelValid : 1;

        /** validity flag of the echo level */
        bool echoLevelValid : 1;

        /** file name of the log file */
        std::string logFileNameValue;

        /** log level */
        megamol::core::utility::log::Log::log_level logLevelValue;

        /** echo level */
        megamol::core::utility::log::Log::log_level echoLevelValue;
    };

    /**
     * Nested helper class for config value names, providing a case
     * insensitive comparison.
     */
    class ConfigValueName {
    public:
        /** Ctor */
        ConfigValueName(void);

        /**
         * Copy ctor.
         *
         * @param src The object to clone from.
         */
        ConfigValueName(const ConfigValueName& src);

        /**
         * Ctor.
         *
         * @param name The name to store.
         */
        ConfigValueName(const char* name);

        /**
         * Ctor.
         *
         * @param name The name to store.
         */
        ConfigValueName(const wchar_t* name);

        /**
         * Ctor.
         *
         * @param name The name to store.
         */
        ConfigValueName(const vislib::StringA& name);

        /**
         * Ctor.
         *
         * @param name The name to store.
         */
        ConfigValueName(const vislib::StringW& name);

        /** Dtor. */
        ~ConfigValueName(void);

        /**
         * Assignment operator.
         *
         * @param src The right hand side operand.
         *
         * @return Reference to 'this'.
         */
        ConfigValueName& operator=(const ConfigValueName& src);

        /**
         * Assignment operator.
         *
         * @param name The right hand side operand.
         *
         * @return Reference to 'this'.
         */
        ConfigValueName& operator=(const char* name);

        /**
         * Assignment operator.
         *
         * @param name The right hand side operand.
         *
         * @return Reference to 'this'.
         */
        ConfigValueName& operator=(const wchar_t* name);

        /**
         * Assignment operator.
         *
         * @param name The right hand side operand.
         *
         * @return Reference to 'this'.
         */
        ConfigValueName& operator=(const vislib::StringA& name);

        /**
         * Assignment operator.
         *
         * @param name The right hand side operand.
         *
         * @return Reference to 'this'.
         */
        ConfigValueName& operator=(const vislib::StringW& name);

        /**
         * Test for equality. The strings are compared case insensitive.
         *
         * @param src The right hand side operand.
         *
         * @return 'true' if the names are equal, 'false' otherwise.
         */
        bool operator==(const ConfigValueName& src) const;

        /**
         * Test for equality. The strings are compared case insensitive.
         *
         * @param name The right hand side operand.
         *
         * @return 'true' if the names are equal, 'false' otherwise.
         */
        bool operator==(const char* name) const;

        /**
         * Test for equality. The strings are compared case insensitive.
         *
         * @param name The right hand side operand.
         *
         * @return 'true' if the names are equal, 'false' otherwise.
         */
        bool operator==(const wchar_t* name) const;

        /**
         * Test for equality. The strings are compared case insensitive.
         *
         * @param name The right hand side operand.
         *
         * @return 'true' if the names are equal, 'false' otherwise.
         */
        bool operator==(const vislib::StringA& name) const;

        /**
         * Test for equality. The strings are compared case insensitive.
         *
         * @param name The right hand side operand.
         *
         * @return 'true' if the names are equal, 'false' otherwise.
         */
        bool operator==(const vislib::StringW& name) const;

    private:
        /** the name string */
        vislib::StringW name;
    };

    /**
     * Utility struct holding all info about plugin searches
     */
    typedef struct _pluginloadinfo_t {

        /** The search directory */
        vislib::TString directory;

        /** The search name pattern */
        vislib::TString name;

        /** The action: include/exclude */
        bool inc;

        /**
         * Test for equality. The strings are compared case insensitive.
         *
         * @param name The right hand side operand.
         *
         * @return 'true' if the names are equal, 'false' otherwise.
         */
        inline bool operator==(const struct _pluginloadinfo_t& rhs) const {
            return this->directory.Equals(rhs.directory) && this->name.Equals(rhs.name) && (this->inc == rhs.inc);
        }

    } PluginLoadInfo;

    /**
     * Sets a configuration value. Both string parameters are UTF8 encoded.
     *
     * @param name The name of the config value.
     * @param value The value of the config value.
     */
    void setConfigValue(const char* name, const char* value) {
        this->setConfigValue(A2W(name), A2W(value));
    }

    /**
     * Sets a configuration value. Both string parameters are UTF8 encoded.
     *
     * @param name The name of the config value.
     * @param value The value of the config value.
     */
    void setConfigValue(const wchar_t* name, const wchar_t* value);

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */

    /** stores the file name of the loaded configuration. */
    vislib::StringW cfgFileName;

    /** indicates an critical error while the last configuration file was parsed. */
    bool criticalParserError;

    /** the list of configuration files. Only valid while parsing. */
    vislib::SingleLinkedList<vislib::StringW> cfgFileLocations;

    /** flag whether or not to allow setting a new log level from the configuration file */
    static bool logLevelLocked;

    /** flag whether or not to allow setting a new log level from the configuration file */
    static bool logEchoLevelLocked;

    /** flag whether or not to allow setting a new log file name from the configuration file */
    static bool logFilenameLocked;

    /** the application directory */
    vislib::StringW appDir;

    /** the shader sourcecode directories */
    vislib::Array<vislib::StringW> resourceDirs;

    /** the shader sourcecode directories */
    vislib::Array<vislib::StringW> shaderDirs;

    /** map of generic configuration values */
    vislib::Map<ConfigValueName, vislib::StringW> configValues;

    /** List of instance requests from the configuration file */
    vislib::SingleLinkedList<InstanceRequest> instanceRequests;

    /** The plugin loading informations */
    vislib::SingleLinkedList<PluginLoadInfo> pluginLoadInfos;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CONFIGURATION_H_INCLUDED */
