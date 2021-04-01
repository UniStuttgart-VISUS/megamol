
#include "CLIConfigParsing.h"

#include <cxxopts.hpp>

// Filesystem
#if defined(_HAS_CXX17) || ((defined(_MSC_VER) && (_MSC_VER > 1916))) // C++2017 or since VS2019
#include <filesystem>
namespace stdfs = std::filesystem;
#else
// WINDOWS
#ifdef _WIN32
#include <filesystem>
namespace stdfs = std::experimental::filesystem;
#else
// LINUX
#include <experimental/filesystem>
namespace stdfs = std::experimental::filesystem;
#endif
#endif

// find user home
#include <stdlib.h>
#include <stdio.h>
static std::string getHomeDir() {
#ifdef _WIN32
    return std::string(getenv("HOMEDRIVE")) + std::string(getenv("HOMEPATH"));
#else // LINUX
    return std::string(getenv("HOME"));
#endif
}

using megamol::frontend_resources::RuntimeConfig;
using megamol::frontend_resources::GlobalValueStore;

// called by main and returns config struct filled with parsed CLI values
std::pair<RuntimeConfig, GlobalValueStore> megamol::frontend::handle_cli_and_config(const int argc, const char** argv, megamol::core::LuaAPI& lua) {
    RuntimeConfig config;

    // config files are already checked to exist in file system
    config.configuration_files = extract_config_file_paths(argc, argv);

    // overwrite default values with values from config file
    config = handle_config(config, lua);

    // overwrite default and config values with CLI inputs
    config = handle_cli(config, argc, argv);

    GlobalValueStore global_value_store;
    for (auto& pair : config.global_values) {
        global_value_store.insert(pair.first, pair.second);
    }

    return {config, global_value_store};
}

static void exit(std::string const& reason) {
    std::cout << "Error: " << reason << std::endl;
    std::cout << "Shut down MegaMol" << std::endl;
    std::exit(1);
}

// only for --config pass
static std::string config_option      = "c,config";

// basic options
static std::string appdir_option      = "appdir";
static std::string resourcedir_option = "resourcedir";
static std::string shaderdir_option   = "shaderdir";
static std::string logfile_option     = "logfile";
static std::string loglevel_option    = "loglevel";
static std::string echolevel_option   = "echolevel";
static std::string project_option     = "p,project";
static std::string global_option      = "g,global";

// service-specific options
// --project and loose project files are both valid ways to provide lua project files
static std::string project_files_option = "project-files";
static std::string host_option          = "host";
static std::string khrdebug_option      = "khrdebug";
static std::string vsync_option         = "vsync";
static std::string window_option        = "w,window";
static std::string fullscreen_option    = "f,fullscreen";
static std::string nodecoration_option  = "nodecoration";
static std::string topmost_option       = "topmost";
static std::string nocursor_option      = "nocursor";
static std::string interactive_option   = "i,interactive";
static std::string guishow_option       = "guishow";
static std::string guiscale_option      = "guiscale";
static std::string help_option          = "h,help";

static void files_exist(std::vector<std::string> vec, std::string const& type) {
    for (const auto& file : vec) {
        if (!stdfs::exists(file)) {
            exit(type + " \"" + file + "\" does not exist!");
        }
    }
}

// option handlers fill the config struct with passed options
// this is a handler template
static void empty_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    // config.option = parsed_options[option_name].as<bool>();
};

static void guishow_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.gui_show = parsed_options[option_name].as<bool>();
};

static void guiscale_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.gui_scale = parsed_options[option_name].as<float>();
};

static void config_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    // is already done by first CLI pass which checks config files before running them through Lua
};

static void appdir_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    auto appdir = parsed_options[option_name].as<std::string>();
    files_exist({appdir}, "Application directory");

    config.application_directory = appdir;
};

static void resourcedir_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    auto v = parsed_options[option_name].as<std::vector<std::string>>();
    files_exist(v, "Resource directory");

    config.resource_directories.insert(config.resource_directories.end(), v.begin(), v.end());
};

static void shaderdir_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    auto v = parsed_options[option_name].as<std::vector<std::string>>();
    files_exist(v, "Shader directory");

    config.shader_directories.insert(config.shader_directories.end(), v.begin(), v.end());
};

static void logfile_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.log_file = parsed_options[option_name].as<std::string>();
};

static void loglevel_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.log_level = parsed_options[option_name].as<unsigned int>();
};

static void echolevel_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.echo_level = parsed_options[option_name].as<unsigned int>();
};

static void project_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    auto v = parsed_options[option_name].as<std::vector<std::string>>();
    files_exist(v, "Project file");

    config.project_files.insert(config.project_files.end(), v.begin(), v.end());
};

static void global_value_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    auto v = parsed_options[option_name].as<std::vector<std::string>>();

    for (auto& key_value : v) {
        auto delimiter = key_value.find(':');
        if (delimiter == std::string::npos)
            exit("Config Key-Value pair \"" + key_value + "\" not valid. Needs colon (:) delimiter between key and value.");

        auto key   = key_value.substr(0, delimiter);
        auto value = key_value.substr(delimiter+1);

        config.global_values.push_back({key, value});
    }
};

static void host_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.lua_host_address = parsed_options[option_name].as<std::string>();
};

static void khrdebug_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.opengl_khr_debug = parsed_options[option_name].as<bool>();
};
static void vsync_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.opengl_vsync = parsed_options[option_name].as<bool>();
};

static void fullscreen_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::fullscreen;
};
static void nodecoration_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::nodecoration;
};
static void topmost_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::topmost;
};
static void nocursor_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::nocursor;
};

static void interactive_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    config.interactive = parsed_options[option_name].as<bool>();
};

static void window_handler(std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
{
    auto s = parsed_options[option_name].as<std::string>();
    // 'WIDTHxHEIGHT[+POSX+POSY]'
    // 'wxh+x+y' with optional '+x+y', e.g. 600x800+0+0 opens window in upper left corner
    std::regex geometry("(\\d+)x(\\d+)(?:\\+(\\d+)\\+(\\d+))?");
    std::smatch match;
    if (std::regex_match(s, match, geometry)) {
        unsigned int width = std::stoul(match[1].str(), nullptr, 10);
        unsigned int height = std::stoul(match[2].str(), nullptr, 10);
        config.window_size = {{width, height}};

        if (match[3].matched) {
            unsigned int x = std::stoul(match[3].str(), nullptr, 10);
            unsigned int y = std::stoul(match[4].str(), nullptr, 10);
            config.window_position = {{x, y}};
        }
    } else {
        exit("window option needs to be in the following format: wxh+x+y or wxh");
    }
};

using OptionsListEntry = std::tuple<std::string, std::string, std::shared_ptr<cxxopts::Value>, std::function<void(std::string const&, cxxopts::ParseResult const&, megamol::frontend::RuntimeConfig&)>>;

std::vector<OptionsListEntry> cli_options_list =
    {     // config name         option description                                                                 type                                        handler
          {config_option,        "Path to Lua configuration file(s)",                                               cxxopts::value<std::vector<std::string>>(), config_handler}
        , {appdir_option,        "Set application directory",                                                       cxxopts::value<std::string>(),              appdir_handler}
        , {resourcedir_option,   "Add resource directory(ies)",                                                     cxxopts::value<std::vector<std::string>>(), resourcedir_handler}
        , {shaderdir_option,     "Add shader directory(ies)",                                                       cxxopts::value<std::vector<std::string>>(), shaderdir_handler}
        , {logfile_option,       "Set log file",                                                                    cxxopts::value<std::string>(),              logfile_handler}
        , {loglevel_option,      "Set logging level",                                                               cxxopts::value<unsigned int>(),             loglevel_handler}
        , {echolevel_option,     "Set echo level",                                                                  cxxopts::value<unsigned int>(),             echolevel_handler}
        , {project_option,       "Project file(s) to load at startup",                                              cxxopts::value<std::vector<std::string>>(), project_handler}
        , {global_option,        "Set global key-value pair(s) in MegaMol environment, syntax: --global key:value", cxxopts::value<std::vector<std::string>>(), global_value_handler}

        , {host_option,          "Address of lua host server",                                                      cxxopts::value<std::string>(),              host_handler         }
        , {khrdebug_option,      "Enable OpenGL KHR debug messages",                                                cxxopts::value<bool>(),                     khrdebug_handler     }
        , {vsync_option,         "Enable VSync in OpenGL window",                                                   cxxopts::value<bool>(),                     vsync_handler        }
        , {window_option,        "Set the window size and position, syntax: --window WIDTHxHEIGHT[+POSX+POSY]",     cxxopts::value<std::string>(),              window_handler       }
        , {fullscreen_option,    "Open maximized window",                                                           cxxopts::value<bool>(),                     fullscreen_handler   }
        , {nodecoration_option,  "Open window without decorations",                                                 cxxopts::value<bool>(),                     nodecoration_handler }
        , {topmost_option,       "Open window that stays on top of all others",                                     cxxopts::value<bool>(),                     topmost_handler      }
        , {nocursor_option,      "Do not show mouse cursor inside window",                                          cxxopts::value<bool>(),                     nocursor_handler     }
        , {interactive_option,   "Run MegaMol even if some project file failed to load",                            cxxopts::value<bool>(),                     interactive_handler  }
        , {project_files_option, "Project file(s) to load at startup",                                              cxxopts::value<std::vector<std::string>>(), project_handler}
        , {guishow_option,       "Render GUI overlay, use '=false' to disable",                                     cxxopts::value<bool>(),                     guishow_handler}
        , {guiscale_option,      "Set scale of GUI, expects float >= 1.0. e.g. 1.0 => 100%, 2.1 => 210%",           cxxopts::value<float>(),                    guiscale_handler}
        , {help_option,          "Print help message",                                                              cxxopts::value<bool>(),                     empty_handler}
    };

static std::string loong(std::string const& option) {
    auto f = option.find(',');
    if (f == std::string::npos)
        return option;

    return option.substr(f+1);
}

std::vector<std::string> megamol::frontend::extract_config_file_paths(const int argc, const char** argv) {
    // load config files from default paths
    // setting Config files from Lua is not possible
    // multiple Config files can be passed via CLI - then default config file paths are ignored

    // find config options in CLI string and overwrite default paths
    cxxopts::Options options("Config option pass", "MegaMol Config Parsing");
    options.add_options()
        (config_option, "", cxxopts::value<std::vector<std::string>>());

    options.allow_unrecognised_options();

    try {
        int _argc = argc;
        auto _argv = const_cast<char**>(argv);
        auto parsed_options = options.parse(_argc, _argv);

        std::vector<std::string> config_files;

        auto personal_config = stdfs::path(getHomeDir()) / stdfs::path(".megamol_config.lua");
        if (stdfs::exists(personal_config)) {
            config_files.push_back(personal_config.string());
        }

        // the personal config should be loaded after the default config to overwrite it
        // but before configs passed via CLI to be overwritten by them
        if (parsed_options.count(loong(config_option)) == 0) {
            // no config files given
            // use defaults
            RuntimeConfig config;
            config_files.insert(config_files.begin(), config.configuration_files.begin(), config.configuration_files.end());
        } else {
            auto cli_config_files = parsed_options[loong(config_option)].as<std::vector<std::string>>();
            config_files.insert(config_files.end(), cli_config_files.begin(), cli_config_files.end());
        }

        // remove empty files: enables to start megamol without config file
        std::remove_if(config_files.begin(), config_files.end(), [](auto const& file) { return file.empty(); });

        // check remaining files exist
        files_exist(config_files, "Config file");

        return config_files;

    } catch (cxxopts::OptionException ex) {
        exit(ex.what());
    }
}

#define add_option(X) (std::get<0>(X), std::get<1>(X), std::get<2>(X))

megamol::frontend_resources::RuntimeConfig megamol::frontend::handle_config(RuntimeConfig config, megamol::core::LuaAPI& lua) {

    // the parsing of CLI options inside Lua callbacks is somewhat of a mess
    // we create lua callbacks on the fly and pass them to Lua to execute during config interpretation
    // those callbacks fill our local fake "cli inputs" arrays, which we then feed into the CLI interpreter (cxxopt)

    using StringPair = megamol::frontend_resources::RuntimeConfig::StringPair;
    std::vector<StringPair> cli_options_from_configs;

    using megamol::frontend_resources::LuaCallbacksCollection;
    using Error = megamol::frontend_resources::LuaCallbacksCollection::LuaError;
    using VoidResult = megamol::frontend_resources::LuaCallbacksCollection::VoidResult;

    #define sane(s) \
                if (s.empty() || s.find_first_of(" =") != std::string::npos) \
                    return Error{"Value \"" + s + "\" is empty, has space or ="};

    #define file_exists(f) \
                if (!stdfs::exists(f)) \
                    return Error{"File does not exist: " + f};

    #define add_cli(o,v) \
                cli_options_from_configs.push_back({loong(o),v});

    // helper to make options that take a string and maybe check for a file to exist
    auto make_option_callback = [&](std::string const& optname, bool file_check = false) {
        return [file_check, optname, &cli_options_from_configs](std::string value) -> VoidResult {
            sane(value);

            if (file_check)
                file_exists(value);

            add_cli(optname, value);
            return VoidResult{};
        };
    };

    // helper to make options that parse a log level
    auto make_log_level_callback = [&](std::string const& optname) {
        return [optname, &cli_options_from_configs](std::string value) -> VoidResult {

            sane(value);

            if (value.front()=='-')
                return Error {"log level value string seems to be negative"};

            unsigned int value_as_uint = 0;

            try {
                if (std::string(value).find_first_of("0123456789") != std::string::npos) {
                    value_as_uint = std::stoi(value);
                } else {
                    value_as_uint = megamol::core::utility::log::Log::ParseLevelAttribute(value);
                }
            }
            catch (...) {
                return Error{"Could not parse valid log level string or positive integer from argument \"" + value + "\""};
            }

            add_cli(optname, std::to_string(value_as_uint));
            return VoidResult{};
        };
    };

    std::vector<std::string> all_options_separate;
    for (auto& opt : cli_options_list) {
        // split "h,help"
        auto& name = std::get<0>(opt);
        auto delimiter = name.find(",");
        if (delimiter == std::string::npos) {
            all_options_separate.push_back(name);
        } else {
            all_options_separate.push_back(name.substr(0, delimiter));
            all_options_separate.push_back(name.substr(delimiter+1));
        }
    }

    LuaCallbacksCollection lua_config_callbacks;

    // mmSetCliOption
    // VoidResult, std::string, std::string
    lua_config_callbacks.add<VoidResult, std::string, std::string>(
        "mmSetCliOption",
        "(string name, string value)\n\tSet CLI option to a specific value.",
        {[&](std::string clioption, std::string value) -> VoidResult {
            // the usual CLI options
            sane(clioption);
            sane(value);

            // we assume that "on" and "off" are used only for boolean cxxopts values
            // and so we can map them to "true" and "false"
            auto map_value = [](std::string const& value) -> std::string {
                if (value == std::string("on")) {
                    return "true";
                }
                if (value == std::string("off")) {
                    return "false";
                }

                return value;
            };

            auto option_it = std::find(all_options_separate.begin(), all_options_separate.end(), clioption);
            bool option_unknown = option_it == all_options_separate.end();
            if (option_unknown)
                return Error{"unknown option: " + clioption};

            // if has option of length one, take the long version
            auto finalopt = clioption;
            if (option_it->size() == 1)
                finalopt = *(option_it+1);

            add_cli(finalopt,map_value(value));
            return VoidResult{};
        }});

    // mmSetGlobalValue
    // std::string, std::string 
    lua_config_callbacks.add<VoidResult, std::string, std::string>(
        "mmSetGlobalValue",
        "(string key, string value)\n\tSets the value of name <key> to <value> in the global key-value store.",
        {[&](std::string key, std::string value) -> VoidResult {
            sane(key);
            sane(value);
            add_cli(global_option, key + ":" + value);
            return VoidResult{};
        }});

#undef sane
#undef file_exists
#undef add_cli

    lua_config_callbacks.add<VoidResult, std::string>(
        "mmSetAppDir",
        "(string dir)\n\tSets the path where the mmconsole.exe is located.",
        { make_option_callback(appdir_option, true) });

    lua_config_callbacks.add<VoidResult, std::string>(
        "mmAddShaderDir",
        "(string dir)\n\tAdds a shader/btf search path.",
        { make_option_callback(shaderdir_option, true) });

    lua_config_callbacks.add<VoidResult, std::string>(
        "mmAddResourceDir",
        "(string dir)\n\tAdds a resource search path.",
        { make_option_callback(resourcedir_option, true) });

    lua_config_callbacks.add<VoidResult, std::string>(
        "mmLoadProject",
        "(string path)\n\tLoad lua (project) file after MegaMol startup.",
        { make_option_callback(project_option, true) });

    lua_config_callbacks.add<VoidResult, std::string>(
        "mmSetLogFile",
        "(string path)\n\tSets the full path of the log file.",
        { make_option_callback(logfile_option) });

    lua_config_callbacks.add<VoidResult, std::string>(
        "mmSetLogLevel",
        "(string level)\n\tSets the level of log events to include. Level values: ('error', 'warn', 'warning', 'info', 'none', 'null', 'zero', 'all', '*')",
        { make_log_level_callback(loglevel_option) });

    lua_config_callbacks.add<VoidResult, std::string>(
        "mmSetEchoLevel",
        "(string level)\n\tSets the level of log events to output to the console. Level values: ('error', 'warn', 'warning', 'info', 'none', 'null', 'zero', 'all', '*')",
        { make_log_level_callback(echolevel_option) });

    lua.AddCallbacks(lua_config_callbacks);

    for (auto& file : config.configuration_files) {
        cli_options_from_configs.clear();
        std::ifstream stream(file);
        std::string file_contents = std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());

        // interpret lua config commands as CLI commands
        std::string lua_result_string;
        bool lua_config_ok = lua.RunString(file_contents, lua_result_string);

        if (!lua_config_ok) {
            exit("Error in Lua config file " + file + "\n Lua Error: " + lua_result_string);
        }

        // feed the options coming from Lua into CLI parser, which writes option changes into the RuntimeConfig
         std::vector<std::string> file_contents_as_cli {cli_options_from_configs.size()};
         for (auto& pair : cli_options_from_configs)
             file_contents_as_cli.push_back("--" + pair.first + "=" + pair.second);

         int argc = file_contents_as_cli.size()+1;
         std::vector<const char*> argv{static_cast<size_t>(argc)};
         auto zero = "Lua Config Pass";
         argv[0] = zero;

         // cxopts can only parse const char**
         int i = 1;
         for (auto& arg : file_contents_as_cli) {
             argv[i++] = arg.data();
         }

        cxxopts::Options options("Lua Config Pass", "MegaMol Lua Config Parsing");
        for (auto& opt : cli_options_list) {
            options.add_options()
                add_option(opt);
        }

        // actually process passed options
        try {
            auto parsed_options = options.parse(argc, argv.data());
            std::string res;

            for (auto& option : cli_options_list) {
                auto option_name = loong(std::get<0>(option));
                if (parsed_options.count(option_name)) {
                    auto& option_handler = std::get<3>(option);
                    option_handler(option_name, parsed_options, config);
                }
            }

        } catch (cxxopts::OptionException ex) {
            exit(std::string(ex.what()) + "\nIn file: " + file);
        }

        config.configuration_file_contents.push_back(file_contents);
        config.configuration_file_contents_as_cli.push_back(
            std::accumulate(file_contents_as_cli.begin(), file_contents_as_cli.end(), std::string(""),
                [](std::string const& init, std::string const& elem) { return init + elem + " "; }));
    }

    lua.ClearCallbacks();

    return config;
}


megamol::frontend_resources::RuntimeConfig megamol::frontend::handle_cli(RuntimeConfig config, const int argc, const char** argv) {

    cxxopts::Options options(argv[0], "MegaMol Frontend 3000");

    for (int i = 0; i < argc; i++)
        config.program_invocation_string += std::string{argv[i]} + " ";

    int _argc = argc;
    auto _argv = const_cast<char**>(argv);

    // parse input project files
    options.positional_help("<additional project files>");
    for (auto& opt : cli_options_list) {
        options.add_options()
            add_option(opt);
    }
    options.parse_positional({project_files_option});

    // actually process passed options
    try {
        auto parsed_options = options.parse(_argc, _argv);
        std::string res;

        if (parsed_options.count("help")) {
            std::cout << options.help({""}) << std::endl;
            std::exit(0);
        }

        for (auto& option : cli_options_list) {
            auto option_name = loong(std::get<0>(option));
            if (parsed_options.count(option_name)) {
                auto& option_handler = std::get<3>(option);
                option_handler(option_name, parsed_options, config);
            }
        }

    } catch (cxxopts::OptionException ex) {
        exit(std::string(ex.what()) + "\n" + options.help({""}));
    }

    return config;
}

