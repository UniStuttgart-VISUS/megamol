
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

static std::string config_option      = "--config";
static std::string appdir_option      = "--appdir";
static std::string resourcedir_option = "--resourcedir";
static std::string shaderdir_option   = "--shaderdir";
static std::string logfile_option     = "--logfile";
static std::string loglevel_option    = "--loglevel";
static std::string echolevel_option   = "--echolevel";
static std::string project_option     = "--project";
static std::string var_option         = "--var";

static auto option_name = [](std::string const& s) { return s.substr(2); };
static auto config_name = option_name(config_option);

static void exit(std::string const& reason) {
    std::cout << "Error :" << reason << std::endl;
    std::cout << "Shut down MegaMol" << std::endl;
    std::exit(1);
}

megamol::frontend_resources::RuntimeConfig megamol::frontend::handle_cli_and_config(const int argc, const char** argv, megamol::core::LuaAPI& lua) {
    RuntimeConfig config;

    // config files are already checked to exist in file system
    config.configuration_files = extract_config_file_paths(argc, argv);

    // overwrite default values with values from config file
    config = handle_config(config, lua);

    // overwrite default and config values with CLI inputs
    config = handle_cli(config, argc, argv);

    return config;
}

std::vector<std::string> megamol::frontend::extract_config_file_paths(const int argc, const char** argv) {
    // load config files from default paths
    // setting Config files from Lua is not possible
    // multiple Config files can be passed via CLI - then default config file paths are ignored

    // clang-format off
    // find config options in CLI string and overwrite default paths
    cxxopts::Options options(argv[0], "MegaMol Config Parsing");
    options.add_options()
        (config_name, "", cxxopts::value<std::vector<std::string>>());
    // clang-format on

    options.allow_unrecognised_options();

    try {
        int _argc = argc;
        auto _argv = const_cast<char**>(argv);
        auto parsed_options = options.parse(_argc, _argv);

        std::vector<std::string> config_files;
        auto config_name = option_name(config_option);

        if (parsed_options.count(config_name) == 0) {
            // no config files given
            // use defaults
            RuntimeConfig config;
            config_files = config.configuration_files;
        } else {
            config_files = parsed_options[config_name].as<std::vector<std::string>>();
        }

        // check files exist
        for (const auto& file : config_files) {
            if (!stdfs::exists(file)) {
                exit("Config file \"" + file + "\" does not exist!");
            }
        }

        return config_files;

    } catch (cxxopts::option_not_exists_exception ex) {
        exit(ex.what());
    } catch (cxxopts::missing_argument_exception ex) {
        exit(ex.what());
    }
}

megamol::frontend_resources::RuntimeConfig megamol::frontend::handle_config(RuntimeConfig config, megamol::core::LuaAPI& lua) {

    // load config file
    auto& files = config.configuration_files;

    // holds CLI options in each for-loop iteration
    // gets cleared for each new iteration
    std::vector<std::string> file_contents_as_cli;
    auto is_weird = [](std::string const& s) { return (s.empty() || s.find_first_of(" =") != std::string::npos); };
    #define check(s) \
                if (is_weird(s)) \
                    return false;
    #define opt(o) \
                std::string option = (o[0] == '-' ? "" : "--") + o;
    #define add(o,v) \
                file_contents_as_cli.push_back(o + "=" + v);

    auto make_option_callback = [&](std::string const& optname) {
        return [&](std::string const& value) {
            check(value);
            add(optname, value);

            return true;
        };
    };

    for (auto& file : files) {
        std::ifstream stream(file);
        std::string file_contents = std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
        file_contents_as_cli.clear();

        auto callbacks = megamol::core::LuaAPI::LuaConfigCallbacks{
            // mmSetConfig_callback_ std::function<void(std::string const&, std::string const&)> ;
            [&](std::string const& config, std::string const& value) {
                // the usual CLI options

                check(config);
                check(value);

                opt(config);

                if (value == std::string("on")) {
                    file_contents_as_cli.push_back(option);
                } else 
                if (value == std::string("off")) {
                    std::remove(file_contents_as_cli.begin(), file_contents_as_cli.end(), option);
                } else {
                    add(option,value);
                }

                return true;
            } ,
            make_option_callback(appdir_option), // mmSetAppDir_callback_ std::function<void(std::string const&)> ;
            make_option_callback(resourcedir_option), // mmAddResourceDir_callback_ std::function<void(std::string const&)> ;
            make_option_callback(shaderdir_option), // mmAddShaderDir_callback_ std::function<void(std::string const&)> ;
            make_option_callback(logfile_option), // mmSetLogFile_callback_ std::function<void(std::string const&)> ;
            // mmSetLogLevel_callback_ std::function<void(int const)> ;
            [&](const int log_level) {
                // Lua checked string to int conversion already
                add(loglevel_option, std::to_string(log_level));
                return true;
            } ,
            // std::function<void(int const)> mmSetEchoLevel_callback_;
            [&](const int echo_level) {
                // Lua checked string to int conversion already
                add(echolevel_option, std::to_string(echo_level));
                return true;
            } ,
            make_option_callback(project_option), // mmLoadProject_callback_ std::function<void(std::string const&)> ;
            // mmSetKeyValue_callback_ std::function<void(std::string const&, std::string const&)> ;
            [&](std::string const& key, std::string const& value) {
                check(key); // no space or = in key

                // no space or : in value
                if (value.find_first_of(" :") != std::string::npos)
                    return false;

                add(var_option, key + ":" + value);
                return true;
            }
            //// mmGetKeyValue_callback_ std::function<std::string(std::string const&)> ;
            //[&](std::string const& maybe_value) {}
        };

        // interpret lua config commands as CLI commands
        std::string lua_result_string;
        bool lua_config_ok = lua.FillConfigFromString(file_contents, lua_result_string, callbacks);

        if (!lua_config_ok) {
            exit("Error in Lua config file " + file + "\n Lua Error: " + lua_result_string);
        }

        auto summarize = [](std::vector<std::string> const& cli) -> std::string
        {
            return std::accumulate(cli.begin(), cli.end(), std::string(""),
                [](std::string const& init, std::string const& elem) { return init + " " + elem; });
        };

        config.configuration_file_contents.push_back(file_contents);
        config.configuration_file_contents_as_cli.push_back(summarize(file_contents_as_cli));
    }

    return config;
}


megamol::frontend_resources::RuntimeConfig megamol::frontend::handle_cli(RuntimeConfig config, const int argc, const char** argv) {

    cxxopts::Options options(argv[0], "MegaMol Frontend 3000");

    config.program_invocation_string = std::string{argv[0]};

    // option handlers fill config struct with passed options
    auto empty_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
    };

    auto config_files_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        // is already done by first CLI pass which checks config files before running them through Lua
    };

    auto project_files_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        const auto& v = parsed_options[option_name].as<std::vector<std::string>>();
        for (const auto& p : v) {
            if (!stdfs::exists(p)) {
                exit("Project file \"" + p + "\" does not exist!");
            }
        }

        config.project_files = v;
    };

    auto host_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.lua_host_address = parsed_options[option_name].as<std::string>();
    };

    auto khrdebug_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.opengl_khr_debug = parsed_options[option_name].as<bool>();
    };
    auto vsync_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.opengl_vsync = parsed_options[option_name].as<bool>();
    };

    auto fullscreen_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::fullscreen;
    };
    auto nodecoration_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::nodecoration;
    };
    auto topmost_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::topmost;
    };
    auto nocursor_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::nocursor;
    };

    auto interactive_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.interactive = parsed_options[option_name].as<bool>();
    };

    auto window_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        auto s = parsed_options[option_name].as<std::string>();
        // 'WIDTHxHEIGHT[+POSX+POSY]'
        // 'wxh+x+y' with optional '+x+y', e.g. 600x800+0+0 opens window in upper left corner
        std::regex geometry("(\\d+)x(\\d+)(?:\\+(\\d+)\\+(\\d+))?");
        std::smatch match;
        if (std::regex_match(s, match, geometry)) {
            config.window_size.push_back( /*width*/ std::stoul(match[1].str(), nullptr, 10) );
            config.window_size.push_back( /*height*/ std::stoul(match[2].str(), nullptr, 10) );
            if (match[3].matched) {
                config.window_position.push_back( /*x*/ std::stoul(match[3].str(), nullptr, 10) );
                config.window_position.push_back( /*y*/ std::stoul(match[4].str(), nullptr, 10) );
            }
        } else {
            exit("window option needs to be in the following format: wxh+x+y or wxh");
        }
    };

    // clang-format off
    std::vector<std::tuple<std::string, std::string, std::shared_ptr<cxxopts::Value>, std::function<void(std::string const&, cxxopts::ParseResult const&, RuntimeConfig&)>>>
    options_list =
    {
          {config_name,     "provide Lua configuration file",                                              cxxopts::value<std::vector<std::string>>(), config_files_handler }
        , {"project-files", "projects to load",                                                            cxxopts::value<std::vector<std::string>>(), project_files_handler}
        , {"host",          "address of lua host server, default: "+config.lua_host_address,               cxxopts::value<std::string>(),              host_handler         }
        , {"khrdebug",      "enable OpenGL KHR debug messages",                                            cxxopts::value<bool>(),                     khrdebug_handler     }
        , {"vsync",         "enable VSync in OpenGL window",                                               cxxopts::value<bool>(),                     vsync_handler        }
        , {"window",        "set the window size and position, accepted format: WIDTHxHEIGHT[+POSX+POSY]", cxxopts::value<std::string>(),              window_handler       }
        , {"fullscreen",    "open maximized window",                                                       cxxopts::value<bool>(),                     fullscreen_handler   }
        , {"nodecoration",  "open window without decorations",                                             cxxopts::value<bool>(),                     nodecoration_handler }
        , {"topmost",       "open window that stays on top of all others",                                 cxxopts::value<bool>(),                     topmost_handler      }
        , {"nocursor",      "do not show mouse cursor inside window",                                      cxxopts::value<bool>(),                     nocursor_handler     }
        , {"interactive",   "open MegaMol even if project files via CLI could not be loaded",              cxxopts::value<bool>(),                     interactive_handler  }
    };

    #define add_option(X) (std::get<0>(X), std::get<1>(X), std::get<2>(X))

    // parse input project files
    options.positional_help("<additional project files>");
    options.add_options()
        add_option(options_list[0])
        add_option(options_list[1])
        add_option(options_list[2])
        add_option(options_list[3])
        add_option(options_list[4])
        add_option(options_list[5])
        add_option(options_list[6])
        add_option(options_list[7])
        add_option(options_list[8])
        add_option(options_list[9])
        add_option(options_list[10])
        ("help", "print help")
        ;
    // clang-format on

    options.parse_positional({"project-files"});

    // actually process passed options
    try {
        int _argc = argc;
        auto _argv = const_cast<char**>(argv);
        auto parsed_options = options.parse(_argc, _argv);
        std::string res;

        if (parsed_options.count("help")) {
            std::cout << options.help({""}) << std::endl;
            std::exit(0);
        }

        for (auto& option : options_list) {
            auto& option_name = std::get<0>(option);
            if (parsed_options.count(option_name)) {
                auto& option_handler = std::get<3>(option);
                option_handler(option_name, parsed_options, config);
            }
        }

    } catch (cxxopts::option_not_exists_exception ex) {
        exit(std::string(ex.what()) + "\n" + options.help({""}));
    }

    return config;
}

