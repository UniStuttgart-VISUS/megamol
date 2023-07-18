
#include "CLIConfigParsing.h"

#define CXXOPTS_VECTOR_DELIMITER '\0'
#include <cxxopts.hpp>

#include <filesystem>

// find user home
#include <stdio.h>
#include <stdlib.h>
static std::filesystem::path getHomeDir() {
#ifdef _WIN32
    return std::filesystem::absolute(std::string(getenv("HOMEDRIVE")) + std::string(getenv("HOMEPATH")));
#else // LINUX
    return std::filesystem::absolute(std::string(getenv("HOME")));
#endif
}

// find megamol executable path
static std::filesystem::path getExecutableDirectory() {
    std::filesystem::path path;
#ifdef _WIN32
    std::vector<wchar_t> filename;
    DWORD length;
    do {
        filename.resize(filename.size() + 1024);
        length = GetModuleFileNameW(NULL, filename.data(), static_cast<DWORD>(filename.size()));
    } while (length >= filename.size());
    filename.resize(length);
    path = {std::wstring(filename.begin(), filename.end())};
#else
    std::filesystem::path p("/proc/self/exe");
    if (!std::filesystem::exists(p) || !std::filesystem::is_symlink(p)) {
        throw std::runtime_error("Cannot read process name!");
    }
    path = std::filesystem::read_symlink(p);
#endif
    // path points to exeutable. remove executable filename and return directory.
    return std::filesystem::absolute(path).remove_filename();
}

using megamol::frontend_resources::GlobalValueStore;
using megamol::frontend_resources::RuntimeConfig;

// called by main and returns config struct filled with parsed CLI values
std::pair<RuntimeConfig, GlobalValueStore> megamol::frontend::handle_cli_and_config(
    const int argc, const char** argv, megamol::core::LuaAPI& lua) {
    RuntimeConfig config;

    config.megamol_executable_directory = getExecutableDirectory().u8string();

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

    // set delimiter ; in lua commands to newlines, so lua can actually execute
    for (auto& character : config.cli_execute_lua_commands) {
        if (character == ';')
            character = '\n';
    }

    return {config, global_value_store};
}

static void exit(std::string const& reason) {
    std::cout << "Error: " << reason << std::endl;
    std::cout << "Shut down MegaMol" << std::endl;
    std::exit(1);
}

// only for --config pass
static std::string config_option = "c,config";

// basic options
static std::string appdir_option = "appdir";
static std::string resourcedir_option = "resourcedir";
static std::string shaderdir_option = "shaderdir";
static std::string logfile_option = "logfile";
static std::string loglevel_option = "loglevel";
static std::string echolevel_option = "echolevel";
static std::string project_option = "p,project";
static std::string execute_lua_option = "e,execute";
static std::string global_option = "g,global";

// service-specific options
// --project and loose project files are both valid ways to provide lua project files
static std::string project_files_option = "project-files";
static std::string host_option = "host";
static std::string opengl_context_option = "opengl";
static std::string khrdebug_option = "khrdebug";
static std::string disable_opengl_option = "nogl";
static std::string vsync_option = "vsync";
static std::string window_option = "w,window";
static std::string fullscreen_option = "f,fullscreen";
static std::string force_window_size_option = "force-window-size";
static std::string nodecoration_option = "nodecoration";
static std::string topmost_option = "topmost";
static std::string nocursor_option = "nocursor";
static std::string hidden_option = "hidden";
static std::string interactive_option = "i,interactive";
static std::string guishow_option = "guishow";
static std::string nogui_option = "nogui";
static std::string guiscale_option = "guiscale";
static std::string privacynote_option = "privacynote";
static std::string versionnote_option = "versionnote";
static std::string profile_log_option = "profiling-log";
static std::string flush_frequency_option = "flush-frequency";
static std::string profile_log_no_autostart_option = "pause-profiling";
static std::string profile_log_include_events_option = "profiling-include-events";
static std::string param_option = "param";
static std::string remote_head_option = "headnode";
static std::string remote_render_option = "rendernode";
static std::string remote_mpi_option = "mpi";
static std::string remote_mpi_broadcast_rank_option = "mpi-broadcaster-rank";
static std::string remote_headnode_zmq_target_option = "headnode-zmq-target";
static std::string remote_rendernode_zmq_source_option = "rendernode-zmq-source";
static std::string remote_headnode_broadcast_quit_option = "headnode-broadcast-quit";
static std::string remote_headnode_broadcast_project_option = "headnode-broadcast-project";
static std::string remote_headnode_connect_at_start_option = "headnode-connect-at-start";
static std::string framebuffer_option = "framebuffer";
static std::string viewport_tile_option = "tile";
static std::string vr_service_option = "vr";
static std::string help_option = "h,help";

static void files_exist(std::vector<std::string> vec, std::string const& type) {
    for (const auto& file : vec) {
        if (!std::filesystem::exists(file)) {
            exit(type + " \"" + file + "\" does not exist!");
        }
    }
}

// option handlers fill the config struct with passed options
// this is a handler template
static void empty_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config){
    // config.option = parsed_options[option_name].as<bool>();
};

static void guishow_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.gui_show = parsed_options[option_name].as<bool>();
};

static void nogui_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.gui_show = !parsed_options[option_name].as<bool>();
};

static void guiscale_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.gui_scale = parsed_options[option_name].as<float>();
};

static void privacynote_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.screenshot_show_privacy_note = parsed_options[option_name].as<bool>();
};

static void versionnote_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.show_version_note = parsed_options[option_name].as<bool>();
};

static void profile_log_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.profiling_output_file = parsed_options[option_name].as<std::string>();
}

static void flush_frequency_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.flush_frequency = parsed_options[option_name].as<uint32_t>();
}

static void profile_log_autostart_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.autostart_profiling = !parsed_options[option_name].as<bool>();
}

static void profile_log_include_events_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.include_graph_events = parsed_options[option_name].as<bool>();
}

static void remote_head_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.remote_headnode = parsed_options[option_name].as<bool>();
};

static void remote_render_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.remote_rendernode = parsed_options[option_name].as<bool>();
};

static void remote_mpirender_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.remote_mpirendernode = parsed_options[option_name].as<bool>();
};

static void remote_mpirank_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.remote_mpi_broadcast_rank = parsed_options[option_name].as<unsigned int>();
};

static void remote_zmqtarget_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.remote_headnode_zmq_target_address = parsed_options[option_name].as<std::string>();
};

static void remote_zmqsource_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.remote_rendernode_zmq_source_address = parsed_options[option_name].as<std::string>();
};

static void remote_head_broadcast_quit_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.remote_headnode_broadcast_quit = parsed_options[option_name].as<bool>();
};

static void remote_head_broadcast_project_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.remote_headnode_broadcast_initial_project = parsed_options[option_name].as<bool>();
};

static void remote_head_connect_at_start_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.remote_headnode_connect_on_start = parsed_options[option_name].as<bool>();
};

static void config_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config){
    // is already done by first CLI pass which checks config files before running them through Lua
};

static void appdir_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    auto appdir = parsed_options[option_name].as<std::string>();
    files_exist({appdir}, "Application directory");

    config.application_directory = appdir;
};

static void resourcedir_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    auto v = parsed_options[option_name].as<std::vector<std::string>>();
    files_exist(v, "Resource directory");

    config.resource_directories.insert(config.resource_directories.end(), v.begin(), v.end());
};

static void shaderdir_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    auto v = parsed_options[option_name].as<std::vector<std::string>>();
    files_exist(v, "Shader directory");

    config.shader_directories.insert(config.shader_directories.end(), v.begin(), v.end());
};

static void logfile_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.log_file = parsed_options[option_name].as<std::string>();
};

static const std::string accepted_log_level_strings =
    "('error', 'warn', 'warning', 'info', 'none', 'null', 'zero', 'all', '*')";

static void loglevel_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.log_level =
        megamol::core::utility::log::Log::ParseLevelAttribute(parsed_options[option_name].as<std::string>());
};

static void echolevel_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.echo_level =
        megamol::core::utility::log::Log::ParseLevelAttribute(parsed_options[option_name].as<std::string>());
};

static void project_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    auto v = parsed_options[option_name].as<std::vector<std::string>>();
    files_exist(v, "Project file");

    config.project_files.insert(config.project_files.end(), v.begin(), v.end());
};

static void execute_lua_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    auto commands = parsed_options[option_name].as<std::string>();
    config.cli_execute_lua_commands += commands + ";";
    //for (auto& cmd : commands) {
    //    config.cli_execute_lua_commands += cmd + ";";
    //}
};

static void param_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    auto strings = parsed_options[option_name].as<std::vector<std::string>>();
    std::string cmds;

    std::regex param_value("(.+)=(.+)");

    auto handle_param = [&](std::string const& string) {
        std::smatch match;
        if (std::regex_match(string, match, param_value)) {
            auto param = "\"" + match[1].str() + "\"";
            auto value = "\"" + match[2].str() + "\"";

            std::string cmd = "mmSetParamValue(" + param + "," + value + ")";
            cmds += cmd + ";";
        } else {
            exit("param option needs to be in the following format: param=value");
        }
    };

    for (auto& paramstring : strings) {
        handle_param(paramstring);
    }

    // prepend param value changes before other CLI Lua commands
    config.cli_execute_lua_commands = cmds + config.cli_execute_lua_commands;
};

static void global_value_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    auto v = parsed_options[option_name].as<std::vector<std::string>>();

    for (auto& key_value : v) {
        auto delimiter = key_value.find(':');
        if (delimiter == std::string::npos)
            exit("Config Key-Value pair \"" + key_value +
                 "\" not valid. Needs colon (:) delimiter between key and value.");

        auto key = key_value.substr(0, delimiter);
        auto value = key_value.substr(delimiter + 1);

        config.global_values.push_back({key, value});
    }
};

static void host_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.lua_host_address = parsed_options[option_name].as<std::string>();
};


static void opengl_context_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    auto string = parsed_options[option_name].as<std::string>();

    std::regex version("(\\d+).(\\d+)(core|compat)?");
    std::smatch match;
    if (std::regex_match(string, match, version)) {
        unsigned int major = std::stoul(match[1].str(), nullptr, 10);
        unsigned int minor = std::stoul(match[2].str(), nullptr, 10);
        bool profile = false;

        if (match[3].matched) {
            profile = match[3].str() == std::string("core");
        }

        config.opengl_context_version = {{major, minor, profile}};
    } else {
        exit("opengl option needs to be in the following format: major.minor[core|compat]");
    }
};

static void khrdebug_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.opengl_khr_debug = parsed_options[option_name].as<bool>();
};
static void vsync_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.opengl_vsync = parsed_options[option_name].as<bool>();
};
static void no_opengl_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    // User cannot overwrite default value when there is no openGL present
#ifdef MEGAMOL_USE_OPENGL
    config.no_opengl = parsed_options[option_name].as<bool>();
#endif
};
static void force_window_size_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.force_window_size = parsed_options[option_name].as<bool>();
};
static void fullscreen_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::fullscreen;
};
static void nodecoration_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::nodecoration;
};
static void topmost_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::topmost;
};
static void nocursor_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::nocursor;
};
static void hidden_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::hidden;
};


static void interactive_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    config.interactive = parsed_options[option_name].as<bool>();
};

static void window_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
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

static void framebuffer_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    auto string = parsed_options[option_name].as<std::string>();
    // WIDTHxHEIGHT
    std::regex geometry("(\\d+)x(\\d+)");
    std::smatch match;
    if (std::regex_match(string, match, geometry)) {
        unsigned int width = std::stoul(match[1].str(), nullptr, 10);
        unsigned int height = std::stoul(match[2].str(), nullptr, 10);
        config.local_framebuffer_resolution = {{width, height}};
    } else {
        exit("framebuffer option needs to be in the following format: WIDTHxHEIGHT, e.g. 200x100");
    }
};

static void viewport_tile_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    auto string = parsed_options[option_name].as<std::string>();
    // x,y;LWIDTHxLHEIGHT;GWIDTHxGHEIGHT
    std::regex geometry("(\\d+),(\\d+):(\\d+)x(\\d+):(\\d+)x(\\d+)");
    std::smatch match;
    if (std::regex_match(string, match, geometry)) {
        using UintPair = std::pair<unsigned int, unsigned int>;

        auto read_pair = [&](int index) {
            return UintPair{
                std::stoul(match[index].str(), nullptr, 10), std::stoul(match[index + 1].str(), nullptr, 10)};
        };

        UintPair start_pixel = read_pair(1);
        UintPair local_resolution = read_pair(3);
        UintPair global_resolution = read_pair(5);

        config.local_viewport_tile = {global_resolution, start_pixel, local_resolution};
    } else {
        exit("viewport tile option needs to be in the following format: x,y:LWIDTHxLHEIGHT:GWIDTHxGHEIGHT, e.g. "
             "0,0:200x100:400x200");
    }
};

static void vr_service_handler(
    std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config) {
    auto string = parsed_options[option_name].as<std::string>();
    // --vr=[off|unitykolab]

    std::vector<std::pair<std::string, RuntimeConfig::VRMode>> options = {
        {"off", RuntimeConfig::VRMode::Off},
#ifdef MEGAMOL_USE_VR_INTEROP
        {"unitykolab", RuntimeConfig::VRMode::UnityKolab},
#endif // MEGAMOL_USE_VR_INTEROP
    };

    auto match = [&](std::string const& string) -> RuntimeConfig::VRMode {
        auto find = std::find_if(options.begin(), options.end(), [&](auto const& opt) { return opt.first == string; });

        if (find != options.end())
            return find->second;

        exit("vr service cli option needs to be one of the following: " +
             std::accumulate(options.begin(), options.end(), std::string{},
                 [](auto const& a, auto const& b) { return a + "  " + b.first; }));
    };

    config.vr_mode = match(string);
};

using OptionsListEntry = std::tuple<std::string, std::string, std::shared_ptr<cxxopts::Value>,
    std::function<void(std::string const&, cxxopts::ParseResult const&, megamol::frontend::RuntimeConfig&)>>;

std::vector<OptionsListEntry> cli_options_list =
    { // config name         option description                                                                 type                                        handler
        {config_option, "Path to Lua configuration file(s)", cxxopts::value<std::vector<std::string>>(),
            config_handler},
        {appdir_option, "Set application directory", cxxopts::value<std::string>(), appdir_handler},
        {resourcedir_option, "Add resource directory(ies)", cxxopts::value<std::vector<std::string>>(),
            resourcedir_handler},
        {shaderdir_option, "Add shader directory(ies)", cxxopts::value<std::vector<std::string>>(), shaderdir_handler},
        {logfile_option, "Set log file", cxxopts::value<std::string>(), logfile_handler},
        {loglevel_option, "Set logging level, accepted values: " + accepted_log_level_strings,
            cxxopts::value<std::string>(), loglevel_handler},
        {echolevel_option, "Set echo level, accepted values see above", cxxopts::value<std::string>(),
            echolevel_handler},
        {project_option, "Project file(s) to load at startup", cxxopts::value<std::vector<std::string>>(),
            project_handler},
        {execute_lua_option, "Execute Lua command(s). Commands separated by ;", cxxopts::value<std::string>(),
            execute_lua_handler},
        {global_option, "Set global key-value pair(s) in MegaMol environment, syntax: --global key:value",
            cxxopts::value<std::vector<std::string>>(), global_value_handler}

        ,
        {host_option, "Address of lua host server", cxxopts::value<std::string>(), host_handler},
        {opengl_context_option, "OpenGL context to request: major.minor[core|compat], e.g. --opengl 3.2compat",
            cxxopts::value<std::string>(), opengl_context_handler},
        {khrdebug_option, "Enable OpenGL KHR debug messages", cxxopts::value<bool>(), khrdebug_handler},
        {vsync_option, "Enable VSync in OpenGL window", cxxopts::value<bool>(), vsync_handler},
        {disable_opengl_option, "Disable OpenGL. Always TRUE if not built with OpenGL", cxxopts::value<bool>(),
            no_opengl_handler},
        {window_option, "Set the window size and position, syntax: --window WIDTHxHEIGHT[+POSX+POSY]",
            cxxopts::value<std::string>(), window_handler},
        {force_window_size_option,
            "Force size of the window, otherwise the given size is only a recommendation for the window manager",
            cxxopts::value<bool>(), force_window_size_handler},
        {fullscreen_option, "Open maximized window", cxxopts::value<bool>(), fullscreen_handler},
        {nodecoration_option, "Open window without decorations", cxxopts::value<bool>(), nodecoration_handler},
        {topmost_option, "Open window that stays on top of all others", cxxopts::value<bool>(), topmost_handler},
        {nocursor_option, "Do not show mouse cursor inside window", cxxopts::value<bool>(), nocursor_handler},
        {hidden_option, "Do not show the window", cxxopts::value<bool>(), hidden_handler},
        {interactive_option, "Run MegaMol even if some project file failed to load", cxxopts::value<bool>(),
            interactive_handler},
        {project_files_option, "Project file(s) to load at startup", cxxopts::value<std::vector<std::string>>(),
            project_handler},
        {guishow_option, "Render GUI overlay", cxxopts::value<bool>(), guishow_handler},
        {nogui_option, "Dont render GUI overlay", cxxopts::value<bool>(), nogui_handler},
        {guiscale_option, "Set scale of GUI, expects float >= 1.0. e.g. 1.0 => 100%, 2.1 => 210%",
            cxxopts::value<float>(), guiscale_handler},
        {privacynote_option, "Show privacy note when taking screenshot, use '=false' to disable",
            cxxopts::value<bool>(), privacynote_handler},
        {versionnote_option, "Show version warning when loading a project, use '=false' to disable",
            cxxopts::value<bool>(), versionnote_handler},
        {flush_frequency_option, "Flush logs (performance, power, ...) every that many frames",
            cxxopts::value<uint32_t>(), flush_frequency_handler}
#ifdef MEGAMOL_USE_PROFILING
        ,
        {profile_log_option, "Enable performance counters and set output to file", cxxopts::value<std::string>(),
            profile_log_handler},
        {profile_log_no_autostart_option, "Do not automatically start writing the profiling log",
            cxxopts::value<bool>(), profile_log_autostart_handler},
        {profile_log_include_events_option, "Include graph events in the profiling log", cxxopts::value<bool>(),
            profile_log_include_events_handler}

#endif
        ,
        {param_option, "Set MegaMol Graph parameter to value: --param param=value",
            cxxopts::value<std::vector<std::string>>(), param_handler},
        {remote_head_option, "Start HeadNode server and run Remote_Service test ", cxxopts::value<bool>(),
            remote_head_handler},
        {remote_render_option, "Start RenderNode client and run Remote_Service test ", cxxopts::value<bool>(),
            remote_render_handler},
        {remote_mpi_option, "Start MPI RenderNode client and run Remote_Service test ", cxxopts::value<bool>(),
            remote_mpirender_handler},
        {remote_mpi_broadcast_rank_option, "MPI rank that broadcasts to others, default: 0",
            cxxopts::value<unsigned int>(), remote_mpirank_handler},
        {remote_headnode_zmq_target_option, "Address and port where to send state via ZMQ",
            cxxopts::value<std::string>(), remote_zmqtarget_handler},
        {remote_rendernode_zmq_source_option, "Address and port where to receive state via ZMQ from",
            cxxopts::value<std::string>(), remote_zmqsource_handler},
        {remote_headnode_broadcast_quit_option, "Headnode broadcasts mmQuit to rendernodes on shutdown",
            cxxopts::value<bool>(), remote_head_broadcast_quit_handler},
        {remote_headnode_broadcast_project_option,
            "Headnode broadcasts initial graph state after project loading at startup", cxxopts::value<bool>(),
            remote_head_broadcast_project_handler},
        {remote_headnode_connect_at_start_option, "Headnode starts sender thread at startup", cxxopts::value<bool>(),
            remote_head_connect_at_start_handler},
        {framebuffer_option, "Size of framebuffer, syntax: --framebuffer WIDTHxHEIGHT", cxxopts::value<std::string>(),
            framebuffer_handler},
        {viewport_tile_option,
            "Geometry of local viewport tile, syntax: --tile x,y:LWIDTHxLHEIGHT:GWIDTHxGHEIGHT"
            "where x,y is the lower left start pixel of the local tile, "
            "LWIDTHxLHEIGHT is the local framebuffer resolution, "
            "GWIDTHxGHEIGHT is the global framebuffer resolution",
            cxxopts::value<std::string>(), viewport_tile_handler},
        {vr_service_option, "VR Service mode: --vr=[off|unitykolab], off by default", cxxopts::value<std::string>(),
            vr_service_handler},
        {help_option, "Print help message", cxxopts::value<bool>(), empty_handler}};

static std::string loong(std::string const& option) {
    auto f = option.find(',');
    if (f == std::string::npos)
        return option;

    return option.substr(f + 1);
}

std::vector<std::string> megamol::frontend::extract_config_file_paths(const int argc, const char** argv) {
    // load config files from default paths
    // setting Config files from Lua is not possible
    // multiple Config files can be passed via CLI - then default config file paths are ignored

    // find config options in CLI string and overwrite default paths
    cxxopts::Options options("Config option pass", "MegaMol Config Parsing");
    options.add_options()(config_option, "", cxxopts::value<std::vector<std::string>>());

    options.allow_unrecognised_options();

    try {
        auto parsed_options = options.parse(argc, argv);

        std::string config_file_name = "megamol_config.lua";
        auto user_dir_config = getHomeDir() / ("." + config_file_name);
        auto executable_dir_config = getExecutableDirectory() / config_file_name;
        //auto current_dir_config = std::filesystem::absolute(std::filesystem::path(".")) / config_file_name;

        // the personal config should be loaded after the default config to overwrite it
        auto default_paths = {executable_dir_config, user_dir_config};

        std::vector<std::string> config_files;

        if (parsed_options.count(loong(config_option)) == 0) {
            // no config options given, look at default config paths
            for (auto config_path : default_paths) {
                if (std::filesystem::exists(config_path)) {
                    config_files.push_back(config_path.string());
                }
            }
        } else {
            auto cli_config_files = parsed_options[loong(config_option)].as<std::vector<std::string>>();
            config_files.insert(config_files.end(), cli_config_files.begin(), cli_config_files.end());
        }

        // remove empty files: enables to start megamol without config file
        std::remove_if(config_files.begin(), config_files.end(), [](auto const& file) { return file.empty(); });

        // check remaining files exist
        files_exist(config_files, "Config file");

        if (config_files.empty()) {
            std::cout << "WARNING: starting MegaMol without config files! Some MegaMol modules may fail due to missing "
                         "resource paths."
                      << std::endl;
        }

        return config_files;

    } catch (cxxopts::exceptions::exception ex) {
        exit(ex.what());
    }
}

#define add_option(X) (std::get<0>(X), std::get<1>(X), std::get<2>(X))

megamol::frontend_resources::RuntimeConfig megamol::frontend::handle_config(
    RuntimeConfig config, megamol::core::LuaAPI& lua) {

    // the parsing of CLI options inside Lua callbacks is somewhat of a mess
    // we create lua callbacks on the fly and pass them to Lua to execute during config interpretation
    // those callbacks fill our local fake "cli inputs" arrays, which we then feed into the CLI interpreter (cxxopt)

    using StringPair = megamol::frontend_resources::RuntimeConfig::StringPair;
    std::vector<StringPair> cli_options_from_configs;

    using megamol::frontend_resources::LuaCallbacksCollection;
    using Error = megamol::frontend_resources::LuaCallbacksCollection::LuaError;
    using VoidResult = megamol::frontend_resources::LuaCallbacksCollection::VoidResult;
    using StringResult = megamol::frontend_resources::LuaCallbacksCollection::StringResult;

#define sane(s)                                                  \
    if (s.empty() || s.find_first_of(" =") != std::string::npos) \
        return Error{"Value \"" + s + "\" is empty, has space or ="};

#define file_exists(f)               \
    if (!std::filesystem::exists(f)) \
        return Error{"File does not exist: " + f};

#define add_cli(o, v) cli_options_from_configs.push_back({loong(o), v});

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

    std::vector<std::string> all_options_separate;
    for (auto& opt : cli_options_list) {
        // split "h,help"
        auto& name = std::get<0>(opt);
        auto delimiter = name.find(",");
        if (delimiter == std::string::npos) {
            all_options_separate.push_back(name);
        } else {
            all_options_separate.push_back(name.substr(0, delimiter));
            all_options_separate.push_back(name.substr(delimiter + 1));
        }
    }

    LuaCallbacksCollection lua_config_callbacks;

    // mmSetCliOption
    // VoidResult, std::string, std::string
    lua_config_callbacks.add<VoidResult, std::string, std::string>("mmSetCliOption",
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
                finalopt = *(option_it + 1);

            add_cli(finalopt, map_value(value));
            return VoidResult{};
        }});

    // mmSetGlobalValue
    // std::string, std::string
    lua_config_callbacks.add<VoidResult, std::string, std::string>("mmSetGlobalValue",
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

    lua_config_callbacks.add<StringResult>("mmGetMegaMolExecutableDirectory",
        "()\n\tReturns the directory of the running MegaMol executable.",
        {[&]() { return StringResult{config.megamol_executable_directory}; }});

    lua_config_callbacks.add<VoidResult, std::string>("mmSetAppDir",
        "(string dir)\n\tSets the path where the mmconsole.exe is located.",
        {make_option_callback(appdir_option, true)});

    lua_config_callbacks.add<VoidResult, std::string>("mmAddShaderDir",
        "(string dir)\n\tAdds a shader/btf search path.", {make_option_callback(shaderdir_option, true)});

    lua_config_callbacks.add<VoidResult, std::string>("mmAddResourceDir",
        "(string dir)\n\tAdds a resource search path.", {make_option_callback(resourcedir_option, true)});

    lua_config_callbacks.add<VoidResult, std::string>("mmLoadProject",
        "(string path)\n\tLoad lua (project) file after MegaMol startup.",
        {make_option_callback(project_option, true)});

    lua_config_callbacks.add<VoidResult, std::string>(
        "mmSetLogFile", "(string path)\n\tSets the full path of the log file.", {make_option_callback(logfile_option)});

    lua_config_callbacks.add<VoidResult, std::string>("mmSetLogLevel",
        "(string level)\n\tSets the level of log events to include. Accepted values: " + accepted_log_level_strings,
        {make_option_callback(loglevel_option)});

    lua_config_callbacks.add<VoidResult, std::string>("mmSetEchoLevel",
        "(string level)\n\tSets the level of log events to output to the console. Accepted values: " +
            accepted_log_level_strings,
        {make_option_callback(echolevel_option)});

    lua.AddCallbacks(lua_config_callbacks);

    for (auto& file : config.configuration_files) {
        cli_options_from_configs.clear();
        std::ifstream stream(file);
        std::string file_contents =
            std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());

        // interpret lua config commands as CLI commands
        std::string lua_result_string;
        bool lua_config_ok = lua.RunString(file_contents, lua_result_string);

        if (!lua_config_ok) {
            exit("Error in Lua config file " + file + "\n Lua Error: " + lua_result_string);
        }

        // feed the options coming from Lua into CLI parser, which writes option changes into the RuntimeConfig
        std::vector<std::string> file_contents_as_cli{cli_options_from_configs.size()};
        for (auto& pair : cli_options_from_configs)
            file_contents_as_cli.push_back("--" + pair.first + "=" + pair.second);

        int argc = file_contents_as_cli.size() + 1;
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
            options.add_options() add_option(opt);
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

        } catch (cxxopts::exceptions::exception ex) {
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


megamol::frontend_resources::RuntimeConfig megamol::frontend::handle_cli(
    RuntimeConfig config, const int argc, const char** argv) {

    cxxopts::Options options(argv[0], "MegaMol Frontend 3000");

    for (int i = 0; i < argc; i++)
        config.program_invocation_string += std::string{argv[i]} + " ";

    // parse input project files
    options.positional_help("<additional project files>");
    for (auto& opt : cli_options_list) {
        options.add_options() add_option(opt);
    }
    options.parse_positional({project_files_option});

    // actually process passed options
    try {
        auto parsed_options = options.parse(argc, argv);
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

    } catch (cxxopts::exceptions::exception ex) {
        exit(std::string(ex.what()) + "\n" + options.help({""}));
    }

    return config;
}
