/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>

#include <cxxopts.hpp>

#include "Connection.h"
#include "ConnectionFactory.h"

std::string setHammerIndex(std::string const& str, uint32_t idx) {
    static const std::regex r("%%i%%");
    return std::regex_replace(str, r, std::to_string(idx));
}

void execCommand(std::vector<std::unique_ptr<Connection>> const& connections, std::string const& cmd, uint32_t hammer) {
    if (hammer == 0) {
        std::cout << "Exec: " << cmd << std::endl;
        std::cout << "Reply: " << connections[0]->sendCommand(cmd) << std::endl;
    } else {
        for (uint32_t i = 0; i < hammer; i++) {
            const std::string cmd_i = setHammerIndex(cmd, i);
            std::cout << "Exec: " << cmd_i << std::endl;
            std::cout << "Reply [" << i << "]: " << connections[i]->sendCommand(cmd_i) << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    cxxopts::Options options("remoteconsole.exe", "MegaMol Remote Lua Console Client");
    // clang-format off
    options.add_options()
        ("c,connect", "Remote host address.", cxxopts::value<std::string>())
        ("e,exec", "Execute lua command.", cxxopts::value<std::string>())
        ("s,script", "Execute lua script file.", cxxopts::value<std::string>())
        ("scriptdelay", "Delay between lines in script (in ms).", cxxopts::value<int>())
        ("timeout", "Time to wait for a MegaMol response (in ms, default 10000)", cxxopts::value<int>())
        ("hammer", "Run with multiple connections. Replaces %%i%% with index.", cxxopts::value<int>())
        ("h,help", "Print help.");
    // clang-format on

    std::string host = "tcp://127.0.0.1:33333";
    std::string exec;
    std::string script;
    uint32_t scriptDelay = 0;
    uint32_t timeout = 10000;
    uint32_t hammer = 0;

    try {
        auto parseResult = options.parse(argc, argv);

        if (parseResult.count("help")) {
            std::cout << options.help({""}) << std::endl;
            exit(0);
        }

        if (parseResult.count("connect")) {
            host = parseResult["connect"].as<std::string>();
        }
        if (parseResult.count("exec")) {
            exec = parseResult["exec"].as<std::string>();
        }
        if (parseResult.count("script")) {
            if (!exec.empty()) {
                throw std::runtime_error("Cannot use exec and script option together!");
            }
            script = parseResult["script"].as<std::string>();
            if (parseResult.count("scriptdelay")) {
                const int val = parseResult["scriptdelay"].as<int>();
                if (val < 0) {
                    throw std::runtime_error("Invalid value for scriptdelay!");
                }
                scriptDelay = static_cast<uint32_t>(val);
            }
        }
        if (parseResult.count("timeout")) {
            const int val = parseResult["timeout"].as<int>();
            if (val < 0) {
                throw std::runtime_error("Invalid value for timeout!");
            }
            timeout = static_cast<uint32_t>(val);
        }
        if (parseResult.count("hammer")) {
            const int val = parseResult["hammer"].as<int>();
            if (val < 0) {
                throw std::runtime_error("Invalid value for hammer!");
            }
            hammer = static_cast<uint32_t>(val);
        }
    } catch (std::exception const& ex) {
        std::cerr << "Error parsing arguments: " << ex.what() << std::endl;
        std::cout << options.help({""}) << std::endl;
        exit(1);
    } catch (...) {
        std::cout << options.help({""}) << std::endl;
        exit(1);
    }

    try {
        std::cout << "MegaMol Remote Lua Console Client" << std::endl << std::endl;

        ConnectionFactory factory(host);
        std::vector<std::unique_ptr<Connection>> connections;

        for (uint32_t i = 0; i < std::max(hammer, 1u); i++) {
            connections.emplace_back(factory.createConnection(timeout));
        }

        if (!exec.empty()) {
            execCommand(connections, exec, hammer);
        } else if (!script.empty()) {
            std::ifstream file(script);
            if (scriptDelay == 0) {
                std::stringstream buf;
                buf << file.rdbuf();
                execCommand(connections, buf.str(), hammer);
            } else {
                std::string line;
                while (std::getline(file, line)) {
                    if (!line.empty()) {
                        execCommand(connections, line, hammer);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(scriptDelay));
                }
            }
        } else {
            bool running = true;
            while (running) {
                std::string cmd;
                std::cout << "> ";
                std::getline(std::cin, cmd);
                if (cmd == "exit") {
                    running = false;
                } else {
                    execCommand(connections, cmd, hammer);
                }
            }
        }

        // Cleanup sockets
        connections.clear();
    } catch (std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "Error: Unknown!" << std::endl;
        exit(1);
    }

    return 0;
}
