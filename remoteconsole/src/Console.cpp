#include "Console.h"

#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

void printGreeting() {
    std::cout << std::endl
              << "MegaMol Remote Lua Console Client" << std::endl
              << "Copyright 2017 by MegaMol Team" << std::endl
              << std::endl;
}

bool execCommand(Connection& conn, const std::string& command, const int index) {
    using std::cout;
    using std::endl;
    using std::string;

    const std::string from = "%%i%%";
    const std::string to = std::to_string(index);
    std::string c2 = command;

    if (index > -1) {
        size_t start_pos = 0;
        while ((start_pos = c2.find(from, start_pos)) != std::string::npos) {
            c2.replace(start_pos, from.length(), to);
            start_pos += to.length();
        }
    }

    if (!conn.Connected()) {
        cout << "Socket not connected" << endl << endl;
        return false;
    }
    cout << "Reply: " << endl << conn.sendCommand(c2) << endl << endl;
    return true;
}

void runScript(Connection& conn, const std::string& scriptfile, const bool singleSend, const int index) {
    using std::cin;
    using std::cout;
    using std::endl;
    using std::string;

    cout << "Loading commands from \"" << scriptfile << "\"" << endl;

    std::ifstream file(scriptfile);

    if (singleSend) {
        while (!file.eof()) {
            std::string line;
            // if (std::getline(file, line).eof()) break;
            std::getline(file, line);

            cout << line << endl;

            if (!line.empty()) {
                if (!execCommand(conn, line, index)) {
                    cout << "\tFailed" << endl;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } else {
        std::stringstream buf;
        buf << file.rdbuf();
        cout << buf.str() << endl;
        if (!execCommand(conn, buf.str(), index)) {
            cout << "\tFailed" << endl;
        }
    }

    cout << "Script completed" << endl << endl;
}

void interactiveConsole(Connection& conn) {
    using std::cin;
    using std::cout;
    using std::endl;
    using std::string;

    bool running = true;
    while (running) {
        std::string command;
        cout << "> ";
        std::getline(cin, command);

        // todo broken
        std::string opCode;
        std::istringstream iss(command);
        std::getline(iss, opCode, ' ');
        if (opCode == "open") {
            //  OPEN [host]              -  Establishes connection to the host
            if (conn.Disconnect()) {
                cout << "Socket closed" << endl << endl;
            }
            try {
                std::string host;
                std::getline(iss, host);
                cout << "Connecting \"" << host << "\" ... ";

                bool rv = conn.Connect(host);
                assert(rv);

                cout << endl << "\tConnected" << endl << endl;

            } catch (std::exception& ex) {
                cout << endl << "ERR Socket connection failed: " << ex.what() << endl << endl;
            } catch (...) { cout << endl << "ERR Socket connection failed: unknown exception" << endl << endl; }

        } else if (opCode == "close") {
            //  CLOSE                    -  Closes connection to the host
            if (conn.Disconnect()) {
                cout << "Socket closed" << endl << endl;
            } else {
                cout << "Socket not connected" << endl << endl;
            }

        } else if (opCode == "status") {
            //  STATUS                   -  Informs about the current connection
            cout << "Socket " << (conn.Connected() ? "connected" : "not connected") << endl << endl;

        } else {
            if (!execCommand(conn, command)) {
                cout << "ERR unknown command" << endl << endl;
            }
        }
    }
}
