#include "stdafx.h"
#include "Console.h"
#include <iostream>
#include <cctype>
#include <fstream>
#include <sstream>
#include <thread>

void printGreeting() {
    std::cout << std::endl
        << "MegaMol Remote Lua Console Client" << std::endl
        << "Copyright 2017 by MegaMol Team" << std::endl
        << std::endl;
}

namespace {

bool execCommand(Connection& conn, std::string command) {
    using std::cout;
    using std::endl;
    using std::string;

    if (!conn.Connected()) {
        cout << "Socket not connected" << endl
            << endl;
        return false;
    }
    cout << "Reply: " << endl
        << conn.sendCommand(command) << endl
        << endl;
    return true;
}

}

void runScript(Connection& conn, const std::string& scriptfile) {
    using std::cout;
    using std::cin;
    using std::endl;
    using std::string;

    cout << "Loading commands from \"" << scriptfile << "\"" << endl;

    std::ifstream file(scriptfile);
    while (!file.eof()) {
        std::string line;
        if (std::getline(file, line).eof()) break;

        cout << line << endl;

        if (!execCommand(conn, line)) {
            cout << "\tFailed" << endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    }

    cout << "Script completed" << endl
        << endl;

}

void interactiveConsole(Connection &conn) {
    using std::cout;
    using std::cin;
    using std::endl;
    using std::string;

    bool running = true;
    while (running) {
        std::string command;
        cout << "> ";
        std::getline(cin, command);

        // todo broken
        if (command == "open") {
            //  OPEN [host]              -  Establishes connection to the host
            if (conn.Disconnect()) {
                cout << "Socket closed" << endl
                    << endl;
            }
            try {
                std::string host;
                cin >> host;
                cout << "Connecting \"" << host << "\" ... ";

                bool rv = conn.Connect(host);
                assert(rv);

                cout << endl
                    << "\tConnected" << endl
                    << endl;

            } catch (std::exception& ex) {
                cout << endl
                    << "ERR Socket connection failed: " << ex.what() << endl
                    << endl;
            } catch (...) {
                cout << endl
                    << "ERR Socket connection failed: unknown exception" << endl
                    << endl;
            }

        } else if (command == "close") {
            //  CLOSE                    -  Closes connection to the host
            if (conn.Disconnect()) {
                cout << "Socket closed" << endl
                    << endl;
            } else {
                cout << "Socket not connected" << endl
                    << endl;
            }

        } else if (command == "status") {
            //  STATUS                   -  Informs about the current connection
            cout << "Socket " << (conn.Connected() ? "connected" : "not connected") << endl
                << endl;

        } else {
            if (!execCommand(conn, command)) {
                cout << "ERR unknown command" << endl
                    << endl;
            }
        }

    }


}
