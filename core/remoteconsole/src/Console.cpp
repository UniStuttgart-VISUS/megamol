#include "stdafx.h"
#include "Console.h"
#include <iostream>
#include <cctype>
#include <fstream>
#include <sstream>
#include <thread>

void printGreeting() {
    std::cout << std::endl
        << "MegaMol SimpleParamRemote ConClient" << std::endl
        << "Copyright 2016 by MegaMol Team, TU Dresden" << std::endl
        << std::endl;
}

void printHelp() {
    using std::cout;
    using std::endl;

    printGreeting();

    cout << "Syntax:" << endl
        << "\tConClient.exe [-o host [-s file [-k] ] ]" << endl
        << endl;
    cout << "\t-o [host]  -  Establishes connection the specified host" << endl
        << "\t-s [file]  -  opens the specified text file and treats each line as command to be sent to the host." << endl
        << "\t              You may only use the four commands from the last command block (see below)." << endl
        << "\t-k         -  Keeps the interactive command prompt open after the script file was executed." << endl
        << endl;

    cout << "Commands:" << endl
        << endl
        << "\tOPEN [host]              -  Establishes connection to the host" << endl
        << "\tCLOSE                    -  Closes connection to the host" << endl
        << "\tSTATUS                   -  Informs about the current connection" << endl
        << endl
        << "\tHELP                     -  Prints command line syntax and this info" << endl
        << "\tEXIT                     -  Closes this program" << endl
        << endl
        << "\tQUERYPARAMS              -  Answers all parameter names" << endl
        << "\tGETTYPE [name]           -  Answers the type descriptor of one parameter" << endl
        << "\tGETDESC [name]           -  Answers the human-readable description of one parameter" << endl
        << "\tGETVALUE [name]          -  Answers the value of one parameter" << endl
        << "\tSETVALUE [name] [value]  -  Sets the value of one parameter" << endl
        << "\tGETPROCESSID             -  Answers the native process id of the host (if protocol MMSPRHOSTINFO is supported)" << endl
        << endl
        << "\tGETMODULEPARAMS [name]   -  Lists all parameters of a module along with their description, type, and value" << endl
        << "\tQUERYMODULES             -  Lists all instantiated modules along with their description and type" << endl
        << "\tQUERYCALLS               -  Lists all instantiated calls along with their description and typee" << endl
        << endl;
}

namespace {

template<class T>
bool execCommand(Connection& conn, std::string command, T& paramStream) {
    using std::cout;
    using std::endl;
    using std::string;

    if (command == "QUERYPARAMS") {
        //  QUERYPARAMS              -  Answers all parameter names
        if (!conn.Connected()) {
            cout << "Socket not connected" << endl
                << endl;
            return false;
        }

        cout << "Reply: " << endl
            << conn.sendCommand(command) << endl
            << endl;

    } else if (command == "GETTYPE") {
        //  GETTYPE [name]           -  Answers the type descriptor of one Parameter
        string name;
        paramStream >> name;
        if (!conn.Connected()) {
            cout << "Socket not connected" << endl
                << endl;
            return false;
        }

        command += " " + name;

        std::cout << "Reply: " << endl
            << conn.sendCommand(command) << std::endl
            << std::endl;

    } else if (command == "GETDESC") {
        //  GETDESC [name]           -  Answers the human-readable description of one parameter
        string name;
        paramStream >> name;
        if (!conn.Connected()) {
            cout << "Socket not connected" << endl
                << endl;
            return false;
        }

        command += " " + name;

        std::cout << "Reply: " << endl
            << conn.sendCommand(command) << std::endl
            << std::endl;

    } else if (command == "GETVALUE") {
        //  GETVALUE [name]          -  Answers the value of one Parameter
        string name;
        paramStream >> name;
        if (!conn.Connected()) {
            cout << "Socket not connected" << endl
                << endl;
            return false;
        }

        command += " " + name;

        std::cout << "Reply: " << endl
            << conn.sendCommand(command) << std::endl
            << std::endl;

    } else if (command == "SETVALUE") {
        //  SETVALUE [name] [value]  -  Sets the value of one Parameter
        string name, value, vx;
        paramStream >> name >> value;
        std::getline(paramStream, vx);
        value += vx;
        if (!conn.Connected()) {
            cout << "Socket not connected" << endl
                << endl;
            return false;
        }

        command += " " + name + " " + value;

        std::cout << "Reply: " << endl
            << conn.sendCommand(command) << std::endl
            << std::endl;

    } else if (command == "GETPROCESSID") {
        if (!conn.Connected()) {
            cout << "Socket not connected" << endl
                << endl;
            return false;
        }
        if (!conn.supportProtocol("MMSPRHOSTINFO")) {
            cout << "Host does not support MMSPRHOSTINFO protocol" << endl
                << endl;
            return false;
        }
        std::cout << "Reply: " << endl
            << conn.sendCommand(command) << std::endl
            << std::endl;

    } else if (command == "GETMODULEPARAMS") {
        string name;
        paramStream >> name;

        if (!conn.Connected()) {
            cout << "Socket not connected" << endl
                << endl;
            return false;
        }
        if (!conn.supportProtocol("MMSPR2")) {
            cout << "Host does not support MMSPR2 protocol" << endl
                << endl;
            return false;
        }

        command += " " + name;

        std::cout << "Reply: " << endl
            << conn.sendCommand(command) << std::endl
            << std::endl;

    } else if (command == "QUERYMODULES") {
        if (!conn.Connected()) {
            cout << "Socket not connected" << endl
                << endl;
            return false;
        }
        if (!conn.supportProtocol("MMSPR2")) {
            cout << "Host does not support MMSPR2 protocol" << endl
                << endl;
            return false;
        }

        std::cout << "Reply: " << endl
            << conn.sendCommand(command) << std::endl
            << std::endl;

    } else if (command == "QUERYCALLS") {

        if (!conn.Connected()) {
            cout << "Socket not connected" << endl
                << endl;
            return false;
        }
        if (!conn.supportProtocol("MMSPR2")) {
            cout << "Host does not support MMSPR2 protocol" << endl
                << endl;
            return false;
        }

        std::cout << "Reply: " << endl
            << conn.sendCommand(command) << std::endl
            << std::endl;

    } else {
        return false;
    }

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

        std::stringstream linestream(line);
        std::string command;
        linestream >> command;
        for (char& c : command) c = std::toupper(c);

        if (!execCommand(conn, command, linestream)) {
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
        cin >> command;
        for (char& c : command) c = std::toupper(c);

        if (command == "OPEN") {
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

        } else if (command == "CLOSE") {
            //  CLOSE                    -  Closes connection to the host
            if (conn.Disconnect()) {
                cout << "Socket closed" << endl
                    << endl;
            } else {
                cout << "Socket not connected" << endl
                    << endl;
            }

        } else if (command == "STATUS") {
            //  STATUS                   -  Informs about the current connection
            cout << "Socket " << (conn.Connected() ? "connected" : "not connected") << endl
                << endl;

        } else if (command == "HELP") {
            //  HELP                     -  Prints command line syntax and this info
            printHelp();

        } else if (command == "EXIT") {
            //  EXIT                     -  Closes this program
            running = false; // leave loop

        } else {
            if (!execCommand(conn, command, cin)) {
                cout << "ERR unknown command" << endl
                    << endl;
            }
        }

    }


}
