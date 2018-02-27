// ConClient.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <zmq.hpp>
#include <string>
#include <iostream>
#include <thread>
#include "Console.h"
#include "cxxopts.hpp"

#ifdef _WIN32
int _tmain(int argc, _TCHAR* argv[]) {
#else /* _WIN32 */
int main(int argc, char* argv[]) {
#endif /* _WIN32 */
    using std::cout;
    using std::endl;

    std::string host;
    std::string file, script;
    bool keepOpen = false;

    cxxopts::Options options("remoteconsole.exe", "MegaMol Remote Lua Console Client");
    options.add_options()
        ("open", "open host", cxxopts::value<std::string>())
        ("source", "source file", cxxopts::value<std::string>())
        ("exec", "execute script", cxxopts::value<std::string>())
        ("keep-open", "keep open")
        ("help", "print help")
        ;

    try {


        auto parseRes = options.parse(argc, argv);

        if (parseRes.count("help")) {
            std::cout << options.help({ "" }) << std::endl;
            exit(0);
        }

        // greeting text
        printGreeting();

        if (parseRes.count("open")) host = parseRes["open"].as<std::string>();
        if (parseRes.count("source")) file = parseRes["source"].as<std::string>();
        if (parseRes.count("exec")) script = options["exec"].as<std::string>();
        if (parseRes.count("keep-open")) keepOpen = parseRes["keep-open"].as<bool>();


        //  Prepare our context and socket
        zmq::context_t context(1);
        zmq::socket_t socket(context, ZMQ_REQ);
        Connection conn(socket);

        if (!host.empty()) {
            cout << "Connecting \"" << host << "\" ... ";
            try {
                conn.Connect(host);
                cout << endl
                    << "\tConnected" << endl
                    << endl;

            }
            catch (std::exception& ex) {
                cout << endl
                    << "ERR Socket connection failed: " << ex.what() << endl
                    << endl;
            }
            catch (...) {
                cout << endl
                    << "ERR Socket connection failed: unknown exception" << endl
                    << endl;
            }
        }

        if (!file.empty()) {
            runScript(conn, file);
        } else if (!script.empty()) {
            if (!execCommand(conn, script)) {
                cout << "\tFailed" << endl;
            }
        } else {
            keepOpen = true;
        }

        if (keepOpen) {
            interactiveConsole(conn);
        }

    }
    catch (...) {
        std::cout << options.help({ "" }) << std::endl;
        exit(0);
    }

    return 0;
}

