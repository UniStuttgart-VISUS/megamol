// ConClient.cpp : Defines the entry point for the console application.
//

#include "Console.h"
#include "stdafx.h"

#undef min
#undef max
#include <cxxopts.hpp>
#include <zmq.hpp>

#include <iostream>
#include <string>
#include <thread>

#ifdef _WIN32
int _tmain(int argc, _TCHAR* argv[]) {
#else  /* _WIN32 */
int main(int argc, char* argv[]) {
#endif /* _WIN32 */
    using std::cout;
    using std::endl;

    std::string host;
    std::string file, script;
    int hammerFactor = 1;
    int timeOutSeconds = 0;
    bool keepOpen = false;
    bool singleSend = false;

    cxxopts::Options options("remoteconsole.exe", "MegaMol Remote Lua Console Client");
    // clang-format off
    options.add_options()
        ("open", "open host", cxxopts::value<std::string>())
        ("source", "source file", cxxopts::value<std::string>())
        ("exec", "execute script", cxxopts::value<std::string>())
        ("keep-open", "keep open")
        ("hammer", "multi-connect, works only with exec or source. replaces %%i%% with index", cxxopts::value<int>())
        ("timeout", "max seconds to wait until MegaMol replies (default 10)", cxxopts::value<int>()->default_value("10"))
        ("single", "send whole file or script in one go")
        ("help", "print help");
    // clang-format on

    try {


        auto parseRes = options.parse(argc, argv);

        if (parseRes.count("help")) {
            std::cout << options.help({""}) << std::endl;
            exit(0);
        }

        // greeting text
        printGreeting();

        if (parseRes.count("open"))
            host = parseRes["open"].as<std::string>();
        if (parseRes.count("source"))
            file = parseRes["source"].as<std::string>();
        if (parseRes.count("exec"))
            script = parseRes["exec"].as<std::string>();
        if (parseRes.count("keep-open"))
            keepOpen = parseRes["keep-open"].as<bool>();
        if (parseRes.count("hammer"))
            hammerFactor = parseRes["hammer"].as<int>();
        timeOutSeconds = parseRes["timeout"].as<int>();
        if (parseRes.count("single"))
            singleSend = parseRes["single"].as<bool>();

        if (!parseRes.count("exec") && !parseRes.count("source")) {
            hammerFactor = 1;
        }

        //  Prepare our context and socket
        zmq::context_t context(1);
        zmq::socket_t pre_socket(context, ZMQ_REQ);
        const int replyLength = 1024;
        char portReply[replyLength];

        std::vector<zmq::socket_t> sockets;
        std::vector<Connection> connections;

        //zmq::socket_t socket(context, ZMQ_PAIR);
        //Connection conn(socket);

        for (int i = 0; i < hammerFactor; ++i) {
            sockets.emplace_back(context, ZMQ_PAIR);
            connections.emplace_back(sockets.back(), timeOutSeconds);
        }

        if (!host.empty()) {
            cout << "Connecting \"" << host << "\" ... ";
            try {

                pre_socket.connect(host);
                for (int i = 0; i < hammerFactor; ++i) {
                    pre_socket.send("ola", 3);
                    pre_socket.recv(portReply, replyLength);

                    int p2 = std::atoi(portReply);
                    const auto portPos = host.find_last_of(":");
                    const auto hostStr = host.substr(0, portPos);
                    std::stringstream newHost;
                    newHost << hostStr << ":" << p2;

                    cout << "Connecting to pair socket: " << newHost.str() << " ...";

                    connections[i].Connect(newHost.str());
                    cout << endl << "\tConnected to " << p2 << endl << endl;
                }
            } catch (zmq::error_t& zmqex) {
                cout << endl << "ERR Socket connection failed: " << zmqex.what() << endl << endl;
            } catch (std::exception& ex) {
                cout << endl << "ERR Socket connection failed: " << ex.what() << endl << endl;
            } catch (...) { cout << endl << "ERR Socket connection failed: unknown exception" << endl << endl; }
        }


        if (!file.empty()) {
            for (int i = 0; i < hammerFactor; ++i) {
                runScript(connections[i], file, singleSend, i);
            }
        } else if (!script.empty()) {
            for (int i = 0; i < hammerFactor; ++i) {
                if (!execCommand(connections[i], script, i)) {
                    cout << "\tFailed" << endl;
                }
            }
        } else {
            keepOpen = true;
        }

        if (keepOpen) {
            interactiveConsole(connections[0]);
        }

    } catch (...) {
        std::cout << options.help({""}) << std::endl;
        exit(0);
    }

    return 0;
}
