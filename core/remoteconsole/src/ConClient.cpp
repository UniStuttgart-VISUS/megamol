// ConClient.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <zmq.hpp>
#include <string>
#include <iostream>
#include <thread>
#include "Console.h"
#include "cxxopts.hpp"


std::string makeFrom(const char *s) {
    return s;
}

bool is_equal(const char *l, const char *r) {
    return std::string(l) == std::string(r);
}

std::string makeFrom(const wchar_t *s) {
    std::wstring ws(s);
    const std::locale locale("");
    typedef std::codecvt<wchar_t, char, std::mbstate_t> converter_type;
    const converter_type& converter = std::use_facet<converter_type>(locale);
    std::vector<char> to(ws.length() * converter.max_length());
    std::mbstate_t state;
    const wchar_t* from_next;
    char* to_next;
    const converter_type::result result = converter.out(state, ws.data(), ws.data() + ws.length(), from_next, &to[0], &to[0] + to.size(), to_next);
    if (result == converter_type::ok || result == converter_type::noconv) {
        const std::string s(&to[0], to_next);
        return s;
    }

    return "";
}

bool is_equal(const wchar_t *l, const char *r) {
    return makeFrom(l) == std::string(r);
}

template<class T>
void parseCommandLine(int argc, T** argv, std::string& outHost, std::string& outFile, bool& outKeepOpen) {
    using std::cout;
    using std::endl;
    // ConClient.exe [-o host [-s file [-k] ] ]
    outHost.clear();
    outFile.clear();
    outKeepOpen = false;

    if (argc < 3) return;
    if (!is_equal(argv[1], "-o")) {
        cout << "Error: arg1 expected -o instead of \"" << argv[1] << endl;
        return;
    }
    outHost = makeFrom(argv[2]);

    if (argc < 5) return;
    if (!is_equal(argv[3], "-s")) {
        cout << "Error: arg3 expected -s instead of \"" << argv[3] << endl;
        return;
    }
    outFile = makeFrom(argv[4]);

    if (argc < 6) return;
    if (!is_equal(argv[5], "-k")) {
        cout << "Error: arg5 expected -k instead of \"" << argv[5] << endl;
        return;
    }
    outKeepOpen = true;

}

#ifdef _WIN32
int _tmain(int argc, _TCHAR* argv[]) {
#else /* _WIN32 */
int main(int argc, char* argv[]) {
#endif /* _WIN32 */
    using std::cout;
    using std::endl;

    std::string host;
    std::string file;
    bool keepOpen = false;

    parseCommandLine(argc, argv, host, file, keepOpen);

    // greeting text
    printGreeting();

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

        } catch (std::exception& ex) {
            cout << endl
                << "ERR Socket connection failed: " << ex.what() << endl
                << endl;
        } catch (...) {
            cout << endl
                << "ERR Socket connection failed: unknown exception" << endl
                << endl;
        }
    }

    if (!file.empty()) { 
        runScript(conn, file);

    } else {
        keepOpen = true;
    }

    if (keepOpen) {
        interactiveConsole(conn);
    }

    return 0;
}

