#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_HAS_CXX17) || ((defined(_MSC_VER) && (_MSC_VER > 1916))) // C++2017 or since VS2019
#include <filesystem>
namespace fs = std::filesystem;
#elif _WIN32
#include <filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#if _WIN32
#define the_popen _popen
#define the_pclose _pclose
#else
#define the_popen popen
#define the_pclose pclose
#endif

void append(std::stringstream& out_stream, std::istream* infile, const std::string& prefix, const std::string& suffix) {
    std::string line;
    out_stream << prefix << std::endl;
    while (std::getline(*infile, line)) {
        out_stream << "    R\"MM_Delim(" << line << ")MM_Delim\"" << std::endl;
    }
    out_stream << suffix << std::endl;
}

std::string exec(fs::path cmd) {
    std::array<char, 1024> buffer;
    std::string result;

    auto pipe = the_popen(cmd.string().c_str(), "r"); // get rid of shared_ptr

    if (!pipe)
        throw std::runtime_error("popen() failed!");

    while (!feof(pipe)) {
        if (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
            result += buffer.data();
    }

    auto rc = the_pclose(pipe);

    if (rc == EXIT_SUCCESS) { // == 0
        return result;
    } else {
        return "unable to execute " + cmd.string();
    }
}

int main(int argc, char* argv[]) {
    using namespace std::string_literals;

    if (argc != 5) {
        std::cerr << "state" << std::endl
                  << "Usage: ./state <buildtime.cpp> <Git executable> <CMake Cache file> <License file>" << std::endl;
        return 1;
    }

    fs::path buildtime(argv[1]);
    fs::path git_exe(argv[2]);
    fs::path cache_file(argv[3]);
    fs::path license(argv[4]);

    if (!fs::is_regular_file(buildtime)) {
        std::cerr << "buildtime config file not found: " << buildtime.string() << std::endl;
        return 1;
    }
    if (!fs::is_regular_file(git_exe)) {
        std::cerr << "Git executable not found: " << git_exe.string() << std::endl;
        return 1;
    }
    if (!fs::is_regular_file(cache_file)) {
        std::cerr << "CMake cache not found: " << cache_file.string() << std::endl;
        return 1;
    }
    if (!fs::is_regular_file(license)) {
        std::cerr << "License not found: " << git_exe.string() << std::endl;
        return 1;
    }

    std::stringstream out_stream;


    std::stringstream diff_stream(exec("\"" + git_exe.string() + "\" diff HEAD"));
    append(out_stream, &diff_stream, "char const* megamol::build_info::MEGAMOL_GIT_DIFF = ", ";");
    out_stream << std::endl;
    auto cstr = std::ifstream(cache_file);
    append(out_stream, &cstr, "char const* megamol::build_info::MEGAMOL_CMAKE = ", ";");
    out_stream << std::endl;
    cstr = std::ifstream(license);
    append(out_stream, &cstr, "char const* megamol::build_info::MEGAMOL_LICENSE = ", ";");

    std::ofstream out(buildtime, std::ios::app);
    out << out_stream.str();

    return 0;
}
