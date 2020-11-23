#include "EnzymeClassProvider.h"

#include <sstream>
#include "mmcore/utility/log/Log.h"

using namespace megamol;
using namespace megamol::molsurfmapcluster;

/*
 * EnzymeClassProvider::classMap
 */
std::multimap<std::string, std::array<int, 4>> EnzymeClassProvider::classMap =
    std::multimap<std::string, std::array<int, 4>>();

/*
 * EnzymeClassProvider::EnzymeClassProvider
 */
EnzymeClassProvider::EnzymeClassProvider(void) {
    // intentionally empty
}

/*
 * EnzymeClassProvider::~EnzymeClassProvider
 */
EnzymeClassProvider::~EnzymeClassProvider(void) {
    // intentionally empty
}

/*
 * EnzymeClassProvider::RetrieveEnzymeClassMap
 */
const std::multimap<std::string, std::array<int, 4>>& EnzymeClassProvider::RetrieveEnzymeClassMap(
    const core::CoreInstance& coreInstance) {
    if (classMap.size() == 0) {
        loadMapFromFile(coreInstance);
    }
    return classMap;
}

/*
 * EnzymeClassProvider::RetrieveClassForPdbId
 */
std::array<int, 4> EnzymeClassProvider::RetrieveClassForPdbId(
    std::string pdbId, const core::CoreInstance& coreInstance) {
    std::array<int, 4> result = {-1, -1, -1, -1};
    if (classMap.size() == 0) {
        loadMapFromFile(coreInstance);
    }
    auto it = classMap.lower_bound(pdbId);
    if (it != classMap.end()) {
        result = (*it).second;
    }
    return result;
}

/*
 * EnzymeClassProvider::RetrieveClassesForPdbId
 */
std::vector<std::array<int, 4>> EnzymeClassProvider::RetrieveClassesForPdbId(
    std::string pdbId, const core::CoreInstance& coreInstance) {
    std::vector<std::array<int, 4>> result;
    if (classMap.size() == 0) {
        loadMapFromFile(coreInstance);
    }
    auto lower = classMap.lower_bound(pdbId);
    auto upper = classMap.upper_bound(pdbId);
    for (; lower != upper; lower++) {
        result.push_back(lower->second);
    }
    return result;
}

/*
 * EnzymeClassProvider::ConvertEnzymeClassToString
 */
std::string EnzymeClassProvider::ConvertEnzymeClassToString(const std::array<int, 4>& enzymeClass) {
    std::string result = "";
    result += enzymeClass[0] != -1 ? std::to_string(enzymeClass[0]) : "";
    result += enzymeClass[1] != -1 ? "." + std::to_string(enzymeClass[1]) : "";
    result += enzymeClass[2] != -1 ? "." + std::to_string(enzymeClass[2]) : "";
    result += enzymeClass[3] != -1 ? "." + std::to_string(enzymeClass[3]) : "";
    return result;
}

/*
 * EnzymeClassProvider::loadMapFromFile
 */
void EnzymeClassProvider::loadMapFromFile(const core::CoreInstance& coreInstance) {
    const auto filepath = determineFilePath(coreInstance);
    classMap.clear();
    std::ifstream file(filepath);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::pair<std::string, std::array<int, 4>> res;
            res.second = {-1, -1, -1, -1};
            // split line by comma
            std::string substr;
            std::stringstream ss(line);
            int i = 0;
            while (ss.good()) {
                std::getline(ss, substr, ',');
                if (i == 0) {
                    std::string idsubstr;
                    std::stringstream ss2(substr);
                    int j = 0;
                    while (ss2.good()) {
                        std::getline(ss2, idsubstr, '.');
                        int val;
                        try {
                            val = std::stoi(idsubstr);
                        } catch (...) { val = -1; }
                        res.second[j] = val;
                        ++j;
                    }
                } else if (i == 1) {
                    res.first = substr;
                }
                ++i;
            }
            classMap.insert(res);
        }
    } else {
        core::utility::log::Log::DefaultLog.WriteError(
            "Could not load the configuration file \"%s\"", filepath.c_str());
    }
}

/*
 * EnzymeClassProvider::determineFilePath
 */
std::filesystem::path EnzymeClassProvider::determineFilePath(const core::CoreInstance& coreInstance) {
    std::filesystem::path result;
    vislib::StringA shortfile = "brenda_enzyme_map.csv";
    auto fname = core::utility::ResourceWrapper::getFileName(coreInstance.Configuration(), shortfile);
    result = std::filesystem::path(W2A(fname));
    return result.make_preferred();
}
