#include "EnzymeClassProvider.h"

#include "mmcore/utility/log/Log.h"
#include <sstream>

using namespace megamol;
using namespace megamol::molsurfmapcluster_gl;

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
    frontend_resources::RuntimeConfig const& runtimeConf) {
    if (classMap.size() == 0) {
        loadMapFromFile(runtimeConf);
    }
    return classMap;
}

/*
 * EnzymeClassProvider::RetrieveClassForPdbId
 */
std::array<int, 4> EnzymeClassProvider::RetrieveClassForPdbId(
    std::string pdbId, frontend_resources::RuntimeConfig const& runtimeConf) {
    std::array<int, 4> result = {-1, -1, -1, -1};
    if (classMap.size() == 0) {
        loadMapFromFile(runtimeConf);
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
    std::string pdbId, frontend_resources::RuntimeConfig const& runtimeConf) {
    std::vector<std::array<int, 4>> result;
    if (classMap.size() == 0) {
        loadMapFromFile(runtimeConf);
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
 * EnzymeClassProvider::EnzymeClassDistance
 */
float EnzymeClassProvider::EnzymeClassDistance(const std::array<int, 4>& class1, const std::array<int, 4>& class2) {
    if (class1[0] < 0 || class2[0] < 0)
        return 10.0f;
    if (class1[0] == class2[0]) {
        if (class1[1] == class2[1] || class1[2] == class2[2]) {
            if (class1[1] == class2[1] && class1[2] == class2[2]) {
                if (class1[3] == class2[3]) {
                    return 0.0f;
                }
                return 1.0f;
            }
            return 2.0f;
        }
        return 3.0f;
    }
    return 4.0f;
}

/*
 * EnzymeClassProvider::EnzymeClassDistance
 */
float EnzymeClassProvider::EnzymeClassDistance(
    const std::string pdbid1, const std::string pdbid2, frontend_resources::RuntimeConfig const& runtimeConf) {
    auto const first_classes = RetrieveClassesForPdbId(pdbid1, runtimeConf);
    auto const second_classes = RetrieveClassesForPdbId(pdbid2, runtimeConf);
    float min_dist = 4.0f; // 4 is the maximum value
    for (auto const& first : first_classes) {
        for (auto const& second : second_classes) {
            const auto dist = EnzymeClassDistance(first, second);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
    }
    return min_dist;
}


/*
 * EnzymeClassProvider::loadMapFromFile
 */
void EnzymeClassProvider::loadMapFromFile(frontend_resources::RuntimeConfig const& runtimeConf) {
    const auto filepath = determineFilePath(runtimeConf);
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
std::filesystem::path EnzymeClassProvider::determineFilePath(frontend_resources::RuntimeConfig const& runtimeConf) {
    std::string shortfile = "brenda_enzyme_map.csv";
    return core::utility::ResourceWrapper::GetResourcePath(runtimeConf, shortfile).make_preferred();
}
