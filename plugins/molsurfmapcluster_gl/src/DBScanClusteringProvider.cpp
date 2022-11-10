#include "DBScanClusteringProvider.h"

#include "mmcore/utility/log/Log.h"
#include <sstream>

using namespace megamol;
using namespace megamol::molsurfmapcluster;

/*
 * DBScanClusteringProvider::clusterMap
 */
std::multimap<std::string, int> DBScanClusteringProvider::clusterMap = std::multimap<std::string, int>();

/*
 * DBScanClusteringProvider::idMap
 */
std::multimap<int, std::string> DBScanClusteringProvider::idMap = std::multimap<int, std::string>();

/*
 * DBScanClusteringProvider::DBScanClusteringProvider
 */
DBScanClusteringProvider::DBScanClusteringProvider(void) {
    // intentionally empty
}

/*
 * DBScanClusteringProvider::~DBScanClusteringProvider
 */
DBScanClusteringProvider::~DBScanClusteringProvider(void) {
    // intentionally empty
}

/*
 * DBScanClusteringProvider::RetrieveClusterMap
 */
const std::multimap<std::string, int>& DBScanClusteringProvider::RetrieveClusterMap(
    frontend_resources::RuntimeConfig const& runtimeConf) {
    if (clusterMap.size() == 0) {
        loadMapFromFile(runtimeConf);
    }
    return clusterMap;
}

/*
 * DBScanClusteringProvider::RetrieveIdMap
 */
const std::multimap<int, std::string>& DBScanClusteringProvider::RetrieveIdMap(
    frontend_resources::RuntimeConfig const& runtimeConf) {
    if (idMap.size() == 0) {
        loadMapFromFile(runtimeConf);
    }
    return idMap;
}

/*
 * DBScanClusteringProvider::RetrieveClusterForPdbId
 */
int DBScanClusteringProvider::RetrieveClusterForPdbId(
    std::string pdbId, frontend_resources::RuntimeConfig const& runtimeConf) {
    int result = -1;
    if (clusterMap.size() == 0) {
        loadMapFromFile(runtimeConf);
    }
    auto it = clusterMap.lower_bound(pdbId);
    if (it != clusterMap.end()) {
        result = (*it).second;
    }
    return result;
}

/*
 * DBScanClusteringProvider::RetrievePDBIdForCluster
 */
std::string DBScanClusteringProvider::RetrievePDBIdForCluster(
    int clusterId, frontend_resources::RuntimeConfig const& runtimeConf) {
    std::string result = "";
    if (idMap.size() == 0) {
        loadMapFromFile(runtimeConf);
    }
    auto it = idMap.lower_bound(clusterId);
    if (it != idMap.end()) {
        result = (*it).second;
    }
    return result;
}

/*
 * DBScanClusteringProvider::RetrieveClustersForPdbId
 */
std::vector<int> DBScanClusteringProvider::RetrieveClustersForPdbId(
    std::string pdbId, frontend_resources::RuntimeConfig const& runtimeConf) {
    std::vector<int> result;
    if (clusterMap.size() == 0) {
        loadMapFromFile(runtimeConf);
    }
    auto lower = clusterMap.lower_bound(pdbId);
    auto upper = clusterMap.upper_bound(pdbId);
    for (; lower != upper; lower++) {
        if (lower->second != -1) {
            result.push_back(lower->second);
        }
    }
    return result;
}

/*
 * DBScanClusteringProvider::RetrievePDBIdsForCluster
 */
std::vector<std::string> DBScanClusteringProvider::RetrievePDBIdsForCluster(
    int clusterId, frontend_resources::RuntimeConfig const& runtimeConf) {
    std::vector<std::string> result;
    if (idMap.size() == 0) {
        loadMapFromFile(runtimeConf);
    }
    auto lower = idMap.lower_bound(clusterId);
    auto upper = idMap.upper_bound(clusterId);
    for (; lower != upper; lower++) {
        result.push_back(lower->second);
    }
    return result;
}

/*
 * DBScanClusteringProvider::loadMapFromFile
 */
void DBScanClusteringProvider::loadMapFromFile(frontend_resources::RuntimeConfig const& runtimeConf) {
    const auto filepath = determineFilePath(runtimeConf);
    idMap.clear();
    clusterMap.clear();
    std::ifstream file(filepath);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::pair<std::string, int> res;
            // split line by semicolon
            std::string substr;
            std::stringstream ss(line);
            int i = 0;
            while (ss.good()) {
                std::getline(ss, substr, ';');
                if (i == 0) {
                    res.first = substr;
                } else if (i == 1) {
                    try {
                        res.second = std::stoi(substr);
                    } catch (...) { res.second = -1; }
                }
                ++i;
            }
            if (res.second != -1) {
                clusterMap.insert(res);
                idMap.insert(std::make_pair(res.second, res.first));
            }
        }
    } else {
        core::utility::log::Log::DefaultLog.WriteError(
            "Could not load the configuration file \"%s\"", filepath.c_str());
    }
}

/*
 * DBScanClusteringProvider::determineFilePath
 */
std::filesystem::path DBScanClusteringProvider::determineFilePath(
    frontend_resources::RuntimeConfig const& runtimeConf) {
    std::string shortfile = "dbscan_protein_cluster_map.csv";
    return core::utility::ResourceWrapper::GetResourcePath(runtimeConf, shortfile).make_preferred();
}
