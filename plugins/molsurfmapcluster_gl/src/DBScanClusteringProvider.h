#pragma once

#include "RuntimeConfig.h"
#include "mmcore/utility/ResourceWrapper.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace megamol {
namespace molsurfmapcluster {
class DBScanClusteringProvider {
public:
    /** Ctor. */
    DBScanClusteringProvider(void);

    /** Dtor. */
    virtual ~DBScanClusteringProvider(void);

    /**
     * Returns a reference to the enzyme class map.
     * The map will be loaded beforehand, if it is not loaded yet.
     *
     * @param coreInstance The core instance needed for file path retrieval.
     * @return Reference to the stored map
     */
    static const std::multimap<std::string, int>& RetrieveClusterMap(
        frontend_resources::RuntimeConfig const& runtimeConf);

    static const std::multimap<int, std::string>& RetrieveIdMap(frontend_resources::RuntimeConfig const& runtimeConf);

    /**
     * Returns the class of a given PDB id.
     * If there are multiple classifications, only the first is returned.
     * If there is none, the result will be {-1,-1,-1,-1}.
     * The map will be loaded beforehand, if it is not loaded yet.
     *
     * @param pdbId The requested protein pdbId.
     * @param coreInstance The core instance needed for file path retrieval.
     * @return Array with four entries, one for each subclass. If one subclass is not listed, the value will be -1.
     */
    static int RetrieveClusterForPdbId(std::string pdbId, frontend_resources::RuntimeConfig const& runtimeConf);

    static std::string RetrievePDBIdForCluster(int clusterId, frontend_resources::RuntimeConfig const& runtimeConf);

    /**
     * Returns all classes a given PDB id lies in.
     * If there is none, the returned vector will be empty
     * The map will be loaded beforehand, if it is not loaded yet.
     *
     * @param pdbId The requested protein pdbId.
     * @param coreInstance The core instance needed for file path retrieval.
     * @return Array with four entries, one for each subclass. If one subclass is not listed, the value will be -1.
     */
    static std::vector<int> RetrieveClustersForPdbId(
        std::string pdbId, frontend_resources::RuntimeConfig const& runtimeConf);

    static std::vector<std::string> RetrievePDBIdsForCluster(
        int clusterId, frontend_resources::RuntimeConfig const& runtimeConf);

private:
    /**
     * Loads the enzyme class map from the file in the resources folder
     *
     * @param coreInstance The core instance needed for file path retrieval.
     */
    static void loadMapFromFile(frontend_resources::RuntimeConfig const& runtimeConf);

    /**
     * Determines the file path of the file to load
     *
     * @param coreInstance The core instance needed for file path retrieval.
     * @return Path to the map file
     */
    static std::filesystem::path determineFilePath(frontend_resources::RuntimeConfig const& runtimeConf);

    /** map mapping pdb ids to cluster ids */
    static std::multimap<std::string, int> clusterMap;

    /** map mapping cluster ids to pdb ids */
    static std::multimap<int, std::string> idMap;
};
} // namespace molsurfmapcluster
} // namespace megamol
