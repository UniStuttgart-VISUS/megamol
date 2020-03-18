#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/ResourceWrapper.h"

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
    static const std::multimap<std::string, int>& RetrieveClusterMap(const core::CoreInstance& coreInstance);

    static const std::multimap<int, std::string>& RetrieveIdMap(const core::CoreInstance& coreInstance);

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
    static int RetrieveClusterForPdbId(std::string pdbId, const core::CoreInstance& coreInstance);

    static std::string RetrievePDBIdForCluster(int clusterId, const core::CoreInstance& coreInstance);

    /**
     * Returns all classes a given PDB id lies in.
     * If there is none, the returned vector will be empty
     * The map will be loaded beforehand, if it is not loaded yet.
     *
     * @param pdbId The requested protein pdbId.
     * @param coreInstance The core instance needed for file path retrieval.
     * @return Array with four entries, one for each subclass. If one subclass is not listed, the value will be -1.
     */
    static std::vector<int> RetrieveClustersForPdbId(std::string pdbId, const core::CoreInstance& coreInstance);

    static std::vector<std::string> RetrievePDBIdsForCluster(int clusterId, const core::CoreInstance& coreInstance);

private:
    /**
     * Loads the enzyme class map from the file in the resources folder
     *
     * @param coreInstance The core instance needed for file path retrieval.
     */
    static void loadMapFromFile(const core::CoreInstance& coreInstance);

    /**
     * Determines the file path of the file to load
     *
     * @param coreInstance The core instance needed for file path retrieval.
     * @return Path to the map file
     */
    static std::filesystem::path determineFilePath(const core::CoreInstance& coreInstance);

    /** map mapping pdb ids to cluster ids */
    static std::multimap<std::string, int> clusterMap;

    /** map mapping cluster ids to pdb ids */
    static std::multimap<int, std::string> idMap;
};
} // namespace molsurfmapcluster
} // namespace megamol
