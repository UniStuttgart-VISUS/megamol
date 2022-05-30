#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <array>
#include <vector>
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/CoreInstance.h"

namespace megamol {
namespace molsurfmapcluster_gl {
class EnzymeClassProvider {
public:
    /** Ctor. */
    EnzymeClassProvider(void);

    /** Dtor. */
    virtual ~EnzymeClassProvider(void);

    /**
     * Returns a reference to the enzyme class map.
     * The map will be loaded beforehand, if it is not loaded yet.
     *
     * @param coreInstance The core instance needed for file path retrieval.
     * @return Reference to the stored map
     */
    static const std::multimap<std::string, std::array<int, 4>>& RetrieveEnzymeClassMap(const core::CoreInstance& coreInstance);

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
    static std::array<int, 4> RetrieveClassForPdbId(std::string pdbId, const core::CoreInstance& coreInstance);

    /**
     * Returns all classes a given PDB id lies in.
     * If there is none, the returned vector will be empty
     * The map will be loaded beforehand, if it is not loaded yet.
     *
     * @param pdbId The requested protein pdbId.
     * @param coreInstance The core instance needed for file path retrieval.
     * @return Array with four entries, one for each subclass. If one subclass is not listed, the value will be -1.
     */
    static std::vector<std::array<int, 4>> RetrieveClassesForPdbId(
        std::string pdbId, const core::CoreInstance& coreInstance);

    /** 
     * Converts a given enzyme class to the string representing it
     *
     * @param enzymeClass Array containing the four class definitions
     * @return String representing the given enzyme class
     */
    static std::string ConvertEnzymeClassToString(const std::array<int, 4>& enzymeClass);

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

    /** map mapping pdb ids to classification numbers */
    static std::multimap<std::string, std::array<int, 4>> classMap;
};
} // namespace MolSurfMapCluster
} // namespace megamol
