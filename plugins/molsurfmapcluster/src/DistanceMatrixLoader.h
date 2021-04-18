#pragma once

#include <map>
#include <vector>
#include <istream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <utility>

namespace megamol {
namespace molsurfmapcluster {

class DistanceMatrixLoader {
public:
    DistanceMatrixLoader(void);
    static double GetDistance(std::string pdbid1, std::string pdbid2);
    static bool load(const std::filesystem::path& path, bool force = false);
    static std::map<std::string, std::map<std::string, double>> getDistanceMap(void);
    static float distanceEps;

private:
    static std::map<std::string, std::map<std::string, double>> distanceMap;
    static std::filesystem::path curPath;
};

} // namespace molsurfmapcluster
} // namespace megamol
