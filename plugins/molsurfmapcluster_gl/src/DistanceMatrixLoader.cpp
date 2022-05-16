#include "DistanceMatrixLoader.h"

using namespace megamol::molsurfmapcluster;

std::filesystem::path DistanceMatrixLoader::curPath = std::filesystem::path();
std::map<std::string, std::map<std::string, double>> DistanceMatrixLoader::distanceMap =
    std::map<std::string, std::map<std::string, double>>();

float DistanceMatrixLoader::distanceEps = std::numeric_limits<float>::epsilon();

DistanceMatrixLoader::DistanceMatrixLoader(void) {
    // intentionally empty
}

bool DistanceMatrixLoader::load(const std::filesystem::path& path, bool force) {
    if (path == curPath && !force)
        return false;
    distanceMap.clear();
    curPath = path;
    std::ifstream file(path);
    std::vector<std::string> idvec;
    if (file.is_open()) {
        bool initialized = false;
        std::string line, id;
        while (std::getline(file, line)) {
            if (!initialized) {
                // the first line contains all availabe protein ids
                std::stringstream ss(line);
                while (ss.good()) {
                    std::getline(ss, id, ';');
                    if (!id.empty()) {
                        idvec.push_back(id);
                    }
                }
                for (const auto& i : idvec) {
                    std::map<std::string, double> curmap;
                    for (const auto& j : idvec) {
                        curmap.insert(std::make_pair(j, 0.0));
                    }
                    distanceMap.insert(std::make_pair(i, curmap));
                }
                initialized = true;
            } else {
                // all other lines contain the distance values
                std::stringstream ss(line);
                bool first = true;
                std::string localid;
                uint32_t index = 0;
                while (ss.good()) {
                    std::getline(ss, id, ';');
                    if (!id.empty()) {
                        if (first) {
                            localid = id;
                            first = false;
                        } else {
                            if (localid.compare(idvec[index]) != 0) {
                                double val = std::stod(id);
                                distanceMap[localid][idvec[index]] = val;
                            }
                            index++;
                        }
                    }
                }
            }
        }
    } else {
        return false;
    }
    return true;
}

double DistanceMatrixLoader::GetDistance(std::string pdbid1, std::string pdbid2) {
    if (pdbid1.compare(pdbid2) == 0)
        return 1.0; // both strings are equal
    if (distanceMap.size() == 0)
        return -1.0; // no distances available
    if (!distanceMap.count(pdbid1) > 0)
        return -1.0; // no entry for pdbid1
    if (!distanceMap.count(pdbid2) > 0)
        return -1.0; // no entry for pdbid2
    double dir1 = distanceMap.at(pdbid1).at(pdbid2);
    double dir2 = distanceMap.at(pdbid2).at(pdbid1);
    return std::min(dir1, dir2);
}

std::map<std::string, std::map<std::string, double>> DistanceMatrixLoader::getDistanceMap(void) {
    return DistanceMatrixLoader::distanceMap;
}
