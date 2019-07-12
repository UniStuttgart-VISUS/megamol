#include "stdafx.h"

#include <fstream>
#include <sstream>

#include "FEMLoader.h"

#include "mmcore/param/FilePathParam.h"

namespace {
/*
 * Hidden, local helper function for parsing strings.
 * Source: https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
 */
std::vector<std::string> split(std::string const& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}
} // namespace

namespace megamol {
namespace archvis {

FEMLoader::FEMLoader()
    : core::Module()
    , m_femNodes_filename_slot("FEM node list filename", "The name of the txt file containing the FEM nodes")
    , m_femElements_filename_slot("FEM element list filename", "The name of the txt file containing the FEM elemets")
    , m_femDeformation_filename_slot(
          "FEM displacment filename", "The name of the txt file containing displacements for the FEM nodes")
    , m_getData_slot("getData", "The slot for publishing the loaded data") {
    this->m_getData_slot.SetCallback(FEMDataCall::ClassName(), "GetData", &FEMLoader::getDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_femNodes_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_femNodes_filename_slot);

    this->m_femDeformation_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_femDeformation_filename_slot);

    this->m_femElements_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_femElements_filename_slot);
}

FEMLoader::~FEMLoader() {}

bool FEMLoader::create(void) {
    m_fem_data = std::make_shared<FEMModel>();
    return true;
}

bool FEMLoader::getDataCallback(core::Call& caller) {
    FEMDataCall* cd = dynamic_cast<FEMDataCall*>(&caller);

    if (cd == NULL) return false;

    //      cd->clearUpdateFlag();
    //      m_update_flag = std::max(0, m_update_flag - 1);
    //      
    //      if (this->m_femNodes_filename_slot.IsDirty()) {
    //          this->m_femNodes_filename_slot.ResetDirty();
    //      
    //          auto vislib_filename = m_femNodes_filename_slot.Param<megamol::core::param::FilePathParam>()->Value();
    //          std::string filename(vislib_filename.PeekBuffer());
    //      
    //          m_fem_data->setNodes(loadNodesFromFile(filename));
    //      
    //          m_update_flag = std::min(2, m_update_flag + 2);
    //      }
    //      
    //      if (this->m_femElements_filename_slot.IsDirty()) {
    //          this->m_femElements_filename_slot.ResetDirty();
    //      
    //          auto vislib_filename = m_femElements_filename_slot.Param<megamol::core::param::FilePathParam>()->Value();
    //          std::string filename(vislib_filename.PeekBuffer());
    //      
    //          m_fem_data->setElements(loadElementsFromFile(filename));
    //      
    //          m_update_flag = std::min(2, m_update_flag + 2);
    //      }
    //      
    //      if (this->m_femDeformation_filename_slot.IsDirty()) {
    //          this->m_femDeformation_filename_slot.ResetDirty();
    //      
    //          auto vislib_filename = m_femDeformation_filename_slot.Param<megamol::core::param::FilePathParam>()->Value();
    //          std::string filename(vislib_filename.PeekBuffer());
    //      
    //          m_fem_data->setNodeDeformations(loadNodeDeformationsFromFile(filename));
    //      
    //          m_update_flag = std::min(2, m_update_flag + 2);
    //      }
    //      
    //      cd->setFEMData(m_fem_data);
    //      if (m_update_flag > 0) cd->setUpdateFlag();

    return true;
}

void FEMLoader::release() {}

std::vector<FEMModel::Vec3> FEMLoader::loadNodesFromFile(std::string const& filename) {
    std::vector<FEMModel::Vec3> retval;

    std::ifstream file;
    file.open(filename, std::ifstream::in);

    if (file.is_open()) {
        file.seekg(0, std::ifstream::beg);

        unsigned int lines_read = 0;
        while (!file.eof()) {
            std::string line;
            std::getline(file, line, '\n');

            auto sl = split(line, ',');

            if (sl.size() == 4)
                retval.push_back(FEMModel::Vec3(std::stof(sl[1]), std::stof(sl[2]), std::stof(sl[3])));
        }
    }

    return retval;
}

std::vector<std::array<size_t, 8>> FEMLoader::loadElementsFromFile(std::string const& filename) {
    std::vector<std::array<size_t, 8>> retval;

    std::ifstream file;
    file.open(filename, std::ifstream::in);

    if (file.is_open()) {
        file.seekg(0, std::ifstream::beg);

        unsigned int lines_read = 0;
        while (!file.eof()) {
            std::string line;
            std::getline(file, line, '\n');

            auto sl = split(line, ',');

            if (sl.size() == 9)
                retval.push_back({std::stoul(sl[1]), std::stoul(sl[2]), std::stoul(sl[3]), std::stoul(sl[4]),
                    std::stoul(sl[5]), std::stoul(sl[6]), std::stoul(sl[7]), std::stoul(sl[8])});
        }
    }

    return retval;
}

std::vector<FEMModel::Vec4> FEMLoader::loadNodeDeformationsFromFile(std::string const& filename) {
    std::vector<FEMModel::Vec4> retval;

    std::ifstream file;
    file.open(filename, std::ifstream::in);

    if (file.is_open()) {
        file.seekg(0, std::ifstream::beg);

        unsigned int lines_read = 0;
        while (!file.eof()) {
            std::string line;
            std::getline(file, line, '\n');

            auto sl = split(line, ',');

            if (sl.size() == 4)
                retval.push_back(
                    FEMModel::Vec4(std::stof(sl[1]), std::stof(sl[2]), std::stof(sl[3]), 0.0f /*padding*/));
        }
    }

    return retval;
}

} // namespace archvis
} // namespace megamol
