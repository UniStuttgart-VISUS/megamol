#include "stdafx.h"

#include <fstream>
#include <sstream>

#include "FEMLoader.h"

#include "mmcore/param/FilePathParam.h"

namespace
{
	/*
	 * Hidden, local helper function for parsing strings.
	 * Source: https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
	 */
	std::vector<std::string> split(std::string const& s, char delimiter)
	{
		std::vector<std::string> tokens;
		std::string token;
		std::istringstream tokenStream(s);
		while (std::getline(tokenStream, token, delimiter))
		{
			tokens.push_back(token);
		}
		return tokens;
	}
}

namespace megamol
{
	namespace archvis
	{
		FEMLoader::FEMLoader()
			: core::Module(),
			m_femNodes_filename_slot("FEM node list filename", "The name of the txt file containing the FEM nodes"),
			m_femElements_filename_slot("FEM element list filename", "The name of the txt file containing the FEM elemets"),
			m_getData_slot("getData", "The slot for publishing the loaded data")
		{
			this->m_getData_slot.SetCallback(FEMDataCall::ClassName(), "GetData", &FEMLoader::getDataCallback);
			this->MakeSlotAvailable(&this->m_getData_slot);

			this->m_femNodes_filename_slot << new core::param::FilePathParam("");
			this->MakeSlotAvailable(&this->m_femNodes_filename_slot);

			this->m_femElements_filename_slot << new core::param::FilePathParam("");
			this->MakeSlotAvailable(&this->m_femElements_filename_slot);
		}

		FEMLoader::~FEMLoader()
		{
		}

		bool FEMLoader::create(void)
		{
			return false;
		}

		bool FEMLoader::getDataCallback(core::Call & caller)
		{
			if (this->m_femNodes_filename_slot.IsDirty())
			{
				this->m_femNodes_filename_slot.ResetDirty();

				auto vislib_filename = m_femNodes_filename_slot.Param<megamol::core::param::FilePathParam>()->Value();
				std::string filename(vislib_filename.PeekBuffer());
				
				m_fem_data->setNodes(loadNodesFromFile(filename));
			}

			if (this->m_femElements_filename_slot.IsDirty())
			{
				this->m_femElements_filename_slot.ResetDirty();

				auto vislib_filename = m_femElements_filename_slot.Param<megamol::core::param::FilePathParam>()->Value();
				std::string filename(vislib_filename.PeekBuffer());

				m_fem_data->setElements(loadElementsFromFile(filename));
			}

			return true;
		}

		void FEMLoader::release()
		{
		}

		std::vector<FEMDataStorage::Vec3> FEMLoader::loadNodesFromFile(std::string const & filename)
		{
			std::vector<FEMDataStorage::Vec3> retval;

			std::ifstream file;
			file.open(filename, std::ifstream::in);

			if (file.is_open())
			{
				file.seekg(0, std::ifstream::beg);

				unsigned int lines_read = 0;
				while (!file.eof())
				{
					std::string line;
					std::getline(file, line, '\n');
					
					auto sl = split(line, ',');

					if(sl.size() == 4)
						retval.push_back(FEMDataStorage::Vec3(std::stof(sl[1]), std::stof(sl[2]), std::stof(sl[3])));
				}
			}

			return retval;
		}

		std::vector<std::array<size_t, 8>> FEMLoader::loadElementsFromFile(std::string const & filename)
		{
			std::vector<std::array<size_t, 8>> retval;

			std::ifstream file;
			file.open(filename, std::ifstream::in);

			if (file.is_open())
			{
				file.seekg(0, std::ifstream::beg);

				unsigned int lines_read = 0;
				while (!file.eof())
				{
					std::string line;
					std::getline(file, line, '\n');

					auto sl = split(line, ',');

					if (sl.size() == 9)
						retval.push_back({
							std::stoul(sl[1]),
							std::stoul(sl[2]),
							std::stoul(sl[3]),
							std::stoul(sl[4]),
							std::stoul(sl[5]),
							std::stoul(sl[6]),
							std::stoul(sl[7]),
							std::stoul(sl[8]) });
				}
			}

			return retval;
		}
	}
}
