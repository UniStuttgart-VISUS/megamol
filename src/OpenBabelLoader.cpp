#include "stdafx.h"
#include "OpenBabelLoader.h"

#ifdef WITH_OPENBABEL

#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_cuda;
using namespace megamol::core::moldyn;

OpenBabelLoader::OpenBabelLoader() :
	filenameSlot("filename","The path to the data file to be loaded"),
	dataOutSlot("dataout", "The slot providing the converted data")
{
	this->filenameSlot << new param::FilePathParam("");
	this->MakeSlotAvailable(&this->filenameSlot);

	this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &OpenBabelLoader::getData);
	this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &OpenBabelLoader::getExtent);
	this->MakeSlotAvailable(&this->dataOutSlot);

	residues = NULL;
}

OpenBabelLoader::~OpenBabelLoader()
{
	this->Release();
}

/*
* OpenBabelLoader::create
*/
bool OpenBabelLoader::create(void)
{
	return true;
}

/*
* OpenBabelLoader::release
*/
void OpenBabelLoader::release(void)
{
}

/*
* OpenBabelLoader::getData
*/
bool OpenBabelLoader::getData(core::Call& call)
{
	MolecularDataCall* mdc_in = dynamic_cast<MolecularDataCall*>(&call);
	if (mdc_in == NULL) return false;
	
	vislib::TString file = this->filenameSlot.Param<core::param::FilePathParam>()->Value();
	auto seperator_list_linux = vislib::StringTokeniserA::Split(file, "/");
	vislib::TString tmp = seperator_list_linux[seperator_list_linux.Count() - 1];
	auto seperator_list_win = vislib::StringTokeniserA::Split(tmp, "\\");
	std::string file_exists = seperator_list_win[seperator_list_win.Count() - 1];
	if (!file_exists.empty())
	{
		std::ifstream* input = new std::ifstream((const char*)T2A(file), std::ios::in);
#ifdef WITH_CURL
		if (!input->is_open())
		{
			loadFromPDB(file_exists,(const char*)T2A(file));
		}
#else
		if (!input->is_open())
		{
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Could not load file %s", (const char*)T2A(file)); // DEBUG
			return false;
		}
#endif
		input->close();

		auto list = vislib::StringTokeniserA::Split(file, ".");
		std::string file_type = list[list.Count() - 1];

		OpenBabel::OBConversion conv = OpenBabel::OBConversion();
		std::vector<std::string> support = conv.GetSupportedInputFormat();
		conv.SetInFormat(file_type.c_str());
		OpenBabel::OBMol* mol = new OpenBabel::OBMol();
		bool not_end = conv.ReadFile(mol, T2A(file));
		atom_count = 0;
		res_count = 0;
		while (not_end)
		{
			atom_count += mol->NumAtoms();
			res_count += mol->NumResidues();
			delete mol;
			mol = new OpenBabel::OBMol();
			not_end = conv.Read(mol);
		}
		delete mol;
		pos = std::vector<float>(atom_count * 3);
		charge = std::vector<float>(atom_count);
		bfactor = std::vector<float>(atom_count);
		occupancy = std::vector<float>(atom_count);
		atomType_arr.AssertCapacity(atom_count);
		atomTypeIdx_arr.SetCount(atom_count);
		residueIdx = std::vector<int>(atom_count);
		//res_bbox = vislib::Array<vislib::math::Cuboid<float> >(atom_count);
		vislib::math::Cuboid<float> tmpBBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
		for (unsigned int i = 0; i < atom_count; i++)
		{
			res_bbox.Add(tmpBBox);
		}
		if (residues != NULL) delete residues;
		residues = new std::vector<MolecularDataCall::Residue*>(res_count);
		charge_min = std::numeric_limits<float>::max();
		charge_max = std::numeric_limits<float>::min();
		bfactor_min = std::numeric_limits<float>::max();
		bfactor_max = std::numeric_limits<float>::min();
		occupancy_min = std::numeric_limits<float>::max();
		occupancy_max = std::numeric_limits<float>::min();

		mol = new OpenBabel::OBMol();
		not_end = conv.ReadFile(mol, T2A(file));
		unsigned int global_atomIdx = 0;
		unsigned int global_resIdx = 0;
		unsigned int global_molIdx = 0;
		while (not_end)
		{
			local_atom_count = mol->NumAtoms();
			OpenBabel::OBAtomIterator iter;
			for (iter = mol->BeginAtoms(); iter != mol->EndAtoms(); iter++)
			{
				vislib::StringA tmpStr = (*iter)->GetType();
				float radius = getElementRadius((*iter));
				vislib::math::Vector<unsigned char, 3> color = getElementColor((*iter));
				MolecularDataCall::AtomType type(tmpStr, radius, color.X(), color.Y(), color.Z());
			
				INT_PTR atomTypeIdx = atomType_arr.IndexOf(type);
				if (atomTypeIdx == vislib::Array<MolecularDataCall::AtomType>::INVALID_POS)
				{
					atomTypeIdx_arr[global_atomIdx] = static_cast<unsigned int>(atomType_arr.Count());
					atomType_arr.Add(type);
				}
				else {
					atomTypeIdx_arr[global_atomIdx] = static_cast<unsigned int>(atomTypeIdx);
				}

				charge[global_atomIdx] = (float)(*iter)->GetPartialCharge();
				if (charge[global_atomIdx] < charge_min) charge_min = charge[global_atomIdx];
				if (charge[global_atomIdx] > charge_max) charge_max = charge[global_atomIdx];
			
				bfactor[global_atomIdx] = (float)(rand() % 2);
				if (bfactor[global_atomIdx] < bfactor_min) bfactor_min = bfactor[global_atomIdx];
				if (bfactor[global_atomIdx] > bfactor_max) bfactor_max = bfactor[global_atomIdx];
			
				occupancy[global_atomIdx] = (float)(rand() % 2);
				if (occupancy[global_atomIdx] < occupancy_min) occupancy_min = occupancy[global_atomIdx];
				if (occupancy[global_atomIdx] > occupancy_max) occupancy_max = occupancy[global_atomIdx];

				// update the bounding box
				vislib::math::Cuboid<float> atomBBox(
					pos[global_atomIdx * 3 + 0] - this->atomType_arr[this->atomTypeIdx_arr[global_atomIdx]].Radius(),
					pos[global_atomIdx * 3 + 1] - this->atomType_arr[this->atomTypeIdx_arr[global_atomIdx]].Radius(),
					pos[global_atomIdx * 3 + 2] - this->atomType_arr[this->atomTypeIdx_arr[global_atomIdx]].Radius(),
					pos[global_atomIdx * 3 + 0] + this->atomType_arr[this->atomTypeIdx_arr[global_atomIdx]].Radius(),
					pos[global_atomIdx * 3 + 1] + this->atomType_arr[this->atomTypeIdx_arr[global_atomIdx]].Radius(),
					pos[global_atomIdx * 3 + 2] + this->atomType_arr[this->atomTypeIdx_arr[global_atomIdx]].Radius());
				if (global_atomIdx == 0) {
					this->bbox = atomBBox;
				}
				else {
					this->bbox.Union(atomBBox);
				}

				residueIdx[global_atomIdx] = (*iter)->GetResidue()->GetIdx();

				res_bbox[residueIdx[global_atomIdx]].Union(atomBBox);
			
				pos[global_atomIdx * 3 + 0] = (*iter)->GetX();
				pos[global_atomIdx * 3 + 1] = (*iter)->GetY();
				pos[global_atomIdx * 3 + 2] = (*iter)->GetZ();
			
				global_atomIdx++;
			}

			con_count = mol->NumBonds();
			connections = std::vector<unsigned int>(2 * con_count);
			for (unsigned int i = 0; i < con_count; i++)
			{
				connections[(2 * i) + 0] = mol->GetBond(i)->GetBeginAtomIdx();
				connections[(2 * i) + 1] = mol->GetBond(i)->GetEndAtomIdx();
			}
			
			unsigned int local_res_count = mol->NumResidues();
			
			/* TODO:
			* For now every molecule is in one chain. If there is a way to figure
			* out what molecule belongs to what chain in the OpenBabel datastructure
			* fix this.
			*/
			MolecularDataCall::Molecule mdc_mol
				= MolecularDataCall::Molecule(global_resIdx, local_res_count, 0);
			molecules.push_back(mdc_mol);

			for (unsigned int i = 0; i < local_res_count; i++)
			{
				unsigned int first_atom_index = (*mol->GetResidue(i)->BeginAtoms())->GetIdx();
				unsigned int res_atom_count = mol->GetResidue(i)->GetNumAtoms();
				
				mol->GetResidue(i)->GetAtoms();
			
				vislib::StringA resName = mol->GetResidue(i)->GetName().c_str();
				unsigned int typeIdx;
				INT_PTR resTypeNameIdx = this->residueTypeName.IndexOf(resName);
				if (resTypeNameIdx == vislib::Array<vislib::StringA>::INVALID_POS)
				{
					typeIdx = static_cast<unsigned int>(this->residueTypeName.Count());
					this->residueTypeName.Add(resName);
					if (mol->GetResidue(i)->GetResidueProperty(8) || mol->GetResidue(i)->GetResidueProperty(9))
					{
						this->solventResidueIdx.Add(typeIdx);
					}
				}
				else
				{
					typeIdx = static_cast<unsigned int>(resTypeNameIdx);
					if (mol->GetResidue(i)->GetResidueProperty(8) || mol->GetResidue(i)->GetResidueProperty(9))
					{
						this->solventResidueIdx.Add(typeIdx);
					}
				}

				int moleculeIdx = global_molIdx;
			
				unsigned int origResIdx = mol->GetResidue(i)->GetIdx();

				residues->at(global_resIdx) = 
					new MolecularDataCall::Residue(
					first_atom_index, res_atom_count, 
					res_bbox[global_resIdx],
					typeIdx, moleculeIdx, origResIdx);
				global_resIdx++;
			}
			
			global_molIdx++;
			delete mol;
			mol = new OpenBabel::OBMol();
			not_end = conv.Read(mol);
		}
			
		mdc_in->SetAtoms(atom_count, atomType_arr.Count(), atomTypeIdx_arr.PeekElements(), pos.data(),
			atomType_arr.PeekElements(), residueIdx.data(), bfactor.data(), charge.data(), occupancy.data());
		
		mdc_in->SetBFactorRange(bfactor_min, bfactor_max);
		mdc_in->SetChargeRange(charge_min, charge_max);
		mdc_in->SetOccupancyRange(occupancy_min, occupancy_max);
		
		mdc_in->SetConnections(con_count, connections.data());
		
		mdc_in->SetResidues(res_count, (const MolecularDataCall::Residue**)residues->data());
		
		mdc_in->SetSolventResidueIndices(solventResidueIdx.Count(), solventResidueIdx.PeekElements());
		mdc_in->SetResidueTypeNames(residueTypeName.Count(), residueTypeName.PeekElements());
		mdc_in->SetMolecules(molecules.size(), molecules.data());
		
		/* TODO:
		* For now every molecule is in one chain. If there is a way to figure
		* out what molecule belongs to what chain in the OpenBabel datastructure
		* fix this.
		*/
		unsigned int firstMolIdx = 0;
		unsigned int molCnt = global_molIdx;
		char name = ' ';
		MolecularDataCall::Chain::ChainType chainType = MolecularDataCall::Chain::ChainType::UNSPECIFIC;
		MolecularDataCall::Chain new_chain = MolecularDataCall::Chain(firstMolIdx, molCnt, name, chainType);
		chain.Add(new_chain);
		
		mdc_in->SetChains(chain.Count(), chain.PeekElements());
		filter.SetCount(atom_count);
		for (unsigned int i = 0; i < filter.Count(); i++)
			filter[i] = 0;
		mdc_in->SetFilter(filter.PeekElements());
		return true;
	}
	return false;
}

/*
* OpenBabelLoader::getExtent
*/
bool OpenBabelLoader::getExtent(core::Call& call)
{
	MolecularDataCall *dc = dynamic_cast<MolecularDataCall*>(&call);
	if (dc == NULL) return false;

	vislib::math::Cuboid<float> bBoxPlus3;
	bBoxPlus3.Grow(25.0f);

	dc->AccessBoundingBoxes().Clear();
	dc->AccessBoundingBoxes().SetObjectSpaceBBox(bBoxPlus3);
	dc->AccessBoundingBoxes().SetObjectSpaceClipBox(bBoxPlus3);

	return true;
}

/*
* OpenBabelLoader::getElementRadius
*/
float OpenBabelLoader::getElementRadius(OpenBabel::OBAtom* atom)
{
	// --- van der Waals radii ---
	if (atom->IsHydrogen())
		return 1.2f;
	if (atom->IsCarbon())
		return 1.7f;
	if (atom->IsNitrogen())
		return 1.55f;
	if (atom->IsOxygen())
		return 1.52f;
	if (atom->IsSulfur())
		return 1.8f;
	if (atom->IsPhosphorus())
		return 1.8f;

	return 1.5f;
}

/*
* Get the color of the element
*/
vislib::math::Vector<unsigned char, 3> OpenBabelLoader::getElementColor(OpenBabel::OBAtom* atom)
{
	if (atom->IsHydrogen()) // white or light grey
		return vislib::math::Vector<unsigned char, 3>(240, 240, 240);
	if (atom->IsCarbon()) // (dark) grey or green
		return vislib::math::Vector<unsigned char, 3>(90, 90, 90);
	if (atom->IsNitrogen()) // blue
		return vislib::math::Vector<unsigned char, 3>(37, 136, 195);
	if (atom->IsOxygen()) // red
		return vislib::math::Vector<unsigned char, 3>(206, 34, 34);
	if (atom->IsSulfur()) // yellow
		return vislib::math::Vector<unsigned char, 3>(255, 215, 0);
	if (atom->IsPhosphorus()) // orange
		return vislib::math::Vector<unsigned char, 3>(255, 128, 64);

	return vislib::math::Vector<unsigned char, 3>(191, 191, 191);
}

#ifdef WITH_CURL
void OpenBabelLoader::loadFromPDB(std::string filename, const char* path)
{
	CURL* curl;
	FILE* fp;
	CURLcode res;
	std::string url = "http://www.rcsb.org/pdb/files/";
	url.append(filename);
	curl = curl_easy_init();
	if (curl)
	{
		fp = fopen(path, "wb");
		curl_easy_setopt(curl, CURLOPT_URL, url);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
		res = curl_easy_perform(curl);
		curl_easy_cleanup(curl);
		fclose(fp);
	}
}
#endif

#endif