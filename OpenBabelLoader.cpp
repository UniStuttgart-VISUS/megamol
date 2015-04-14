#include "stdafx.h"
#include "OpenBabelLoader.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
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

	/**
	 * Converter can be used to convert from one data type to another.
	 * In the following only the read function is used.
	 * 
	 */
	OpenBabel::OBConversion conv;

	std::istream* in_file;
	conv = OpenBabel::OBConversion(in_file);

	vislib::TString file = this->filenameSlot.Param<core::param::FilePathParam>()->Value();
	auto list = vislib::StringTokeniserA::Split(file, ".");
	std::string file_type = list[list.Count() - 1];
	if (conv.SetInFormat(file_type.c_str()))
	{
		OpenBabel::OBMol* mol = new OpenBabel::OBMol();
		bool not_end = conv.ReadFile(mol, T2A(file));
		while (not_end)
		{
			atom_count += mol->NumAtoms();
			res_count += mol->NumResidues();
			mol = new OpenBabel::OBMol();
			not_end = conv.Read(mol);
		}
		pos = std::vector<float>(atom_count * 3);
		charge = std::vector<float>(atom_count);
		bfactor = std::vector<float>(atom_count);
		occupancy = std::vector<float>(atom_count);
		atomType_arr.AssertCapacity(atom_count);
		atomTypeIdx_arr.SetCount(atom_count);
		residueIdx = std::vector<int>(atom_count);
		res_bbox = vislib::Array<vislib::math::Cuboid<float> >(atom_count);
		for (unsigned int i = 0; i < atom_count; i++)
		{
			res_bbox[i] = vislib::math::Cuboid<float>();
		}
		residues = std::vector<MolecularDataCall::Residue*>(res_count);
		charge_min = std::numeric_limits<float>::max();
		charge_max = std::numeric_limits<float>::lowest();
		bfactor_min = std::numeric_limits<float>::max();
		bfactor_max = std::numeric_limits<float>::lowest();
		occupancy_min = std::numeric_limits<float>::max();
		occupancy_max = std::numeric_limits<float>::lowest();

		mol = new OpenBabel::OBMol();
		not_end = conv.ReadFile(mol, T2A(file));
		unsigned int global_atomIdx = 0;
		unsigned int global_resIdx = 0;
		unsigned int global_molIdx = 0;
		while (not_end)
		{
			unsigned int local_atom_count = mol->NumAtoms();
			for (unsigned int i = 0; i < local_atom_count; i++)
			{
				vislib::StringA tmpStr = mol->GetAtom(i)->GetType();
				float radius = getElementRadius(tmpStr);
				vislib::math::Vector<unsigned char, 3> color = getElementColor(tmpStr);
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

				charge[global_atomIdx] = (float)mol->GetAtom(i)->GetPartialCharge();
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

				residueIdx[global_atomIdx] = mol->GetAtom(i)->GetResidue()->GetIdx();

				res_bbox[residueIdx[global_atomIdx]].Union(atomBBox);

				pos[global_atomIdx * 3 + 0] = mol->GetAtom(i)->GetX();
				pos[global_atomIdx * 3 + 1] = mol->GetAtom(i)->GetY();
				pos[global_atomIdx * 3 + 2] = mol->GetAtom(i)->GetZ();

				global_atomIdx++;
			}

			unsigned int con_count = mol->NumBonds();
			for (unsigned int i = 0; i < con_count; i++)
			{
				connections.push_back(mol->GetBond(i)->GetBeginAtomIdx());
				connections.push_back(mol->GetBond(i)->GetEndAtomIdx());
			}

			unsigned int local_res_count = mol->NumResidues();

			/* TODO:
			 * For now every molecule is in one chain. If there is a way to figure
			 * out what molecule belongs to what chain in the OpenBabel datastructure
			 * fix this.
			 */
			MolecularDataCall::Molecule mdc_mol
				= MolecularDataCall::Molecule(global_resIdx, local_res_count,0);
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

				residues[global_resIdx] = new MolecularDataCall::Residue(first_atom_index, res_atom_count, res_bbox[global_resIdx],
					typeIdx, moleculeIdx, origResIdx); 

				global_resIdx++;
			}

			global_molIdx++;
			mol = new OpenBabel::OBMol();
			not_end = conv.Read(mol);
		}

		mdc_in->SetAtoms(atom_count, atomType_arr.Count(), atomTypeIdx_arr.PeekElements(), pos.data(),
			atomType_arr.PeekElements(), residueIdx.data(), bfactor.data(), charge.data(), occupancy.data());

		mdc_in->SetBFactorRange(bfactor_min, bfactor_max);
		mdc_in->SetChargeRange(charge_min, charge_max);
		mdc_in->SetOccupancyRange(occupancy_min, occupancy_max);

		mdc_in->SetConnections(connections.size(), connections.data());

		mdc_in->SetResidues(res_count, (const MolecularDataCall::Residue**)residues.data());

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

		mdc_in->SetChains(chain.Count(),chain.PeekElements());
		filter.SetCount(atom_count);
		for (unsigned int i = 0; i < filter.Count(); i++)
			filter[i] = 0;
		mdc_in->SetFilter(filter.PeekElements());
	}	
	return true;
}

/*
* OpenBabelLoader::getExtent
*/
bool OpenBabelLoader::getExtent(core::Call& call)
{
	return true;
}

/*
* OpenBabelLoader::getElementRadius
*/
float OpenBabelLoader::getElementRadius(vislib::StringA name)
{
	// extract the element symbol from the name
	unsigned int cnt = 0;
	vislib::StringA element;
	while (vislib::CharTraitsA::IsDigit(name[cnt])) {
		cnt++;
	}

#ifdef SFB716DEMO
	if (name.Equals("Po")) // Pore
		return 1.5f;
	if (name.Equals("P1")) // Pore (coarse)
		return 0.3f;
	if (name.Equals("XX")) // CL
		return 1.5f / 2.0f;
	if (name.Equals("YY")) // NA
		return 1.5f / 2.0f;
	if (name.Equals("ZZ")) // DNA center
		return 1.5f;
	if (name.Equals("QQ")) // DNA base
		return 1.5f;
#endif

	// --- van der Waals radii ---
	if (name[cnt] == 'H')
		return 1.2f;
	if (name[cnt] == 'C')
		return 1.7f;
	if (name[cnt] == 'N')
		return 1.55f;
	if (name[cnt] == 'O')
		return 1.52f;
	if (name[cnt] == 'S')
		return 1.8f;
	if (name[cnt] == 'P')
		return 1.8f;
	if (name[cnt] == 'C')
		return 1.7f;

	return 1.5f;
}

/*
* Get the color of the element
*/
vislib::math::Vector<unsigned char, 3> OpenBabelLoader::getElementColor(vislib::StringA name) 
{
	// extract the element symbol from the name
	unsigned int cnt = 0;
	vislib::StringA element;
#ifdef SFB716DEMO
	if (name.Equals("Po")) // Pore
		return vislib::math::Vector<unsigned char, 3>(149, 149, 149);
	if (name.Equals("P1")) // Pore (coarse)
		return vislib::math::Vector<unsigned char, 3>(149, 149, 149);
	if (name.Equals("XX")) // CL
		return vislib::math::Vector<unsigned char, 3>(154, 205, 50);
	if (name.Equals("YY")) // NA
		return vislib::math::Vector<unsigned char, 3>(255, 215, 20);
	if (name.Equals("ZZ")) // DNA center
		return vislib::math::Vector<unsigned char, 3>(240, 240, 240);
	if (name.Equals("QQ")) // DNA base
		return vislib::math::Vector<unsigned char, 3>(240, 80, 50);
#endif

	while (vislib::CharTraitsA::IsDigit(name[cnt])) {
		cnt++;
	}
	if (name[cnt] == 'H') // white or light grey
		return vislib::math::Vector<unsigned char, 3>(240, 240, 240);
	if (name[cnt] == 'C') // (dark) grey or green
		return vislib::math::Vector<unsigned char, 3>(90, 90, 90);
	//return vislib::math::Vector<unsigned char, 3>( 125, 125, 125);
	//return vislib::math::Vector<unsigned char, 3>( 90, 175, 50);
	if (name[cnt] == 'N') // blue
		//return vislib::math::Vector<unsigned char, 3>( 37, 136, 195);
		return vislib::math::Vector<unsigned char, 3>(37, 136, 195);
	if (name[cnt] == 'O') // red
		//return vislib::math::Vector<unsigned char, 3>( 250, 94, 82);
		return vislib::math::Vector<unsigned char, 3>(206, 34, 34);
	if (name[cnt] == 'S') // yellow
		//return vislib::math::Vector<unsigned char, 3>( 250, 230, 50);
		return vislib::math::Vector<unsigned char, 3>(255, 215, 0);
	if (name[cnt] == 'P') // orange
		return vislib::math::Vector<unsigned char, 3>(255, 128, 64);
	if (name[cnt] == 'M' /*&& name[cnt+1] == 'e'*/) // Methanol? -> same as carbon ...
		return vislib::math::Vector<unsigned char, 3>(90, 90, 90);
	/*
	if( name[cnt] == 'H' ) // white or light grey
	return vislib::math::Vector<unsigned char, 3>( 240, 240, 240);
	if( name[cnt] == 'C' ) // (dark) grey
	return vislib::math::Vector<unsigned char, 3>( 175, 175, 175);
	if( name[cnt] == 'N' ) // blue
	return vislib::math::Vector<unsigned char, 3>( 40, 160, 220);
	if( name[cnt] == 'O' ) // red
	return vislib::math::Vector<unsigned char, 3>( 230, 50, 50);
	if( name[cnt] == 'S' ) // yellow
	return vislib::math::Vector<unsigned char, 3>( 255, 215, 0);
	if( name[cnt] == 'P' ) // orange
	return vislib::math::Vector<unsigned char, 3>( 255, 128, 64);
	*/

	return vislib::math::Vector<unsigned char, 3>(191, 191, 191);
}