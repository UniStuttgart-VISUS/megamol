#ifdef WITH_OPENBABEL

#ifndef MMPROTEINCUDAPLUGIN_OPENBABELLOADER_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_OPENBABELLOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "stdafx.h"
#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/moldyn/MolecularDataCall.h"
#include "vislib/StringTokeniser.h"
#include "vislib/StringConverter.h"
#include "vislib/sys/Log.h"

#include <openbabel/obconversion.h>
#include <openbabel/mol.h>
#include <openbabel/atom.h>
#ifdef WITH_CURL
#include <curl/curl.h>
#endif

using namespace megamol::core::moldyn;

namespace megamol {
namespace protein_cuda {

class OpenBabelLoader : public megamol::core::Module
{
public:
	/**
	* Answer the name of this module.
	*
	* @return The name of this module.
	*/
	static const char *ClassName(void) {
		return "OpenBabelLoader";
	}

	/**
	* Answer a human readable description of this module.
	*
	* @return A human readable description of this module.
	*/
	static const char *Description(void) {
		return "Offers data conversations.";
	}

	/**
	* Answers whether this module is available on the current system.
	*
	* @return 'true' if the module is available, 'false' otherwise.
	*/
	static bool IsAvailable(void) {
		return true;
	}

	OpenBabelLoader();
	~OpenBabelLoader();

protected:
	/**
	* Implementation of 'Create'.
	*
	* @return 'true' on success, 'false' otherwise.
	*/
	virtual bool create(void);

	/**
	* Implementation of 'release'.
	*/
	virtual void release(void);

	/**
	* Call for get data.
	*/
	bool getData(megamol::core::Call& call);

	/**
	* Call for get extent.
	*/
	bool getExtent(megamol::core::Call& call);

private:
	float getElementRadius(OpenBabel::OBAtom* atom);
	vislib::math::Vector<unsigned char, 3> getElementColor(OpenBabel::OBAtom* atom);
#ifdef WITH_CURL
	/**
	*
	* @param filename for the pdb file in the database
	*/
	void loadFromPDB(std::string filename, const char* path);
#endif
	
	megamol::core::CalleeSlot dataOutSlot;
	core::param::ParamSlot filenameSlot;

	unsigned int atom_count;
	unsigned int con_count;
	vislib::Array<unsigned int> atomTypeIdx_arr;
	vislib::Array<megamol::core::moldyn::MolecularDataCall::AtomType> atomType_arr;
	std::vector<float> charge;
	std::vector<float> bfactor;
	std::vector<float> occupancy;
	std::vector<float> pos;
	std::vector<int> residueIdx;
	float charge_min;
	float charge_max;
	float bfactor_min;
	float bfactor_max;
	float occupancy_min;
	float occupancy_max;

	unsigned int local_atom_count;

	std::vector<unsigned int> connections;

	std::vector<MolecularDataCall::Residue*>* residues;
	unsigned int res_count;
	vislib::Array<unsigned int> resTypeIdx_arr;
	vislib::Array<vislib::StringA> residueTypeName;
	vislib::Array<unsigned int> solventResidueIdx;
	vislib::math::Cuboid<float> bbox;
	vislib::Array<vislib::math::Cuboid<float> > res_bbox;

	std::vector<MolecularDataCall::Molecule> molecules;
	vislib::Array<int> filter;

	vislib::Array<megamol::core::moldyn::MolecularDataCall::Chain> chain;
};

} // namespace protein_cuda
} // namespace megamol

#endif // MMPROTEINCUDAPLUGIN_OPENBABELLOADER_H_INCLUDED

#endif