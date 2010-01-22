/*
 * NetCDFData.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */


#include "stdafx.h"
#include "NetCDFData.h"

#if (defined(WITH_NETCDF) && (WITH_NETCDF))

#include "param/StringParam.h"
#include "param/BoolParam.h"
#include "vislib/BufferedFile.h"
#include "vislib/sysfunctions.h"
#include "vislib/Log.h"
#include <string>
#include <fstream>

using namespace megamol::core;

/* defines for the frame cache size */
// minimum number of frames in the cache (2 for interpolation; 1 for loading)
#define CACHE_SIZE_MIN 3
// maximum number of frames in the cache (just a nice number)
#define CACHE_SIZE_MAX 1000
// factor multiplied to the frame size for estimating the overhead to the pure data.
#define CACHE_FRAME_FACTOR 1.15f

/*****************************************************************************/

/*
 * protein::NetCDFData::Frame::Frame
 */
protein::NetCDFData::Frame::Frame(const NetCDFData *parent) : 
    view::AnimDataModule::Frame(*const_cast<protein::NetCDFData*>(parent)),
	parent(parent), coords(NULL), time(NULL) 
{

}


/*
 * protein::NetCDFData::Frame::~Frame
 */
protein::NetCDFData::Frame::~Frame() 
{
	parent = NULL;
	delete[] coords;
	delete[] time;
}


/*
 * protein::NetCDFData::Frame::Clear
 */
void protein::NetCDFData::Frame::Clear(void) 
{

}

/*
 * protein::NetCDFData::Frame::GetCoords
 */
const float* protein::NetCDFData::Frame::GetCoords() const
{
	// std::cout<<"GetCoords of frame "<<frame<<", starting with "<<std::endl;
	// std::cout<<"( "<<coords[0]<<", "<<coords[1]<<", "<<coords[2]<<")"<<std::endl;
	return &coords[0];
}

/*
 * protein::NetCDFData::Frame::GetDisulfidBonds
 */
std::vector<protein::CallProteinData::IndexPair> protein::NetCDFData::Frame::GetDisulfidBonds()
{
	return disulfidBondsVec;
}

/*****************************************************************************/

/*
 * protein::NetCDFData::NetCDFData (CTOR)
 */
protein::NetCDFData::NetCDFData(void): 
      view::AnimDataModule(), m_lastFrame(NULL), m_lastTime(-1.0f), m_interpolFrame(this),
      m_generalDataFrame(this), m_RMSFrame1(this), m_RMSFrame2(this),
      m_frameID(0), m_numFrames(0), m_numAtoms(0), m_numAtomTypes(0), m_numres(0), m_scale(1.0f),
	  m_numAminoAcidNames(0), m_numAtomsPerMol(NULL) ,m_animTimer(),
      m_filenameParam("filename", "The path to the NetCDF data file to load."),
	  m_generalDataSlot("provideGeneralData", "Connects the protein rendering with netcdf data storage"),
      m_RMSFrameDataSlot1("provideRMSFrameData1", "Connects the protein rendering with netcdf data storage"),
      m_RMSFrameDataSlot2("provideRMSFrameData2", "Connects the protein rendering with netcdf data storage")
{
	//numberOfConnections = 0;
	//checkBondLength = false;
	//checkBondLength = true;

    this->m_animTimer.SetSource(this);

    CallProteinDataDescription cpdd;
    this->m_generalDataSlot.SetCallback(cpdd.ClassName(), "GetData", &NetCDFData::NetCDFDataCallback);
    this->MakeSlotAvailable(&this->m_generalDataSlot);

    this->m_RMSFrameDataSlot1.SetCallback(cpdd.ClassName(), "GetData", &NetCDFData::NetCDFDataCallback);
    this->MakeSlotAvailable(&this->m_RMSFrameDataSlot1);

    this->m_RMSFrameDataSlot2.SetCallback(cpdd.ClassName(), "GetData", &NetCDFData::NetCDFDataCallback);
    this->MakeSlotAvailable(&this->m_RMSFrameDataSlot2);

    this->m_filenameParam.SetParameter(new param::StringParam(""));
    this->MakeSlotAvailable(&this->m_filenameParam);

	// access to parameter from AnimDataTimer class
	this->MakeSlotAvailable(&this->m_animTimer.AnimationActiveParameter());
    this->MakeSlotAvailable(&this->m_animTimer.AnimationSpeedParameter());

    this->m_animation = false;
}


/*
 * protein::NetCDFData::~NetCDFData (DTOR)
 */
protein::NetCDFData::~NetCDFData(void)
{
    if (m_netcdfOpen) 
	{
        nc_close(m_netcdfFileID);
        m_netcdfFileID = 0;
        m_netcdfOpen = false;
    }
	delete[] m_numAtomsPerMol;

    this->Release ();
}


/*
 * protein::NetCDFData::create
 */
bool protein::NetCDFData::create(void)
{
    this->tryLoadFile();
    this->m_filenameParam.ResetDirty();
    return true;
}

/*
 * protein::NetCDFData::tryLoadFile
 */
bool protein::NetCDFData::tryLoadFile(void)
{
    //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, 
	//                                      "protein::NetCDFData::create called \n");

    vislib::StringA tmp_netcdffile(this->m_filenameParam.Param<param::StringParam>()->Value());
    vislib::StringA tmp_topfile(this->m_filenameParam.Param<param::StringParam>()->Value());

	if (tmp_netcdffile.IsEmpty()) {
        // no file to load
        return false;
    }
    const char *netcdffile = tmp_netcdffile.PeekBuffer();

    //TODO: Ueber Namen oder eigenes Fileformat fuer top+netCDF???
    //TODO: .parm erlauben!
    vislib::StringA topfiletype("top");
    // set filename of topology file
    tmp_topfile.Truncate( tmp_netcdffile.Length()-6);
    tmp_topfile += topfiletype;
    const char *topfile = tmp_topfile.PeekBuffer();

    if (!(readTopfile(topfile))) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
				"NetCDFData: Failed to read topology file %s \n",topfile);
        return false;
    }

    // now read amber netCDF file

    // open netcdf file
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "Reading netCDF file %s \n",netcdffile);

    int status = nc_open(netcdffile, 0, &m_netcdfFileID);
    if (status != NC_NOERR) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "NetCDFData: Error reading netCDF file: %s \n", nc_strerror(status));
        return false;
    }
    m_netcdfOpen = true;
    
    // read header from netCDF file
    if (!readHeaderFromNetCDF()) {
        nc_close(m_netcdfFileID);
        m_netcdfOpen = false;
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "NetCDFData: Error reading header from netCDF file. \n");
    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "NetCDFData: Read header from netCDF file. \n");
    }
	if (m_numFrames > 0) {
		setFrameCount(m_numFrames);
	} else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "NetCDFData: No frame count found in netCDF file. \n");
		return false;
	}

    // read first frame from trajectory:
	loadFrame(&this->m_interpolFrame, 0);
    loadFrame(&this->m_RMSFrame1, 0);
    loadFrame(&this->m_RMSFrame2, 0);
    loadFrame(&this->m_generalDataFrame, 0);

	// all atom and amino acid data is read, make the connectivity table
	/* TODO: Connections aus netcdf!
	if( !this->makeConnections() ){
		retval = false;
		m_netcdfOpen = false;
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
		"NetCDFData: makeConnections failed!");
	} else {
		// try to read the secondary structure from a file
		// THIS HAS TO BE DONE AFTER makeConnections() WAS CALLED!!!
		this->readSecondaryStructure();
	}
	*/

	// TODO #frames im Cache berechnen!
	initFrameCache(10);

    return m_netcdfOpen;
}

/*
 * protein::NetCDFData::NetCDFDataCallback
 */
bool protein::NetCDFData::NetCDFDataCallback( Call& call) 
{
	Frame *af, *bf;
	unsigned int at;

    protein::CallProteinData *pdi = dynamic_cast<protein::CallProteinData*>(&call);

    // parameter refresh
    if (this->m_filenameParam.IsDirty()) 
	{
		this->tryLoadFile();
		this->m_filenameParam.ResetDirty();

        if(pdi->GetRMSUse())
        {
            /* Reset frame cache because of conflict with loadFrame.
             * If cache is needed too, precautions for synchronised 
             * access to shared ressources have to be done.
             */
            resetFrameCache();
        }
    }

	if(pdi)
	{
		if(pdi->GetRMSUse())
		{
			this->m_lastFrame = NULL;
            // load requested frame directly
            if(call.PeekCalleeSlot()->Name() == this->m_generalDataSlot.Name())
            {
        	    this->loadFrame(&this->m_generalDataFrame, pdi->GetRequestedRMSFrame());
		        this->m_lastFrame = &this->m_generalDataFrame;
            }
            else if(call.PeekCalleeSlot()->Name() == this->m_RMSFrameDataSlot1.Name()) 
            {
        	    this->loadFrame(&this->m_RMSFrame1, pdi->GetRequestedRMSFrame());
		        this->m_lastFrame = &this->m_RMSFrame1;
            }
            else if(call.PeekCalleeSlot()->Name() == this->m_RMSFrameDataSlot2.Name())
            {
        	    this->loadFrame(&this->m_RMSFrame2, pdi->GetRequestedRMSFrame());
		        this->m_lastFrame = &this->m_RMSFrame2;
            }
		}
        else
		{
            // parameter refresh
	        if(this->m_animTimer.AnimationActiveParameter().IsDirty())
	        {
		        this->m_animation = this->m_animTimer.AnimationActiveParameter().Param<param::BoolParam>()->Value();
                this->m_animTimer.AnimationActiveParameter().ResetDirty();
	        }
            // animation control
	        if(this->m_animation)
	        {
		        if (!this->m_animTimer.IsRunning())
			        this->m_animTimer.Start();
	        }
	        else
	        {
		        if (this->m_animTimer.IsRunning())
			        this->m_animTimer.Stop();
	        }

            // unlock last frame
            if(this->m_lastFrame != NULL)
            {
                this->m_lastFrame->Unlock();
                this->m_lastFrame = NULL;    // DO NOT DELETE
            }
			// from AnimDataTimer (VIMDataSource ?)...
			float animTime = this->m_animTimer.Time();

			if (!vislib::math::IsEqual(animTime, this->m_lastTime)) {
				at = static_cast<unsigned int>(animTime);
				af = dynamic_cast<Frame*>(this->requestLockedFrame(at)); // unlock true

				if (af == NULL) {
					// no data at all?
					this->m_lastFrame = NULL;
					this->m_lastTime = -1.0f; // because loading might change the situation

				} else if ((af->FrameNumber() == at) && (at + 1 < this->FrameCount())) {
					bf = dynamic_cast<Frame*>(this->requestLockedFrame(at + 1));  // unlock false
					if ((bf != NULL) && (bf->FrameNumber() == at + 1)) {
						// both frames present, so try to interpolate
                        this->m_lastFrame = const_cast<Frame*>(this->m_interpolFrame.MakeInterpolationFrame(animTime - float(at), *af, *bf));
						this->m_lastTime = animTime;
                        af->Unlock();
                        bf->Unlock();

					} else {
						// correct bf not available
						this->m_lastFrame = af;
						this->m_lastTime = -1.0f; // because loading might change the situation
                        if(bf != NULL)
                            bf->Unlock();
					}
				} else {
					// not the right data, so don't even try to interpolate anything
					this->m_lastFrame = af;
					if (at + 1 >= this->FrameCount()) {
						this->m_lastTime = animTime;
					}
				}
			}
			// ...from AnimDataTimer (VIMDataSource ?)
		}

		if (this->m_lastFrame != NULL) {
			// assign protAtomData
			pdi->SetProteinAtomCount(this->m_numProteinAtoms);
			pdi->SetProteinAtomDataPointer(&this->m_proteinAtomTable[0]);
			// TODO hier muss interpoliert werden, falls noetig!
			const float *coords = this->m_lastFrame->GetCoords();
			pdi->SetProteinAtomPositionPointer(const_cast<float*>(coords));
			pdi->SetAtomTypeTable (this->m_atomTypeTable.size(), &this->m_atomTypeTable[0]);

			//assign aminoAcidNameCnt
			pdi->SetAminoAcidNameCount(this->m_numAminoAcidNames);
			pdi->SetAminoAcidNameTable(this->m_numAminoAcidNames, &this->m_aminoAcidTypeTable[0]);

			// assign solAtomData
			pdi->SetSolventAtomCount(this->m_numSolventAtoms);
			pdi->SetSolventAtomDataPointer(&this->m_solventAtomTable[0]);
			pdi->SetSolventAtomPositionPointer (const_cast<float*>(&coords[3*this->m_numProteinAtoms]));
			pdi->SetSolventMoleculeTypeCount(this->m_numSolventMolTypes);
			for (unsigned int idx=0; idx<m_numSolventMolTypes; idx++) {
				pdi->SetSolventMoleculeCount(idx, this->m_solventTypeVec[idx].numMolsPerType);
				pdi->SetSolventMoleculeData(idx, this->m_solventData[idx]);
			}

			// assign bounding box
			pdi->SetBoundingBox((-1.0f)*this->m_boundingBox[0]/2.0f, (-1.0f)*this->m_boundingBox[1]/2.0f,
					(-1.0f)*this->m_boundingBox[2]/2.0f, this->m_boundingBox[0]/2.0f,
					this->m_boundingBox[1]/2.0f, this->m_boundingBox[2]/2.0f);

			// assign the disulfide bond table
			std::vector<protein::CallProteinData::IndexPair> disulfidBondsVec(m_interpolFrame.GetDisulfidBonds());
			pdi->SetDisulfidBondsPointer(disulfidBondsVec.size(), &disulfidBondsVec[0]);
			pdi->SetScaling(this->m_scale);
		} 
		else {
			// assign protAtomData
			pdi->SetProteinAtomCount(m_numProteinAtoms);
			pdi->SetProteinAtomDataPointer(NULL);
			// TODO hier muss interpoliert werden, falls noetig!
			pdi->SetProteinAtomPositionPointer(NULL);
			pdi->SetAtomTypeTable (0, NULL);

			//assign aminoAcidNameCnt
			std::cout<<"m_numAminoAcidNames="<<m_numAminoAcidNames<<std::endl;
			pdi->SetAminoAcidNameCount(m_numAminoAcidNames);
			pdi->SetAminoAcidNameTable(0, NULL);

			// assign solAtomData
			std::cout<<"m_numSolventAtoms="<<m_numSolventAtoms<<std::endl;
			pdi->SetSolventAtomCount(m_numSolventAtoms);
			pdi->SetSolventAtomDataPointer(NULL);
			pdi->SetSolventAtomPositionPointer (NULL);
			std::cout<<"m_numSolventMolTypes="<<m_numSolventMolTypes<<std::endl;
			pdi->SetSolventMoleculeTypeCount(m_numSolventMolTypes);
			/*
			for (unsigned int idx=0; idx<m_numSolventMolTypes; idx++) {
				pdi->SetSolventMoleculeCount(idx, m_solventTypeVec[idx].m_numMolsPerType);
				pdi->SetSolventMoleculeData(idx, m_solventData[idx]);
			}
			*/

			// assign bounding box
			pdi->SetBoundingBox((-1.0f)*this->m_boundingBox[0]/2.0f, (-1.0f)*this->m_boundingBox[1]/2.0f,
					(-1.0f)*this->m_boundingBox[2]/2.0f, this->m_boundingBox[0]/2.0f,
					this->m_boundingBox[1]/2.0f, this->m_boundingBox[2]/2.0f);

			// assign the disulfide bond table
			std::vector<protein::CallProteinData::IndexPair> disulfidBondsVec(m_interpolFrame.GetDisulfidBonds());
			pdi->SetDisulfidBondsPointer(0, NULL);
			pdi->SetScaling(1.0);
		}

		// to start animation by default
		/*if (!this->m_animTimer.IsRunning())
			this->m_animTimer.Start();*/
	}
	return true;
}


/**********************************************************************
 * 'netcdf'-functions
 **********************************************************************/

/*
 * protein::NetCDFData::readTopfile
 *
 * Reads the topology file (.top)
 */
bool protein::NetCDFData::readTopfile(const char* &topofile)
{
	protein::CallProteinData::AtomData atom;
	protein::CallProteinData::AtomType atomtype;
	protein::CallProteinData::AminoAcid aminoacid;

    //std::vector<unsigned int> m_sulfides;

    //bool for new parm file without flags and format comments:
    bool newparm=0;

    std::string firstline;
    std::string title;
    std::string dummy_line;

    //std::cout<<"topfile = "<<topofile<<std::endl;
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "NetCDFData: Reading top/parm file: %s \n", topofile);
    std::ifstream topfile(topofile);
    if (!topfile.is_open()) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "NetCDFData: Failed to open top/parm file: %s \n", topofile);
        return false;
    } else {
        getline(topfile, firstline);

        char buffer[8];
        for (int i=0; i<8; i++)
            buffer[i]=firstline[i];

        if (!strncmp( buffer, "%VERSION", 8 ) ){
            newparm = 1;
            //std::cout<<"New parm file type"<<std::endl;
            for(int i=0; i<3; i++){
                getline(topfile, title);
            }
        } else{
            newparm=0;
            title=firstline;
            //std::cout<<"Old parm file type"<<std::endl;
        }

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }

        /** POINTERS; FORMAT(10I8) */
        /**
         *  0)NATOM,   1)NTYPES,  2)NBONH,   3)MBONA,   4)NTHETH,  5)MTHETA,
         *  6)NPHIH,   7)MPHIA,   8)NHPARM,  9)NPARM,  10)NEXT,   11)NRES,
         * 12)NBONA,  13)NTHETA, 14)NPHIA,  15)NUMBND, 16)NUMANG, 17)NPTRA,
         * 18)NATYP,  19)NPHB,   20)IFPERT, 21)NBPER,  22)NGPER,  23)NDPER,
         * 24)MBPER,  25)MGPER,  26)MDPER,  27)IFBOX,  28)NMXRS,  29)IFCAP
         *
         * NATOM  : total number of atoms 
         * NTYPES : total number of distinct atom types
         * NBONH  : number of bonds containing hydrogen
         * MBONA  : number of bonds not containing hydrogen
         * NTHETH : number of angles containing hydrogen
         * MTHETA : number of angles not containing hydrogen
         * NPHIH  : number of dihedrals containing hydrogen
         * MPHIA  : number of dihedrals not containing hydrogen
         * NHPARM : currently not used
         * NPARM  : currently not used
         * NEXT   : number of excluded atoms
         * NRES   : number of residues
         * NBONA  : MBONA + number of constraint bonds
         * NTHETA : MTHETA + number of constraint angles
         * NPHIA  : MPHIA + number of constraint dihedrals
         * NUMBND : number of unique bond types
         * NUMANG : number of unique angle types
         * NPTRA  : number of unique dihedral types
         * NATYP  : number of atom types in parameter file, see SOLTY below
         * NPHB   : number of distinct 10-12 hydrogen bond pair types
         * IFPERT : set to 1 if perturbation info is to be read in
         * NBPER  : number of bonds to be perturbed
         * NGPER  : number of angles to be perturbed
         * NDPER  : number of dihedrals to be perturbed
         * MBPER  : number of bonds with atoms completely in perturbed group
         * MGPER  : number of angles with atoms completely in perturbed group
         * MDPER  : number of dihedrals with atoms completely in perturbed groups
         * IFBOX  : set to 1 if standard periodic box, 2 when truncated octahedral
         * NMXRS  : number of atoms in the largest residue
         * IFCAP  : set to 1 if the CAP option from edit was specified
         */

        // array of parameters needed for file read
        int nums[31];

        for (int num=0; num<30; num++)
            topfile >> nums[num];

        if (newparm)
            topfile >> nums[30];
        else
            nums[30]=0;
        getline(topfile, dummy_line);


		// total number of atoms
        m_numAtoms = nums[0];
		// total number of atom types
        m_numAtomTypes = nums[1];
		// total number of residues
        m_numres = nums[11];

		/** 
         *  ifbox: set to 1 if standard periodic box,
		 *         2 when truncated octahedral 
		 *  decides whether some data is contained in the coordinate file or not
		 */
        int ifbox = nums[27];
		/** 
         *  ifcap: set to 1 if the CAP option from edit was specified
		 *  decides whether some data is contained in the coordinate file or not
		 */
        int ifcap = nums[29];
		/** 
         *  ifpert : set to 1 if perturbation info is to be read in
		 */
        int ifpert = nums[20];

        //array of atoms and their attributes:
        // atoms = new Atom[m_numAtoms];

        m_atomTable.clear();
        m_atomTable.reserve(m_numAtoms);

        m_aminoAcidTable.clear();
        m_aminoAcidTable.reserve(m_numres);

		m_atomTypeTable.clear();
		m_atomTypeTable.reserve(m_numAtomTypes);


        // ATOM_NAME; FORMAT(20a4)
        /* FORMAT(20a4)  (IGRAPH(i), i=1,NATOM)
         * IGRAPH : the user atoms names
         */
		// Both, protein and solvent, because here we don't know yet which is which
        if (newparm) {
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        char tmp_name[5];
        for (unsigned int i=0; i<m_numAtoms; i++) {
			//protein::CallProteinData::AtomType atomtype;
            topfile.width(5);
            topfile >> tmp_name;
//std::cout<<"atomtype "<<tmp_name<<"\n";
			// check whether it's a new atom type
			atomtype.SetName(vislib::StringA(tmp_name));
			unsigned int typeIdx = 0;
			unsigned int idx=0;
			//unsigned int m_atomTypeTableSize = m_atomTypeTable.size();
			if (m_atomTypeTable.size() > 0) {
				for (idx=0; idx<m_atomTypeTable.size(); idx++) {
					if (m_atomTypeTable[idx].Name() == atomtype.Name()) {
						typeIdx = idx;
						break;
					}
				}
			}
			if (m_atomTypeTable.empty() || idx == m_atomTypeTable.size()) {
				m_atomTypeTable.push_back(atomtype);
				typeIdx = m_atomTypeTable.size()-1;
			}

			atom.SetTypeIndex(typeIdx);

            // if the current atom is the S_gamma of CYStein add index to
            // m_sulfides vector
            if (atomtype.Name().Compare("SG")) {
                m_sulfides.push_back(i);
            }

            // initially add atom to m_atomTable
            m_atomTable.push_back(atom);
        }

        getline(topfile, dummy_line);


        // CHARGE; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (CHRG(i), i=1,NATOM)
         * CHRG: the atom charges.  (Divide by 18.2223 to convert to charge
         *       in units of the electron charge)
         */
		// Both, protein and solvent, because here we don't know yet which is which
        if (newparm) {
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
		float charge;
        for (unsigned int i=0; i<m_numAtoms; i++) {
            //topfile >> m_atomTable[i].charge;
			topfile >> charge;
			m_atomTable[i].SetCharge(charge);
        }
        getline(topfile, dummy_line);


        // MASS; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (AMASS(i), i=1,NATOM)
         * AMASS  : the atom masses
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // TODO: add to atomData?
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        double tmp_mass;
        for(unsigned int i=0; i<m_numAtoms; i++){
            topfile >> tmp_mass;
            //      topfile >> m_atomTable[i].mass;
        }
        getline(topfile, dummy_line);


        // ATOM_TYPE_INDEX; FORMAT(10I8)
        /* FORMAT(12I6)  (IAC(i), i=1,NATOM)
         * IAC: index for the atom types involved in Lennard Jones (6-12) 
         *      interactions.  See ICO below.
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // TODO: ???
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        int tmp_iac;
        for(unsigned int i=0; i<m_numAtoms; i++){
            topfile >> tmp_iac;
            //      topfile >> atoms[i].iac;
        }
        getline(topfile, dummy_line);


        // NUMBER_EXCLUDED_ATOMS; FORMAT(10I8)
        /* FORMAT(12I6)  (NUMEX(i), i=1,NATOM)
         * NUMEX: total number of excluded atoms for atom "i".  See
         * NATEX below.
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // TODO: ???
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        int tmp_numex;
        for(unsigned int i=0; i<m_numAtoms; i++){
            topfile >> tmp_numex;
            //topfile >> atoms[i].numex;
        }
        getline(topfile, dummy_line);

        // NONBONDED_PARM_INDEX; FORMAT(10I8)
        /* FORMAT(12I6)  (ICO(i), i=1,NTYPES*NTYPES)
         * ICO: provides the index to the nonbon parameter
         *      arrays CN1, CN2 and ASOL, BSOL.  All possible 6-12
         *      or 10-12 atoms type interactions are represented.
         *      NOTE: A particular atom type can have either a 10-12
         *      or a 6-12 interaction, but not both.  The index is
         *      calculated as follows:
         *        index = ICO(NTYPES*(IAC(i)-1)+IAC(j))
         *      If index is positive, this is an index into the
         *      6-12 parameter arrays (CN1 and CN2) otherwise it
         *      is an index into the 10-12 parameter arrays (ASOL
         *      and BSOL).
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // TODO: ???
        // int ico[nums[1]*nums[1]];
        int tmp_ico;
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[1]*nums[1]; i++){
            topfile >> tmp_ico;
            //topfile >> ico[i];
        }
        getline(topfile, dummy_line);


        // RESIDUE_LABEL; FORMAT(20a4)
        /* FORMAT(20A4)  (LABRES(i), i=1,NRES)
         * LABRES : the residue labels
         */
		// Both, protein and solvent, because here we don't know yet which is which
        //labres=new std::string[m_numres];
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        char tmp_label[5];
		//std::cout<<"m_numres = "<<m_numres<<std::endl;
        for(int i=0; i<m_numres; i++){
            topfile.width(5);
            topfile >> tmp_label;

			// check whether it's a new amino acid type
			unsigned int m_aminoAcidTypeTableSize = m_aminoAcidTypeTable.size();
			if (m_aminoAcidTypeTableSize > 0) {
				unsigned int aaIdx = 0;
				for (aaIdx = 0; aaIdx < m_aminoAcidTypeTable.size(); aaIdx++) {
					if (m_aminoAcidTypeTable[aaIdx] == tmp_label) {
						aminoacid.SetNameIndex(aaIdx);
						break;
					} 
				}
				if (aaIdx == m_aminoAcidTypeTable.size()) {
					m_aminoAcidTypeTable.push_back(tmp_label);
					aminoacid.SetNameIndex(m_aminoAcidTypeTable.size()-1);
				}
			} else {
				//std::cout<<i<<": Found new amino acid ("<<tmp_label<<"), new idx = 0"<<std::endl;
				//Add first amino acid type
				m_aminoAcidTypeTable.push_back(tmp_label);
				aminoacid.SetNameIndex(0);
			}
			m_aminoAcidTable.push_back(aminoacid);
        }
		m_numAminoAcidNames = m_aminoAcidTypeTable.size();
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
				"NetCDFData: m_numAminoAcidNames = %i", m_numAminoAcidNames);
        getline(topfile, dummy_line);


        // RESIDUE_POINTER; FORMAT(10I8)
        /* FORMAT(12I6)  (IPRES(i), i=1,NRES)
         * IPRES: atoms in each residue are listed for atom "i" in
         *	     IPRES(i) to IPRES(i+1)-1
         */
		// Both, protein and solvent, because here we don't know yet which is which
        //ipres=new int[m_numres];
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
		unsigned int prevfirstAtom;
		unsigned int firstAtom;

		topfile >> prevfirstAtom;
        for(int i=1; i<m_numres; i++){
            topfile >> firstAtom;
			m_aminoAcidTable[i-1].SetPosition(prevfirstAtom, firstAtom-prevfirstAtom);
			prevfirstAtom = firstAtom;
        }
		m_aminoAcidTable[m_numres-1].SetPosition(prevfirstAtom, (m_numAtoms+1)-prevfirstAtom);
		//m_aminoAcidTable[i-1].SetPosition(prevfirstAtom, (m_numAtoms+1)-prevfirstAtom);

        getline(topfile, dummy_line);

        // BOND_FORCE_CONSTANT; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (RK(i), i=1,NUMBND)
         * RK     : force constant for the bonds of each type, kcal/mol
         */
        // ???

        //   double bondforces[nums[15]];
        double bondforces;
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[15]; i++){
            topfile >> bondforces;
            //topfile >> bondforces[i];
        }
        getline(topfile, dummy_line);


        // BOND_EQUIL_VALUE; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (REQ(i), i=1,NUMBND)
         * REQ    : the equilibrium bond length for the bonds of each type, angstroms
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        //   double bondlenghts[nums[15]];
        double bondlenghts;
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[15]; i++){
            topfile >> bondlenghts;
            // topfile >> bondlenghts[i];
        }
        getline(topfile, dummy_line);


        // ANGLE_FORCE_CONSTANT; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (TK(i), i=1,NUMANG)
         * TK     : force constant for the angles of each type, kcal/mol A**2
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        //   double angleforces[nums[16]];
        double angleforces;
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[16]; i++){
            topfile >> angleforces;
            // topfile >> angleforces[i];
        }
        getline(topfile, dummy_line);


        // ANGLE_EQUIL_VALUE; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (TEQ(i), i=1,NUMANG)
         * TEQ    : the equilibrium angle for the angles of each type, radians
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        //   double equil_angle[nums[16]];
        double equil_angle;
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[16]; i++){
            topfile >> equil_angle;
            // topfile >> equil_angle[i];
        }
        getline(topfile, dummy_line);


        // DIHEDRAL_FORCE_CONSTANT; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (PK(i), i=1,NPTRA)
         * PK     : force constant for the dihedrals of each type, kcal/mol
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        //   double dihedralforces[nums[17]];
        double dihedralforces;
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[17]; i++){
            topfile >> dihedralforces;
            // topfile >> dihedralforces[i];
        }
        getline(topfile, dummy_line);


        // DIHEDRAL_PERIODICITY; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (PN(i), i=1,NPTRA)
         * PN     : periodicity of the dihedral of a given type
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        //   double dihedralperiod[nums[17]];
        double dihedralperiod;
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[17]; i++){
            topfile >> dihedralperiod;
            // topfile >> dihedralperiod[i];
        }
        getline(topfile, dummy_line);


        // DIHEDRAL_PHASE; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (PHASE(i), i=1,NPTRA)
         * PHASE  : phase of the dihedral of a given type, radians
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        //   double dihedralphase[nums[17]];
        double dihedralphase;
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[17]; i++){
            topfile >> dihedralphase;
            //topfile >> dihedralphase[i];
        }
        getline(topfile, dummy_line);


        // SOLTY; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (SOLTY(i), i=1,NATYP)
         * SOLTY  : currently unused (reserved for future use)
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        //double solty[nums[18]];
        double solty;
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[18]; i++){
            topfile >> solty; 
            //topfile >> solty[i]; 
        }
        getline(topfile, dummy_line);

        // LENNARD_JONES_ACOEF; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (CN1(i), i=1,NTYPES*(NTYPES+1)/2)
         * CN1: Lennard Jones r**12 terms for all possible atom type
         *      interactions, indexed by ICO and IAC; for atom i and j
         *      where i < j, the index into this array is as follows
         *      (assuming the value of ICO(index) is positive):
         *      CN1(ICO(NTYPES*(IAC(i)-1)+IAC(j))).
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        int size_cn1=nums[1]*(nums[1]+1)/2;
        double cn1;
        //double cn1[size_cn1];
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<size_cn1; i++){
            topfile >> cn1; 
            //topfile >> cn1[i]; 
        }
        getline(topfile, dummy_line);


        // LENNARD_JONES_BCOEF; FORMAT(5E16.8)
        /* FORMAT(5E16.8)  (CN2(i), i=1,NTYPES*(NTYPES+1)/2)
         * CN2: Lennard Jones r**6 terms for all possible atom type
         *      interactions.  Indexed like CN1 above.
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        int size_cn2=nums[1]*(nums[1]+1)/2;
        //double cn2[size_cn2];
        double cn2;
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<size_cn2; i++){
            topfile >> cn2; 
            //topfile >> cn2[i]; 
        }
        getline(topfile, dummy_line);


        /* NOTE: the atom numbers in the following arrays that describe bonds, angles, and
         * dihedrals are coordinate array indexes for runtime speed. The true atom number
         * equals the absolute value of the number divided by three, plus one. In the case
         * of the dihedrals, if the fourth atom is negative, this implies that the
         * dihedral is an improper. If the third atom is negative, this implies that the
         * end group interations are to be ignored. End group interactions are ignored,
         * for example, in dihedrals of various ring systems (to prevent double counting
         * of 1-4 interactions) and in multiterm dihedrals.
         */

        // TODO: Bond info in Connectivity eintragen !!!

        // BONDS_INC_HYDROGEN; FORMAT(10I8)
        // IBH, JBH, ICBH
        /* FORMAT(12I6)  (IBH(i),JBH(i),ICBH(i), i=1,NBONH)
         * IBH    : atom involved in bond "i", bond contains hydrogen
         * JBH    : atom involved in bond "i", bond contains hydrogen
         * ICBH   : index into parameter arrays RK and REQ
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // TODO!

        struct bonds_hydro_struct{
            int ibh;
            int jbh;
            int icbh;
        };

        bonds_hydro_struct bonds_hydro;
        // bonds_hydro_struct bonds_hydro[nums[2]];

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[2]; i++){
            topfile >> bonds_hydro.ibh;
            topfile >> bonds_hydro.jbh;
            topfile >> bonds_hydro.icbh;
            //topfile >> bonds_hydro[i].ibh;
            //topfile >> bonds_hydro[i].jbh;
            //topfile >> bonds_hydro[i].icbh;
        }
        getline(topfile, dummy_line);


        // BONDS_WITHOUT_HYDROGEN; FORMAT(10I8)
        // IB, JB, ICB
        /* FORMAT(12I6)  (IB(i),JB(i),ICB(i), i=1,NBONA)
         * IB     : atom involved in bond "i", bond does not contain hydrogen
         * JB     : atom involved in bond "i", bond does not contain hydrogen
         * ICB    : index into parameter arrays RK and REQ
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // TODO!

        struct bonds_nohydro_struct{
            int ibh;
            int jbh;
            int icbh;
        };

        //bonds_nohydro_struct bonds_nohydro[nums[3]];
        bonds_nohydro_struct bonds_nohydro;

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[3]; i++){
            topfile >> bonds_nohydro.ibh;
            topfile >> bonds_nohydro.jbh;
            topfile >> bonds_nohydro.icbh;
            //topfile >> bonds_nohydro[i].ibh;
            //topfile >> bonds_nohydro[i].jbh;
            //topfile >> bonds_nohydro[i].icbh;
        }
        getline(topfile, dummy_line);


        // ANGLES_INC_HYDROGEN; FORMAT(10I8)
        // ITH, JTH, KTH, ICTH
        /* FORMAT(12I6)  (ITH(i),JTH(i),KTH(i),ICTH(i), i=1,NTHETH)
         * ITH    : atom involved in angle "i", angle contains hydrogen
         * JTH    : atom involved in angle "i", angle contains hydrogen
         * KTH    : atom involved in angle "i", angle contains hydrogen
         * ICTH   : index into parameter arrays TK and TEQ for angle
         *          ITH(i)-JTH(i)-KTH(i)
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        struct angles_hydro_struct{
            int ith;
            int jth;
            int kth;
            int icth;
        };

        angles_hydro_struct angles_hydro;
        //   angles_hydro_struct angles_hydro[nums[4]];

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[4]; i++){
            topfile >> angles_hydro.ith;
            topfile >> angles_hydro.jth;
            topfile >> angles_hydro.kth;
            topfile >> angles_hydro.icth;
            //topfile >> angles_hydro[i].ith;
            //topfile >> angles_hydro[i].jth;
            //topfile >> angles_hydro[i].kth;
            //topfile >> angles_hydro[i].icth;
        }
        getline(topfile, dummy_line);


        // ANGLES_WITHOUT_HYDROGEN; FORMAT(10I8)
        // IT, JT, KT, ICT
        /* FORMAT(12I6)  (IT(i),JT(i),KT(i),ICT(i), i=1,NTHETA)
         * IT     : atom involved in angle "i", angle does not contain hydrogen
         * JT     : atom involved in angle "i", angle does not contain hydrogen
         * KT     : atom involved in angle "i", angle does not contain hydrogen
         * ICT    : index into parameter arrays TK and TEQ for angle
         *          ITH(i)-JTH(i)-KTH(i)
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        struct angles_nohydro_struct{
            int it;
            int jt;
            int kt;
            int ict;
        };

        angles_nohydro_struct angles_nohydro;
        //   angles_nohydro_struct angles_nohydro[nums[5]];

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[5]; i++){
            topfile >> angles_nohydro.it;
            topfile >> angles_nohydro.jt;
            topfile >> angles_nohydro.kt;
            topfile >> angles_nohydro.ict;
            //topfile >> angles_nohydro[i].it;
            //topfile >> angles_nohydro[i].jt;
            //topfile >> angles_nohydro[i].kt;
            //topfile >> angles_nohydro[i].ict;
        }
        getline(topfile, dummy_line);


        // DIHEDRALS_INC_HYDROGEN; FORMAT(10I8)
        // IPH, JPH, KPH, LPH, ICPH
        /* FORMAT(12I6)  (IPH(i),JPH(i),KPH(i),LPH(i),ICPH(i), i=1,NPHIH)
         * IPH    : atom involved in dihedral "i", dihedral contains hydrogen
         * JPH    : atom involved in dihedral "i", dihedral contains hydrogen
         * KPH    : atom involved in dihedral "i", dihedral contains hydrogen
         * LPH    : atom involved in dihedral "i", dihedral contains hydrogen
         * ICPH   : index into parameter arrays PK, PN, and PHASE for
         *          dihedral IPH(i)-JPH(i)-KPH(i)-LPH(i)
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        struct dihedral_hydro_struct{
            int iph;
            int jph;
            int kph;
            int lph;
            int icph;
        };

        dihedral_hydro_struct dihedral_hydro;
        //dihedral_hydro_struct dihedral_hydro[nums[6]];

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[6]; i++){
            topfile >> dihedral_hydro.iph;
            topfile >> dihedral_hydro.jph;
            topfile >> dihedral_hydro.kph;
            topfile >> dihedral_hydro.lph;
            topfile >> dihedral_hydro.icph;
            //topfile >> dihedral_hydro[i].iph;
            //topfile >> dihedral_hydro[i].jph;
            //topfile >> dihedral_hydro[i].kph;
            //topfile >> dihedral_hydro[i].lph;
            //topfile >> dihedral_hydro[i].icph;
        }
        getline(topfile, dummy_line);


        // DIHEDRALS_WITHOUT_HYDROGEN, FORMAT(10I8)
        // IP, JP, KP, LP, ICP
        /* FORMAT(12I6)  (IP(i),JP(i),KP(i),LP(i),ICP(i), i=1,NPHIA)
         * IP: atom involved in dihedral "i", dihedral does not contain hydrogen
         * JP: atom involved in dihedral "i", dihedral does not contain hydrogen
         * KP: atom involved in dihedral "i", dihedral does not contain hydrogen
         * LP: atom involved in dihedral "i", dihedral does not contain hydrogen
         * ICP: index into parameter arrays PK, PN, and PHASE for
         *      dihedral IPH(i)-JPH(i)-KPH(i)-LPH(i).  Note, if the
         *      periodicity is negative, this implies the following entry
         *      in the PK, PN, and PHASE arrays is another term in a
         *      multitermed dihedral.  
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        struct dihedral_nohydro_struct{
            int ip;
            int jp;
            int kp;
            int lp;
            int icp;
        };

        dihedral_nohydro_struct dihedral_nohydro;
        //dihedral_nohydro_struct dihedral_nohydro[nums[7]];

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[7]; i++){
            topfile >> dihedral_nohydro.ip;
            topfile >> dihedral_nohydro.jp;
            topfile >> dihedral_nohydro.kp;
            topfile >> dihedral_nohydro.lp;
            topfile >> dihedral_nohydro.icp;
            //topfile >> dihedral_nohydro[i].ip;
            //topfile >> dihedral_nohydro[i].jp;
            //topfile >> dihedral_nohydro[i].kp;
            //topfile >> dihedral_nohydro[i].lp;
            //topfile >> dihedral_nohydro[i].icp;
        }
        getline(topfile, dummy_line);


        // EXCLUDED_ATOMS_LIST, FORMAT(10I8)
        // NATEX
        /* FORMAT(12I6)  (NATEX(i), i=1,NEXT)
         * NATEX: the excluded atom list.  To get the excluded list for atom 
         *        "i" you need to traverse the NUMEX list, adding up all
         *        the previous NUMEX values, since NUMEX(i) holds the number
         *        of excluded atoms for atom "i", not the index into the 
         *        NATEX list.  Let IEXCL = SUM(NUMEX(j), j=1,i-1), then
         *        excluded atoms are NATEX(IEXCL) to NATEX(IEXCL+NUMEX(i)).
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        int natex;
        //int natex[nums[10]];

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[10]; i++){
            topfile >> natex;
            //topfile >> natex[i];
        }
        getline(topfile, dummy_line);


        // HBOND_ACOEF; FORMAT(5E16.8)
        // ASOL
        /* FORMAT(5E16.8)  (ASOL(i), i=1,NPHB)
         * ASOL: the value for the r**12 term for hydrogen bonds of all
         *       possible types.  Index into these arrays is equivalent
         *       to the CN1 and CN2 arrays, however the index is negative.
         *       For example, for atoms i and j, with i < j, the index is
         *       -ICO(NTYPES*(IAC(i)-1+IAC(j)).
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        double asol;
        //double asol[nums[19]];

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[19]; i++){
            topfile >> asol;
            //topfile >> asol[i];
        }
        getline(topfile, dummy_line);


        // HBOND_BCOEF; FORMAT(5E16.8)
        // BSOL
        /* FORMAT(5E16.8)  (BSOL(i), i=1,NPHB)
         * BSOL: the value for the r**10 term for hydrogen bonds of all
         *       possible types.  Indexed like ASOL.
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        double bsol;
        //double bsol[nums[19]];

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[19]; i++){
            topfile >> bsol;
            //topfile >> bsol[i];
        }
        getline(topfile, dummy_line);


        // HBCUT; FORMAT(5E16.8)
        // HBCUT
        /* FORMAT(5E16.8)  (HBCUT(i), i=1,NPHB)
         * HBCUT  : no longer in use
		// Both, protein and solvent, because here we don't know yet which is which
         */
        // ???

        double hbcut;
        //double hbcut[nums[19]];

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        for(int i=0; i<nums[19]; i++){
            topfile >> hbcut;
            //topfile >> hbcut[i];
        }
        getline(topfile, dummy_line);


        // AMBER_ATOM_TYPE; FORMAT(20a4)
        // ISYMBL
        /* FORMAT(20A4)  (ISYMBL(i), i=1,NATOM)
         * ISYMBL : the AMBER atom types for each atom
		// Both, protein and solvent, because here we don't know yet which is which
         */
        // we use ATOM_NAME instead!
		// TODO: stimmt das noch so?
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        char tmp_type[5];
        for(unsigned int i=0; i<m_numAtoms; i++){
            topfile.width(5);
            topfile >> tmp_type;
        }
        getline(topfile, dummy_line);


        // TREE_CHAIN_CLASSIFICATION; FORMAT(20a4)
        // ITREE
        /* FORMAT(20A4)  (ITREE(i), i=1,NATOM)
         * ITREE: the list of tree joining information, classified into five
         *        types.  M -- main chain, S -- side chain, B -- branch point, 
         *        3 -- branch into three chains, E -- end of the chain
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // TODO: Fuer Connectivity sinnvoll?
        // ???
        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        char tmp_itree[5];
        for(unsigned int i=0; i<m_numAtoms; i++){
            topfile.width(5);
            topfile >> tmp_itree;
            //atoms[i].itree=std::string(tmp_itree);
        }
        getline(topfile, dummy_line);


        // JOIN_ARRAY; FORMAT(10I8)
        // JOIN
        /* FORMAT(12I6)  (JOIN(i), i=1,NATOM)
         * JOIN: tree joining information, potentially used in ancient
         *       analysis programs.  Currently unused in sander or gibbs.
         */
		// Both, protein and solvent, because here we don't know yet which is which
        // ???

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        //int tmp_join[m_numAtoms];
        int tmp_join;
        for(unsigned int i=0; i<m_numAtoms; i++){
            topfile >> tmp_join;
            //topfile >> tmp_join[i];
        }
        getline(topfile, dummy_line);


        // IROTAT; FORMAT(10I8)
        // IROTAT
        /* FORMAT(12I6)  (IROTAT(i), i = 1, NATOM)
         * IROTAT: apparently the last atom that would move if atom i was
         *         rotated, however the meaning has been lost over time.
         *         Currently unused in sander or gibbs.
		// Both, protein and solvent, because here we don't know yet which is which
         */

        if(newparm){
            getline(topfile, dummy_line);
            getline(topfile, dummy_line);
        }
        int tmp_irotat;
        for(unsigned int i=0; i<m_numAtoms; i++){
            topfile >> tmp_irotat;
        }
        getline(topfile, dummy_line);

        /**************************************************/
        /* The following are only present if IFBOX .gt. 0 */
        /**************************************************/

		//std::cout<<"ifbox="<<ifbox<<std::endl;
        if(ifbox>0){
			//std::cout<<"NetCDFData: ifbox present"<<std::endl;
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
					"NetCDFData: ifbox present");

            //SOLVENT_POINTERS; FORMAT(3I8)
            // IPTRES, NSPM, NSPSOL; 
            /* FORMAT(12I6)  IPTRES, NSPM, NSPSOL
             * IPTRES : final residue that is considered part of the solute,
             *          reset in sander and gibbs
             * NSPM   : total number of molecules
             * NSPSOL : the first solvent "molecule"
             */

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            topfile >> m_finSoluteRes;
            topfile >> m_numMols;
            topfile >> m_firstSolventMol;
			/*
            topfile >> iptres;
            topfile >> nspm;
            topfile >> nspsol;
			*/
            getline(topfile, dummy_line);
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
					"NetCDFData: final residue in solute: %d, total number of molecules: %d, first solvent molecule %d", m_finSoluteRes, m_numMols, m_firstSolventMol);


            // ATOMS_PER_MOLECULE; FORMAT(10I8)
            // NSP
            /* FORMAT(12I6)  (NSP(i), i=1,NSPM)
             * NSP: the total number of atoms in each molecule, necessary
             *      to correctly perform the pressure scaling.
             */
			// molecule corresponds to chain for protein molecules

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
			m_numProteinAtoms = 0;
            //nsp= new int[nspm];
            m_numAtomsPerMol= new int[m_numMols];
            for(int i=0; i<m_numMols; i++){
                topfile >> m_numAtomsPerMol[i];
            }
            for(int i=0; i<m_firstSolventMol-1; i++)
                m_numProteinAtoms += m_numAtomsPerMol[i];

			m_numSolventAtoms = m_numAtoms-m_numProteinAtoms;

            getline(topfile, dummy_line);
            initChains();

			/**
             * Split atom and amino acid vectors into protein and solvent part
			 * Free radicals (e.g. Na+) are part of the solute!
			 * at least until we find out more about it...
			 */

			/* Position of first solvent atom in table of all atoms in dataset */
			std::vector<protein::CallProteinData::AtomData>::iterator solventPos;
			//solventPos = m_atomTable.begin()+(m_numProteinAtoms-1);
			solventPos = m_atomTable.begin()+(m_numProteinAtoms);

			/* Reserve memory for all protein atoms and copy them from m_atomTable */
			m_proteinAtomTable.reserve(m_numProteinAtoms);
			m_proteinAtomTable.insert(m_proteinAtomTable.begin(), m_atomTable.begin(), solventPos);

			/* Reserve memory for all solvent atoms and copy them from m_atomTable */
			m_solventAtomTable.reserve(m_numSolventAtoms);
			m_solventAtomTable.insert(m_solventAtomTable.begin(), solventPos, m_atomTable.end());

			/* make m_atomTable empty */
			m_atomTable.clear();

			/**
             * Set solvent-specific data
			 */
            initSolvent();

            // BOX_DIMENSIONS
            // BETA, BOX(1), BOX(2), BOX(3)
            /* FORMAT(5E16.8)  BETA, BOX(1), BOX(2), BOX(3)
             * BETA: periodic box, angle between the XY and YZ planes in
             *       degrees.
             * BOX : the periodic box lengths in the X, Y, and Z directions
             */
            // currently MegaMol only supports cubic bounding box,
			// angle will be ignored

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            topfile >> m_box_beta;
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
					"box angle = %f", m_box_beta);
            for(int i=0; i<3; i++){
                topfile >> m_boundingBox[i];
            }
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
					"box dimensions [%f, %f, %f]", m_boundingBox[0], m_boundingBox[1],
					m_boundingBox[2]);

            getline(topfile, dummy_line);
        } //if(ifbox>0)
		else{
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
					"no ifbox (ifbox<=0)");
		}

        /**************************************************/
        /* The following are only present if IFCAP .gt. 0 */
        /**************************************************/

        if(ifcap>0){
            // 
            // NATCAP
            /* FORMAT(12I6)  NATCAP
             * NATCAP: last atom before the start of the cap of waters
             *         placed by edit
             */
            int natcap=0;

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            topfile >> natcap;
            getline(topfile, dummy_line);

        } //if(ifcap>0)

        /***************************************************/
        /* The following are only present if IFPERT .gt. 0 */
        /***************************************************/

        if(ifpert>0){
			/*
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
					"ifpert present (ifpert>0)");
					*/

            // ???
            // IBPER, JBPER
            /* FORMAT(12I6)  (IBPER(i), JBPER(i), i=1,NBPER)
             * IBPER: atoms involved in perturbed bonds
             * JBPER: atoms involved in perturbed bonds
             */
            // brauchen wir hier nicht...

            int nbper=nums[21];
            //int ibper[nbper];
            //int jbper[nbper];
            int ibper;
            int jbper;

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            for(int i=0; i<nbper; i++){
                topfile >> ibper;
                topfile >> jbper;
                //topfile >> ibper[i];
                //topfile >> jbper[i];
            }
            getline(topfile, dummy_line);


            // ???
            // ICBPER
            /* FORMAT(12I6)  (ICBPER(i), i=1,2*NBPER)
             * ICBPER : pointer into the bond parameter arrays RK and REQ for the
             *          perturbed bonds.  ICBPER(i) represents lambda=1 and 
             *          ICBPER(i+NBPER) represents lambda=0.
             */
            // brauchen wir hier nicht...

            //int nbper=nums[21];
            //int icbper[2*nbper];
            int icbper;

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            for(int i=0; i<2*nbper; i++){
                topfile >> icbper;
                //	 topfile >> icbper[i];
            }
            getline(topfile, dummy_line);


            // ???
            // IPTER, JTPER, KTPER
            /* FORMAT(12I6)  (ITPER(i), JTPER(i), KTPER(i), i=1,NGPER)
             * IPTER  : atoms involved in perturbed angles
             * JTPER  : atoms involved in perturbed angles
             * KTPER  : atoms involved in perturbed angles
             */
            // brauchen wir hier nicht...

            int ngper=nums[22];
            int ipter;
            int jpter;
            int kpter;
            //int ipter[ngper];
            //int jpter[ngper];
            //int kpter[ngper];

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            for(int i=0; i<ngper; i++){
                topfile >> ipter;
                topfile >> jpter;
                topfile >> kpter;
                //topfile >> ipter[i];
                //topfile >> jpter[i];
                //topfile >> kpter[i];
            }
            getline(topfile, dummy_line);


            // ???
            // ICTPER
            /* FORMAT(12I6)  (ICBTER(i), i=1,2*NGPER)
             * ICTPER : pointer into the angle parameter arrays TK and TEQ for 
             * the perturbed angles.  ICTPER(i) represents lambda=0 and 
             * ICTPER(i+NGPER) represents lambda=1.
             */
            // brauchen wir hier nicht...

            //int ngper=nums[22];
            //int ictper[2*ngper];
            int ictper;

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            for(int i=0; i<2*ngper; i++){
                topfile >> ictper;
                //topfile >> ictper[i];
            }
            getline(topfile, dummy_line);


            // ???
            // IPPER, JPPER, KPPER, LPPER
            /* FORMAT(12I6)  (IPPER(i), JPPER(i), KPPER(i), LPPER(i), i=1,NDPER)
             * IPPER  : atoms involved in perturbed dihedrals
             * JPPER  : atoms involved in perturbed dihedrals
             * KPPER  : atoms involved in perturbed dihedrals
             * LPPER  : atoms involved in pertrubed dihedrals
             */
            // brauchen wir hier nicht...

            int ndper=nums[23];
            int ipper;
            int jpper;
            int kpper;
            int lpper;
            //int ipper[ngper];
            //int jpper[ngper];
            //int kpper[ngper];
            //int lpper[ngper];

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            for(int i=0; i<ndper; i++){
                topfile >> ipper;
                topfile >> jpper;
                topfile >> kpper;
                topfile >> lpper;
                //topfile >> ipper[i];
                //topfile >> jpper[i];
                //topfile >> kpper[i];
                //topfile >> lpper[i];
            }
            getline(topfile, dummy_line);


            // ???
            // ICPPER
            /* FORMAT(12I6)  (ICPPER(i), i=1,2*NDPER)
             * ICPPER : pointer into the dihedral parameter arrays PK, PN and
             *          PHASE for the perturbed dihedrals.  ICPPER(i) represents 
             *          lambda=1 and ICPPER(i+NGPER) represents lambda=0.
             */
            // brauchen wir hier nicht...

            //int ndper=nums[23];
            //int icpper[2*ndper];
            int icpper;

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            for(int i=0; i<2*ndper; i++){
                topfile >> icpper;
                //topfile >> icpper[i];
            }
            getline(topfile, dummy_line);

            // ???
            // LABRES
            /* FORMAT(20A4)  (LABRES(i), i=1,NRES)
             * LABRES : the residue labels
             */

            //std::string labres0[nums[11]];
            std::string labres0;
            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            //char tmp_label[5];
            for(int i=0; i<m_numres; i++){
                topfile.width(5);
                topfile >> tmp_label;
                labres0=std::string(tmp_label);
                //labres0[i]=std::string(tmp_label);
            }
            getline(topfile, dummy_line);

            // ???
            // IGRPER
            /* FORMAT(20A4)  (IGRPER(i), i=1,NATOM)
             * IGRPER : atomic names at lambda=0
             */
            // brauchen wir hier nicht...

            //std::string atomnames0[m_numAtoms];
            std::string atomnames0;
            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            char tmp_name[5];
            for(unsigned int i=0; i<m_numAtoms; i++){
                topfile.width(5);
                topfile >> tmp_name;
                atomnames0=std::string(tmp_name);
                //atomnames0[i]=std::string(tmp_name);
            }
            getline(topfile, dummy_line);


            // ???
            // ISMPER
            /* FORMAT(20A4)  (ISMPER(i), i=1,NATOM)
             * ISMPER : atomic symbols at lambda=0
             */
            // brauchen wir hier nicht...

            //std::string atomsyms0[m_numAtoms];
            std::string atomsyms0;
            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            //char tmp_name[5];
            for(unsigned int i=0; i<m_numAtoms; i++){
                topfile.width(5);
                topfile >> tmp_name;
                atomsyms0=std::string(tmp_name);
                //	 atomsyms0[i]=std::string(tmp_name);
            }
            getline(topfile, dummy_line);

            // ???
            // ALMPER
            /* FORMAT(5E16.8)  (ALMPER(i), i=1,NATOM)
             * ALMPER : unused currently in gibb
             */
            // brauchen wir hier nicht...

            //double almper[m_numAtoms];
            double almper;

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            for(unsigned int i=0; i<m_numAtoms; i++){
                topfile >> almper;
                //topfile >> almper[i];
            }
            getline(topfile, dummy_line);


            // ???
            // IAPER
            /* FORMAT(12I6)  (IAPER(i), i=1,NATOM)
             * IAPER: IAPER(i)=1 if the atom is being perturbed
             */
            // brauchen wir hier nicht...

            //int iaper[m_numAtoms];
            int iaper;

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            for(unsigned int i=0; i<m_numAtoms; i++){
                topfile >> iaper;
                //topfile >> iaper[i];
            }
            getline(topfile, dummy_line);

            // ???
            // IACPER
            /* FORMAT(12I6)  (IACPER(i), i=1,NATOM)
             * IACPER: index for the atom types involved in Lennard Jones
             *         interactions at lambda=0.  Similar to IAC above.
             *	 See ICO above.
             */
            // brauchen wir hier nicht...

            int iacper;
            //int iacper[m_numAtoms];

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            for(unsigned int i=0; i<m_numAtoms; i++){
                topfile >> iacper;
                //topfile >> iacper[i];
            }
            getline(topfile, dummy_line);


            // ???
            // CGPER
            /* FORMAT(5E16.8)  (CGPER(i), i=1,NATOM)
             * CGPER  : atomic charges at lambda=0
             */
            // brauchen wir hier nicht...

            double cgper0;
            //double cgper0[m_numAtoms];

            if(newparm){
                getline(topfile, dummy_line);
                getline(topfile, dummy_line);
            }
            for(unsigned int i=0; i<m_numAtoms; i++){
                topfile >> cgper0;
                //topfile >> cgper0[i];
            }
            getline(topfile, dummy_line);

        } //if(ifpert>0)

		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
				"NetCDFData: Finished reading topology file");
        topfile.close();

        return true;
    }
}

/*
 * protein::NetCDFData::readHeaderFromNetCDF
 */
bool protein::NetCDFData::readHeaderFromNetCDF()
{
    // check if file is still open
    if (!m_netcdfOpen) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "protein::NetCDFData::readHeaderFromNetCDF: failed to read frame (file not open) \n");
        return false;
    }

    /* 
     * check global attributes
     *
     * required attributes:
     *   Conventions
     *   ConventionVersion
     *   program
     *   programVersion
     */

    // Conventions
    vislib::StringA conventions;
    if (!netCDFGetAttrString(NC_GLOBAL, "Conventions", conventions)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "protein::NetCDFData::readHeaderFromNetCDF Conventions attribute not found! \n");
        return false;
    }
    if (!conventions.Equals("AMBER")) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "protein::NetCDFData::readHeaderFromNetCDF: Conventions attribute is not AMBER as it should be but %s!\n", conventions.PeekBuffer());
        return false;
    }

    // ConventionVersion
    vislib::StringA conventionVersion;
    if (!netCDFGetAttrString(NC_GLOBAL, "ConventionVersion", conventionVersion)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
        "ConventionVersion attribute not found!\n");
        return false;
    }
    if (!conventionVersion.Equals("1.0")) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
        "ConventionVersion attribute is not 1.0 as it should be but %s!\n", conventionVersion.PeekBuffer());
        return false;
    }

    // title
    vislib::StringA title;
    if (netCDFGetAttrString(NC_GLOBAL, "title", title)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Dataset: %s\n", title.PeekBuffer());
    }

    // program
    vislib::StringA program;
    if (!netCDFGetAttrString(NC_GLOBAL, "program", program)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
        "program attribute not found!\n");
    } else {
        // programVersion
        vislib::StringA programVersion;
        if (!netCDFGetAttrString(NC_GLOBAL, "programVersion", programVersion)) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
            "programVersion attribute not found!\n");
        }
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "File created by %s (Version %s)\n", program.PeekBuffer(), programVersion.PeekBuffer());
    }

    // application
    vislib::StringA application;
    if (netCDFGetAttrString(NC_GLOBAL, "application", application)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Part of %s\n", application.PeekBuffer());
    }

    /*
     * required dimensions:
     *   frame(length unlimited)                    - data from different time steps
     *   spatial(length 3; X,Y,Z)                   - spatial dimensions
     *   atom                                       - indices of particles
     *
     * optional dimensions:
     *   cell_spatial(length 3, a,b,c)              - size of the unit cell
     *   cell_angular(length 3, alpha,beta,gamma)   - shape of the unit cell
     *   label                                      - longest string of label variables
     */
    // frame
    if ((nc_inq_dimid(m_netcdfFileID, "frame", &m_frameDimID) != NC_NOERR) ||
            (nc_inq_dimlen(m_netcdfFileID, m_frameDimID, &m_framesize) != NC_NOERR)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
        "Dimension 'frame' is not valid!\n");
        return false;
    }
    // search for unlimited dimension id
    if (nc_inq_unlimdim(m_netcdfFileID, &m_unlimID) != NC_NOERR) {
        m_unlimID = m_frameDimID - 1;
    }
    if (m_unlimID != m_frameDimID) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
        "Dimension 'frame' is not unlimited but should be!\n");
    }
    m_numFrames = static_cast<unsigned int>(m_framesize);
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
    "Dimension 'frame' has size %u\n", m_numFrames);

    // spatial
    if ((nc_inq_dimid(m_netcdfFileID, "spatial", &m_spatialDimID) != NC_NOERR) ||
            (nc_inq_dimlen(m_netcdfFileID, m_spatialDimID, &m_spatialsize) != NC_NOERR)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
        "Dimension 'spatial' is not valid!\n");
        return false;
    }
    if (m_spatialsize != 3) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
        "Dimension 'spatial' has not length 3!\n");
        return false;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
    "Dimension 'spatial' has size %d\n", m_spatialsize);

    // atom
    if ((nc_inq_dimid(m_netcdfFileID, "atom", &m_atomDimID) != NC_NOERR) ||
            (nc_inq_dimlen(m_netcdfFileID, m_atomDimID, &m_atomsize) != NC_NOERR)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
        "Dimension 'atom' is not valid!\n");
        return false;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
    "Dimension 'atom' has size %d\n", m_atomsize);

    if (m_numAtoms != m_atomsize) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
        "Number of atoms changed!\n");
        return false;
    }

    // cell_spatial
    if ((nc_inq_dimid(m_netcdfFileID, "cell_spatial", &m_cellspatialDimID) != NC_NOERR) ||
            (nc_inq_dimlen(m_netcdfFileID, m_cellspatialDimID, &m_cell_spatial_size) != NC_NOERR)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
        "No dimension 'cell_spatial' found.\n");
        return false;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
    "Dimension 'cell_spatial' has size %d\n", m_cell_spatial_size);

    // cell_angular
    if ((nc_inq_dimid(m_netcdfFileID, "cell_angular", &this->m_cellangularDimID) != NC_NOERR) ||
            (nc_inq_dimlen(m_netcdfFileID, m_cellangularDimID, &this->m_cell_angular_size) != NC_NOERR)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
        "No dimension 'cell_angular' found.\n");
        return false;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
    "Dimension 'cell_angular' has size %d\n", this->m_cell_angular_size);

    // label
    if ((nc_inq_dimid(this->m_netcdfFileID, "label", &m_labelDimID) != NC_NOERR) ||
            (nc_inq_dimlen(this->m_netcdfFileID, this->m_labelDimID, &m_label_size) != NC_NOERR)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
        "No dimension 'label' found.\n");
        return false;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
    "Dimension 'label' has size %d\n", m_label_size);

    /* 
     * variables:
     * Label variables (to ignore by read)
     * Data values (possibly m_scale_factor attribute!) all optional!!
     *   - time (frame; units=picosecond)                   - time ellapsed since 1st frame
     *   - coordinates (frame,atom,spatial; units=angstrom) - cartesian coordinates
     *   - cell_length (frame,cell_spatial; units=angstrom) - if periodic boundaries
     *   - cell_angles (frame,cell_angles; units=angstrom)  - if cell_length set
     */
#if 0
	// TODO test it!
    //std::cout<<"Checking Data Variables!"<<std::endl;
    //time[frame]
    int timeVarID;
    NcVar* var_time;
    //float* time;
	if (nc_inq_varid(m_netcdfFileID, "time", &timeVarID) == NC_NOERR) {
		vislib::StringA timeUnit;
		if (netCDFGetAttrString(timeVarID, "units", timeUnit)) {
			if (!timeUnit.Equals("angstrom", false)) {
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
						"time:units= %s, but should be 'picosecond'!",
						timeUnit.PeekBuffer());
			}
		}
		size_t num_vals;
        if (nc_inq_attlen(m_netcdfFileID, timeVarID, "m_framesize", &num_vals) == NC_NOERR) {
			if (m_framesize != num_vals && m_framesize>0) {
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
						"m_framesize=%i != num_vals=%i", m_framesize, num_vals);
				//std::cerr<<"m_framesize="<<m_framesize<<" != num_vals="<<num_vals<<std::endl;
			}
		} else {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
					"Error reading number of frames from 'time'");
		}
        time = new float[m_framesize];
		size_t starts = 0;
		if (nc_get_vara_float(m_netcdfFileID, timeVarID, &starts, m_framesize, time) != NC_NOERR) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Error reading 'time'");
		} 

		float m_scale_factor = 1.0;
		if (nc_get_att_float(m_netcdfFileID, timeVarID, "m_scale_factor", &m_scale_factor) != NC_NOERR) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "Variable 'time' has m_scale_factor of %f", m_scale_factor);
            for (int i=0; i<num_vals; i++) {
                time[i] *= m_scale_factor;
            }
        }
    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
        "No variable 'time' found.");
        //std::cout<<"No variable 'time' found. "<<std::endl;
    }
#endif

    // coords[frame][atom][spatial]
    // coords[(frame + atom * m_framesize) * m_atomsize + spatial]
    //int m_coordsVarID;
    if (nc_inq_varid(m_netcdfFileID, "coordinates", &m_coordsVarID) != NC_NOERR) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "No variable 'coordinates' found!\n");
        return false;
    }
    // check unit (warn and ignore on missmatch)
    vislib::StringA coordsUnit;
    if (netCDFGetAttrString(m_coordsVarID, "units", coordsUnit)) {
        if (!coordsUnit.Equals("angstrom", false)) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                    "Coordinate units are not in 'angstrom', but %s! No conversion applied!\n",
                    coordsUnit.PeekBuffer());
        }
    }
    // check dimensions
    int numDimensions;
    if (nc_inq_varndims(m_netcdfFileID, m_coordsVarID, &numDimensions) != NC_NOERR) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to retrieve dimension count of variable 'coordinates'!\n");
        return false;
    }
    if (numDimensions != 3) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Variable 'coordinates' has %d dimensions (3 expected)!\n", numDimensions);
        return false;
    }
    int dims[3];
    if (nc_inq_vardimid(m_netcdfFileID, m_coordsVarID, dims) != NC_NOERR) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to retreive dimension types of variable 'coordinates'!\n");
        return false;
    }
    if ((dims[0] != m_frameDimID) || (dims[1] != m_atomDimID) || (dims[2] != m_spatialDimID)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Dimensions of variable 'coordinates' are of wrong types.\n");
        return false;
    }
    // check variable type == float
    nc_type coordsType;
    if (nc_inq_vartype(m_netcdfFileID, m_coordsVarID, &coordsType) != NC_NOERR) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to retreive type of variable 'coordinates'!\n");
        return false;
    }
    if (coordsType != NC_FLOAT) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                "Type of variable 'coordinates' is %d (NC_FLOAT expected)!\n",
                int(coordsType));
        // no 'return false;' because the netcdf-lib will convert the values.
    }

    return true;
}


/*
 * protein::NetCDFData::loadFrame
 */
void protein::NetCDFData::loadFrame(view::AnimDataModule::Frame* frame,
		unsigned int frameIDx)
{
	Frame *netcdfFrame = dynamic_cast<Frame*>(frame);
    if (!netcdfFrame->readFrameFromNetCDF(frameIDx)) {
        nc_close(this->m_netcdfFileID);
        this->m_netcdfOpen = false;
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "NetCDFData: Error reading frame %u from netCDF file. \n", frameIDx);
    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "NetCDFData: Finished reading frame %u from netCDF file. \n", frameIDx);
    }
	return;
}


/*
 * protein::NetCDFData::readFrameFromNetCDF
 *
 * TODO!!! 
 * time, cell_length, cell_angles and velocities read only frame-specific values!!!
 */
bool protein::NetCDFData::Frame::readFrameFromNetCDF(unsigned int frameIdx)
{
	bool coords_scaled = false;
	size_t framesize = parent->m_framesize;
	size_t atomsize = parent->m_atomsize;
	size_t spatialsize = parent->m_spatialsize;
	int frameID = parent->m_frameID;
	int netcdfFileID = parent->m_netcdfFileID;
	int coordsVarID = parent->m_coordsVarID;

    // check if file is still open
    if (!parent->m_netcdfOpen) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "loadFrame: failed to read frame (file not open) \n");
        return false;
    } else {
		// check requested frame number
		if (frameIdx >= framesize) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
					"loadFrame: Illegal frame number (%d) requested.\n", frameIdx);
			return false;
		} else {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
					"loadFrame: Reading frame %d \n", frameIdx);
		}
    }

//    /* 
//     * variables:
//     * Label variables (to ignore by read)
//     * Data values (possibly scale_factor attribute!) all optional!!
//     *   - time (frame; units=picosecond)                   - time ellapsed since 1st frame
//     *   - coordinates (frame,atom,spatial; units=angstrom) - cartesian coordinates
//     *   - cell_length (frame,cell_spatial; units=angstrom) - if periodic boundaries
//     *   - cell_angles (frame,cell_angles; units=angstrom)  - if cell_length set
//     */
//	//TODO: Nur einmal ueberpruefen!
//#if 0
//    //std::cout<<"Checking Data Variables!"<<std::endl;
//    //time[frame]
//    NcVar* var_time;
//    float* time;
//    if ((var_time=dataFile.get_var("time")) != NULL) {
//        if (strcmp(var_time->get_att(0)->as_string(0), "picosecond")!=0) {
//            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
//            "time:units= %s, but should be 'picosecond'!",
//            var_time->get_att(0)->as_string(0));
//            /*
//            std::cerr<<"time:units='"<<var_time->get_att(0)->as_string(0)
//                <<"', but should be 'picosecond'!"<<std::endl;
//            */
//        }
//        long num_vals=var_time->num_vals();
//        if (framesize != num_vals && framesize>0) {
//            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
//            "framesize=%i != num_vals=%i", framesize, num_vals);
//            //std::cerr<<"framesize="<<framesize<<" != num_vals="<<num_vals<<std::endl;
//        }
//        time = new float[framesize];
//        if (!(var_time->get(time, framesize))) {
//            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
//            "Error reading 'time': datatype (%s) does not match",
//            var_time->type());
//            //std::cerr<<var_time->type()<<" Error reading 'time': datatype does not match"<<std::endl;
//        }
//        if (var_time->get_att("scale_factor") != NULL) {
//            float scale_factor=var_time->get_att("scale_factor")->as_float(0);
//            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
//            "Variable 'time' has scale_factor of %f", scale_factor);
//            //std::cout<<"Variable 'time' has scale_factor of "<< scale_factor<<std::endl;
//            for (int i=0; i<num_vals; i++) {
//                time[i] *= scale_factor;
//            }
//        }
//    } else {
//        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
//        "No variable 'time' found.");
//        //std::cout<<"No variable 'time' found. "<<std::endl;
//    }
//#endif
//
//    // coords[frame][atom][spatial]
//    // coords[(frame + atom * framesize) * atomsize + spatial]
//    int coordsVarID;
//    if (nc_inq_varid(netcdfFileID, "coordinates", &coordsVarID) != NC_NOERR) {
//        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
//                "No variable 'coordinates' found!\n");
//        return false;
//    }
//    // check unit (warn and ignore on missmatch)
//    vislib::StringA coordsUnit;
//    if (netCDFGetAttrString(coordsVarID, "units", coordsUnit)) {
//        if (!coordsUnit.Equals("angstrom", false)) {
//            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
//                    "Coordinate units are not in 'angstrom', but %s! No conversion applied!\n",
//                    coordsUnit.PeekBuffer());
//        }
//    }
//    // check dimensions
//    int numDimensions;
//    if (nc_inq_varndims(netcdfFileID, coordsVarID, &numDimensions) != NC_NOERR) {
//        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
//                "Unable to retreive dimension count of variable 'coordinates'!\n");
//        return false;
//    }
//    if (numDimensions != 3) {
//        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
//                "Variable 'coordinates' has %d dimensions (3 expected)!\n", numDimensions);
//        return false;
//    }
//    int dims[3];
//    if (nc_inq_vardimid(netcdfFileID, moordsVarID, dims) != NC_NOERR) {
//        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
//                "Unable to retreive dimension types of variable 'coordinates'!\n");
//        return false;
//    }
//    if ((dims[0] != frameDimID) || (dims[1] != atomDimID) || (dims[2] != spatialDimID)) {
//        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
//                "Dimensions of variable 'coordinates' are of wrong types.\n");
//        return false;
//    }
//    // check variable type == float
//    nc_type coordsType;
//    if (nc_inq_vartype(netcdfFileID, coordsVarID, &coordsType) != NC_NOERR) {
//        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
//                "Unable to retreive type of variable 'coordinates'!\n");
//        return false;
//    }
//    if (coordsType != NC_FLOAT) {
//        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
//                "Type of variable 'coordinates' is %d (NC_FLOAT expected)!\n",
//                int(coordsType));
//        // no 'return false;' because the netcdf-lib will convert the values.
//    }
    // finally load the requested frame
    if (coords == NULL) {
        coords = new float[atomsize * spatialsize];
    }
    size_t starts[3] = {frameIdx, 0, 0};
    size_t counts[3] = {1, atomsize, spatialsize};
    frameID = 0;
    if (nc_get_vara_float(netcdfFileID, coordsVarID, starts, counts, coords) != NC_NOERR) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
        "Unable to load variable 'coordinate' for the requested frame!\n");
        return false;
    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "loadFrame: Loaded variable 'coordinate' for the requested frame %d.\n", frameIdx);
    }

    frameID = frameIdx;
	this->frame = frameIdx;

	/*
	//TEST
	float minx=coords[0];
	float maxx=coords[0];
	float miny=coords[1];
	float maxy=coords[1];
	float minz=coords[2];
	float maxz=coords[2];
	for(int atom=0; atom<atomsize; atom++) {
		if (coords[3*atom]<minx)
			minx=coords[3*atom];
		else if (coords[3*atom]>minx)
			maxx=coords[3*atom];
		if (coords[3*atom+1]<miny)
			miny=coords[3*atom+1];
		else if (coords[3*atom+1]>miny)
			maxy=coords[3*atom+1];
		if (coords[3*atom+2]<minz)
			minz=coords[3*atom+2];
		else if (coords[3*atom+2]>minz)
			maxz=coords[3*atom+2];
	}
	std::cout<<minx<<"<=x<="<<maxx<<", "<<miny<<"<=y<="<<maxy<<", "<<minz<<"<=z<="<<maxz<<std::endl;
	//TEST
	*/

	//TODO C interface!
	// optional scaling
	int scaleID;
	if (nc_inq_attid(netcdfFileID, coordsVarID, "scale_factor", &scaleID) == NC_NOERR) {
		// scale factor present
		float scale;
		if (nc_get_att_float(netcdfFileID, coordsVarID, "scale_factor", &scale) == NC_NOERR) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
					"scale factor of %f applied to variable 'coordinate'.\n", scale);
			size_t cnt = atomsize * spatialsize;
			for (size_t i = 0; i < cnt; i++) {
				coords[i] *= scale;
			}
			coords_scaled = true;
		}
	} else {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
				"No scale in netcdf file");
	}

	// check for disulfide bonds

	unsigned int num_sulfides = parent->m_sulfides.size();
	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
			"NetCDFData: Checking disulfide bonds of %i sulfide atoms ...",
			num_sulfides);
	for (unsigned int i = 0; i < num_sulfides-1; i++) {
		unsigned int sulfidIdx1 = 3*parent->m_sulfides[i];
		vislib::math::Point<float, 3> sulfCoords1(coords[sulfidIdx1],
				coords[sulfidIdx1+1], coords[sulfidIdx1+2]);
		for (unsigned int j = i+1; j < num_sulfides; j++) {
			//std::cout<<"i="<<i<<", j="<<j<<std::endl;
			unsigned int sulfidIdx2 = 3*parent->m_sulfides[j];
			vislib::math::Point<float, 3> sulfCoords2(coords[sulfidIdx2],
					coords[sulfidIdx2+1], coords[sulfidIdx2+2]);
			if ((sulfCoords2-sulfCoords1).Length() < 3.0f) {
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
						"NetCDFData: Found disulfide bond between %i and %i ",
						i, j);
				protein::CallProteinData::IndexPair disulfidBond(i,j);
				disulfidBondsVec.push_back(disulfidBond);
			}
		}
	}
	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
			"NetCDFData: Found %i disulfide bonds.",
			disulfidBondsVec.size());


#if 0
    //cell_lengths[frame][cell_spatial]
    NcVar* var_cell_lengths;
    //double* cell_lengths;
    if((var_cell_lengths=dataFile.get_var("cell_lengths")) != NULL){
        if(strcmp(var_cell_lengths->get_att(0)->as_string(0), "angstrom")!=0){
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "cell_lengths:units=' %s ', but should be 'angstrom'! \n");
                    /*
            std::cerr<<"cell_lengths:units='"<<var_cell_lengths->get_att(0)->as_string(0)
                <<"', but should be 'angstrom'!"<<std::endl;
                */
        }
        long num_vals=var_cell_lengths->num_vals();
        if(framesize*parent->m_cell_spatial_size != num_vals && framesize>0){
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "frame(%i)*cell_spatial(%i) = %i != num_vals= %i \n",
                    framesize, parent->m_cell_spatial_size, framesize*parent->m_cell_spatial_size, num_vals);
            /*
            std::cerr<<"frame("<<framesize<<")*cell_spatial("
                <<parent->m_cell_spatial_size<<"="<<framesize*parent->m_cell_spatial_size
                <<" != num_vals="<<num_vals<<std::endl;
                */
            if(parent->m_cell_spatial_size<0)
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "'cell_spatial' not set in datafile but needed for 'cell_lengths' \n");
        }
        cell_lengths = new double[num_vals];
        if(!(var_cell_lengths->get(cell_lengths, framesize, parent->m_cell_spatial_size))) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    " Error reading 'cell_lengths': datatype does not match");
            //std::cerr<<var_cell_lengths->type()<<" Error reading 'cell_lengths': datatype does not match"<<std::endl;
        }
        if(var_cell_lengths->get_att("scale_factor") != NULL){
            float scale_factor=var_cell_lengths->get_att("scale_factor")->as_float(0);
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                    "Variable 'cell_lengths' has scale_factor of %f \n",scale_factor);
            /*
            std::cout<<"Variable 'cell_lengths' has scale_factor of "
                << scale_factor<<std::endl;
                */
            for(int i=0; i<num_vals; i++){
                cell_lengths[i] *= scale_factor;
            }
        }
    }else
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
        "No variable 'cell_lengths' found. \n");
        //std::cout<<"No variable 'cell_lengths' found. "<<std::endl;
#endif
#if 0
    //cell_angles[frame][cell_angular]
    NcVar* var_cell_angles;
    //double* cell_angles;
    if((var_cell_angles=dataFile.get_var("cell_angles")) != NULL){
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Variable 'cell_angles' found. \n");
        //std::cout<<"Variable 'cell_angles' found. "<<std::endl;
        if(strcmp(var_cell_angles->get_att(0)->as_string(0), "degree")!=0) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                    "cell_angles:units='%s', but should be 'degree'!",
                    var_cell_angles->get_att(0)->as_string(0));
            /*
               std::cerr<<"cell_angles:units='"<<var_cell_angles->get_att(0)->as_string(0)
               <<"', but should be 'degree'!"<<std::endl;
             */
        }
        long num_vals=var_cell_angles->num_vals();
        if(framesize*parent->m_cell_angular_size != num_vals && framesize>0) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "frame(%i)*cell_angular(%i) = %i != num_vals=%i", framesize,
                    parent->m_cell_angular_size, framesize*parent->m_cell_angular_size, num_vals);
            /*
            std::cerr<<"frame("<<m_framesize<<")*cell_angular("
                <<parent->m_cell_angular_size<<"="<<framesize*parent->m_cell_angular_size
                <<" != num_vals="<<num_vals<<std::endl;
                */
            if(parent->m_cell_angular_size<0) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "'cell_angular' not set in datafile but needed for 'cell_angles'");
                //std::cerr<<"'cell_angular' not set in datafile but needed for 'cell_angles'"<<std::endl;
            }
        }
        cell_angles = new double[num_vals];
        var_cell_angles->get(cell_angles, framesize, parent->m_cell_angular_size);
        if(var_cell_angles->get_att("m_scale_factor") != NULL) {
            float m_scale_factor=var_cell_angles->get_att("m_scale_factor")->as_float(0);
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "Variable 'cell_angles' has m_scale_factor of %f", m_scale_factor);
            /*
            std::cout<<"Variable 'cell_angles' has scale_factor of "
                << scale_factor<<std::endl;
                */
            for(int i=0; i<num_vals; i++){
                cell_angles[i] *= m_scale_factor;
            }
        }
    }else{
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "No variable 'cell_angles' found.");
        //std::cout<<"No variable 'cell_angles' found. "<<std::endl;
    }
#endif
#if 0
    //velocities[frame][atom][spatial]
    NcVar* var_velocities;
    //float* velocities;
    if((var_velocities=dataFile.get_var("velocities")) != NULL) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Variable 'velocities' found.");
        //std::cout<<"Variable 'velocities' found. "<<std::endl;
        if(strcmp(var_velocities->get_att(0)->as_string(0), "angstrom/picosecond")!=0) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                    "velocities:units='%s', but should be 'angstrom/picosecond'!",
                    var_velocities->get_att(0)->as_string(0) );
            /*
            std::cout<<"velocities:units='"<<var_velocities->get_att(0)->as_string(0)
                <<"', but should be 'angstrom/picosecond'!"<<std::endl;
            */
        }
        long num_vals=var_velocities->num_vals();
        if(framesize*atomsize*spatialsize != num_vals && 
                (framesize>0 && atomsize>0 && spatialsize>0)){
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "frame(%i)*atom(%i)*spatial(%i) = %i != num_vals=%i",
                    framesize, atomsize, spatialsize, framesize*atomsize*spatialsize,
                    num_vals);
            /*
            std::cerr<<"frame("<<framesize<<")*atom("<<atomsize<<")*spatial("
                <<spatialsize<<"="<<framesize*atomsize*spatialsize
                <<" != num_vals="<<num_vals<<std::endl;
                */
        }
        velocities = new float[num_vals];
        var_velocities->get(cell_angles, framesize, atomsize, spatialsize);
        if(var_velocities->get_att("scale_factor") != NULL) {
            float scale_factor=var_velocities->get_att("scale_factor")->as_float(0);
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                    "Variable 'velocities' has scale_factor of %f", scale_factor);
            /*
            std::cout<<"Variable 'velocities' has scale_factor of "
                << scale_factor<<std::endl;
                */
            for(int i=0; i<num_vals; i++){
                velocities[i] *= scale_factor;
            }
        }
    }else{
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "No variable 'velocities' found.");
        //std::cout<<"No variable 'velocities' found. "<<std::endl;
    }
#endif

    return true;
}


/*
 * protein::NetCDFData::Frame::MakeInterpolationFrame
 */
const protein::NetCDFData::Frame* protein::NetCDFData::Frame::MakeInterpolationFrame(float alpha,
		const Frame &a, const Frame &b)
{
	size_t numCoords = parent->m_numAtoms * parent->m_spatialsize;
	if (alpha < 0.0000001f) return &a;
	if (alpha > 0.9999999f) return &b;
	float beta = 1.0f - alpha;
	// TODO Aufteilung Protein - Solvent davor oder danach???
	for (unsigned int idx=0; idx<numCoords; idx++) {
		coords[idx] = alpha*b.coords[idx]+beta*a.coords[idx];
	}

	return this;
}



/*
 * protein::NetCDFData::constructFrame
 */
view::AnimDataModule::Frame* protein::NetCDFData::constructFrame(void) const {
	view::AnimDataModule::Frame *frame = new Frame(this);
	return frame;
}


/*
 * protein::NetCDFData::netCDFGetAttrString
 */
bool protein::NetCDFData::netCDFGetAttrString(int varid,
		const char *name, vislib::StringA& outStr)
{
    size_t length;
    if (nc_inq_attlen(this->m_netcdfFileID, varid, name, &length) != NC_NOERR) {
        return false;
    }
    if (nc_get_att_text(this->m_netcdfFileID, varid, name, outStr.AllocateBuffer(int(length + 1))) != NC_NOERR) {
        outStr.Clear();
        return false;
    }
    outStr.Truncate(int(length));
    return true;
}


/*
 * protein::NetCDFData::makeConnections
 *
 * read the atom/amino acid table and build the connectivity table
 * TODO: Kein HETATM und TER in NetCDF. Tut es trotzdem?!
 */
bool protein::NetCDFData::makeConnections(void)
{
    bool retval = true;
	/*
    bool missingAtom = false;
    int index;
    Atom amino;
    Atom carboxyl;
    Atom tmp01;
    Backbone bb;
    std::vector<Atom> aa;
    bool aaComplete;
    bool aaContainsH = false;
    unsigned int backboneIndex = 0;

    // TODO: wird nie auf was anderes als "EMPTY" gesetzt. Wozu also?!
    // carboxyl.atomName = "EMPTY";

    // clear old connectivity table
    this->connectivityTable.clear();
    this->connectivityTable.resize(this->m_atomTable.size());
    // clear old backbone table
    this->backboneTable.clear();
    this->backboneTable.resize(1);

    // iterate over all atoms in the table
    //TODO i nur bis Ende Protein!!!
    for (unsigned int i = 0; i < m_numProteinAtoms; i++ ) {
        // if a TER atom is reached, we don't add it to the current amino acid
        if (this->m_atomTable[i].atomName.Compare("TER") ) {
            // the amino acid is not read completely
            aaComplete = false;
            // a new chain starts: the carboxyl term is empty
            carboxyl.atomName = "EMPTY";
            // proceed to the next row of the backbone table
            this->backboneTable.resize( this->backboneTable.size()+1);
            backboneIndex++;
        }
        // if a HETATM is reached, we don't add it to the current amino acid
        else if (this->m_atomTable[i].atomName.Compare("HETATM") ) {
            // the amino acid is not read completely
            aaComplete = false;
        } else {
            // add the atom to the current amino acid
            aa.push_back( this->m_atomTable[i]);
            if (this->m_atomTable[i].atomName.StartsWith("H") )
                aaContainsH = true;
            // check, if the vector contains exactly all atoms of one amino acid
            // --> true, if the last atom is reached
            if (i == (this->m_atomTable.size()-1) )
                aaComplete = true;
            // --> true, if last atom is not reached and next atom belongs to different amino acid or a 'TER' is following
            else {
                if( this->m_atomTable[i].aminoAcidId != this->m_atomTable[i+1].aminoAcidId ||
                        this->m_atomTable[i+1].atomName.Compare("TER") )
                    aaComplete = true;
                // --> false for all other cases
                else
                    aaComplete = false;
            }
        }

        if (aaComplete) {
            // check error: no valid amino acid
			index = this->m_atomTable[i].aminoAcidId;
            if (index > -1) {
                if (this->m_aminoAcidTable[index].type == 'Z' ) {
                    this->connectivityTable.clear();
                    return false;
                }
            } else{
                return false;
			}

			//freie Radikale weglassen
			if (aa.size() > 1) {

				/////////////////////////////////////////////////////
				// write the CA, C and O atom to the backbone list //
				/////////////////////////////////////////////////////
				int CA_ID = findAtomByName("CA", aa);
				int C_ID = findAtomByName("C", aa);
				int O_ID = findAtomByName("O", aa);


				if (CA_ID != -1 && C_ID != -1 && O_ID != -1 ) {
					bb.CA = aa[CA_ID].index;
					bb.C = aa[C_ID].index;
					bb.O = aa[O_ID].index;
					this->backboneTable[backboneIndex].push_back( bb);
				} else {
					return false;
				}

				//////////////////////////////
				// make backbone connection //
				//////////////////////////////
				// find and store the amino terminus
				int N_ID = findAtomByName( "N", aa);
				if(N_ID != -1)
					amino = aa[N_ID];
				else{
					return false;
				}

				// if the amino acid is not the first (carboxyl not EMPTY )
				// --> connect it to its predecessors carboxyl end
				if( !carboxyl.atomName.CompareInsensitive( "EMPTY") ) {
					this->connectivityTable[carboxyl.index].id.Add( amino.index);
				}
				// find and store the carboxyl terminus of the current amino acid
				if(C_ID != -1)
					carboxyl = aa[C_ID];
				else {
					return false;
				}

				//////////////////////////////////////////////////////////////////////////////////////
				// make connections for C alpha, C, O and N (these are the same in all amino acids) //
				//////////////////////////////////////////////////////////////////////////////////////
				// connect CA with N
				if (CA_ID != -1) {
					if (aa[CA_ID].id > aa[N_ID].id)
						this->connectivityTable[aa[N_ID].index].id.Add( aa[CA_ID].index);
					else
						this->connectivityTable[aa[CA_ID].index].id.Add( aa[N_ID].index);
				} else{
					return false;
				}
				// connect CA with C
				if( aa[CA_ID].id > aa[findAtomByName( "C", aa)].id )
					this->connectivityTable[aa[findAtomByName( "C", aa)].index].id.Add( aa[findAtomByName( "CA", aa)].index);
				else
					this->connectivityTable[aa[CA_ID].index].id.Add( aa[findAtomByName( "C", aa)].index);
				// connect O with C
				if (O_ID != -1) {
					if (aa[O_ID].id > aa[C_ID].id)
						this->connectivityTable[aa[C_ID].index].id.Add(aa[O_ID].index);
					else
						this->connectivityTable[aa[O_ID].index].id.Add(aa[C_ID].index);
				} else{
					return false;
				}

				/////////////////////////////////////////////////////////////////////////////////
				// make connection for C and (if available) OXT (only for the last amino acid) //
				/////////////////////////////////////////////////////////////////////////////////
				// connect C with OXT
				int OXT_ID = findAtomByName("OXT", aa);
				//if( findAtomByName( "C", aa) != -1 && findAtomByName( "OXT", aa) != -1 )
				if(OXT_ID != -1 )
					if( aa[C_ID].id > aa[OXT_ID].id )
						this->connectivityTable[aa[OXT_ID].index].id.Add(aa[C_ID].index);
					else
						this->connectivityTable[aa[C_ID].index].id.Add(aa[OXT_ID].index);

				//////////////////////////////
				// make connections for ALA //
				//////////////////////////////
				//index = this->getm_aminoAcidTableIndex(this->m_atomTable[i].aminoAcidId, this->m_atomTable[i].chainId);
				index = this->m_atomTable[i].aminoAcidId;
				if (index > -1) {
					// TODO: Weiter Gruppieren bzw. schachtel. Einige Abfragen sinnlos mehrfach neu!!!
					int CB_ID = findAtomByName( "CB", aa);
					if (this->m_aminoAcidTable[index].type == 'A') {
						// connect CA with CB
						//int CB_ID = findAtomByName( "CB", aa);
						if(CA_ID != -1 && CB_ID != -1 )
							if( aa[CA_ID].id > aa[CB_ID].id )
								this->connectivityTable[aa[CB_ID].index].id.Add( aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add( aa[CB_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for ARG //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'R') {
						// connect CA with CB
						//int CB_ID = findAtomByName( "CB", aa);
						//if (CA_ID != -1 && CB_ID != -1 )
						if (CB_ID != -1 )
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName("CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CD with CG
						int CD_ID = findAtomByName("CD", aa);
						if (CD_ID != -1 && CG_ID != -1 )
							if (aa[CD_ID].id > aa[CG_ID].id )
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD_ID].index);
							else
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;

						// connect CD with NE
						int NE_ID = findAtomByName("NE", aa);
						if (CD_ID != -1 && NE_ID != -1)
							if (aa[CD_ID].id > aa[NE_ID].id)
								this->connectivityTable[aa[NE_ID].index].id.Add(aa[CD_ID].index);
							else
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[NE_ID].index);
						else missingAtom = true;

						// connect CZ with NE
						int CZ_ID = findAtomByName("CZ", aa);
						if (CZ_ID != -1 && NE_ID != -1)
							if (aa[CZ_ID].id > aa[NE_ID].id)
								this->connectivityTable[aa[NE_ID].index].id.Add(aa[CZ_ID].index);
							else
								this->connectivityTable[aa[CZ_ID].index].id.Add(aa[NE_ID].index);
						else missingAtom = true;

						// connect CZ with NH1
						int NH1_ID = findAtomByName("NH1", aa);
						if (CZ_ID != -1 && NH1_ID != -1 )
							if (aa[CZ_ID].id > aa[NH1_ID].id )
								this->connectivityTable[aa[NH1_ID].index].id.Add(aa[CZ_ID].index);
							else
								this->connectivityTable[aa[CZ_ID].index].id.Add(aa[NH1_ID].index);
						else missingAtom = true;

						// connect CZ with NH2
						int NH2_ID = findAtomByName("NH2", aa);
						if (CZ_ID != -1 && NH2_ID != -1)
							if (aa[CZ_ID].id > aa[NH2_ID].id)
								this->connectivityTable[aa[NH2_ID].index].id.Add(aa[CZ_ID].index);
							else
								this->connectivityTable[aa[CZ_ID].index].id.Add(aa[NH2_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for ASN //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'N') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName("CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect OD1 with CG
						int OD1_ID = findAtomByName("OD1", aa);
						if (OD1_ID != -1 && CG_ID != -1)
							if (aa[OD1_ID].id > aa[CG_ID].id)
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[OD1_ID].index);
							else
								this->connectivityTable[aa[OD1_ID].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;

						// connect ND2 with CG
						int ND2 = findAtomByName("ND2", aa);
						if (ND2 != -1 && CG_ID != -1)
							if (aa[ND2].id > aa[CG_ID].id)
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[ND2].index);
							else
								this->connectivityTable[aa[ND2].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for ASP //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'D') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName("CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect OD1 with CG
						int OD1_ID = findAtomByName( "OD1", aa);
						if (OD1_ID != -1 && CG_ID != -1)
							if (aa[OD1_ID].id > aa[CG_ID].id)
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[OD1_ID].index);
							else
								this->connectivityTable[aa[OD1_ID].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;

						// connect OD2 with CG
						int OD2_ID = findAtomByName( "OD2", aa);
						if (OD2_ID != -1 && CG_ID != -1)
							if (aa[OD2_ID].id > aa[CG_ID].id)
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[OD2_ID].index);
							else
								this->connectivityTable[aa[OD2_ID].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for CYS //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'C') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1 )
							if (aa[CA_ID].id > aa[CB_ID].id )
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect SG with CB
						int SG_ID = findAtomByName( "SG", aa);
						if (SG_ID != -1 && CB_ID != -1)
							if (aa[SG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[SG_ID].index);
							else
								this->connectivityTable[aa[SG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for GLU //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'E') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName( "CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CD with CG
						int CD_ID = findAtomByName( "CD", aa);
						if (CD_ID != -1 && CG_ID != -1)
							if (aa[CD_ID].id > aa[CG_ID].id)
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD_ID].index);
							else
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;

						// connect OE1 with CD
						int OE1_ID = findAtomByName( "OE1", aa);
						if (OE1_ID != -1 && CD_ID != -1)
							if (aa[OE1_ID].id > aa[CD_ID].id)
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[OE1_ID].index);
							else
								this->connectivityTable[aa[OE1_ID].index].id.Add(aa[CD_ID].index);
						else missingAtom = true;

						// connect OE2 with CD
						int OE2_ID = findAtomByName( "OE2", aa);
						if (OE2_ID != -1 && CD_ID != -1)
							if (aa[OE2_ID].id > aa[CD_ID].id)
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[OE2_ID].index);
							else
								this->connectivityTable[aa[OE2_ID].index].id.Add(aa[CD_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for GLN //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'Q') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName( "CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CD with CG
						int CD_ID = findAtomByName( "CD", aa);
						if (CD_ID != -1 && CG_ID != -1)
							if (aa[CD_ID].id > aa[CG_ID].id)
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD_ID].index);
							else
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;

						// connect OE1 with CD
						int OE1_ID = findAtomByName( "OE1", aa);
						if (OE1_ID != -1 && CD_ID != -1)
							if (aa[OE1_ID].id > aa[CD_ID].id)
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[OE1_ID].index);
							else
								this->connectivityTable[aa[OE1_ID].index].id.Add(aa[CD_ID].index);
						else missingAtom = true;

						// connect NE2 with CD
						int NE2_ID = findAtomByName( "NE2", aa);
						if (NE2_ID != -1 && CD_ID != -1)
							if (aa[NE2_ID].id > aa[CD_ID].id)
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[NE2_ID].index);
							else
								this->connectivityTable[aa[NE2_ID].index].id.Add(aa[CD_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for HIS //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'H') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName( "CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect ND1 with CG
						int ND1_ID = findAtomByName( "ND1", aa);
						if (ND1_ID != -1 && CG_ID != -1)
							if (aa[ND1_ID].id > aa[CG_ID].id)
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[ND1_ID].index);
							else
								this->connectivityTable[aa[ND1_ID].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;

						// connect CD2 with CG
						int CD2_ID = findAtomByName( "CD2", aa);
						if (CD2_ID != -1 && CG_ID != -1)
							if (aa[CD2_ID].id > aa[CG_ID].id)
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD2_ID].index);
							else
								this->connectivityTable[aa[CD2_ID].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;

						// connect ND1 with CE1
						int CE1_ID = findAtomByName( "CE1", aa);
						if (ND1_ID != -1 && CE1_ID != -1)
							if (aa[ND1_ID].id > aa[CE1_ID].id)
								this->connectivityTable[aa[CE1_ID].index].id.Add(aa[ND1_ID].index);
							else
								this->connectivityTable[aa[ND1_ID].index].id.Add(aa[CE1_ID].index);
						else missingAtom = true;

						// connect CD2 with NE2
						int NE2_ID = findAtomByName( "NE2", aa);
						if (CD2_ID != -1 && NE2_ID != -1)
							if (aa[CD2_ID].id > aa[NE2_ID].id)
								this->connectivityTable[aa[NE2_ID].index].id.Add(aa[CD2_ID].index);
							else
								this->connectivityTable[aa[CD2_ID].index].id.Add(aa[NE2_ID].index);
						else missingAtom = true;

						// connect CE1 with NE2
						if (CE1_ID != -1 && NE2_ID != -1)
							if (aa[CE1_ID].id > aa[NE2_ID].id)
								this->connectivityTable[aa[NE2_ID].index].id.Add(aa[CE1_ID].index);
							else
								this->connectivityTable[aa[CE1_ID].index].id.Add(aa[NE2_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for ILE //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'I') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG1 with CB
						int CG1_ID = findAtomByName( "CG1", aa);
						if (CG1_ID != -1 && CB_ID != -1)
							if (aa[CG1_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG1_ID].index);
							else
								this->connectivityTable[aa[CG1_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG2 with CB
						int CG2_ID = findAtomByName( "CG2", aa);
						if (CG2_ID != -1 && CB_ID != -1)
							if (aa[CG2_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG2_ID].index);
							else
								this->connectivityTable[aa[CG2_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG1 with CD1
						int CD1_ID = findAtomByName( "CD1", aa);
						if (CG1_ID != -1 && CD1_ID != -1)
							if (aa[CG1_ID].id > aa[CD1_ID].id)
								this->connectivityTable[aa[CD1_ID].index].id.Add(aa[CG1_ID].index);
							else
								this->connectivityTable[aa[CG1_ID].index].id.Add(aa[CD1_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for LEU //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'L') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName( "CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CD1
						int CD1_ID = findAtomByName( "CD1", aa);
						if (CG_ID != -1 && CD1_ID != -1)
							if (aa[CG_ID].id > aa[CD1_ID].id)
								this->connectivityTable[aa[CD1_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD1_ID].index);
						else missingAtom = true;

						// connect CG with CD2
						int CD2_ID = findAtomByName( "CD2", aa);
						if (CG_ID != -1 && CD2_ID != -1)
							if (aa[CG_ID].id > aa[CD2_ID].id)
								this->connectivityTable[aa[CD2_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD2_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for LYS //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'K') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName( "CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CD with CG
						int CD_ID = findAtomByName( "CD", aa);
						if (CD_ID != -1 && CG_ID != -1)
							if (aa[CD_ID].id > aa[CG_ID].id)
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD_ID].index);
							else
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;

						// connect CD with CE
						int CE_ID = findAtomByName( "CE", aa);
						if (CD_ID != -1 && CE_ID != -1)
							if (aa[CD_ID].id > aa[CE_ID].id)
								this->connectivityTable[aa[CE_ID].index].id.Add(aa[CD_ID].index);
							else
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[CE_ID].index);
						else missingAtom = true;

						// connect NZ with CE
						int NZ_ID = findAtomByName( "NZ", aa); 
						if (NZ_ID != -1 && CE_ID != -1)
							if (aa[NZ_ID].id > aa[CE_ID].id)
								this->connectivityTable[aa[CE_ID].index].id.Add(aa[NZ_ID].index);
							else
								this->connectivityTable[aa[NZ_ID].index].id.Add(aa[CE_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for MET //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'M') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName( "CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect SD with CG
						int SD_ID = findAtomByName( "SD", aa);
						if (SD_ID != -1 && CG_ID != -1)
							if (aa[SD_ID].id > aa[CG_ID].id)
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[SD_ID].index);
							else
								this->connectivityTable[aa[SD_ID].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;

						// connect SD with CE
						int CE_ID = findAtomByName( "CE", aa);
						if (SD_ID != -1 && CE_ID != -1)
							if (aa[SD_ID].id > aa[CE_ID].id)
								this->connectivityTable[aa[CE_ID].index].id.Add(aa[SD_ID].index);
							else
								this->connectivityTable[aa[SD_ID].index].id.Add(aa[CE_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for PHE //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'F') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName( "CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CD1
						int CD1_ID = findAtomByName( "CD1", aa);
						if (CG_ID != -1 && CD1_ID != -1)
							if (aa[CG_ID].id > aa[CD1_ID].id)
								this->connectivityTable[aa[CD1_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD1_ID].index);
						else missingAtom = true;

						// connect CG with CD2
						int CD2_ID = findAtomByName( "CD2", aa);
						if (CG_ID != -1 && CD2_ID != -1)
							if (aa[CG_ID].id > aa[CD2_ID].id)
								this->connectivityTable[aa[CD2_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD2_ID].index);
						else missingAtom = true;

						// connect CE1 with CD1
						int CE1_ID = findAtomByName( "CE1", aa);
						if (CE1_ID != -1 && CD1_ID != -1)
							if (aa[CE1_ID].id > aa[CD1_ID].id)
								this->connectivityTable[aa[CD1_ID].index].id.Add(aa[CE1_ID].index);
							else
								this->connectivityTable[aa[CE1_ID].index].id.Add(aa[CD1_ID].index);
						else missingAtom = true;

						// connect CE2 with CD2
						int CE2_ID = findAtomByName( "CE2", aa);
						if (CE2_ID != -1 && CD2_ID != -1)
							if (aa[CE2_ID].id > aa[CD2_ID].id)
								this->connectivityTable[aa[CD2_ID].index].id.Add(aa[CE2_ID].index);
							else
								this->connectivityTable[aa[CE2_ID].index].id.Add(aa[CD2_ID].index);
						else missingAtom = true;

						// connect CE1 with CZ
						int CZ_ID = findAtomByName( "CZ", aa);
						if (CE1_ID != -1 && CZ_ID != -1)
							if (aa[CE1_ID].id > aa[CZ_ID].id)
								this->connectivityTable[aa[CZ_ID].index].id.Add(aa[CE1_ID].index);
							else
								this->connectivityTable[aa[CE1_ID].index].id.Add(aa[CZ_ID].index);
						else missingAtom = true;

						// connect CE2 with CZ
						if (CE2_ID != -1 && CZ_ID != -1)
							if (aa[CE2_ID].id > aa[CZ_ID].id)
								this->connectivityTable[aa[CZ_ID].index].id.Add(aa[CE2_ID].index);
							else
								this->connectivityTable[aa[CE2_ID].index].id.Add(aa[CZ_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for PRO //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'P') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName( "CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CD with CG
						int CD_ID = findAtomByName( "CD", aa);
						if (CD_ID != -1 && CG_ID != -1)
							if (aa[CD_ID].id > aa[CG_ID].id)
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD_ID].index);
							else
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[CG_ID].index);
						else missingAtom = true;

						// connect CD with N
						if (CD_ID != -1 && N_ID != -1)
							if (aa[CD_ID].id > aa[N_ID].id)
								this->connectivityTable[aa[N_ID].index].id.Add(aa[CD_ID].index);
							else
								this->connectivityTable[aa[CD_ID].index].id.Add(aa[N_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for SER //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'S') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect OG with CB
						int OG_ID = findAtomByName( "OG", aa);
						if (OG_ID != -1 && CB_ID != -1)
							if (aa[OG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[OG_ID].index);
							else
								this->connectivityTable[aa[OG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for THR //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'T') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect OG1 with CB
						int OG1_ID = findAtomByName( "OG1", aa);
						if (OG1_ID != -1 && CB_ID != -1)
							if (aa[OG1_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[OG1_ID].index);
							else
								this->connectivityTable[aa[OG1_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG2 with CB
						int CG2_ID = findAtomByName( "CG2", aa);
						if (CG2_ID != -1 && CB_ID != -1)
							if (aa[CG2_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG2_ID].index);
							else
								this->connectivityTable[aa[CG2_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for TRP //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'W') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName( "CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CD1
						int CD1_ID = findAtomByName( "CD1", aa);
						if (CG_ID != -1 && CD1_ID != -1)
							if (aa[CG_ID].id > aa[CD1_ID].id)
								this->connectivityTable[aa[CD1_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD1_ID].index);
						else missingAtom = true;

						// connect CG with CD2
						int CD2_ID = findAtomByName( "CD2", aa);
						if (CG_ID != -1 && CD2_ID != -1)
							if (aa[CG_ID].id > aa[CD2_ID].id)
								this->connectivityTable[aa[CD2_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD2_ID].index);
						else missingAtom = true;

						// connect NE1 with CD1
						int NE1_ID = findAtomByName( "NE1", aa);
						if (NE1_ID != -1 && CD1_ID != -1)
							if (aa[NE1_ID].id > aa[CD1_ID].id)
								this->connectivityTable[aa[CD1_ID].index].id.Add(aa[NE1_ID].index);
							else
								this->connectivityTable[aa[NE1_ID].index].id.Add(aa[CD1_ID].index);
						else missingAtom = true;

						// connect CE2 with CD2
						int CE2_ID = findAtomByName( "CE2", aa);
						if (CE2_ID != -1 && CD2_ID != -1)
							if (aa[CE2_ID].id > aa[CD2_ID].id)
								this->connectivityTable[aa[CD2_ID].index].id.Add(aa[CE2_ID].index);
							else
								this->connectivityTable[aa[CE2_ID].index].id.Add(aa[CD2_ID].index);
						else missingAtom = true;

						// connect CE3 with CD2
						int CE3_ID = findAtomByName( "CE3", aa);
						if (CE3_ID != -1 && CD2_ID != -1)
							if (aa[CE3_ID].id > aa[CD2_ID].id)
								this->connectivityTable[aa[CD2_ID].index].id.Add(aa[CE3_ID].index);
							else
								this->connectivityTable[aa[CE3_ID].index].id.Add(aa[CD2_ID].index);
						else missingAtom = true;

						// connect CE2 with NE1
						if (CE2_ID != -1 && NE1_ID != -1)
							if (aa[CE2_ID].id > aa[NE1_ID].id)
								this->connectivityTable[aa[NE1_ID].index].id.Add(aa[CE2_ID].index);
							else
								this->connectivityTable[aa[CE2_ID].index].id.Add(aa[NE1_ID].index);
						else missingAtom = true;

						// connect CE2 with CZ2
						int CZ2_ID = findAtomByName( "CZ2", aa);
						if (CE2_ID != -1 && CZ2_ID != -1)
							if (aa[CE2_ID].id > aa[CZ2_ID].id)
								this->connectivityTable[aa[CZ2_ID].index].id.Add(aa[CE2_ID].index);
							else
								this->connectivityTable[aa[CE2_ID].index].id.Add(aa[CZ2_ID].index);
						else missingAtom = true;

						// connect CE3 with CZ3
						int CZ3_ID = findAtomByName( "CZ3", aa);
						if (CE3_ID != -1 && CZ3_ID != -1)
							if (aa[CE3_ID].id > aa[CZ3_ID].id)
								this->connectivityTable[aa[CZ3_ID].index].id.Add(aa[CE3_ID].index);
							else
								this->connectivityTable[aa[CE3_ID].index].id.Add(aa[CZ3_ID].index);
						else missingAtom = true;

						// connect CH2 with CZ3
						int CH2_ID = findAtomByName( "CH2", aa);
						if (CH2_ID != -1 && CZ3_ID != -1)
							if (aa[CH2_ID].id > aa[CZ3_ID].id)
								this->connectivityTable[aa[CZ3_ID].index].id.Add(aa[CH2_ID].index);
							else
								this->connectivityTable[aa[CH2_ID].index].id.Add(aa[CZ3_ID].index);
						else missingAtom = true;

						// connect CH2 with CZ2
						if (CH2_ID != -1 && CZ2_ID != -1)
							if (aa[CH2_ID].id > aa[CZ2_ID].id)
								this->connectivityTable[aa[CZ2_ID].index].id.Add(aa[CH2_ID].index);
							else
								this->connectivityTable[aa[CH2_ID].index].id.Add(aa[CZ2_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for TYR //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'Y') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CB
						int CG_ID = findAtomByName( "CG", aa);
						if (CG_ID != -1 && CB_ID != -1)
							if (aa[CG_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;

						// connect CG with CD1
						int CD1_ID = findAtomByName( "CD1", aa);
						if (CG_ID != -1 && CD1_ID != -1)
							if (aa[CG_ID].id > aa[CD1_ID].id)
								this->connectivityTable[aa[CD1_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD1_ID].index);
						else missingAtom = true;

						// connect CG with CD2
						int CD2_ID = findAtomByName( "CD2", aa);
						if (CG_ID != -1 && CD2_ID != -1)
							if (aa[CG_ID].id > aa[CD2_ID].id)
								this->connectivityTable[aa[CD2_ID].index].id.Add(aa[CG_ID].index);
							else
								this->connectivityTable[aa[CG_ID].index].id.Add(aa[CD2_ID].index);
						else missingAtom = true;
						// connect CE1 with CD1
						int CE1_ID = findAtomByName( "CE1", aa);
						if (CE1_ID != -1 && CD1_ID != -1)
							if (aa[CE1_ID].id > aa[CD1_ID].id)
								this->connectivityTable[aa[CD1_ID].index].id.Add(aa[CE1_ID].index);
							else
								this->connectivityTable[aa[CE1_ID].index].id.Add(aa[CD1_ID].index);
						else missingAtom = true;
						// connect CE2 with CD2
						int CE2_ID = findAtomByName( "CE2", aa);
						if (CE2_ID != -1 && CD2_ID != -1)
							if (aa[CE2_ID].id > aa[CD2_ID].id)
								this->connectivityTable[aa[CD2_ID].index].id.Add(aa[CE2_ID].index);
							else
								this->connectivityTable[aa[CE2_ID].index].id.Add(aa[CD2_ID].index);
						else missingAtom = true;
						// connect CE2 with CZ
						int CZ_ID = findAtomByName( "CZ", aa);
						if (CE2_ID != -1 && CZ_ID != -1)
							if (aa[CE2_ID].id > aa[CZ_ID].id)
								this->connectivityTable[aa[CZ_ID].index].id.Add(aa[CE2_ID].index);
							else
								this->connectivityTable[aa[CE2_ID].index].id.Add(aa[CZ_ID].index);
						else missingAtom = true;
						// connect CE1 with CZ
						if (CE1_ID != -1 && CZ_ID != -1)
							if (aa[CE1_ID].id > aa[CZ_ID].id)
								this->connectivityTable[aa[CZ_ID].index].id.Add(aa[CE1_ID].index);
							else
								this->connectivityTable[aa[CE1_ID].index].id.Add(aa[CZ_ID].index);
						else missingAtom = true;
						// connect OH with CZ
						int OH_ID = findAtomByName( "OH", aa);
						if (OH_ID != -1 && CZ_ID != -1)
							if (aa[OH_ID].id > aa[CZ_ID].id)
								this->connectivityTable[aa[CZ_ID].index].id.Add(aa[OH_ID].index);
							else
								this->connectivityTable[aa[OH_ID].index].id.Add(aa[CZ_ID].index);
						else missingAtom = true;
					}

					//////////////////////////////
					// make connections for VAL //
					//////////////////////////////
					else if (this->m_aminoAcidTable[index].type == 'V') {
						// connect CA with CB
						if (CA_ID != -1 && CB_ID != -1)
							if (aa[CA_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CA_ID].index);
							else
								this->connectivityTable[aa[CA_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;
						// connect CG1 with CB
						int CG1_ID = findAtomByName( "CG1", aa);
						if (CG1_ID != -1 && CB_ID != -1)
							if (aa[CG1_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG1_ID].index);
							else
								this->connectivityTable[aa[CG1_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;
						// connect CG2 with CB
						int CG2_ID = findAtomByName( "CG2", aa);
						if (CG2_ID != -1 && CB_ID != -1)
							if (aa[CG2_ID].id > aa[CB_ID].id)
								this->connectivityTable[aa[CB_ID].index].id.Add(aa[CG2_ID].index);
							else
								this->connectivityTable[aa[CG2_ID].index].id.Add(aa[CB_ID].index);
						else missingAtom = true;
					}

					////////////////////////
					// make H connections //
					////////////////////////
					if (aaContainsH) {
						for (unsigned int i = 0; i < aa.size(); i++) {
							if (aa.at(i).atomName.StartsWith("H")) {
								vislib::StringA HName = aa.at(i).atomName;
								int HName_ID = findAtomByName( HName, aa);

								if (HName == "HA")
									if (aa[CA_ID].id > aa[HName_ID].id)
										this->connectivityTable[aa[HName_ID].index].id.Add(aa[CA_ID].index);
									else
										this->connectivityTable[aa[CA_ID].index].id.Add(aa[HName_ID].index);
								else if (HName == "H1")
									if (aa[N_ID].id > aa[HName_ID].id)
										this->connectivityTable[aa[HName_ID].index].id.Add(aa[N_ID].index);
									else
										this->connectivityTable[aa[N_ID].index].id.Add(aa[HName_ID].index);
								else if (HName == "H")
									if (aa[N_ID].id > aa[HName_ID].id )
										this->connectivityTable[aa[HName_ID].index].id.Add(aa[N_ID].index);
									else
										this->connectivityTable[aa[N_ID].index].id.Add(aa[HName_ID].index);
								else {
									bool found = false;
									for (unsigned int j = 0; j < aa.size(); j++) {

										if (j != i && aa.at(j).atomName.Substring(1) == HName.Substring(1)) {
											vislib::StringA jHName = aa.at(j).atomName;
											if (aa[findAtomByName(jHName, aa)].id > aa[HName_ID].id)
												this->connectivityTable[aa[HName_ID].index].id.Add( aa[findAtomByName( jHName, aa)].index);
											else
												this->connectivityTable[aa[findAtomByName( jHName, aa)].index].id.Add( aa[HName_ID].index);
											found = true;
										}
									}
									if (!found) {
										for (unsigned int j = 0; j < aa.size(); j++) {
											vislib::StringA jHName = aa.at(j).atomName;
											if (j != i && jHName.Substring(1) == HName.Substring(1, HName.Length()-2)) {
												if( aa[findAtomByName( jHName, aa)].id > aa[HName_ID].id )
													this->connectivityTable[aa[HName_ID].index].id.Add( aa[findAtomByName( jHName, aa)].index);
												else
													this->connectivityTable[aa[findAtomByName( jHName, aa)].index].id.Add( aa[HName_ID].index);
												found = true;
											}
										}
									}
								}
							}
						}
						aaContainsH = false;
					}
				}
				// clear the aa vector
				aa.clear();
			} else //aa.size() > 1
				break;
        }
    }

    // erase empty backbone entries
    while (this->backboneTable[this->backboneTable.size()-1].size() == 0) {
        this->backboneTable.resize(this->backboneTable.size()-1);
    }

    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
    "connectivity table successfully created...\n");

    if (missingAtom)
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
        "Some amino acids are not comlete, one or more side chain atoms are missing!\n");

    // check, if a bond is longer than 3 Angstrom
    if (this->checkBondLength) {
        //unsigned int count = 0;
        for (unsigned int i = 0; i < this->connectivityTable.size(); i++) {
            unsigned int j = 0;
            while (j < this->connectivityTable[i].id.Count() ) {
                if ((this->m_atomTable[i].position - this->m_atomTable[connectivityTable[i].id[j]].position).Length() > 3.0f) {
                    this->connectivityTable[i].id.Erase( j);
                } else {
                    j++;
                }
            }
        }
    }

    // count the number of connections
    this->numberOfConnections = 0;
    for (unsigned int i = 0; i < this->connectivityTable.size(); i++) {
        this->numberOfConnections += (int)this->connectivityTable.at(i).id.Count();
    }

*/
    return retval;
}


/*
 * protein::NetCDFData::readSecondaryStructure
 */
bool protein::NetCDFData::readSecondaryStructure(void)
{
	bool retval = true;
	/*

    vislib::sys::File *file = new vislib::sys::BufferedFile();
    // filename is .top
	vislib::StringW filename = this->GetFilename();
	vislib::StringA extension( "stride");
	// change the files extension from '.top' to '.stride'
	filename.Truncate( filename.Length()-3);
	filename += extension;

	if (file->Open (filename, vislib::sys::File::READ_ONLY,
				vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
		vislib::StringA line, str;
		SecondaryStructure sec;
		unsigned int firstId;
		unsigned int lastId;
		vislib::StringA firstChainId;
		vislib::StringA lastChainId;
		
		while (!file->IsEOF()) {
			line = vislib::sys::ReadLineFromFileA( *file);

			// LOC contains information about the start and end of the structures
			if (line.StartsWithInsensitive("LOC")) {
				// read first residue PDB number
				str = line.Substring( 22, 5);
				str.TrimSpaces();
				firstId = atoi( str.PeekBuffer());

				// read last residue PDB number
				str = line.Substring( 40, 5);
				str.TrimSpaces();
				lastId = atoi( str.PeekBuffer());

				// read first chain id
				str = line.Substring( 28, 1);
				str.TrimSpaces();
				firstChainId = str;
				// read last chain id
				str = line.Substring( 46, 1);
				str.TrimSpaces();
				lastChainId = str;

				// search for the C-alpha of amino acid 'firstId' in chain 'firstChainId'
				// and for the C-alpha of amino acid 'lastId' in chain 'lastChainId'
				bool firstSet = false;
				for (unsigned int i = 0; i < this->m_atomTable.size(); i++) {
					if (this->m_atomTable[i].aminoAcidId == firstId && 
						this->m_atomTable[i].chainId == firstChainId &&
						!firstSet) {
							sec.start = this->m_atomTable[i].index;
							sec.startAminoAcidIndex = this->m_atomTable[i].aminoAcidIndex;
							firstSet = true;
					} else if (this->m_atomTable[i].aminoAcidId == lastId && 
						this->m_atomTable[i].chainId == lastChainId) {
						sec.end = this->m_atomTable[i].index;
						sec.endAminoAcidIndex = this->m_atomTable[i].aminoAcidIndex;
					}
				}

				// read the structure type
				str = line.Substring( 5, 11);
				str.TrimSpaces();
				if (str.Find( "Helix") != vislib::StringA::INVALID_POS ||
					str.Find( "helix") != vislib::StringA::INVALID_POS) {
					sec.type = 'H';
				} else if (str.Find( "Strand") != vislib::StringA::INVALID_POS || 
					str.Find( "strand") != vislib::StringA::INVALID_POS) {
					sec.type = 'E';
				} else if (str.Find( "Turn") != vislib::StringA::INVALID_POS || 
					str.Find( "turn") != vislib::StringA::INVALID_POS) {
					sec.type = 'T';
				} else {
					sec.type = 'U'; // unknown type
				}
				// add the current secondary structure 'sec' to the secondary structure table, if it has a known type
				if( sec.type != 'U' )
					this->secondaryStructureTable.push_back(sec);
			}
		}

		// close the file
		file->Close();

		// write the index to the sec struct entries and to the atoms
		for (unsigned int i = 0; i < this->secondaryStructureTable.size(); i++) {
			this->secondaryStructureTable[i].index = i;
			for (unsigned int j = this->secondaryStructureTable[i].start; j <= this->secondaryStructureTable[i].end; j++) {
				this->m_atomTable[j].secStructIndex = i;
			}
		}
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
		"STRIDE file containing secondary structure successfully loaded...\n");
	} else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
		"STRIDE file not found, no secondary structure information available.\n");
		retval = false;
	}
*/

	return retval;
}



/*
 * protein::NetCDFData::initChains
 *
 * Set aminoAcids in each chain 
 * and set special atom indices (C alpha, etc.) per amino acid
 */
void protein::NetCDFData::initChains() {
	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
			"protein::NetCDFData::initChains called");
	m_proteinChains.reserve(m_firstSolventMol);
	protein::CallProteinData::Chain protChain;
	unsigned int chainsFirstAminoAcid = 0; 
	unsigned int aminoAcidIdx = 0;

    for (int molIdx = 0; molIdx < m_firstSolventMol; molIdx++) {
		unsigned int numChainAtoms = m_numAtomsPerMol[molIdx];
		chainsFirstAminoAcid = aminoAcidIdx; 
		while (m_aminoAcidTable[aminoAcidIdx].FirstAtomIndex() < numChainAtoms) {
			//search N, C alpha, C and O
			protein::CallProteinData::AminoAcid tmp_aminoAcid = m_aminoAcidTable[aminoAcidIdx];
			unsigned int atomCnt = tmp_aminoAcid.AtomCount();
			for (unsigned int atomIdx=0; atomIdx<atomCnt; atomIdx++) {
				unsigned int tmp_atomIdx = tmp_aminoAcid.FirstAtomIndex()+atomIdx;
				protein::CallProteinData::AtomData tmp_atom = m_atomTable[tmp_atomIdx];
				if (m_atomTypeTable[tmp_atom.TypeIndex ()].Name() == "N"){
					tmp_aminoAcid.SetNIndex(atomIdx);
				} 
				else if (m_atomTypeTable[tmp_atom.TypeIndex ()].Name() == "CA"){
					tmp_aminoAcid.SetCAlphaIndex(atomIdx);
				} 
				else if (m_atomTypeTable[tmp_atom.TypeIndex ()].Name() == "C"){
					tmp_aminoAcid.SetCCarbIndex(atomIdx);
				} 
				else if (m_atomTypeTable[tmp_atom.TypeIndex ()].Name() == "O")
					tmp_aminoAcid.SetOIndex(atomIdx);
			}
			aminoAcidIdx++;
		}
		protChain.SetAminoAcidCount(aminoAcidIdx-chainsFirstAminoAcid);
		protChain.SetAminoAcid(aminoAcidIdx-chainsFirstAminoAcid,
				               &m_aminoAcidTable[chainsFirstAminoAcid]);
		m_proteinChains.push_back(protChain);
    }
	// next amino acid is label of first solvent molecule
	// TODO: this or next???
	m_firstSolRes = ++aminoAcidIdx;
    return;
}

/*
 * protein::NetCDFData::initSolvent
 *
 * Sets solvent entries m_numSolventMolTypes, m_numSolventMolsPerTypes, m_solventData
 * TODO: Add connection!
 */
void protein::NetCDFData::initSolvent() {
	/*
	 * TODO: Sicherstellen, dass die Reihenfolge stimmt!!! Performanzproblem falls nicht!!!
	 *       Koordinaten in gleicher Reihenfolge jeweils...
	 *       evtl. gleiche Typen die nicht zusammenhaengend sind als unterschiedlich behandeln?
	 *        -> nur letzten Eintrag auf gleichen nameIdx pruefen, sonst neuer Eintrag
	 */
	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
			"protein::NetCDFData::initSolvent called");
	// search molecule types in m_aminoAcidTable and add new types to m_solventTypeVec:
	int numAminoAcids = m_aminoAcidTable.size();
	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
			"m_aminoAcidTable.size = %i", numAminoAcids);
	for (int aminoAcidIdx=m_firstSolRes; aminoAcidIdx<numAminoAcids; aminoAcidIdx++) {
		//unsigned int nameIdx = m_aminoAcidTable[aminoAcidIdx].NameIndex();
		protein::CallProteinData::AminoAcid tmp_aminoAcid = m_aminoAcidTable[aminoAcidIdx];
		unsigned int nameIdx = tmp_aminoAcid.NameIndex();
		//std::cout<<"--- tmp_aminoAcid.NameIndex = "<<nameIdx<<std::endl;
		solvent tmp_solvent;
		int vecsize = m_solventTypeVec.size();
		int vecIdx=0;
		if (!m_solventTypeVec.empty()) {
			// Search for matching molecule type
			for (vecIdx =0; vecIdx<vecsize; vecIdx++) {
				if (m_solventTypeVec[vecIdx].aminoAcidNameIdx == nameIdx) {
					++(m_solventTypeVec[vecIdx].numMolsPerType);
					break;
				}
			}
		}
		if (m_solventTypeVec.empty() || vecIdx == vecsize) {
			// Add new entry
			tmp_solvent.aminoAcidNameIdx = nameIdx;
			tmp_solvent.numMolsPerType = 1;
			tmp_solvent.numAtomsPerType = tmp_aminoAcid.AtomCount();
			m_solventTypeVec.push_back(tmp_solvent);
			//std::cout<<"protein::NetCDFData::initSolvent: Added residue with nameIdx ="<<nameIdx<<std::endl;
		}
	}

	// Add data to m_solventData vector
	m_numSolventMolTypes = m_solventTypeVec.size();
	m_solventData.reserve(m_numSolventMolTypes);
	protein::CallProteinData::SolventMoleculeData tmp_solventMol;
	int vecsize = m_solventTypeVec.size();
	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
			"protein::NetCDFData::initSolvent Found %i residue types in solvent",
			m_solventTypeVec.size());

	for (int vecIdx=0; vecIdx<vecsize; vecIdx++) {
		solvent tmp_solvent = m_solventTypeVec[vecIdx];
		//std::cout<<"protein::NetCDFData::initSolvent: aminoAcidNameIdx ="<<tmp_solvent.aminoAcidNameIdx<<std::endl;
		/*
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
				"protein::NetCDFData::initSolvent: Calling SetName ");
				*/
		tmp_solventMol.SetName(m_aminoAcidTypeTable[tmp_solvent.aminoAcidNameIdx]);
		tmp_solventMol.SetAtomCount(tmp_solvent.numAtomsPerType);
		m_solventData.push_back(tmp_solventMol);
	}
	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
			"NetCDFData: Leaving initSolvent ");

    return;
}

#endif /* (defined(WITH_NETCDF) && (WITH_NETCDF)) */
