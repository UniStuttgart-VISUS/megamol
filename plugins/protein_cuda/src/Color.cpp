/*
 * Color.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "Color.h"
#include <omp.h>
#include "mmcore/utility/ColourParser.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "mmcore/CoreInstance.h"
#include <string>
#include <iostream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_cuda;


/*
 * FillAminoAcidColorTable
 */
void Color::FillAminoAcidColorTable(
        vislib::Array<vislib::math::Vector<float, 3> > &aminoAcidColorTable) {

    aminoAcidColorTable.Clear();
    aminoAcidColorTable.SetCount( 25);
    aminoAcidColorTable[0].Set( 0.5f, 0.5f, 0.5f);
    aminoAcidColorTable[1].Set( 1.0f, 0.0f, 0.0f);
    aminoAcidColorTable[2].Set( 1.0f, 1.0f, 0.0f);
    aminoAcidColorTable[3].Set( 0.0f, 1.0f, 0.0f);
    aminoAcidColorTable[4].Set( 0.0f, 1.0f, 1.0f);
    aminoAcidColorTable[5].Set( 0.0f, 0.0f, 1.0f);
    aminoAcidColorTable[6].Set( 1.0f, 0.0f, 1.0f);
    aminoAcidColorTable[7].Set( 0.5f, 0.0f, 0.0f);
    aminoAcidColorTable[8].Set( 0.5f, 0.5f, 0.0f);
    aminoAcidColorTable[9].Set( 0.0f, 0.5f, 0.0f);
    aminoAcidColorTable[10].Set( 0.0f, 0.5f, 0.5f);
    aminoAcidColorTable[11].Set( 0.0f, 0.0f, 0.5f);
    aminoAcidColorTable[12].Set( 0.5f, 0.0f, 0.5f);
    aminoAcidColorTable[13].Set( 1.0f, 0.5f, 0.0f);
    aminoAcidColorTable[14].Set( 0.0f, 0.5f, 1.0f);
    aminoAcidColorTable[15].Set( 1.0f, 0.5f, 1.0f);
    aminoAcidColorTable[16].Set( 0.5f, 0.25f, 0.0f);
    aminoAcidColorTable[17].Set( 1.0f, 1.0f, 0.5f);
    aminoAcidColorTable[18].Set( 0.5f, 1.0f, 0.5f);
    aminoAcidColorTable[19].Set( 0.75f, 1.0f, 0.0f);
    aminoAcidColorTable[20].Set( 0.5f, 0.0f, 0.75f);
    aminoAcidColorTable[21].Set( 1.0f, 0.5f, 0.5f);
    aminoAcidColorTable[22].Set( 0.75f, 1.0f, 0.75f);
    aminoAcidColorTable[23].Set( 0.75f, 0.75f, 0.5f);
    aminoAcidColorTable[24].Set( 1.0f, 0.75f, 0.5f);
}


/*
 * Color::GetModeByIndex
 */
Color::ColoringMode Color::GetModeByIndex(const megamol::protein_calls::MolecularDataCall *mol,
    unsigned int idx) {

    switch(idx) {
        case 0 : return ELEMENT;
        case 1 : return RESIDUE;
        case 2 : return STRUCTURE;
        case 3 : return BFACTOR;
        case 4 : return CHARGE;
        case 5 : return OCCUPANCY;
        case 6 : return CHAIN;
        case 7 : return MOLECULE;
        case 8 : return RAINBOW;
        default : return ELEMENT;
    }
}

/*
 * Color::GetModeByIndex
 */
Color::ColoringMode Color::GetModeByIndex(const megamol::protein_calls::MolecularDataCall *mol,
	const protein_calls::BindingSiteCall *bs, unsigned int idx) {
    switch(idx) {
        case 0 : return ELEMENT;
        case 1 : return RESIDUE;
        case 2 : return STRUCTURE;
        case 3 : return BFACTOR;
        case 4 : return CHARGE;
        case 5 : return OCCUPANCY;
        case 6 : return CHAIN;
        case 7 : return MOLECULE;
        case 8 : return RAINBOW;
        case 9 : return BINDINGSITE;
        default : return ELEMENT;
    }
}

/*
 * Color::GetName
 */
std::string Color::GetName(Color::ColoringMode col) {

    switch(col) {
        case ELEMENT     : return "Element";
        case STRUCTURE   : return "Structure";
        case RAINBOW     : return "Rainbow";
        case BFACTOR     : return "BFactor";
        case CHARGE      : return "Charge";
        case OCCUPANCY   : return "Occupancy";
        case CHAIN       : return "Chain";
        case MOLECULE    : return "Molecule";
        case RESIDUE     : return "Residue";
        case CHAINBOW    : return "Chainbow";
        case AMINOACID   : return "Aminoacid";
        case VALUE       : return "Value";
        case CHAIN_ID    : return "ChainID";
        case MOVEMENT    : return "Movement";
        case BINDINGSITE : return "BindingSite";
        default : return "";
    }
}


/*
 * MakeColorTable routine for molecular data call
 */
void Color::MakeColorTable(const megamol::protein_calls::MolecularDataCall *mol,
        ColoringMode currentColoringMode,
        vislib::Array<float> &atomColorTable,
        vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
        vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
        vislib::TString minGradColor,
        vislib::TString midGradColor,
        vislib::TString maxGradColor,
        bool forceRecompute,
		const protein_calls::BindingSiteCall *bs) {

    // temporary variables
    unsigned int cnt, idx, cntAtom, cntRes, cntChain, cntMol, cntSecS, atomIdx,
        atomCnt;
    vislib::math::Vector<float, 3> color;
    float r, g, b;

    // if recomputation is forced: clear current color table
    if( forceRecompute ) {
        atomColorTable.Clear();
    }
    // reserve memory for all atoms
    atomColorTable.AssertCapacity( mol->AtomCount() * 3 );

	// TODO Remove
	/*if( atomColorTable.IsEmpty() ) {
		for( cnt = 0; cnt < mol->AtomCount(); ++cnt) {

			if( cnt < 200 ) {
				atomColorTable.Add(1.0);
				atomColorTable.Add(0.0);
				atomColorTable.Add(1.0);
			} else {
				atomColorTable.Add(0.0);
				atomColorTable.Add(1.0);
				atomColorTable.Add(0.0);
			}
		}
	}*/

    // only compute color table if necessary
    if( atomColorTable.IsEmpty() ) {
        if( currentColoringMode == ELEMENT ) {
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {

                atomColorTable.Add( float( mol->AtomTypes()[mol->
                    AtomTypeIndices()[cnt]].Colour()[0]) / 255.0f);

                atomColorTable.Add( float( mol->AtomTypes()[mol->
                    AtomTypeIndices()[cnt]].Colour()[1]) / 255.0f);

                atomColorTable.Add( float( mol->AtomTypes()[mol->
                    AtomTypeIndices()[cnt]].Colour()[2]) / 255.0f);
            }
        } // ... END coloring mode ELEMENT
        else if( currentColoringMode == RESIDUE ) {
            unsigned int resTypeIdx;
            // loop over all residues
            for( cntRes = 0; cntRes < mol->ResidueCount(); ++cntRes ) {
                // loop over all atoms of the current residue
                idx = mol->Residues()[cntRes]->FirstAtomIndex();
                cnt = mol->Residues()[cntRes]->AtomCount();
                // get residue type index
                resTypeIdx = mol->Residues()[cntRes]->Type();
                for( cntAtom = idx; cntAtom < idx + cnt; ++cntAtom ) {
                    // Special cases for water/toluol/methanol
                    if (mol->ResidueTypeNames()[resTypeIdx].Equals("SOL")) {
                        // Water
                        atomColorTable.Add(0.3f);
                        atomColorTable.Add(0.3f);
                        atomColorTable.Add(1.0f);
                    } else if(mol->ResidueTypeNames()[resTypeIdx].Equals("MeOH")) {
                        // methanol
                        atomColorTable.Add(1.0f);
                        atomColorTable.Add(0.5f);
                        atomColorTable.Add(0.0f);
                    }else if(mol->ResidueTypeNames()[resTypeIdx].Equals("TOL")) {
                        // toluol
                        atomColorTable.Add(1.0f);
                        atomColorTable.Add(1.0f);
                        atomColorTable.Add(1.0f);
                    }
                    else {
                    atomColorTable.Add( colorLookupTable[resTypeIdx%
                        colorLookupTable.Count()].X());
                    atomColorTable.Add( colorLookupTable[resTypeIdx%
                        colorLookupTable.Count()].Y());
                    atomColorTable.Add( colorLookupTable[resTypeIdx%
                        colorLookupTable.Count()].Z());
                    }
                }
            }
        } // ... END coloring mode RESIDUE
        else if( currentColoringMode == STRUCTURE ) {
            utility::ColourParser::FromString( "#00ff00", r, g, b);
            vislib::math::Vector<float, 3> colNone( r, g, b);
            utility::ColourParser::FromString( "#ff0000", r, g, b);
            vislib::math::Vector<float, 3> colHelix( r, g, b);
            utility::ColourParser::FromString( "#0000ff", r, g, b);
            vislib::math::Vector<float, 3> colSheet( r, g, b);
            utility::ColourParser::FromString( "#888888", r, g, b);
            vislib::math::Vector<float, 3> colRCoil( r, g, b);
            // loop over all atoms and fill the table with the default color
            for( cntAtom = 0; cntAtom < mol->AtomCount(); ++cntAtom ) {
                atomColorTable.Add( colNone.X());
                atomColorTable.Add( colNone.Y());
                atomColorTable.Add( colNone.Z());
            }
            // write colors for sec structure elements
			megamol::protein_calls::MolecularDataCall::SecStructure::ElementType elemType;

            for( cntSecS = 0; cntSecS < mol->SecondaryStructureCount();
                 ++cntSecS ) {

                idx = mol->SecondaryStructures()[cntSecS].FirstAminoAcidIndex();
                cnt = idx +
                    mol->SecondaryStructures()[cntSecS].AminoAcidCount();
                elemType = mol->SecondaryStructures()[cntSecS].Type();

                for( cntRes = idx; cntRes < cnt; ++cntRes ) {
                    atomIdx = mol->Residues()[cntRes]->FirstAtomIndex();
                    atomCnt = atomIdx + mol->Residues()[cntRes]->AtomCount();
                    for( cntAtom = atomIdx; cntAtom < atomCnt; ++cntAtom ) {
                        if( elemType ==
							megamol::protein_calls::MolecularDataCall::SecStructure::TYPE_HELIX) {
                            atomColorTable[3*cntAtom+0] = colHelix.X();
                            atomColorTable[3*cntAtom+1] = colHelix.Y();
                            atomColorTable[3*cntAtom+2] = colHelix.Z();
                        } else if( elemType ==
							megamol::protein_calls::MolecularDataCall::SecStructure::TYPE_SHEET) {
                            atomColorTable[3*cntAtom+0] = colSheet.X();
                            atomColorTable[3*cntAtom+1] = colSheet.Y();
                            atomColorTable[3*cntAtom+2] = colSheet.Z();
                        } else if( elemType ==
							megamol::protein_calls::MolecularDataCall::SecStructure::TYPE_COIL) {
                            atomColorTable[3*cntAtom+0] = colRCoil.X();
                            atomColorTable[3*cntAtom+1] = colRCoil.Y();
                            atomColorTable[3*cntAtom+2] = colRCoil.Z();
                        }
                    }
                }
            }
        } // ... END coloring mode STRUCTURE
        else if( currentColoringMode == BFACTOR ) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString(
               minGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMin( r, g, b);
            // get mid color
            utility::ColourParser::FromString(
                midGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMid( r, g, b);
            // get max color
            utility::ColourParser::FromString(
                maxGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMax( r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min_val( mol->MinimumBFactor());
            float max_val( mol->MaximumBFactor());
            float mid( ( max_val - min_val)/2.0f + min_val );
            float val;

            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min_val == max_val ) {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                    continue;
                }

                val = mol->AtomBFactors()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min_val) ) *
                        ( val - min_val );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max_val - mid) ) *
                        ( val - mid );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                }
            }
        } // ... END coloring mode BFACTOR
        else if( currentColoringMode == CHARGE ) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString(
                minGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMin( r, g, b);
            // get mid color
            utility::ColourParser::FromString(
                midGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMid( r, g, b);
            // get max color
            utility::ColourParser::FromString(
                maxGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMax( r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min_val( mol->MinimumCharge());
            float max_val( mol->MaximumCharge());
            float mid( ( max_val - min_val)/2.0f + min_val );
            float val;

            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min_val == max_val ) {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                    continue;
                }

                val = mol->AtomCharges()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min_val) ) *
                        ( val - min_val );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max_val - mid) ) *
                        ( val - mid );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                }
            }
        } // ... END coloring mode CHARGE
        else if( currentColoringMode == OCCUPANCY ) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString(
                minGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMin( r, g, b);
            // get mid color
            utility::ColourParser::FromString(
                midGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMid( r, g, b);
            // get max color
            utility::ColourParser::FromString(
                maxGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMax( r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min_val( mol->MinimumOccupancy());
            float max_val( mol->MaximumOccupancy());
            float mid( ( max_val - min_val)/2.0f + min_val );
            float val;

            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min_val == max_val ) {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                    continue;
                }

                val = mol->AtomOccupancies()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min_val) ) *
                        ( val - min_val );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max_val - mid) ) *
                        ( val - mid );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                }
            }
        } // ... END coloring mode OCCUPANCY
        else if( currentColoringMode == CHAIN ) {
            // get the last atom of the last res of the last mol of the first chain
            cntChain = 0;
            cntMol = mol->Chains()[cntChain].MoleculeCount() - 1;
            cntRes = mol->Molecules()[cntMol].FirstResidueIndex()
                + mol->Molecules()[cntMol].ResidueCount() - 1;
            cntAtom = mol->Residues()[cntRes]->FirstAtomIndex()
                + mol->Residues()[cntRes]->AtomCount() - 1;
            // get the first color
            idx = 0;
            color = colorLookupTable[idx%colorLookupTable.Count()];
            // loop over all atoms
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                // check, if the last atom of the current chain is reached
                if( cnt > cntAtom ) {
                    // get the last atom of the last res of the last mol of the next chain
                    cntChain++;
                    cntMol = mol->Chains()[cntChain].FirstMoleculeIndex()
                        + mol->Chains()[cntChain].MoleculeCount() - 1;
                    cntRes = mol->Molecules()[cntMol].FirstResidueIndex()
                        + mol->Molecules()[cntMol].ResidueCount() - 1;
                    cntAtom = mol->Residues()[cntRes]->FirstAtomIndex()
                        + mol->Residues()[cntRes]->AtomCount() - 1;
                    // get the next color
                    idx++;
                    color = colorLookupTable[idx%colorLookupTable.Count()];

                }
                atomColorTable.Add( color.X());
                atomColorTable.Add( color.Y());
                atomColorTable.Add( color.Z());
            }
        } // ... END coloring mode CHAIN
        else if( currentColoringMode == MOLECULE ) {
            // get the last atom of the last res of the first mol
            cntMol = 0;
            cntRes = mol->Molecules()[cntMol].FirstResidueIndex()
                + mol->Molecules()[cntMol].ResidueCount() - 1;
            cntAtom = mol->Residues()[cntRes]->FirstAtomIndex()
                + mol->Residues()[cntRes]->AtomCount() - 1;
            // get the first color
            idx = 0;
            color = colorLookupTable[idx%colorLookupTable.Count()];
            // loop over all atoms
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                // check, if the last atom of the current chain is reached
                if( cnt > cntAtom ) {
                    // get the last atom of the last res of the next mol
                    cntMol++;
                    cntRes = mol->Molecules()[cntMol].FirstResidueIndex()
                        + mol->Molecules()[cntMol].ResidueCount() - 1;
                    cntAtom = mol->Residues()[cntRes]->FirstAtomIndex()
                        + mol->Residues()[cntRes]->AtomCount() - 1;
                    // get the next color
                    idx++;
                    color = colorLookupTable[idx%colorLookupTable.Count()];

                }
                atomColorTable.Add( color.X());
                atomColorTable.Add( color.Y());
                atomColorTable.Add( color.Z());
            }
        } // ... END coloring mode MOLECULE
        else if( currentColoringMode == BINDINGSITE ) {
            // initialize all colors as white
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                atomColorTable.Add( 1.0f);
                atomColorTable.Add( 1.0f);
                atomColorTable.Add( 1.0f);
            }
            // search for binding sites if BindingSiteCall is available
            if( bs ) {
                // temporary variables
                unsigned int firstMol;
                unsigned int firstRes;
                unsigned int firstAtom;
                unsigned int atomIdx;
                for( unsigned int cCnt = 0; cCnt < mol->ChainCount(); cCnt++ ) {
                    firstMol = mol->Chains()[cCnt].FirstMoleculeIndex();
                    for( unsigned int mCnt = firstMol; mCnt < firstMol + mol->Chains()[cCnt].MoleculeCount(); mCnt++ ) {
                        firstRes = mol->Molecules()[mCnt].FirstResidueIndex();
                        for( unsigned int rCnt = 0; rCnt < mol->Molecules()[mCnt].ResidueCount(); rCnt++ ) {
                            // try to match binding sites
                            vislib::Pair<char, unsigned int> bsRes;
                            // loop over all binding sites
                            for( unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++ ) {
                                for( unsigned int bsResCnt = 0; bsResCnt < bs->GetBindingSite(bsCnt)->Count(); bsResCnt++ ) {
                                    bsRes = bs->GetBindingSite(bsCnt)->operator[](bsResCnt);
                                    if( mol->Chains()[cCnt].Name() == bsRes.First() &&
                                        mol->Residues()[firstRes+rCnt]->OriginalResIndex() == bsRes.Second() &&
                                        mol->ResidueTypeNames()[mol->Residues()[firstRes+rCnt]->Type()] == bs->GetBindingSiteResNames(bsCnt)->operator[](bsResCnt) ) {
                                            // TODO loop over all atoms and add the color
                                        firstAtom = mol->Residues()[firstRes+rCnt]->FirstAtomIndex();
                                        for( unsigned int aCnt = 0; aCnt < mol->Residues()[firstRes+rCnt]->AtomCount(); aCnt++ ) {
                                            atomIdx = firstAtom + aCnt;
                                            atomColorTable[3 * atomIdx + 0] = bs->GetBindingSiteColor(bsCnt).X();
                                            atomColorTable[3 * atomIdx + 1] = bs->GetBindingSiteColor(bsCnt).Y();
                                            atomColorTable[3 * atomIdx + 2] = bs->GetBindingSiteColor(bsCnt).Z();
                                        }
                                    }
                                }
                            }
                        } // residues
                    } // molecules
                } // chains
            } // BindingSiteCall available
        } // ... END coloring mode BINDINGSITE
        else if( currentColoringMode == RAINBOW ) {
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                idx = int( ( float( cnt) / float( mol->AtomCount())) *
                    float( rainbowColors.Count()));
                color = rainbowColors[idx];
                atomColorTable.Add( color.GetX());
                atomColorTable.Add( color.GetY());
                atomColorTable.Add( color.GetZ());
            }
        } // ... END coloring mode RAINBOW
    }
}


/*
 * MakeColorTable routine for molecular data call interpolating between two
 * colors
 */
void Color::MakeColorTable(const megamol::protein_calls::MolecularDataCall *mol,
        ColoringMode cm0,
        ColoringMode cm1,
        float weight0,
        float weight1,
        vislib::Array<float> &atomColorTable,
        vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
        vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
        vislib::TString minGradColor,
        vislib::TString midGradColor,
        vislib::TString maxGradColor,
        bool forceRecompute,
		const protein_calls::BindingSiteCall *bs) {

    // if recomputation is forced: clear current color table
    if(forceRecompute) {
        atomColorTable.Clear();
    }

    // Clamp weights to zero
    if(weight0 < 0.0) weight0 = 0.0;
    if(weight1 < 0.0) weight1 = 0.0;

    // Normalize weights
    weight0 = weight0 / (weight0 + weight1);
    weight1 = weight1 / (weight0 + weight1);

    // only compute color table if necessary
    if(atomColorTable.IsEmpty()) {

        vislib::Array<float> color0;
        vislib::Array<float> color1;

        // reserve memory for all atoms
        atomColorTable.AssertCapacity(mol->AtomCount()*3);

        // Compute first color table
        Color::MakeColorTable(mol, cm0, color0, colorLookupTable, rainbowColors,
            minGradColor, midGradColor, maxGradColor, true, bs);

        // Compute second color table
        Color::MakeColorTable(mol, cm1, color1, colorLookupTable, rainbowColors,
            minGradColor, midGradColor, maxGradColor, true, bs);

        // Interpolate
        for(unsigned int cnt = 0; cnt < mol->AtomCount()*3; cnt++) {
            atomColorTable.Add(color0[cnt]*weight0+color1[cnt]*weight1);
        }

    }
}

void Color::MakeComparisonColorTable(const megamol::protein_calls::MolecularDataCall *mol1,
	const megamol::protein_calls::MolecularDataCall *mol2,
	ColoringMode currentColoringMode,
	vislib::Array<float> &atomColorTable,
	vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
	vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
	vislib::TString minGradColor,
	vislib::TString midGradColor,
	vislib::TString maxGradColor,
	bool forceRecompute,
	const protein_calls::BindingSiteCall *bs) {

	// temporary variables
    unsigned int cnt, cnt2, idx, idx2, cntAtom, cntRes, cntSecS, atomIdx,
        atomIdx2, atomCnt, atomCnt2;
    vislib::math::Vector<float, 3> color;
    //float r, g, b;

    // if recomputation is forced: clear current color table
    if( forceRecompute ) {
        atomColorTable.Clear();
    }

    // reserve memory for all atoms
    atomColorTable.AssertCapacity( mol1->AtomCount() * 3 );

	// fill table with the default color
	for( cntAtom = 0; cntAtom < mol1->AtomCount(); ++cntAtom ) {
		atomColorTable.Add(0.0);
		atomColorTable.Add(0.0);
		atomColorTable.Add(0.0);
	}

	unsigned int ssc = mol1->SecondaryStructureCount();

	if(ssc > mol2->SecondaryStructureCount())
		ssc = mol2->SecondaryStructureCount();
		
	float max = -2.0f;

	// go through every secondary structure
	for( cntSecS = 0; cntSecS < ssc; ++cntSecS ) {
		
		idx = mol1->SecondaryStructures()[cntSecS].FirstAminoAcidIndex();
		idx2 = mol2->SecondaryStructures()[cntSecS].FirstAminoAcidIndex();
		cnt = idx + mol1->SecondaryStructures()[cntSecS].AminoAcidCount();
		cnt2 = idx + mol2->SecondaryStructures()[cntSecS].AminoAcidCount();

		// go through every single aminoacid
		for( cntRes = idx; cntRes < cnt; ++cntRes ) {

			atomIdx = mol1->Residues()[cntRes]->FirstAtomIndex();
			atomIdx2 = mol2->Residues()[cntRes]->FirstAtomIndex();
			atomCnt = atomIdx + mol1->Residues()[cntRes]->AtomCount();
			atomCnt2 = atomIdx2 + mol2->Residues()[cntRes]->AtomCount(); 

			megamol::protein_calls::MolecularDataCall::AminoAcid * aminoacid1;
			megamol::protein_calls::MolecularDataCall::AminoAcid * aminoacid2;

			const megamol::protein_calls::MolecularDataCall::Residue* t1 = mol2->Residues()[cntRes];

			if(t1 == NULL)
				break;

			if (mol1->Residues()[cntRes]->Identifier() == megamol::protein_calls::MolecularDataCall::Residue::AMINOACID
				&& mol2->Residues()[cntRes]->Identifier() == megamol::protein_calls::MolecularDataCall::Residue::AMINOACID) {
				aminoacid1 = (megamol::protein_calls::MolecularDataCall::AminoAcid*)(mol1->Residues()[cntRes]);
				aminoacid2 = (megamol::protein_calls::MolecularDataCall::AminoAcid*)(mol2->Residues()[cntRes]);
			} else {// TODO check if this is correct
				continue;
			}

			unsigned int cAlphaIdx = aminoacid1->CAlphaIndex();
			unsigned int cAlphaIdx2 = aminoacid2->CAlphaIndex();

			float xDist = mol1->AtomPositions()[cAlphaIdx*3+0] - mol2->AtomPositions()[cAlphaIdx2*3+0];
			float yDist = mol1->AtomPositions()[cAlphaIdx*3+1] - mol2->AtomPositions()[cAlphaIdx2*3+1];
			float zDist = mol1->AtomPositions()[cAlphaIdx*3+2] - mol2->AtomPositions()[cAlphaIdx2*3+2];

			float absoluteDist = sqrt(xDist*xDist + yDist*yDist + zDist*zDist);

			if(absoluteDist > max)
				max = absoluteDist;

			// set the color for every atom
			for( cntAtom = atomIdx; cntAtom < atomCnt; ++cntAtom ) {
				atomColorTable[3*cntAtom+0] = absoluteDist;
				atomColorTable[3*cntAtom+1] = absoluteDist;
				atomColorTable[3*cntAtom+2] = absoluteDist;
			}
		}
	}

	// normalize the colors to [0.0..1.0]
	if( max > 0.0000001f ) {
		for( cntAtom = 0; cntAtom < mol1->AtomCount(); ++cntAtom ) {
				atomColorTable[3*cntAtom+0] /= max;
				atomColorTable[3*cntAtom+1] /= max;
				atomColorTable[3*cntAtom+2] /= max;
		}
	}
}


/*
 * Creates a rainbow color table with 'num' entries.
 */
void Color::MakeRainbowColorTable( unsigned int num,
    vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors) {

    unsigned int n = (num/4);
    // the color table should have a minimum size of 16
    if( n < 4 )
        n = 4;
    rainbowColors.Clear();
    rainbowColors.AssertCapacity( num);
    float f = 1.0f/float(n);
    vislib::math::Vector<float,3> color;
    color.Set( 1.0f, 0.0f, 0.0f);
    for( unsigned int i = 0; i < n; i++) {
        color.SetY( vislib::math::Min( color.GetY() + f, 1.0f));
        rainbowColors.Add( color);
    }
    for( unsigned int i = 0; i < n; i++) {
        color.SetX( vislib::math::Max( color.GetX() - f, 0.0f));
        rainbowColors.Add( color);
    }
    for( unsigned int i = 0; i < n; i++) {
        color.SetZ( vislib::math::Min( color.GetZ() + f, 1.0f));
        rainbowColors.Add( color);
    }
    for( unsigned int i = 0; i < n; i++) {
        color.SetY( vislib::math::Max( color.GetY() - f, 0.0f));
        rainbowColors.Add( color);
    }
}


/*
 * Read color table from file.
 */
void Color::ReadColorTableFromFile( vislib::StringA filename,
          vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable) {

    // file buffer variable
    vislib::sys::ASCIIFileBuffer file;
    // delete old color table
    colorLookupTable.SetCount( 0);
    // try to load the color table file
    if( file.LoadFile( filename) ) {
        float r, g, b;
        colorLookupTable.AssertCapacity( file.Count());
        // get colors from file
        for( unsigned int cnt = 0; cnt < file.Count(); ++cnt ) {
        	vislib::StringA lineStr( file.Line(cnt) );
            if( lineStr.Length() > 0 && utility::ColourParser::FromString( lineStr, r, g, b) ) {
                colorLookupTable.Add( vislib::math::Vector<float, 3>
                  ( r, g, b));
            }
        }
    }
    // if the file could not be loaded or contained no valid colors
    if( colorLookupTable.Count() == 0 ) {
        // set default color table
        colorLookupTable.SetCount( 25);
        colorLookupTable[0].Set( 0.70f, 0.8f, 0.4f);
        colorLookupTable[1].Set( 1.0f, 0.0f, 0.0f);
        colorLookupTable[2].Set( 1.0f, 1.0f, 0.0f);
        colorLookupTable[3].Set( 0.0f, 1.0f, 0.0f);
        colorLookupTable[4].Set( 0.0f, 1.0f, 1.0f);
        colorLookupTable[5].Set( 0.0f, 0.0f, 1.0f);
        colorLookupTable[6].Set( 1.0f, 0.0f, 1.0f);
        colorLookupTable[7].Set( 0.5f, 0.0f, 0.0f);
        colorLookupTable[8].Set( 0.5f, 0.5f, 0.0f);
        colorLookupTable[9].Set( 0.0f, 0.5f, 0.0f);
        colorLookupTable[10].Set( 0.00f, 0.50f, 0.50f);
        colorLookupTable[11].Set( 0.00f, 0.00f, 0.50f);
        colorLookupTable[12].Set( 0.50f, 0.00f, 0.50f);
        colorLookupTable[13].Set( 1.00f, 0.50f, 0.00f);
        colorLookupTable[14].Set( 0.00f, 0.50f, 1.00f);
        colorLookupTable[15].Set( 1.00f, 0.50f, 1.00f);
        colorLookupTable[16].Set( 0.50f, 0.25f, 0.00f);
        colorLookupTable[17].Set( 1.00f, 1.00f, 0.50f);
        colorLookupTable[18].Set( 0.50f, 1.00f, 0.50f);
        colorLookupTable[19].Set( 0.75f, 1.00f, 0.00f);
        colorLookupTable[20].Set( 0.50f, 0.00f, 0.75f);
        colorLookupTable[21].Set( 1.00f, 0.50f, 0.50f);
        colorLookupTable[22].Set( 0.75f, 1.00f, 0.75f);
        colorLookupTable[23].Set( 0.75f, 0.75f, 0.50f);
        colorLookupTable[24].Set( 1.00f, 0.75f, 0.50f);
    }
}

