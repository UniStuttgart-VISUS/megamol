/*
 * Color.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include <array>
#include <iostream>
#include <omp.h>
#include <string>
#include "Color.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/ColourParser.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/sys/ASCIIFileBuffer.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;

//#define MONERA

/*
 * FillAminoAcidColorTable
 */
void Color::FillAminoAcidColorTable(vislib::Array<vislib::math::Vector<float, 3>>& aminoAcidColorTable) {

    aminoAcidColorTable.Clear();
    aminoAcidColorTable.SetCount(25);
    aminoAcidColorTable[0].Set(0.5f, 0.5f, 0.5f);
    aminoAcidColorTable[1].Set(1.0f, 0.0f, 0.0f);
    aminoAcidColorTable[2].Set(1.0f, 1.0f, 0.0f);
    aminoAcidColorTable[3].Set(0.0f, 1.0f, 0.0f);
    aminoAcidColorTable[4].Set(0.0f, 1.0f, 1.0f);
    aminoAcidColorTable[5].Set(0.0f, 0.0f, 1.0f);
    aminoAcidColorTable[6].Set(1.0f, 0.0f, 1.0f);
    aminoAcidColorTable[7].Set(0.5f, 0.0f, 0.0f);
    aminoAcidColorTable[8].Set(0.5f, 0.5f, 0.0f);
    aminoAcidColorTable[9].Set(0.0f, 0.5f, 0.0f);
    aminoAcidColorTable[10].Set(0.0f, 0.5f, 0.5f);
    aminoAcidColorTable[11].Set(0.0f, 0.0f, 0.5f);
    aminoAcidColorTable[12].Set(0.5f, 0.0f, 0.5f);
    aminoAcidColorTable[13].Set(1.0f, 0.5f, 0.0f);
    aminoAcidColorTable[14].Set(0.0f, 0.5f, 1.0f);
    aminoAcidColorTable[15].Set(1.0f, 0.5f, 1.0f);
    aminoAcidColorTable[16].Set(0.5f, 0.25f, 0.0f);
    aminoAcidColorTable[17].Set(1.0f, 1.0f, 0.5f);
    aminoAcidColorTable[18].Set(0.5f, 1.0f, 0.5f);
    aminoAcidColorTable[19].Set(0.75f, 1.0f, 0.0f);
    aminoAcidColorTable[20].Set(0.5f, 0.0f, 0.75f);
    aminoAcidColorTable[21].Set(1.0f, 0.5f, 0.5f);
    aminoAcidColorTable[22].Set(0.75f, 1.0f, 0.75f);
    aminoAcidColorTable[23].Set(0.75f, 0.75f, 0.5f);
    aminoAcidColorTable[24].Set(1.0f, 0.75f, 0.5f);
}


/*
 * Color::GetModeByIndex
 */
Color::ColoringMode Color::GetModeByIndex(const megamol::protein_calls::MolecularDataCall* mol, unsigned int idx) {

    switch (idx) {
    case 0:
        return ELEMENT;
    case 1:
        return RESIDUE;
    case 2:
        return STRUCTURE;
    case 3:
        return BFACTOR;
    case 4:
        return CHARGE;
    case 5:
        return OCCUPANCY;
    case 6:
        return CHAIN;
    case 7:
        return MOLECULE;
    case 8:
        return RAINBOW;
    case 9:
        return HYDROPHOBICITY;
    case 10:
        return HEIGHTMAP_COL;
    case 11:
        return HEIGHTMAP_VAL;
    default:
        return ELEMENT;
    }
}

/*
 * Color::GetModeByIndex
 */
Color::ColoringMode Color::GetModeByIndex(
    const megamol::protein_calls::MolecularDataCall* mol, const protein_calls::BindingSiteCall* bs, unsigned int idx) {
    switch (idx) {
    case 0:
        return ELEMENT;
    case 1:
        return RESIDUE;
    case 2:
        return STRUCTURE;
    case 3:
        return BFACTOR;
    case 4:
        return CHARGE;
    case 5:
        return OCCUPANCY;
    case 6:
        return CHAIN;
    case 7:
        return MOLECULE;
    case 8:
        return RAINBOW;
    case 9:
        return HYDROPHOBICITY;
    case 10:
        return HEIGHTMAP_COL;
    case 11:
        return HEIGHTMAP_VAL;
    case 12:
        return BINDINGSITE;
    default:
        return ELEMENT;
    }
}

/*
 * Color::GetModeByIndex
 */
Color::ColoringMode Color::GetModeByIndex(const megamol::protein_calls::MolecularDataCall* mol,
    const protein_calls::BindingSiteCall* bs, const protein_calls::PerAtomFloatCall* pa, unsigned int idx) {
    switch (idx) {
    case 0:
        return ELEMENT;
    case 1:
        return RESIDUE;
    case 2:
        return STRUCTURE;
    case 3:
        return BFACTOR;
    case 4:
        return CHARGE;
    case 5:
        return OCCUPANCY;
    case 6:
        return CHAIN;
    case 7:
        return MOLECULE;
    case 8:
        return RAINBOW;
    case 9:
        return HYDROPHOBICITY;
    case 10:
        return HEIGHTMAP_COL;
    case 11:
        return HEIGHTMAP_VAL;
    case 12:
        return BINDINGSITE;
    case 13:
        return PER_ATOM_FLOAT;
    case 14:
        return AMINOACID;
    default:
        return ELEMENT;
    }
}

/*
 * Color::GetName
 */
std::string Color::GetName(Color::ColoringMode col) {

    switch (col) {
    case ELEMENT:
        return "Element";
    case STRUCTURE:
        return "Structure";
    case RAINBOW:
        return "Rainbow";
    case BFACTOR:
        return "BFactor";
    case CHARGE:
        return "Charge";
    case OCCUPANCY:
        return "Occupancy";
    case CHAIN:
        return "Chain";
    case MOLECULE:
        return "Molecule";
    case RESIDUE:
        return "Residue";
    case CHAINBOW:
        return "Chainbow";
    case AMINOACID:
        return "Aminoacid";
    case VALUE:
        return "Value";
    case CHAIN_ID:
        return "ChainID";
    case MOVEMENT:
        return "Movement";
    case HYDROPHOBICITY:
        return "Hydrophobicity";
    case BINDINGSITE:
        return "BindingSite";
    case PER_ATOM_FLOAT:
        return "SolventCount";
    case HEIGHTMAP_COL:
        return "HeightMapColor";
    case HEIGHTMAP_VAL:
        return "HeightMapValue";
    default:
        return "";
    }
}

/*
 * Get the hydrophobicity value for an amino acid
 */
float Color::GetHydrophibicityByResName(vislib::StringA resName) {
#ifdef MONERA
    // O.D.Monera, T.J.Sereda, N.E.Zhou, C.M.Kay, R.S.Hodges
    // "Relationship of sidechain hydrophobicity and alpha-helical propensity on the stability of the single-stranded
    // amphipathic alpha-helix." J.Pept.Sci. 1995 Sep - Oct; 1(5) : 319 - 29.
    if (resName.Equals("ALA", false))
        return 41.0f;
    else if (resName.Equals("ARG", false))
        return -14.0f;
    else if (resName.Equals("LEU", false))
        return 97.0f;
    else if (resName.Equals("LYS", false))
        return -23.0f;
    else if (resName.Equals("MET", false))
        return 74.0f;
    else if (resName.Equals("GLN", false))
        return -10.0f;
    else if (resName.Equals("ILE", false))
        return 99.0f;
    else if (resName.Equals("TRP", false))
        return 97.0f;
    else if (resName.Equals("PHE", false))
        return 100.0f;
    else if (resName.Equals("TYR", false))
        return 63.0f;
    else if (resName.Equals("CYS", false))
        return 49.0f;
    else if (resName.Equals("VAL", false))
        return 76.0f;
    else if (resName.Equals("ASN", false))
        return -28.0f;
    else if (resName.Equals("SER", false))
        return -5.0f;
    else if (resName.Equals("HIS", false))
        return 8.0f;
    else if (resName.Equals("GLU", false))
        return -31.0f;
    else if (resName.Equals("THR", false))
        return 13.0f;
    else if (resName.Equals("ASP", false))
        return -55.0f;
    else if (resName.Equals("GLY", false))
        return 0.0f;
    else if (resName.Equals("PRO", false))
        return -46.0f;
#else
    // J. Kyte, R. F. Doolittle
    // "A simple method for displaying the hydropathic character of a protein."
    // J. Mol. Biol. 1982 May 5; 157(1) : 105 - 32.
    if (resName.Equals("Ala", false))
        return 1.8f;
    else if (resName.Equals("Arg", false))
        return -4.5f;
    else if (resName.Equals("Asn", false))
        return -3.5f;
    else if (resName.Equals("Asp", false))
        return -3.5f;
    else if (resName.Equals("Cys", false))
        return 2.5f;
    else if (resName.Equals("Glu", false))
        return -3.5f;
    else if (resName.Equals("Gln", false))
        return -3.5f;
    else if (resName.Equals("Gly", false))
        return -0.4f;
    else if (resName.Equals("His", false))
        return -3.2f;
    else if (resName.Equals("Ile", false))
        return 4.5f;
    else if (resName.Equals("Leu", false))
        return 3.8f;
    else if (resName.Equals("Lys", false))
        return -3.9f;
    else if (resName.Equals("Met", false))
        return 1.9f;
    else if (resName.Equals("Phe", false))
        return 2.8f;
    else if (resName.Equals("Pro", false))
        return -1.6f;
    else if (resName.Equals("Ser", false))
        return -0.8f;
    else if (resName.Equals("Thr", false))
        return -0.7f;
    else if (resName.Equals("Trp", false))
        return -0.9f;
    else if (resName.Equals("Tyr", false))
        return -1.3f;
    else if (resName.Equals("Val", false))
        return 4.2f;
#endif
    return 0.0f;
}

/*
 * MakeColorTable routine for molecular data call
 */
void Color::MakeColorTable(const megamol::protein_calls::MolecularDataCall* mol, ColoringMode currentColoringMode,
    vislib::Array<float>& atomColorTable, vislib::Array<vislib::math::Vector<float, 3>>& colorLookupTable,
    vislib::Array<vislib::math::Vector<float, 3>>& rainbowColors, vislib::TString minGradColor,
    vislib::TString midGradColor, vislib::TString maxGradColor, bool forceRecompute,
    const protein_calls::BindingSiteCall* bs, bool useNeighbors, const protein_calls::PerAtomFloatCall* pa) {

    // temporary variables
    unsigned int cnt, idx, cntAtom, cntRes, cntChain, cntMol, cntSecS, atomIdx, atomCnt;
    vislib::math::Vector<float, 3> color;
    float r, g, b;

    // if recomputation is forced: clear current color table
    if (forceRecompute) {
        atomColorTable.Clear();
    }
    // reserve memory for all atoms
    atomColorTable.AssertCapacity(mol->AtomCount() * 3);

    // only compute color table if necessary
    if (atomColorTable.IsEmpty()) {
        if (currentColoringMode == ELEMENT) {
            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {

                atomColorTable.Add(float(mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Colour()[0]) / 255.0f);

                atomColorTable.Add(float(mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Colour()[1]) / 255.0f);

                atomColorTable.Add(float(mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Colour()[2]) / 255.0f);
            }
        } // ... END coloring mode ELEMENT
        else if (currentColoringMode == RESIDUE) {
            unsigned int resTypeIdx;
            // loop over all residues
            for (cntRes = 0; cntRes < mol->ResidueCount(); ++cntRes) {
                // loop over all atoms of the current residue
                idx = mol->Residues()[cntRes]->FirstAtomIndex();
                cnt = mol->Residues()[cntRes]->AtomCount();
                // get residue type index
                resTypeIdx = mol->Residues()[cntRes]->Type();
                for (cntAtom = idx; cntAtom < idx + cnt; ++cntAtom) {
                    // Special cases for water/toluol/methanol
                    if (mol->ResidueTypeNames()[resTypeIdx].Equals("SOL")) {
                        // Water
                        atomColorTable.Add(0.3f);
                        atomColorTable.Add(0.3f);
                        atomColorTable.Add(1.0f);
                    } else if (mol->ResidueTypeNames()[resTypeIdx].Equals("MeOH")) {
                        // methanol
                        atomColorTable.Add(1.0f);
                        atomColorTable.Add(0.5f);
                        atomColorTable.Add(0.0f);
                    } else if (mol->ResidueTypeNames()[resTypeIdx].Equals("TOL")) {
                        // toluol
                        atomColorTable.Add(1.0f);
                        atomColorTable.Add(1.0f);
                        atomColorTable.Add(1.0f);
                    } else {
                        atomColorTable.Add(colorLookupTable[resTypeIdx % colorLookupTable.Count()].X());
                        atomColorTable.Add(colorLookupTable[resTypeIdx % colorLookupTable.Count()].Y());
                        atomColorTable.Add(colorLookupTable[resTypeIdx % colorLookupTable.Count()].Z());
                    }
                }
            }
        } // ... END coloring mode RESIDUE
        else if (currentColoringMode == AMINOACID) {
            unsigned int resTypeIdx, colour_idx;
            // loop over all residues
            for (cntRes = 0; cntRes < mol->ResidueCount(); ++cntRes) {
                // loop over all atoms of the current residue
                idx = mol->Residues()[cntRes]->FirstAtomIndex();
                cnt = mol->Residues()[cntRes]->AtomCount();
                // get residue type index
                resTypeIdx = mol->Residues()[cntRes]->Type();
                if (mol->ResidueTypeNames()[resTypeIdx].Equals("ALA") ||
                    mol->ResidueTypeNames()[resTypeIdx].Equals("VAL") ||
                    mol->ResidueTypeNames()[resTypeIdx].Equals("MET") ||
                    mol->ResidueTypeNames()[resTypeIdx].Equals("LEU") ||
                    mol->ResidueTypeNames()[resTypeIdx].Equals("ILE") ||
                    mol->ResidueTypeNames()[resTypeIdx].Equals("PRO") ||
                    mol->ResidueTypeNames()[resTypeIdx].Equals("TRP") ||
                    mol->ResidueTypeNames()[resTypeIdx].Equals("PHE")) {
                    colour_idx = 0;
                } else if (mol->ResidueTypeNames()[resTypeIdx].Equals("TYR") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("TYM") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("THR") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("GLN") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("GLY") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("SER") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("CYS") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("CYX") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("CYM") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("ASN") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("ASH")) {
                    colour_idx = 1;
                } else if (mol->ResidueTypeNames()[resTypeIdx].Equals("LYS") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("LYN") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("ARG") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("HIS") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("HID") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("HIE") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("HIP")) {
                    colour_idx = 2;
                } else if (mol->ResidueTypeNames()[resTypeIdx].Equals("GLU") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("GLH") ||
                           mol->ResidueTypeNames()[resTypeIdx].Equals("ASP")) {
                    colour_idx = 3;
                } else {
                    colour_idx = 4;
                }
                for (cntAtom = idx; cntAtom < idx + cnt; ++cntAtom) {
                    if (colour_idx == 0) {
                        atomColorTable.Add(255.0f / 255.0f);
                        atomColorTable.Add(230.0f / 255.0f);
                        atomColorTable.Add(128.0f / 255.0f);
                    } else if (colour_idx == 1) {
                        atomColorTable.Add(170 / 255.0f);
                        atomColorTable.Add(222.0f / 255.0f);
                        atomColorTable.Add(135.0f / 255.0f);
                    } else if (colour_idx == 2) {
                        atomColorTable.Add(128.0f / 255.0f);
                        atomColorTable.Add(179.0f / 255.0f);
                        atomColorTable.Add(255.0f / 255.0f);
                    } else if (colour_idx == 3) {
                        atomColorTable.Add(255.0f / 255.0f);
                        atomColorTable.Add(170.0f / 255.0f);
                        atomColorTable.Add(170.0f / 255.0f);
                    } else {
                        atomColorTable.Add(180.0f / 255.0f);
                        atomColorTable.Add(180.0f / 255.0f);
                        atomColorTable.Add(180.0f / 255.0f);
                    }
                }
            }
        } // ... END coloring mode AMINOACID
        else if (currentColoringMode == HYDROPHOBICITY) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString(minGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMin(r, g, b);
            // get mid color
            utility::ColourParser::FromString(midGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMid(r, g, b);
            // get max color
            utility::ColourParser::FromString(maxGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMax(r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;
#ifdef MONERA
            float min_val(-100.0f);
            float max_val(100.0f);
            float mid(0.0f);
#else
            float min_val(-4.5f);
            float max_val(4.5f);
            float mid(0.0f);
#endif
            float val;

            unsigned int resTypeIdx;
            // loop over all residues
            for (cntRes = 0; cntRes < mol->ResidueCount(); ++cntRes) {
                // loop over all atoms of the current residue
                idx = mol->Residues()[cntRes]->FirstAtomIndex();
                cnt = mol->Residues()[cntRes]->AtomCount();
                // get residue type index
                resTypeIdx = mol->Residues()[cntRes]->Type();
                val = GetHydrophibicityByResName(mol->ResidueTypeNames()[resTypeIdx]);
                for (cntAtom = idx; cntAtom < idx + cnt; ++cntAtom) {
                    // below middle value --> blend between min and mid color
                    if (val < mid) {
                        col = colMin + ((colMid - colMin) / (mid - min_val)) * (val - min_val);
                        atomColorTable.Add(col.GetX());
                        atomColorTable.Add(col.GetY());
                        atomColorTable.Add(col.GetZ());
                    }
                    // above middle value --> blend between max and mid color
                    else if (val > mid) {
                        col = colMid + ((colMax - colMid) / (max_val - mid)) * (val - mid);
                        atomColorTable.Add(col.GetX());
                        atomColorTable.Add(col.GetY());
                        atomColorTable.Add(col.GetZ());
                    }
                    // middle value --> assign mid color
                    else {
                        atomColorTable.Add(colMid.GetX());
                        atomColorTable.Add(colMid.GetY());
                        atomColorTable.Add(colMid.GetZ());
                    }
                }
            }
        } // ... END coloring mode HYDROPHOBICITY
        else if (currentColoringMode == STRUCTURE) {
            utility::ColourParser::FromString("#00ff00", r, g, b);
            vislib::math::Vector<float, 3> colNone(r, g, b);
            utility::ColourParser::FromString("#ff0000", r, g, b);
            vislib::math::Vector<float, 3> colHelix(r, g, b);
            utility::ColourParser::FromString("#0000ff", r, g, b);
            vislib::math::Vector<float, 3> colSheet(r, g, b);
            utility::ColourParser::FromString("#888888", r, g, b);
            vislib::math::Vector<float, 3> colRCoil(r, g, b);
            // loop over all atoms and fill the table with the default color
            for (cntAtom = 0; cntAtom < mol->AtomCount(); ++cntAtom) {
                atomColorTable.Add(colNone.X());
                atomColorTable.Add(colNone.Y());
                atomColorTable.Add(colNone.Z());
            }
            // write colors for sec structure elements
            megamol::protein_calls::MolecularDataCall::SecStructure::ElementType elemType;

            for (cntSecS = 0; cntSecS < mol->SecondaryStructureCount(); ++cntSecS) {

                idx = mol->SecondaryStructures()[cntSecS].FirstAminoAcidIndex();
                cnt = idx + mol->SecondaryStructures()[cntSecS].AminoAcidCount();
                elemType = mol->SecondaryStructures()[cntSecS].Type();

                for (cntRes = idx; cntRes < cnt; ++cntRes) {
                    atomIdx = mol->Residues()[cntRes]->FirstAtomIndex();
                    atomCnt = atomIdx + mol->Residues()[cntRes]->AtomCount();
                    for (cntAtom = atomIdx; cntAtom < atomCnt; ++cntAtom) {
                        if (elemType == megamol::protein_calls::MolecularDataCall::SecStructure::TYPE_HELIX) {
                            atomColorTable[3 * cntAtom + 0] = colHelix.X();
                            atomColorTable[3 * cntAtom + 1] = colHelix.Y();
                            atomColorTable[3 * cntAtom + 2] = colHelix.Z();
                        } else if (elemType == megamol::protein_calls::MolecularDataCall::SecStructure::TYPE_SHEET) {
                            atomColorTable[3 * cntAtom + 0] = colSheet.X();
                            atomColorTable[3 * cntAtom + 1] = colSheet.Y();
                            atomColorTable[3 * cntAtom + 2] = colSheet.Z();
                        } else if (elemType == megamol::protein_calls::MolecularDataCall::SecStructure::TYPE_COIL) {
                            atomColorTable[3 * cntAtom + 0] = colRCoil.X();
                            atomColorTable[3 * cntAtom + 1] = colRCoil.Y();
                            atomColorTable[3 * cntAtom + 2] = colRCoil.Z();
                        }
                    }
                }
            }
        } // ... END coloring mode STRUCTURE
        else if (currentColoringMode == BFACTOR) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString(minGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMin(r, g, b);
            // get mid color
            utility::ColourParser::FromString(midGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMid(r, g, b);
            // get max color
            utility::ColourParser::FromString(maxGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMax(r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min_val(mol->MinimumBFactor());
            float max_val(mol->MaximumBFactor());
            float mid((max_val - min_val) / 2.0f + min_val);
            float val;

            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {
                if (min_val == max_val) {
                    atomColorTable.Add(colMid.GetX());
                    atomColorTable.Add(colMid.GetY());
                    atomColorTable.Add(colMid.GetZ());
                    continue;
                }

                val = mol->AtomBFactors()[cnt];
                // below middle value --> blend between min and mid color
                if (val < mid) {
                    col = colMin + ((colMid - colMin) / (mid - min_val)) * (val - min_val);
                    atomColorTable.Add(col.GetX());
                    atomColorTable.Add(col.GetY());
                    atomColorTable.Add(col.GetZ());
                }
                // above middle value --> blend between max and mid color
                else if (val > mid) {
                    col = colMid + ((colMax - colMid) / (max_val - mid)) * (val - mid);
                    atomColorTable.Add(col.GetX());
                    atomColorTable.Add(col.GetY());
                    atomColorTable.Add(col.GetZ());
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add(colMid.GetX());
                    atomColorTable.Add(colMid.GetY());
                    atomColorTable.Add(colMid.GetZ());
                }
            }
        } // ... END coloring mode BFACTOR
        else if (currentColoringMode == PER_ATOM_FLOAT) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString(minGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMin(r, g, b);
            // get mid color
            utility::ColourParser::FromString(midGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMid(r, g, b);
            // get max color
            utility::ColourParser::FromString(maxGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMax(r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min_val(pa->MinValue());
            float max_val(pa->MaxValue());
            float mid(pa->MidValue());
            float val;
            printf("%.1f ", min_val);
            printf("%.1f ", mid);
            printf("%.1f ", max_val);

            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {
                if (min_val == max_val) {
                    atomColorTable.Add(colMid.GetX());
                    atomColorTable.Add(colMid.GetY());
                    atomColorTable.Add(colMid.GetZ());
                    continue;
                }

                val = pa->GetFloat()[cnt];
                // below middle value --> blend between min and mid color
                if (val < mid) {
                    col = colMin + ((colMid - colMin) / (mid - min_val)) * (val - min_val);
                    atomColorTable.Add(col.GetX());
                    atomColorTable.Add(col.GetY());
                    atomColorTable.Add(col.GetZ());
                }
                // above middle value --> blend between max and mid color
                else if (val > mid) {
                    col = colMid + ((colMax - colMid) / (max_val - mid)) * (val - mid);
                    atomColorTable.Add(col.GetX());
                    atomColorTable.Add(col.GetY());
                    atomColorTable.Add(col.GetZ());
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add(colMid.GetX());
                    atomColorTable.Add(colMid.GetY());
                    atomColorTable.Add(colMid.GetZ());
                }
            }
        } // ... END coloring mode PER_ATOM_FLOAT
        else if (currentColoringMode == CHARGE) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString(minGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMin(r, g, b);
            // get mid color
            utility::ColourParser::FromString(midGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMid(r, g, b);
            // get max color
            utility::ColourParser::FromString(maxGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMax(r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min_val(mol->MinimumCharge());
            float max_val(mol->MaximumCharge());
            float mid((max_val - min_val) / 2.0f + min_val);
            float val;

            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {
                if (min_val == max_val) {
                    atomColorTable.Add(colMid.GetX());
                    atomColorTable.Add(colMid.GetY());
                    atomColorTable.Add(colMid.GetZ());
                    continue;
                }

                val = mol->AtomCharges()[cnt];
                // below middle value --> blend between min and mid color
                if (val < mid) {
                    col = colMin + ((colMid - colMin) / (mid - min_val)) * (val - min_val);
                    atomColorTable.Add(col.GetX());
                    atomColorTable.Add(col.GetY());
                    atomColorTable.Add(col.GetZ());
                }
                // above middle value --> blend between max and mid color
                else if (val > mid) {
                    col = colMid + ((colMax - colMid) / (max_val - mid)) * (val - mid);
                    atomColorTable.Add(col.GetX());
                    atomColorTable.Add(col.GetY());
                    atomColorTable.Add(col.GetZ());
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add(colMid.GetX());
                    atomColorTable.Add(colMid.GetY());
                    atomColorTable.Add(colMid.GetZ());
                }
            }
        } // ... END coloring mode CHARGE
        else if (currentColoringMode == OCCUPANCY) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString(minGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMin(r, g, b);
            // get mid color
            utility::ColourParser::FromString(midGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMid(r, g, b);
            // get max color
            utility::ColourParser::FromString(maxGradColor, r, g, b);
            vislib::math::Vector<float, 3> colMax(r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min_val(mol->MinimumOccupancy());
            float max_val(mol->MaximumOccupancy());
            float mid((max_val - min_val) / 2.0f + min_val);
            float val;

            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {
                if (min_val == max_val) {
                    atomColorTable.Add(colMid.GetX());
                    atomColorTable.Add(colMid.GetY());
                    atomColorTable.Add(colMid.GetZ());
                    continue;
                }

                val = mol->AtomOccupancies()[cnt];
                // below middle value --> blend between min and mid color
                if (val < mid) {
                    col = colMin + ((colMid - colMin) / (mid - min_val)) * (val - min_val);
                    atomColorTable.Add(col.GetX());
                    atomColorTable.Add(col.GetY());
                    atomColorTable.Add(col.GetZ());
                }
                // above middle value --> blend between max and mid color
                else if (val > mid) {
                    col = colMid + ((colMax - colMid) / (max_val - mid)) * (val - mid);
                    atomColorTable.Add(col.GetX());
                    atomColorTable.Add(col.GetY());
                    atomColorTable.Add(col.GetZ());
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add(colMid.GetX());
                    atomColorTable.Add(colMid.GetY());
                    atomColorTable.Add(colMid.GetZ());
                }
            }
        } // ... END coloring mode OCCUPANCY
        else if (currentColoringMode == CHAIN) {
            // get the last atom of the last res of the last mol of the first chain
            cntChain = 0;
            cntMol = mol->Chains()[cntChain].MoleculeCount() - 1;
            cntRes = mol->Molecules()[cntMol].FirstResidueIndex() + mol->Molecules()[cntMol].ResidueCount() - 1;
            cntAtom = mol->Residues()[cntRes]->FirstAtomIndex() + mol->Residues()[cntRes]->AtomCount() - 1;
            // get the first color
            idx = 0;
            color = colorLookupTable[idx % colorLookupTable.Count()];
            // loop over all atoms
            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {
                // check, if the last atom of the current chain is reached
                if (cnt > cntAtom) {
                    // get the last atom of the last res of the last mol of the next chain
                    cntChain++;
                    cntMol = mol->Chains()[cntChain].FirstMoleculeIndex() + mol->Chains()[cntChain].MoleculeCount() - 1;
                    cntRes = mol->Molecules()[cntMol].FirstResidueIndex() + mol->Molecules()[cntMol].ResidueCount() - 1;
                    cntAtom = mol->Residues()[cntRes]->FirstAtomIndex() + mol->Residues()[cntRes]->AtomCount() - 1;
                    // get the next color
                    idx++;
                    color = colorLookupTable[idx % colorLookupTable.Count()];
                }
                atomColorTable.Add(color.X());
                atomColorTable.Add(color.Y());
                atomColorTable.Add(color.Z());
            }
        } // ... END coloring mode CHAIN
        else if (currentColoringMode == MOLECULE) {
            // get the last atom of the last res of the first mol
            cntMol = 0;
            cntRes = mol->Molecules()[cntMol].FirstResidueIndex() + mol->Molecules()[cntMol].ResidueCount() - 1;
            cntAtom = mol->Residues()[cntRes]->FirstAtomIndex() + mol->Residues()[cntRes]->AtomCount() - 1;
            // get the first color
            idx = 0;
            color = colorLookupTable[idx % colorLookupTable.Count()];
            // loop over all atoms
            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {
                // check, if the last atom of the current chain is reached
                if (cnt > cntAtom) {
                    // get the last atom of the last res of the next mol
                    cntMol++;
                    cntRes = mol->Molecules()[cntMol].FirstResidueIndex() + mol->Molecules()[cntMol].ResidueCount() - 1;
                    cntAtom = mol->Residues()[cntRes]->FirstAtomIndex() + mol->Residues()[cntRes]->AtomCount() - 1;
                    // get the next color
                    idx++;
                    color = colorLookupTable[idx % colorLookupTable.Count()];
                }
                atomColorTable.Add(color.X());
                atomColorTable.Add(color.Y());
                atomColorTable.Add(color.Z());
            }
        } // ... END coloring mode MOLECULE
        else if (currentColoringMode == BINDINGSITE) {
            // initialize all colors as white
            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {
                atomColorTable.Add(1.0f);
                atomColorTable.Add(1.0f);
                atomColorTable.Add(1.0f);
            }
            // search for binding sites if BindingSiteCall is available
            if (bs) {
                // temporary variables
                unsigned int firstMol;
                unsigned int firstRes;
                unsigned int firstAtom;
                unsigned int atomIdx;
                for (unsigned int cCnt = 0; cCnt < mol->ChainCount(); cCnt++) {
                    firstMol = mol->Chains()[cCnt].FirstMoleculeIndex();
                    for (unsigned int mCnt = firstMol; mCnt < firstMol + mol->Chains()[cCnt].MoleculeCount(); mCnt++) {
                        firstRes = mol->Molecules()[mCnt].FirstResidueIndex();
                        for (unsigned int rCnt = 0; rCnt < mol->Molecules()[mCnt].ResidueCount(); rCnt++) {
                            // try to match binding sites
                            vislib::Pair<char, unsigned int> bsRes;
                            // loop over all binding sites
                            for (unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++) {
                                for (unsigned int bsResCnt = 0; bsResCnt < bs->GetBindingSite(bsCnt)->Count();
                                     bsResCnt++) {
                                    bsRes = bs->GetBindingSite(bsCnt)->operator[](bsResCnt);
                                    if (mol->Chains()[cCnt].Name() == bsRes.First() &&
                                        mol->Residues()[firstRes + rCnt]->OriginalResIndex() == bsRes.Second() &&
                                        mol->ResidueTypeNames()[mol->Residues()[firstRes + rCnt]->Type()] ==
                                            bs->GetBindingSiteResNames(bsCnt)->operator[](bsResCnt)) {
                                        // TODO loop over all atoms and add the color
                                        firstAtom = mol->Residues()[firstRes + rCnt]->FirstAtomIndex();
                                        for (unsigned int aCnt = 0;
                                             aCnt < mol->Residues()[firstRes + rCnt]->AtomCount(); aCnt++) {
                                            atomIdx = firstAtom + aCnt;
                                            atomColorTable[3 * atomIdx + 0] = bs->GetBindingSiteColor(bsCnt).X();
                                            atomColorTable[3 * atomIdx + 1] = bs->GetBindingSiteColor(bsCnt).Y();
                                            atomColorTable[3 * atomIdx + 2] = bs->GetBindingSiteColor(bsCnt).Z();
                                        }
                                    }
                                }
                            }
                        } // residues
                    }     // molecules
                }         // chains
            }             // BindingSiteCall available
        }                 // ... END coloring mode BINDINGSITE
        else if (currentColoringMode == RAINBOW) {
            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {
                idx = int((float(cnt) / float(mol->AtomCount())) * float(rainbowColors.Count()));
                color = rainbowColors[idx];
                atomColorTable.Add(color.GetX());
                atomColorTable.Add(color.GetY());
                atomColorTable.Add(color.GetZ());
            }
        } // ... END coloring mode RAINBOW
        else if (currentColoringMode == HEIGHTMAP_COL || currentColoringMode == HEIGHTMAP_VAL) {
            // initialize all colors as white
            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {
                atomColorTable.Add(1.0f);
                atomColorTable.Add(1.0f);
                atomColorTable.Add(1.0f);
            }
            // loop over all protein atoms to calculate centroid
            std::array<float, 3> centroid = {0.0f, 0.0f, 0.0f};
            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {
                centroid[0] += mol->AtomPositions()[cnt * 3 + 0];
                centroid[1] += mol->AtomPositions()[cnt * 3 + 1];
                centroid[2] += mol->AtomPositions()[cnt * 3 + 2];
            }
            centroid[0] /= static_cast<float>(mol->AtomCount());
            centroid[1] /= static_cast<float>(mol->AtomCount());
            centroid[2] /= static_cast<float>(mol->AtomCount());
            // calculate maximum distance
            float max_dist = 0.0f;
            float dist, steps;
            unsigned int anz_cols = 15;
            for (unsigned int i = 0; i < mol->AtomCount(); ++i) {
                dist = sqrt(pow(centroid[0] - mol->AtomPositions()[3 * i + 0], 2.0f) +
                            pow(centroid[1] - mol->AtomPositions()[3 * i + 1], 2.0f) +
                            pow(centroid[2] - mol->AtomPositions()[3 * i + 2], 2.0f));
                if (dist > max_dist) max_dist = dist;
            }
            steps = max_dist / static_cast<float>(anz_cols);
            // calculate the colors
            std::vector<std::vector<float>> colours =
                std::vector<std::vector<float>>(anz_cols + 2, std::vector<float>(3));
            colours[0][0] = 160.0f / 255.0f;
            colours[1][0] = 173.0f / 255.0f;
            colours[2][0] = 185.0f / 255.0f;
            colours[3][0] = 196.0f / 255.0f;
            colours[4][0] = 206.0f / 255.0f;
            colours[5][0] = 218.0f / 255.0f;
            colours[0][1] = 194.0f / 255.0f;
            colours[1][1] = 203.0f / 255.0f;
            colours[2][1] = 213.0f / 255.0f;
            colours[3][1] = 223.0f / 255.0f;
            colours[4][1] = 231.0f / 255.0f;
            colours[5][1] = 240.0f / 255.0f;
            colours[0][2] = 222.0f / 255.0f;
            colours[1][2] = 230.0f / 255.0f;
            colours[2][2] = 237.0f / 255.0f;
            colours[3][2] = 244.0f / 255.0f;
            colours[4][2] = 249.0f / 255.0f;
            colours[5][2] = 253.0f / 255.0f;

            colours[6][0] = 172.0f / 255.0f;
            colours[7][0] = 148.0f / 255.0f;
            colours[8][0] = 168.0f / 255.0f;
            colours[9][0] = 209.0f / 255.0f;
            colours[10][0] = 239.0f / 255.0f;
            colours[11][0] = 222.0f / 255.0f;
            colours[6][1] = 208.0f / 255.0f;
            colours[7][1] = 191.0f / 255.0f;
            colours[8][1] = 198.0f / 255.0f;
            colours[9][1] = 215.0f / 255.0f;
            colours[10][1] = 235.0f / 255.0f;
            colours[11][1] = 214.0f / 255.0f;
            colours[6][2] = 165.0f / 255.0f;
            colours[7][2] = 139.0f / 255.0f;
            colours[8][2] = 143.0f / 255.0f;
            colours[9][2] = 171.0f / 255.0f;
            colours[10][2] = 192.0f / 255.0f;
            colours[11][2] = 163.0f / 255.0f;

            colours[12][0] = 211.0f / 255.0f;
            colours[13][0] = 202.0f / 255.0f;
            colours[14][0] = 195.0f / 255.0f;
            colours[15][0] = 195.0f / 255.0f;
            colours[16][0] = 195.0f / 255.0f;
            colours[12][1] = 202.0f / 255.0f;
            colours[13][1] = 185.0f / 255.0f;
            colours[14][1] = 167.0f / 255.0f;
            colours[15][1] = 167.0f / 255.0f;
            colours[16][1] = 167.0f / 255.0f;
            colours[12][2] = 157.0f / 255.0f;
            colours[13][2] = 130.0f / 255.0f;
            colours[14][2] = 107.0f / 255.0f;
            colours[15][2] = 107.0f / 255.0f;
            colours[16][2] = 107.0f / 255.0f;

            unsigned int bin;
            float alpha, upper_bound, lower_bound;
            for (cnt = 0; cnt < mol->AtomCount(); ++cnt) {
                dist = sqrt(pow(centroid[0] - mol->AtomPositions()[3 * cnt + 0], 2.0f) +
                            pow(centroid[1] - mol->AtomPositions()[3 * cnt + 1], 2.0f) +
                            pow(centroid[2] - mol->AtomPositions()[3 * cnt + 2], 2.0f));
                if (currentColoringMode == HEIGHTMAP_COL) {
                    bin = static_cast<unsigned int>(std::truncf(dist / steps));
                    lower_bound = float(bin) * steps;
                    upper_bound = float(bin + 1) * steps;
                    alpha = (dist - lower_bound) / (upper_bound - lower_bound);
                    atomColorTable[3 * cnt + 0] = (1.0f - alpha) * colours[bin][0] + alpha * colours[bin + 1][0];
                    atomColorTable[3 * cnt + 1] = (1.0f - alpha) * colours[bin][1] + alpha * colours[bin + 1][1];
                    atomColorTable[3 * cnt + 2] = (1.0f - alpha) * colours[bin][2] + alpha * colours[bin + 1][2];
                } else if (currentColoringMode == HEIGHTMAP_VAL) {
                    dist /= max_dist;
                    atomColorTable[3 * cnt + 0] = dist;
                    atomColorTable[3 * cnt + 1] = dist;
                    atomColorTable[3 * cnt + 2] = dist;
                }
            }
        }
    }

    // apply the neighborhood colors
    if (useNeighbors && mol->NeighborhoodSizes() != NULL && mol->Neighborhoods() != NULL) {
        for (unsigned int i = 0; i < mol->AtomCount(); i++) {
            vislib::math::ShallowVector<float, 3> myCol(&atomColorTable[i * 3]);

            unsigned int neighSize = mol->NeighborhoodSizes()[i];
            const unsigned int* neighborhood = mol->Neighborhoods()[i];
            float accum[3] = {0.0f, 0.0f, 0.0f};
            vislib::math::ShallowVector<float, 3> accumVec(accum);
            unsigned int actualSize = 0;

            for (unsigned int j = 0; j < neighSize; j++) {
                if (i != neighborhood[j]) {
                    actualSize++;
                    vislib::math::ShallowVector<float, 3> newCol(&atomColorTable[neighborhood[j] * 3]);
                    accumVec += newCol;
                }
            }

            accumVec /= static_cast<float>(actualSize);
            atomColorTable[i * 3 + 0] = accumVec[0];
            atomColorTable[i * 3 + 1] = accumVec[1];
            atomColorTable[i * 3 + 2] = accumVec[2];
        }
    }
}


/*
 * MakeColorTable routine for molecular data call interpolating between two
 * colors
 */
void Color::MakeColorTable(const megamol::protein_calls::MolecularDataCall* mol, ColoringMode cm0, ColoringMode cm1,
    float weight0, float weight1, vislib::Array<float>& atomColorTable,
    vislib::Array<vislib::math::Vector<float, 3>>& colorLookupTable,
    vislib::Array<vislib::math::Vector<float, 3>>& rainbowColors, vislib::TString minGradColor,
    vislib::TString midGradColor, vislib::TString maxGradColor, bool forceRecompute,
    const protein_calls::BindingSiteCall* bs, bool useNeighbors, const protein_calls::PerAtomFloatCall* pa) {

    // if recomputation is forced: clear current color table
    if (forceRecompute) {
        atomColorTable.Clear();
    }

    // Clamp weights to zero
    if (weight0 < 0.0) weight0 = 0.0;
    if (weight1 < 0.0) weight1 = 0.0;

    // Normalize weights
    weight0 = weight0 / (weight0 + weight1);
    weight1 = weight1 / (weight0 + weight1);

    // only compute color table if necessary
    if (atomColorTable.IsEmpty()) {

        vislib::Array<float> color0;
        vislib::Array<float> color1;

        // reserve memory for all atoms
        atomColorTable.AssertCapacity(mol->AtomCount() * 3);

        // Compute first color table
        Color::MakeColorTable(
            mol, cm0, color0, colorLookupTable, rainbowColors, minGradColor, midGradColor, maxGradColor, true, bs, pa);

        // Compute second color table
        Color::MakeColorTable(
            mol, cm1, color1, colorLookupTable, rainbowColors, minGradColor, midGradColor, maxGradColor, true, bs, pa);

        // Interpolate
        for (unsigned int cnt = 0; cnt < mol->AtomCount() * 3; cnt++) {
            atomColorTable.Add(color0[cnt] * weight0 + color1[cnt] * weight1);
        }
    }

    // apply the neighborhood colors
    if (useNeighbors && mol->NeighborhoodSizes() != NULL && mol->Neighborhoods() != NULL) {
        for (unsigned int i = 0; i < mol->AtomCount(); i++) {
            vislib::math::ShallowVector<float, 3> myCol(&atomColorTable[i * 3]);

            unsigned int neighSize = mol->NeighborhoodSizes()[i];
            const unsigned int* neighborhood = mol->Neighborhoods()[i];
            float accum[3] = {0.0f, 0.0f, 0.0f};
            vislib::math::ShallowVector<float, 3> accumVec(accum);
            unsigned int actualSize = 0;

            for (unsigned int j = 0; j < neighSize; j++) {
                if (i != neighborhood[j]) {
                    actualSize++;
                    vislib::math::ShallowVector<float, 3> newCol(&atomColorTable[neighborhood[j] * 3]);
                    accumVec += newCol;
                }
            }

            accumVec /= static_cast<float>(actualSize);
            atomColorTable[i * 3 + 0] = accumVec[0];
            atomColorTable[i * 3 + 1] = accumVec[1];
            atomColorTable[i * 3 + 2] = accumVec[2];
        }
    }
}

void Color::MakeComparisonColorTable(const megamol::protein_calls::MolecularDataCall* mol1,
    const megamol::protein_calls::MolecularDataCall* mol2, ColoringMode currentColoringMode,
    vislib::Array<float>& atomColorTable, vislib::Array<vislib::math::Vector<float, 3>>& colorLookupTable,
    vislib::Array<vislib::math::Vector<float, 3>>& rainbowColors, vislib::TString minGradColor,
    vislib::TString midGradColor, vislib::TString maxGradColor, bool forceRecompute,
    const protein_calls::BindingSiteCall* bs, const protein_calls::PerAtomFloatCall* pa) {

    // temporary variables
    unsigned int cnt, cnt2, idx, idx2, cntAtom, cntRes, cntSecS, atomIdx, atomIdx2, atomCnt, atomCnt2;
    vislib::math::Vector<float, 3> color;
    // float r, g, b;

    // if recomputation is forced: clear current color table
    if (forceRecompute) {
        atomColorTable.Clear();
    }

    // reserve memory for all atoms
    atomColorTable.AssertCapacity(mol1->AtomCount() * 3);

    // fill table with the default color
    for (cntAtom = 0; cntAtom < mol1->AtomCount(); ++cntAtom) {
        atomColorTable.Add(0.0);
        atomColorTable.Add(0.0);
        atomColorTable.Add(0.0);
    }

    unsigned int ssc = mol1->SecondaryStructureCount();

    if (ssc > mol2->SecondaryStructureCount()) ssc = mol2->SecondaryStructureCount();

    float max = -2.0f;

    // go through every secondary structure
    for (cntSecS = 0; cntSecS < ssc; ++cntSecS) {

        idx = mol1->SecondaryStructures()[cntSecS].FirstAminoAcidIndex();
        idx2 = mol2->SecondaryStructures()[cntSecS].FirstAminoAcidIndex();
        cnt = idx + mol1->SecondaryStructures()[cntSecS].AminoAcidCount();
        cnt2 = idx + mol2->SecondaryStructures()[cntSecS].AminoAcidCount();

        // go through every single aminoacid
        for (cntRes = idx; cntRes < cnt; ++cntRes) {

            atomIdx = mol1->Residues()[cntRes]->FirstAtomIndex();
            atomIdx2 = mol2->Residues()[cntRes]->FirstAtomIndex();
            atomCnt = atomIdx + mol1->Residues()[cntRes]->AtomCount();
            atomCnt2 = atomIdx2 + mol2->Residues()[cntRes]->AtomCount();

            megamol::protein_calls::MolecularDataCall::AminoAcid* aminoacid1;
            megamol::protein_calls::MolecularDataCall::AminoAcid* aminoacid2;

            const megamol::protein_calls::MolecularDataCall::Residue* t1 = mol2->Residues()[cntRes];

            if (t1 == NULL) break;

            if (mol1->Residues()[cntRes]->Identifier() ==
                    megamol::protein_calls::MolecularDataCall::Residue::AMINOACID &&
                mol2->Residues()[cntRes]->Identifier() ==
                    megamol::protein_calls::MolecularDataCall::Residue::AMINOACID) {
                aminoacid1 = (megamol::protein_calls::MolecularDataCall::AminoAcid*)(mol1->Residues()[cntRes]);
                aminoacid2 = (megamol::protein_calls::MolecularDataCall::AminoAcid*)(mol2->Residues()[cntRes]);
            } else { // TODO check if this is correct
                continue;
            }

            unsigned int cAlphaIdx = aminoacid1->CAlphaIndex();
            unsigned int cAlphaIdx2 = aminoacid2->CAlphaIndex();

            float xDist = mol1->AtomPositions()[cAlphaIdx * 3 + 0] - mol2->AtomPositions()[cAlphaIdx2 * 3 + 0];
            float yDist = mol1->AtomPositions()[cAlphaIdx * 3 + 1] - mol2->AtomPositions()[cAlphaIdx2 * 3 + 1];
            float zDist = mol1->AtomPositions()[cAlphaIdx * 3 + 2] - mol2->AtomPositions()[cAlphaIdx2 * 3 + 2];

            float absoluteDist = sqrt(xDist * xDist + yDist * yDist + zDist * zDist);

            if (absoluteDist > max) max = absoluteDist;

            // set the color for every atom
            for (cntAtom = atomIdx; cntAtom < atomCnt; ++cntAtom) {
                atomColorTable[3 * cntAtom + 0] = absoluteDist;
                atomColorTable[3 * cntAtom + 1] = absoluteDist;
                atomColorTable[3 * cntAtom + 2] = absoluteDist;
            }
        }
    }

    // normalize the colors to [0.0..1.0]
    if (max > 0.0000001f) {
        for (cntAtom = 0; cntAtom < mol1->AtomCount(); ++cntAtom) {
            atomColorTable[3 * cntAtom + 0] /= max;
            atomColorTable[3 * cntAtom + 1] /= max;
            atomColorTable[3 * cntAtom + 2] /= max;
        }
    }
}


/*
 * Creates a rainbow color table with 'num' entries.
 */
void Color::MakeRainbowColorTable(unsigned int num, vislib::Array<vislib::math::Vector<float, 3>>& rainbowColors) {

    unsigned int n = (num / 4);
    // the color table should have a minimum size of 16
    if (n < 4) n = 4;
    rainbowColors.Clear();
    rainbowColors.AssertCapacity(num);
    float f = 1.0f / float(n);
    vislib::math::Vector<float, 3> color;
    color.Set(1.0f, 0.0f, 0.0f);
    for (unsigned int i = 0; i < n; i++) {
        color.SetY(vislib::math::Min(color.GetY() + f, 1.0f));
        rainbowColors.Add(color);
    }
    for (unsigned int i = 0; i < n; i++) {
        color.SetX(vislib::math::Max(color.GetX() - f, 0.0f));
        rainbowColors.Add(color);
    }
    for (unsigned int i = 0; i < n; i++) {
        color.SetZ(vislib::math::Min(color.GetZ() + f, 1.0f));
        rainbowColors.Add(color);
    }
    for (unsigned int i = 0; i < n; i++) {
        color.SetY(vislib::math::Max(color.GetY() - f, 0.0f));
        rainbowColors.Add(color);
    }
}


/*
 * Read color table from file.
 */
void Color::ReadColorTableFromFile(
    vislib::StringA filename, vislib::Array<vislib::math::Vector<float, 3>>& colorLookupTable) {

    // file buffer variable
    vislib::sys::ASCIIFileBuffer file;
    // delete old color table
    colorLookupTable.SetCount(0);
    // try to load the color table file
    if (file.LoadFile(filename)) {
        float r, g, b;
        colorLookupTable.AssertCapacity(file.Count());
        // get colors from file
        for (unsigned int cnt = 0; cnt < file.Count(); ++cnt) {
            vislib::StringA lineStr(file.Line(cnt));
            if (lineStr.Length() > 0 && utility::ColourParser::FromString(lineStr, r, g, b)) {
                colorLookupTable.Add(vislib::math::Vector<float, 3>(r, g, b));
            }
        }
    }
    // if the file could not be loaded or contained no valid colors
    if (colorLookupTable.Count() == 0) {
        // set default color table
        colorLookupTable.SetCount(25);
        colorLookupTable[0].Set(0.70f, 0.8f, 0.4f);
        colorLookupTable[1].Set(1.0f, 0.0f, 0.0f);
        colorLookupTable[2].Set(1.0f, 1.0f, 0.0f);
        colorLookupTable[3].Set(0.0f, 1.0f, 0.0f);
        colorLookupTable[4].Set(0.0f, 1.0f, 1.0f);
        colorLookupTable[5].Set(0.0f, 0.0f, 1.0f);
        colorLookupTable[6].Set(1.0f, 0.0f, 1.0f);
        colorLookupTable[7].Set(0.5f, 0.0f, 0.0f);
        colorLookupTable[8].Set(0.5f, 0.5f, 0.0f);
        colorLookupTable[9].Set(0.0f, 0.5f, 0.0f);
        colorLookupTable[10].Set(0.00f, 0.50f, 0.50f);
        colorLookupTable[11].Set(0.00f, 0.00f, 0.50f);
        colorLookupTable[12].Set(0.50f, 0.00f, 0.50f);
        colorLookupTable[13].Set(1.00f, 0.50f, 0.00f);
        colorLookupTable[14].Set(0.00f, 0.50f, 1.00f);
        colorLookupTable[15].Set(1.00f, 0.50f, 1.00f);
        colorLookupTable[16].Set(0.50f, 0.25f, 0.00f);
        colorLookupTable[17].Set(1.00f, 1.00f, 0.50f);
        colorLookupTable[18].Set(0.50f, 1.00f, 0.50f);
        colorLookupTable[19].Set(0.75f, 1.00f, 0.00f);
        colorLookupTable[20].Set(0.50f, 0.00f, 0.75f);
        colorLookupTable[21].Set(1.00f, 0.50f, 0.50f);
        colorLookupTable[22].Set(0.75f, 1.00f, 0.75f);
        colorLookupTable[23].Set(0.75f, 0.75f, 0.50f);
        colorLookupTable[24].Set(1.00f, 0.75f, 0.50f);
    }
}
