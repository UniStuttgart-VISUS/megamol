/*
 * Color.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "Color.h"
#include <omp.h>
#include "utility/ColourParser.h"
#include "vislib/ASCIIFileBuffer.h"
#include "CoreInstance.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * FillAminoAcidColorTable
 */
void Color::FillAminoAcidColorTable(
    vislib::Array<vislib::math::Vector<unsigned char, 3> >
      &aminoAcidColorTable) {

    aminoAcidColorTable.Clear();
    aminoAcidColorTable.SetCount( 25);
    aminoAcidColorTable[0].Set( 128, 128, 128);
    aminoAcidColorTable[1].Set( 255, 0, 0);
    aminoAcidColorTable[2].Set( 255, 255, 0);
    aminoAcidColorTable[3].Set( 0, 255, 0);
    aminoAcidColorTable[4].Set( 0, 255, 255);
    aminoAcidColorTable[5].Set( 0, 0, 255);
    aminoAcidColorTable[6].Set( 255, 0, 255);
    aminoAcidColorTable[7].Set( 128, 0, 0);
    aminoAcidColorTable[8].Set( 128, 128, 0);
    aminoAcidColorTable[9].Set( 0, 128, 0);
    aminoAcidColorTable[10].Set( 0, 128, 128);
    aminoAcidColorTable[11].Set( 0, 0, 128);
    aminoAcidColorTable[12].Set( 128, 0, 128);
    aminoAcidColorTable[13].Set( 255, 128, 0);
    aminoAcidColorTable[14].Set( 0, 128, 255);
    aminoAcidColorTable[15].Set( 255, 128, 255);
    aminoAcidColorTable[16].Set( 128, 64, 0);
    aminoAcidColorTable[17].Set( 255, 255, 128);
    aminoAcidColorTable[18].Set( 128, 255, 128);
    aminoAcidColorTable[19].Set( 192, 255, 0);
    aminoAcidColorTable[20].Set( 128, 0, 192);
    aminoAcidColorTable[21].Set( 255, 128, 128);
    aminoAcidColorTable[22].Set( 192, 255, 192);
    aminoAcidColorTable[23].Set( 192, 192, 128);
    aminoAcidColorTable[24].Set( 255, 192, 128);
}


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
 * MakeColorTable
 */
void Color::MakeColorTable(const MolecularDataCall *mol,
        vislib::TString minGradColor,
        vislib::TString midGradColor,
        vislib::TString maxGradColor,
        ColoringMode currentColoringMode,
        vislib::Array<vislib::math::Vector<float, 3> > &atomColorTable,
        vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
        vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
        bool forceRecompute) {

    // temporary variables
    unsigned int cnt, idx, cntAtom, cntRes, cntChain, cntMol, cntSecS;
    unsigned int atomIdx, atomCnt;
    vislib::math::Vector<float, 3> color;
    float r, g, b;

    // if recomputation is forced: clear current color table
    if( forceRecompute ) {
        atomColorTable.Clear();
    }
    // reserve memory for all atoms
    atomColorTable.AssertCapacity( mol->AtomCount() * 3 );

    // only compute color table if necessary
    if( atomColorTable.IsEmpty() ) {
        if( currentColoringMode == ELEMENT ) {
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                color.SetX( float( mol->AtomTypes()[mol->
                  AtomTypeIndices()[cnt]].Colour()[0]) / 255.0f );
                color.SetY( float( mol->AtomTypes()[mol->
                  AtomTypeIndices()[cnt]].Colour()[1]) / 255.0f );
                color.SetZ( float( mol->AtomTypes()[mol->
                  AtomTypeIndices()[cnt]].Colour()[2]) / 255.0f );
                atomColorTable.Add( color);
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
                    atomColorTable.Add( colorLookupTable[resTypeIdx
                      %colorLookupTable.Count()]);
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
            utility::ColourParser::FromString( "#ffffff", r, g, b);
            vislib::math::Vector<float, 3> colRCoil( r, g, b);
            // loop over all atoms and fill the table with the default color
            for( cntAtom = 0; cntAtom < mol->AtomCount(); ++cntAtom ) {
                atomColorTable.Add( colNone);
            }
            // write colors for sec structure elements
            MolecularDataCall::SecStructure::ElementType elemType;
            for( cntSecS = 0; cntSecS < mol->SecondaryStructureCount();
              ++cntSecS ) {
                idx = mol->SecondaryStructures()[cntSecS].FirstAminoAcidIndex();
                cnt = idx +mol->SecondaryStructures()[cntSecS].AminoAcidCount();
                elemType = mol->SecondaryStructures()[cntSecS].Type();
                for( cntRes = idx; cntRes < cnt; ++cntRes ) {
                    atomIdx = mol->Residues()[cntRes]->FirstAtomIndex();
                    atomCnt = atomIdx + mol->Residues()[cntRes]->AtomCount();
                    for( cntAtom = atomIdx; cntAtom < atomCnt; ++cntAtom ) {
                        if( elemType ==
                          MolecularDataCall::SecStructure::TYPE_HELIX ) {
                            atomColorTable[cntAtom] = colHelix;
                        } else if( elemType ==
                          MolecularDataCall::SecStructure::TYPE_SHEET ) {
                            atomColorTable[cntAtom] = colSheet;
                        } else if( elemType ==
                          MolecularDataCall::SecStructure::TYPE_COIL ) {
                            atomColorTable[cntAtom] = colRCoil;
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

            float min( mol->MinimumBFactor());
            float max( mol->MaximumBFactor());
            float mid( ( max - min)/2.0f + min );
            float val;

            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min == max ) {
                    atomColorTable.Add( colMid);
                    continue;
                }

                val = mol->AtomBFactors()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min) )
                      * ( val - min );
                    atomColorTable.Add( col);
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max - mid) )
                      * ( val - mid );
                    atomColorTable.Add( col);
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add( colMid);
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

            float min( mol->MinimumCharge());
            float max( mol->MaximumCharge());
            float mid( ( max - min)/2.0f + min );
            float val;

            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min == max ) {
                    atomColorTable.Add( colMid);
                    continue;
                }

                val = mol->AtomCharges()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min) )
                      * ( val - min );
                    atomColorTable.Add( col);
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max - mid) )
                      * ( val - mid );
                    atomColorTable.Add( col);
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add( colMid);
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

            float min( mol->MinimumOccupancy());
            float max( mol->MaximumOccupancy());
            float mid( ( max - min)/2.0f + min );
            float val;

            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min == max ) {
                    atomColorTable.Add( colMid);
                    continue;
                }

                val = mol->AtomOccupancies()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min) )
                      * ( val - min );
                    atomColorTable.Add( col);
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max - mid) )
                      * ( val - mid );
                    atomColorTable.Add( col);
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add( colMid);
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
                atomColorTable.Add( color);
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
                atomColorTable.Add( color);
            }
        } // ... END coloring mode MOLECULE
        else if( currentColoringMode == RAINBOW ) {
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                idx = int( ( float( cnt) / float( mol->AtomCount()))
                        * float( rainbowColors.Count()));
                color = rainbowColors[idx];
                atomColorTable.Add( color);
            }
        } // ... END coloring mode RAINBOW
    }
}


/*
 * MakeColorTable
 */
void Color::MakeColorTable( const CallProteinData *prot,
    ColoringMode currentColoringMode,
    vislib::Array<unsigned char> &protAtomColorTable,
    vislib::Array<vislib::math::Vector<unsigned char, 3> > &aminoAcidColorTable,
    std::vector<vislib::math::Vector<float,3> > &rainbowColors,
    bool forceRecompute) {

    unsigned int i;
    unsigned int currentChain, currentAminoAcid, currentAtom, currentSecStruct;
    unsigned int cntCha, cntRes, cntAto;
    protein::CallProteinData::Chain chain;
    vislib::math::Vector<float, 3> color;
    // if recomputation is forced: clear current color table
    if( forceRecompute )
    {
        protAtomColorTable.Clear();
    }
    // reserve memory for all atoms
    protAtomColorTable.AssertCapacity( prot->ProteinAtomCount() );
    // only compute color table if necessary
    if( protAtomColorTable.IsEmpty() )
    {
        if( currentColoringMode == ELEMENT )
        {
            for( i = 0; i < prot->ProteinAtomCount(); i++ )
            {
                protAtomColorTable.Add( prot->AtomTypes()[prot->ProteinAtomData()[i].TypeIndex()].Colour()[0]);
                protAtomColorTable.Add( prot->AtomTypes()[prot->ProteinAtomData()[i].TypeIndex()].Colour()[1]);
                protAtomColorTable.Add( prot->AtomTypes()[prot->ProteinAtomData()[i].TypeIndex()].Colour()[2]);
            }
        } // ... END coloring mode ELEMENT
        else if( currentColoringMode == AMINOACID )
        {
            // loop over all chains
            for( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
            {
                chain = prot->ProteinChain( currentChain);
                // loop over all amino acids in the current chain
                for( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
                {
                    // loop over all connections of the current amino acid
                    for( currentAtom = 0;
                         currentAtom < chain.AminoAcid()[currentAminoAcid].AtomCount();
                         currentAtom++ )
                    {
                        i = chain.AminoAcid()[currentAminoAcid].NameIndex()+1;
                        i = i % (unsigned int)(aminoAcidColorTable.Count());
                        protAtomColorTable.Add(
                            aminoAcidColorTable[i].GetX() );
                        protAtomColorTable.Add(
                            aminoAcidColorTable[i].GetY() );
                        protAtomColorTable.Add(
                            aminoAcidColorTable[i].GetZ() );
                    }
                }
            }
        } // ... END coloring mode AMINOACID
        else if( currentColoringMode == STRUCTURE )
        {
            // loop over all chains
            for( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
            {
                chain = prot->ProteinChain( currentChain);
                // loop over all secondary structure elements in this chain
                for( currentSecStruct = 0;
                     currentSecStruct < chain.SecondaryStructureCount();
                     currentSecStruct++ )
                {
                    i = chain.SecondaryStructure()[currentSecStruct].AtomCount();
                    // loop over all atoms in this secondary structure element
                    for( currentAtom = 0; currentAtom < i; currentAtom++ )
                    {
                        if( chain.SecondaryStructure()[currentSecStruct].Type() ==
                            protein::CallProteinData::SecStructure::TYPE_HELIX )
                        {
                            protAtomColorTable.Add( 255);
                            protAtomColorTable.Add( 0);
                            protAtomColorTable.Add( 0);
                        }
                        else if( chain.SecondaryStructure()[currentSecStruct].Type() ==
                            protein::CallProteinData::SecStructure::TYPE_SHEET )
                        {
                            protAtomColorTable.Add( 0);
                            protAtomColorTable.Add( 0);
                            protAtomColorTable.Add( 255);
                        }
                        else if( chain.SecondaryStructure()[currentSecStruct].Type() ==
                            protein::CallProteinData::SecStructure::TYPE_TURN )
                        {
                            protAtomColorTable.Add( 255);
                            protAtomColorTable.Add( 255);
                            protAtomColorTable.Add( 0);
                        }
                        else
                        {
                            protAtomColorTable.Add( 230);
                            protAtomColorTable.Add( 230);
                            protAtomColorTable.Add( 230);
                        }
                    }
                }
            }
            // add missing atom colors
            if ( prot->ProteinAtomCount() > ( protAtomColorTable.Count() / 3 ) )
            {
                currentAtom = protAtomColorTable.Count() / 3;
                for ( ; currentAtom < prot->ProteinAtomCount(); ++currentAtom )
                {
                    protAtomColorTable.Add ( 200 );
                    protAtomColorTable.Add ( 200 );
                    protAtomColorTable.Add ( 200 );
        }
            }
        } // ... END coloring mode STRUCTURE
        else if( currentColoringMode == VALUE )
        {
            //vislib::math::Vector<int, 3> colMax( 255,   0,   0);
            //vislib::math::Vector<int, 3> colMid( 255, 255, 255);
            //vislib::math::Vector<int, 3> colMin(   0,   0, 255);
            vislib::math::Vector<float, 3> colMax( 250,  94,  82);
            vislib::math::Vector<float, 3> colMid( 250, 250, 250);
            vislib::math::Vector<float, 3> colMin(  37, 136, 195);
            vislib::math::Vector<float, 3> col;

            float min( prot->MinimumTemperatureFactor() );
            float max( prot->MaximumTemperatureFactor() );
            float mid( ( max - min)/2.0f + min );
            float val;

            for ( i = 0; i < prot->ProteinAtomCount(); i++ )
            {
                if( min == max )
                {
                    protAtomColorTable.Add( colMid.GetX() );
                    protAtomColorTable.Add( colMid.GetY() );
                    protAtomColorTable.Add( colMid.GetZ() );
                    continue;
                }

                val = prot->ProteinAtomData()[i].TempFactor();
                // below middle value --> blend between min and mid color
                if( val < mid )
                {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min) ) * ( val - min );
                    protAtomColorTable.Add( col.GetX() );
                    protAtomColorTable.Add( col.GetY() );
                    protAtomColorTable.Add( col.GetZ() );
                }
                // above middle value --> blend between max and mid color
                else if( val > mid )
                {
                    col = colMid + ( ( colMax - colMid ) / ( max - mid) ) * ( val - mid );
                    protAtomColorTable.Add( col.GetX() );
                    protAtomColorTable.Add( col.GetY() );
                    protAtomColorTable.Add( col.GetZ() );
        }
                // middle value --> assign mid color
                else
                {
                    protAtomColorTable.Add( colMid.GetX() );
                    protAtomColorTable.Add( colMid.GetY() );
                    protAtomColorTable.Add( colMid.GetZ() );
                }
            }
        } // ... END coloring mode VALUE
        else if( currentColoringMode == CHAIN_ID )
        {
            // loop over all chains
            for( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
            {
                chain = prot->ProteinChain( currentChain);
                // loop over all amino acids in the current chain
                for( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
                {
                    // loop over all connections of the current amino acid
                    for( currentAtom = 0;
                         currentAtom < chain.AminoAcid()[currentAminoAcid].AtomCount();
                         currentAtom++ )
                    {
                        i = (currentChain + 1) % (unsigned int)(aminoAcidColorTable.Count());
                        protAtomColorTable.Add(
                            aminoAcidColorTable[i].GetX() );
                        protAtomColorTable.Add(
                            aminoAcidColorTable[i].GetY() );
                        protAtomColorTable.Add(
                            aminoAcidColorTable[i].GetZ() );
                    }
                }
            }
        } // ... END coloring mode CHAIN_ID
        else if( currentColoringMode == RAINBOW )
        {
            for( cntCha = 0; cntCha < prot->ProteinChainCount(); ++cntCha )
            {
                for( cntRes = 0; cntRes < prot->ProteinChain( cntCha).AminoAcidCount(); ++cntRes )
                {
                    i = int( ( float( cntRes) / float( prot->ProteinChain( cntCha).AminoAcidCount() ) ) * float( rainbowColors.size() ) );
                    color = rainbowColors[i];
                    for( cntAto = 0;
                         cntAto < prot->ProteinChain( cntCha).AminoAcid()[cntRes].AtomCount();
                         ++cntAto )
                    {
                        protAtomColorTable.Add( int(color.GetX() * 255.0f) );
                        protAtomColorTable.Add( int(color.GetY() * 255.0f) );
                        protAtomColorTable.Add( int(color.GetZ() * 255.0f) );
                    }
                }
            }
        } // ... END coloring mode RAINBOW
        else if ( currentColoringMode == CHARGE )
        {
            vislib::math::Vector<int, 3> colMax( 255,   0,   0);
            vislib::math::Vector<int, 3> colMid( 255, 255, 255);
            vislib::math::Vector<int, 3> colMin(   0,   0, 255);
            vislib::math::Vector<int, 3> col;

            float min( prot->MinimumCharge() );
            float max( prot->MaximumCharge() );
            float mid( ( max - min)/2.0f + min );
            float charge;

            for ( i = 0; i < prot->ProteinAtomCount(); i++ ) {
                if( min == max )
                {
                    protAtomColorTable.Add( colMid.GetX() );
                    protAtomColorTable.Add( colMid.GetY() );
                    protAtomColorTable.Add( colMid.GetZ() );
                    continue;
                }

                charge = prot->ProteinAtomData()[i].Charge();
                // below middle value --> blend between min and mid color
                if( charge < mid )
                {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min) ) * ( charge - min );
                    protAtomColorTable.Add( col.GetX() );
                    protAtomColorTable.Add( col.GetY() );
                    protAtomColorTable.Add( col.GetZ() );
                }
                // above middle value --> blend between max and mid color
                else if( charge > mid )
                {
                    col = colMid + ( ( colMax - colMid ) / ( max - mid) ) * ( charge - mid );
                    protAtomColorTable.Add( col.GetX() );
                    protAtomColorTable.Add( col.GetY() );
                    protAtomColorTable.Add( col.GetZ() );
                }
                // middle value --> assign mid color
                else
                {
                    protAtomColorTable.Add( colMid.GetX() );
                    protAtomColorTable.Add( colMid.GetY() );
                    protAtomColorTable.Add( colMid.GetZ() );
                }
            }
        } // ... END coloring mode CHARGE
    }
}


/*
 * Make color table for all atoms
 */
void Color::MakeColorTable( const CallProteinData *prot,
    ColoringMode currentColoringMode,
    vislib::math::Vector<float, 3> minValueColor,
    vislib::math::Vector<float, 3> meanValueColor,
    vislib::math::Vector<float, 3> maxValueColor,
    std::vector<vislib::math::Vector<float, 3> > &atomColor,
    vislib::Array<vislib::math::Vector<float, 3> > &aminoAcidColorTable,
    std::vector<vislib::math::Vector<float,3> > &rainbowColors,
    bool forceRecompute) {

    unsigned int i;
    unsigned int currentChain, currentAminoAcid, currentAtom, currentSecStruct;
    unsigned int cntCha, cntRes, cntAto;
    CallProteinData::Chain chain;
    vislib::math::Vector<float, 3> color;
    // if recomputation is forced: clear current color table
    if( forceRecompute )
        atomColor.clear();
    // only compute color table if necessary
    if( atomColor.empty() ) {
        atomColor.reserve( prot->ProteinAtomCount());
        if( currentColoringMode == ELEMENT ) {
            // resize atom color table
            atomColor.resize( prot->ProteinAtomCount());
            for( i = 0; i < prot->ProteinAtomCount(); i++ ) {
                atomColor[i].SetX( float(prot->AtomTypes()[prot->ProteinAtomData()[i].TypeIndex()].Colour()[0]) / 255.0f);
                atomColor[i].SetY( float(prot->AtomTypes()[prot->ProteinAtomData()[i].TypeIndex()].Colour()[1]) / 255.0f);
                atomColor[i].SetZ( float(prot->AtomTypes()[prot->ProteinAtomData()[i].TypeIndex()].Colour()[2]) / 255.0f);
            }
        } // ===== END coloring mode ELEMENT =====
        else if( currentColoringMode == AMINOACID ) {
            // loop over all chains
            for( currentChain = 0; currentChain < prot->ProteinChainCount();
              currentChain++ ) {
                chain = prot->ProteinChain( currentChain);
                // loop over all amino acids in the current chain
                for( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
                {
                    // loop over all connections of the current amino acid
                    for( currentAtom = 0;
                         currentAtom < chain.AminoAcid()[currentAminoAcid].AtomCount();
                         currentAtom++ )
                    {
                        i = chain.AminoAcid()[currentAminoAcid].NameIndex()+1;
                        i = i % (unsigned int)(aminoAcidColorTable.Count());
                        atomColor.push_back( aminoAcidColorTable[i]);
                    }
                }
            }
        } // ===== END coloring mode AMINOACID =====
        else if( currentColoringMode == STRUCTURE ) {
            // loop over all chains
            for( currentChain = 0; currentChain < prot->ProteinChainCount();
              currentChain++ ) {
                    chain = prot->ProteinChain( currentChain);
                    // loop over all secondary structure elements in this chain
                    for( currentSecStruct = 0;
                        currentSecStruct < chain.SecondaryStructureCount();
                        currentSecStruct++ ) {
                            i = chain.SecondaryStructure()[currentSecStruct].AtomCount();
                            // loop over all atoms in this secondary structure element
                            for( currentAtom = 0; currentAtom < i; currentAtom++ )
                            {
                                    if( chain.SecondaryStructure()[currentSecStruct].Type() ==
                                            protein::CallProteinData::SecStructure::TYPE_HELIX )
                                    {
                                            atomColor.push_back( vislib::math::Vector<float, 3>( 1.0f, 0.0f, 0.0f));
                                    }
                                    else if( chain.SecondaryStructure()[currentSecStruct].Type() ==
                                            protein::CallProteinData::SecStructure::TYPE_SHEET )
                                    {
                                            atomColor.push_back( vislib::math::Vector<float, 3>( 0.0f, 0.0f, 1.0f));
                                    }
                                    else if( chain.SecondaryStructure()[currentSecStruct].Type() ==
                                            protein::CallProteinData::SecStructure::TYPE_TURN )
                                    {
                                            atomColor.push_back( vislib::math::Vector<float, 3>( 1.0f, 1.0f, 0.0f));
                                    }
                                    else
                                    {
                                            atomColor.push_back( vislib::math::Vector<float, 3>( 0.9f, 0.9f, 0.9f));
                                    }
                            }
                    }
                }
                // add missing atom colors
                if ( prot->ProteinAtomCount() > atomColor.size() )
                {
                        currentAtom = (unsigned int)atomColor.size();
                        for ( ; currentAtom < prot->ProteinAtomCount(); ++currentAtom )
                        {
                                atomColor.push_back( vislib::math::Vector<float, 3>( 0.3f, 0.9f, 0.3f));
                        }
                }
        } // ===== END coloring mode STRUCTURE =====
        else if( currentColoringMode == CHAIN_ID )
        {
                // loop over all chains
                for( currentChain = 0; currentChain < prot->ProteinChainCount(); currentChain++ )
                {
                        chain = prot->ProteinChain( currentChain);
                        // loop over all amino acids in the current chain
                        for( currentAminoAcid = 0; currentAminoAcid < chain.AminoAcidCount(); currentAminoAcid++ )
                        {
                                // loop over all connections of the current amino acid
                                for( currentAtom = 0;
                                         currentAtom < chain.AminoAcid()[currentAminoAcid].AtomCount();
                                         currentAtom++ )
                                {
                                        i = (currentChain + 1) % (unsigned int)(aminoAcidColorTable.Count());
                                        atomColor.push_back( aminoAcidColorTable[i]);
                                }
                        }
                }
        } // ===== END coloring mode CHAIN_ID =====
        else if( currentColoringMode == VALUE )
        {
                vislib::math::Vector<float, 3> colMax = maxValueColor;
                vislib::math::Vector<float, 3> colMid = meanValueColor;
                vislib::math::Vector<float, 3> colMin = minValueColor;

                float min( prot->MinimumTemperatureFactor() );
                float max( prot->MaximumTemperatureFactor() );
                float mid( ( max - min)/2.0f + min );
                float val;

                for ( i = 0; i < prot->ProteinAtomCount(); i++ )
                {
                        if( min == max )
                        {
                                atomColor.push_back( colMid);
                                continue;
                        }

                        val = prot->ProteinAtomData()[i].TempFactor();
                        // below middle value --> blend between min and mid color
                        if( val < mid )
                        {
                                color = colMin + ( ( colMid - colMin ) / ( mid - min) ) * ( val - min );
                                atomColor.push_back( color);
                        }
                        // above middle value --> blend between max and mid color
                        else if( val > mid )
                        {
                                color = colMid + ( ( colMax - colMid ) / ( max - mid) ) * ( val - mid );
                                atomColor.push_back( color);
                        }
                        // middle value --> assign mid color
                        else
                        {
                                atomColor.push_back( colMid);
                        }
                }
        } // ===== END coloring mode VALUE =====
        else if( currentColoringMode == RAINBOW )
        {
                for( cntCha = 0; cntCha < prot->ProteinChainCount(); ++cntCha )
                {
                        for( cntRes = 0; cntRes < prot->ProteinChain( cntCha).AminoAcidCount(); ++cntRes )
                        {
                                i = int( ( float( cntRes) / float( prot->ProteinChain( cntCha).AminoAcidCount() ) ) * float( rainbowColors.size() ) ) % rainbowColors.size();
                                color = rainbowColors[i];
                                for( cntAto = 0;
                                     cntAto < prot->ProteinChain( cntCha).AminoAcid()[cntRes].AtomCount();
                                     ++cntAto )
                                {
                                        atomColor.push_back( color);
                                }
                        }
                }
        } // ===== END coloring mode RAINBOW =====
        else if ( currentColoringMode == CHARGE )
        {
                vislib::math::Vector<float, 3> colMax = maxValueColor;
                vislib::math::Vector<float, 3> colMid = meanValueColor;
                vislib::math::Vector<float, 3> colMin = minValueColor;

                float min( prot->MinimumCharge() );
                float max( prot->MaximumCharge() );
                float mid( ( max - min)/2.0f + min );
                float charge;

                for ( i = 0; i < prot->ProteinAtomCount(); i++ )
                {
                        if( min == max )
                        {
                                atomColor.push_back( colMid);
                                continue;
                        }

                        charge = prot->ProteinAtomData()[i].Charge();
                        // below middle value --> blend between min and mid color
                        if( charge < mid )
                        {
                                color = colMin + ( ( colMid - colMin ) / ( mid - min) ) * ( charge - min );
                                atomColor.push_back( color);
                        }
                        // above middle value --> blend between max and mid color
                        else if( charge > mid )
                        {
                                color = colMid + ( ( colMax - colMid ) / ( max - mid) ) * ( charge - mid );
                                atomColor.push_back( color);
                        }
                        // middle value --> assign mid color
                        else
                        {
                                atomColor.push_back( colMid);
                        }
                }
        } // ===== END coloring mode CHARGE =====
        else if ( currentColoringMode == OCCUPANCY )
        {
                vislib::math::Vector<float, 3> colMax = maxValueColor;
                vislib::math::Vector<float, 3> colMid = meanValueColor;
                vislib::math::Vector<float, 3> colMin = minValueColor;

                float min( prot->MinimumOccupancy() );
                float max( prot->MaximumOccupancy() );
                float mid( ( max - min)/2.0f + min );
                float charge;

                for ( i = 0; i < prot->ProteinAtomCount(); i++ )
                {
                        if( min == max )
                        {
                                atomColor.push_back( colMid);
                                continue;
                        }

                        charge = prot->ProteinAtomData()[i].Occupancy();
                        // below middle value --> blend between min and mid color
                        if( charge < mid )
                        {
                                color = colMin + ( ( colMid - colMin ) / ( mid - min) ) * ( charge - min );
                                atomColor.push_back( color);
                        }
                        // above middle value --> blend between max and mid color
                        else if( charge > mid )
                        {
                                color = colMid + ( ( colMax - colMid ) / ( max - mid) ) * ( charge - mid );
                                atomColor.push_back( color);
                        }
                        // middle value --> assign mid color
                        else
                        {
                                atomColor.push_back( colMid);
                        }
                }
        } // ===== END coloring mode OCCUPANCY =====
    else {// default case (wrong parameter):
                // resize atom color table
                atomColor.resize( prot->ProteinAtomCount());
                for( i = 0; i < prot->ProteinAtomCount(); i++ )
                {
                        atomColor[i].SetX( 1.0f);
                        atomColor[i].SetY( 1.0f);
                        atomColor[i].SetZ( 1.0f);
                }
        } // ===== END uniform color =====
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
 * Creates a rainbow color table with 'num' entries.
 */
void Color::MakeRainbowColorTable( unsigned int num,
    std::vector<vislib::math::Vector<float,3> > &rainbowColors) {

{
    unsigned int n = (num/4);
    // the color table should have a minimum size of 16
    if( n < 4 )
        n = 4;
    rainbowColors.clear();
    float f = 1.0f/float(n);
    vislib::math::Vector<float,3> color;
    color.Set( 1.0f, 0.0f, 0.0f);
    for( unsigned int i = 0; i < n; i++)
    {
        color.SetY( vislib::math::Min( color.GetY() + f, 1.0f));
        rainbowColors.push_back( color);
    }
    for( unsigned int i = 0; i < n; i++)
    {
        color.SetX( vislib::math::Max( color.GetX() - f, 0.0f));
        rainbowColors.push_back( color);
    }
    for( unsigned int i = 0; i < n; i++)
    {
        color.SetZ( vislib::math::Min( color.GetZ() + f, 1.0f));
        rainbowColors.push_back( color);
    }
    for( unsigned int i = 0; i < n; i++)
    {
        color.SetY( vislib::math::Max( color.GetY() - f, 0.0f));
        rainbowColors.push_back( color);
    }
}
}


/*
 * Read color table from file
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
            if( utility::ColourParser::FromString( vislib::StringA(
              file.Line( cnt)), r, g, b) ) {
                colorLookupTable.Add( vislib::math::Vector<float, 3>
                  ( r, g, b));
            }
        }
    }
    // if the file could not be loaded or contained no valid colors
    if( colorLookupTable.Count() == 0 ) {
        // set default color table
        colorLookupTable.SetCount( 25);
        colorLookupTable[0].Set( 0.5f, 0.5f, 0.5f);
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


