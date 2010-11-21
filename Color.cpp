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
 * protein::Color::MakeColorTable
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


