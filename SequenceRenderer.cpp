#include "stdafx.h"
#include "SequenceRenderer.h"

#include "SequenceRenderer.h"
#include "CoreInstance.h"
#include "param/IntParam.h"
#include "param/FilePathParam.h"
#include "utility/ColourParser.h"
#include "vislib/Rectangle.h"
#include "vislib/BufferedFile.h"
#include "vislib/sysfunctions.h"
#include "glh/glh_extensions.h"
#include <GL/glu.h>
#include <math.h>
#include "misc/ImageViewer.h"
#include "utility/ResourceWrapper.h"
#include "Color.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace vislib::graphics::gl;
using vislib::sys::Log;

/*
 * SequenceRenderer::SequenceRenderer (CTOR)
 */
SequenceRenderer::SequenceRenderer( void ) : Renderer2DModule (),
        dataCallerSlot( "getData", "Connects the sequence diagram rendering with data storage." ),
        bindingSiteCallerSlot( "getBindingSites", "Connects the sequence diagram rendering with binding site storage." ),
        resCountPerRowParam( "ResiduesPerRow", "The number of residues per row" ),
        colorTableFileParam( "ColorTableFilename", "The filename of the color table."),
        dataPrepared(false), atomCount(0), bindingSiteCount(0), resCount(0), resCols(0), resRows(0), rowHeight( 3.0f)
#ifndef USE_SIMPLE_FONT
        , theFont(FontInfo_Verdana) 
#endif
        {

    // molecular data caller slot
    this->dataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    // binding site data caller slot
    this->bindingSiteCallerSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable(&this->bindingSiteCallerSlot);

    // param slot for number of residues per row
    this->resCountPerRowParam.SetParameter( new param::IntParam( 50, 1));
    this->MakeSlotAvailable( &this->resCountPerRowParam);
    
    // fill color table with default values and set the filename param
    vislib::StringA filename( "colors.txt");
    this->colorTableFileParam.SetParameter(new param::FilePathParam( A2T( filename)));
    this->MakeSlotAvailable( &this->colorTableFileParam);
    Color::ReadColorTableFromFile( T2A(this->colorTableFileParam.Param<param::FilePathParam>()->Value()), this->colorTable);

}

/*
 * SequenceRenderer::~SequenceRenderer (DTOR)
 */
SequenceRenderer::~SequenceRenderer( void ) {
    this->Release();
}

/*
 * SequenceRenderer::create
 */
bool SequenceRenderer::create() {

    return true;
}

/*
 * SequenceRenderer::release
 */
void SequenceRenderer::release() {
}

bool SequenceRenderer::GetExtents(view::CallRender2D& call) {
    // check molecular data
    MolecularDataCall *mol = this->dataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL ) return false;
    if (!(*mol)(MolecularDataCall::CallForGetData)) return false;
    
    // get pointer to BindingSiteCall
    BindingSiteCall *bs = this->bindingSiteCallerSlot.CallAs<BindingSiteCall>();
    if( bs != NULL ) {
        // execute the call
        if( !(*bs)(BindingSiteCall::CallForGetData) ) {
            bs = NULL;
        }
    }
    
    // check whether the number of residues per row or the number of atoms was changed
    if( this->resCountPerRowParam.IsDirty() ||
        this->atomCount != mol->AtomCount() ) {
        this->dataPrepared = false;
    }

    // check whether the number of binding sites has changed
    if( bs && this->bindingSiteCount != bs->GetBindingSiteCount() ) {
        this->dataPrepared = false;
    }

    // prepare the data
    if( !this->dataPrepared ) {
        unsigned int oldRowHeight = this->rowHeight;
        this->dataPrepared = this->PrepareData( mol, bs);
        if( oldRowHeight != this->rowHeight ) {
            this->dataPrepared = this->PrepareData( mol, bs);
        }

        if( this->dataPrepared ) {
            // the data has been prepared for the current number of residues per row
            this->resCountPerRowParam.ResetDirty();
            // store the number of atoms for which the data was prepared
            this->atomCount = mol->AtomCount();
            //store the number of binding sites
            bs ? this->bindingSiteCount = bs->GetBindingSiteCount() : this->bindingSiteCount = 0;
        }
    }
        
    // set the bounding box
    //call.SetBoundingBox( 0.0f, 0.0f, static_cast<float>(this->resCols), static_cast<float>(this->resRows));
    call.SetBoundingBox( 0.0f, -static_cast<float>(this->resRows) * this->rowHeight, static_cast<float>(this->resCols), 0.0f);

    return true;
}


/*
 * SequenceRenderer::Render
 */
bool SequenceRenderer::Render(view::CallRender2D &call) {
    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->dataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL ) return false;
    // execute the call
    if( !(*mol)(MolecularDataCall::CallForGetData) ) return false;
    
    // read and update the color table, if necessary
    if( this->colorTableFileParam.IsDirty() ) {
        Color::ReadColorTableFromFile( T2A(this->colorTableFileParam.Param<param::FilePathParam>()->Value()), this->colorTable);
        this->colorTableFileParam.ResetDirty();
    }

    if( this->dataPrepared ) {
        //draw tiles for structure
        glBegin( GL_QUADS);
        for( unsigned int i = 0; i < this->resIndex.Count(); i++ ) {
            if( this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_HELIX )
                glColor3f( 1.0f, 0.0f, 0.0f);
            else if( this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_SHEET )
                glColor3f( 0.0f, 0.0f, 1.0f);
            else if( this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_TURN )
                glColor3f( 1.0f, 1.0f, 0.0f);
            else
                glColor3f( 0.5f, 0.5f, 0.5f);
            glVertex2f( this->vertices[2*i]       , -this->vertices[2*i+1]);
            glVertex2f( this->vertices[2*i]       , -this->vertices[2*i+1] - 1.0f);
            glVertex2f( this->vertices[2*i] + 1.0f, -this->vertices[2*i+1] - 1.0f);
            glVertex2f( this->vertices[2*i] + 1.0f, -this->vertices[2*i+1]);
        }
        glEnd();
        // draw tiles for binding sites
        glBegin( GL_QUADS);
        for( unsigned int i = 0; i < this->bsIndices.Count(); i++ ) {
            glColor3fv( this->colorTable[this->bsIndices[i]].PeekComponents());
            glVertex2f( this->bsVertices[2*i] + 0.1f, -this->bsVertices[2*i+1]);
            glVertex2f( this->bsVertices[2*i] + 0.1f, -this->bsVertices[2*i+1] - 0.4f);
            glVertex2f( this->bsVertices[2*i] + 0.9f, -this->bsVertices[2*i+1] - 0.4f);
            glVertex2f( this->bsVertices[2*i] + 0.9f, -this->bsVertices[2*i+1]);
        }
        glEnd();
    
        // set test color (inverse background color)
        float bgColor[4];
        float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
        glGetFloatv( GL_COLOR_CLEAR_VALUE, bgColor);
        for( unsigned int i = 0; i < 4; i++ ) {
            fgColor[i] -= bgColor[i];
        }
        
        // labeling
        glColor3fv( fgColor);
        if( theFont.Initialise() ) {
            for( unsigned int i = 0; i < this->aminoAcidStrings.Count(); i++ ) {
                for( unsigned int j = 0; j < this->aminoAcidStrings[i].Length(); j++ ) {
                    // draw the one-letter amino acid code
                    theFont.DrawString( static_cast<float>(j), -( static_cast<float>(i) * this->rowHeight), 1.0f, -1.0f,
                        1.0f, true, this->aminoAcidStrings[i].Substring(j, 1), vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);
                    // draw the chain name and amino acid index
                    theFont.DrawString( static_cast<float>(j), -( static_cast<float>(i) * this->rowHeight + this->rowHeight - 0.5f), 1.0f, 0.5f,
                        0.35f, true, this->aminoAcidIndexStrings[i][j], vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);
                }
            }
        }
                
    } // dataPrepared
    
    return true;
}


/*
 * Callback for mouse events (move, press, and release)
 */
bool SequenceRenderer::MouseEvent(float x, float y, view::MouseFlags flags) {
    bool consumeEvent = false;
    
    return consumeEvent;
}


/*
 *
 */
bool SequenceRenderer::PrepareData( MolecularDataCall *mol, BindingSiteCall *bs) {
    if( !mol ) return false;

    // temporary variables
    unsigned int firstStruct;
    unsigned int firstMol;
    vislib::StringA tmpStr;

    // initialization
    this->resCount = 0;
    this->vertices.Clear();
    this->vertices.AssertCapacity( mol->ResidueCount() * 2);
    this->bsVertices.Clear();
    this->bsVertices.AssertCapacity( mol->ResidueCount() * 2);
    this->bsIndices.Clear();
    this->bsIndices.AssertCapacity( mol->ResidueCount());
    this->resIndex.Clear();
    this->resIndex.AssertCapacity( mol->ResidueCount());
    this->resCols = static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value());
    this->aminoAcidStrings.Clear();
    this->aminoAcidStrings.AssertCapacity( mol->ResidueCount() / this->resCols + 1);
    this->aminoAcidIndexStrings.Clear();
    this->aminoAcidIndexStrings.AssertCapacity( mol->ResidueCount() / this->resCols + 1);
    unsigned int currentRow = 0;
    unsigned int maxNumBindingSitesPerRes = 0;

    // count residues
    for( unsigned int cCnt = 0; cCnt < mol->ChainCount(); cCnt++ ) {
        firstMol = mol->Chains()[cCnt].FirstMoleculeIndex();
        for( unsigned int mCnt = firstMol; mCnt < firstMol + mol->Chains()[cCnt].MoleculeCount(); mCnt++ ) {
            firstStruct = mol->Molecules()[mCnt].FirstSecStructIndex();
            for( unsigned int sCnt = 0; sCnt < mol->Molecules()[mCnt].SecStructCount(); sCnt++ ) {
                for( unsigned int rCnt = 0; rCnt < mol->SecondaryStructures()[firstStruct + sCnt].AminoAcidCount(); rCnt++ ) {
                    // store the original index of the current residue
                    this->resIndex.Add( mol->SecondaryStructures()[firstStruct + sCnt].FirstAminoAcidIndex() + rCnt);
                    // store the secondary structure element type of the current residue
                    this->resSecStructType.Add( mol->SecondaryStructures()[firstStruct + sCnt].Type());
                    // compute the position of the residue icon
                    if( this->resCount == 0 ) {
                        this->vertices.Add( 0.0f);
                        this->vertices.Add( 0.0f);
                        this->aminoAcidStrings.Add("");
                        this->aminoAcidIndexStrings.Add(vislib::Array<vislib::StringA>());
                    } else {
                        if( this->resCount % static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value()) != 0 ) {
                            this->vertices.Add( this->vertices[this->resCount * 2 - 2] + 1.0f);
                            this->vertices.Add( this->vertices[this->resCount * 2 - 1]);
                        } else {
                            this->vertices.Add( 0.0f);
                            this->vertices.Add( this->vertices[this->resCount * 2 - 1] + this->rowHeight);
                            currentRow++;
                            this->aminoAcidStrings.Add("");
                            this->aminoAcidIndexStrings.Add(vislib::Array<vislib::StringA>());
                        }
                    }
                    // store the amino acid name
                    this->aminoAcidStrings[currentRow].Append( this->GetAminoAcidOneLetterCode(mol->ResidueTypeNames()[mol->Residues()[this->resIndex.Last()]->Type()]));
                    // store the chain name and residue index
                    tmpStr.Format("%c %i", mol->Chains()[cCnt].Name(), mol->Residues()[this->resIndex.Last()]->OriginalResIndex());
                    this->aminoAcidIndexStrings[currentRow].Add( tmpStr);
                    
                    // try to match binding sites
                    if( bs ) {
                        vislib::Pair<char, unsigned int> bsRes;
                        unsigned int numBS = 0;
                        // loop over all binding sites
                        for( unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++ ) {
                            for( unsigned int bsResCnt = 0; bsResCnt < bs->GetBindingSite(bsCnt)->Count(); bsResCnt++ ) {
                                bsRes = bs->GetBindingSite(bsCnt)->operator[](bsResCnt);
                                if( mol->Chains()[cCnt].Name() == bsRes.First() &&
                                    mol->Residues()[this->resIndex.Last()]->OriginalResIndex() == bsRes.Second() &&
                                    mol->ResidueTypeNames()[mol->Residues()[this->resIndex.Last()]->Type()] == bs->GetBindingSiteResNames(bsCnt)->operator[](bsResCnt) ) {
                                        //this->aminoAcidIndexStrings[currentRow].Last().Append(" *");
                                        this->bsVertices.Add( this->vertices[this->vertices.Count()-2]);
                                        this->bsVertices.Add( this->vertices[this->vertices.Count()-1] + 2.0f + numBS * 0.5f);
                                        this->bsIndices.Add( bsCnt);
                                        numBS++;
                                        maxNumBindingSitesPerRes = vislib::math::Max( maxNumBindingSitesPerRes, numBS);
                                }
                            }
                        }
                    }

                    this->resCount++;
                } // residues
            } // secondary structures
        } // molecules
    } // chains

    this->rowHeight = 3.0f + maxNumBindingSitesPerRes * 0.5f;
    
    // set the number of columns
    this->resCols = vislib::math::Min(this->resCount,
        static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value()));
    this->resCountPerRowParam.Param<param::IntParam>()->SetValue( this->resCols);
    // compute the number of rows
    this->resRows = static_cast<unsigned int>( std::ceilf(static_cast<float>(this->resCount) / static_cast<float>(this->resCols)));
    
    return true;
}

/*
 * Check if the residue is an amino acid.
 */
char SequenceRenderer::GetAminoAcidOneLetterCode( vislib::StringA resName ) {
    if( resName.Equals( "ALA" ) ) return 'A';
    else if( resName.Equals( "ARG" ) ) return 'R';
    else if( resName.Equals( "ASN" ) ) return 'N';
    else if( resName.Equals( "ASP" ) ) return 'D';
    else if( resName.Equals( "CYS" ) ) return 'C';
    else if( resName.Equals( "GLN" ) ) return 'Q';
    else if( resName.Equals( "GLU" ) ) return 'E';
    else if( resName.Equals( "GLY" ) ) return 'G';
    else if( resName.Equals( "HIS" ) ) return 'H';
    else if( resName.Equals( "ILE" ) ) return 'I';
    else if( resName.Equals( "LEU" ) ) return 'L';
    else if( resName.Equals( "LYS" ) ) return 'K';
    else if( resName.Equals( "MET" ) ) return 'M';
    else if( resName.Equals( "PHE" ) ) return 'F';
    else if( resName.Equals( "PRO" ) ) return 'P';
    else if( resName.Equals( "SER" ) ) return 'S';
    else if( resName.Equals( "THR" ) ) return 'T';
    else if( resName.Equals( "TRP" ) ) return 'W';
    else if( resName.Equals( "TYR" ) ) return 'Y';
    else if( resName.Equals( "VAL" ) ) return 'V';
    else if( resName.Equals( "ASH" ) ) return 'D';
    else if( resName.Equals( "CYX" ) ) return 'C';
    else if( resName.Equals( "CYM" ) ) return 'C';
    else if( resName.Equals( "GLH" ) ) return 'E';
    else if( resName.Equals( "HID" ) ) return 'H';
    else if( resName.Equals( "HIE" ) ) return 'H';
    else if( resName.Equals( "HIP" ) ) return 'H';
    else if( resName.Equals( "MSE" ) ) return 'M';
    else if( resName.Equals( "LYN" ) ) return 'K';
    else if( resName.Equals( "TYM" ) ) return 'Y';
    else return '?';
}

