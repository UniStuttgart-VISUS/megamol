#include "stdafx.h"
#include "SequenceRenderer.h"

#include "SequenceRenderer.h"
#include "CoreInstance.h"
#include "param/IntParam.h"
#include "utility/ColourParser.h"
#include "vislib/SimpleFont.h"
#include "vislib/Rectangle.h"
#include "vislib/BufferedFile.h"
#include "vislib/sysfunctions.h"
#include "glh/glh_extensions.h"
#include <GL/glu.h>
#include <math.h>
#include "misc/ImageViewer.h"
#include "utility/ResourceWrapper.h"

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
        dataPrepared(false), resCount(0), resCols(0), resRows(0), rowHeight( 3.0f) {

    // molecular data caller slot
    this->dataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    // binding site data caller slot
    this->bindingSiteCallerSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable(&this->bindingSiteCallerSlot);

    // param slot for number of residues per row
    this->resCountPerRowParam.SetParameter( new param::IntParam( 50, 1));
    this->MakeSlotAvailable( &this->resCountPerRowParam);
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
    
    // prepare the data
    if( this->dataPrepared = this->PrepareData( mol) ) {
        // try to get binding site information
        BindingSiteCall *site = this->bindingSiteCallerSlot.CallAs<BindingSiteCall>();
        if( site != NULL ) {
            this->getBindingSites( site);
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
    // get pointer to Diagram2DCall
    MolecularDataCall *mol = this->dataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL ) return false;
    
    // execute the call
    if( !(*mol)(MolecularDataCall::CallForGetData) ) return false;
    
    if( this->dataPrepared ) {
        glBegin( GL_TRIANGLES);
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
            glVertex2f( this->vertices[2*i] + 1.0f, -this->vertices[2*i+1]);
        }
        glEnd();

    
        float bgColor[4];
        float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
        glGetFloatv( GL_COLOR_CLEAR_VALUE, bgColor);
        for( unsigned int i = 0; i < 4; i++ ) {
            fgColor[i] -= bgColor[i];
        }

        // one-letter amino acid code
        //glColor3f( 0.8f, 0.8f, 0.8f);
        glColor3fv( fgColor);
        if( theFont.Initialise() ) {
            for( unsigned int i = 0; i < this->aminoAcidStrings.Count(); i++ ) {
                //theFont.DrawString( 0.0, -( static_cast<float>(i) * this->rowHeight + 1.0f), 1.0f, true, this->aminoAcidStrings[i].PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                for( unsigned int j = 0; j < this->aminoAcidStrings[i].Length(); j++ ) {
                    theFont.DrawString( static_cast<float>(j), -( static_cast<float>(i) * this->rowHeight), 1.0f, -1.0f,
                        1.0f, true, this->aminoAcidStrings[i].Substring(j, 1), vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);
                    theFont.DrawString( static_cast<float>(j), -( static_cast<float>(i) * this->rowHeight + 3.0f), 1.0f, 0.5f,
                        0.4f, true, this->aminoAcidIndexStrings[i][j], vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);
                }
            }
        }

        // amino acid number
        /*
        vislib::StringA tmpStr;
        int cnt = 0;
        if( theFont.Initialise() ) {
            for( unsigned int i = 0; i < this->aminoAcidStrings.Count(); i++ ) {
                //theFont.DrawString( 0.0, -( static_cast<float>(i) * this->rowHeight + 1.0f), 1.0f, true, this->aminoAcidStrings[i].PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                for( unsigned int j = 0; j < this->aminoAcidStrings[i].Length(); j++ ) {
                    tmpStr.Format( "%i", this->resIndex[cnt]);
                    theFont.DrawString( static_cast<float>(j), -( static_cast<float>(i) * this->rowHeight + 3.0f), 1.0f, 0.5f,
                        0.4f, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);
                    cnt++;
                }
            }
        }
        */

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
bool SequenceRenderer::PrepareData( MolecularDataCall *mol) {
    if( !mol ) return false;

    // temporary variables
    unsigned int firstStruct;
    unsigned int firstMol;
    vislib::StringA tmpStr;

    // initialization
    this->resCount = 0;
    this->vertices.Clear();
    this->vertices.AssertCapacity( mol->ResidueCount() * 2);
    this->resIndex.Clear();
    this->resIndex.AssertCapacity( mol->ResidueCount());
    this->resCols = static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value());
    this->aminoAcidStrings.Clear();
    this->aminoAcidStrings.AssertCapacity( mol->ResidueCount() / this->resCols + 1);
    this->aminoAcidIndexStrings.Clear();
    this->aminoAcidIndexStrings.AssertCapacity( mol->ResidueCount() / this->resCols + 1);
    unsigned int currentRow = 0;
    // count residues
    for( unsigned int cCnt = 0; cCnt < mol->ChainCount(); cCnt++ ) {
        firstMol = mol->Chains()[cCnt].FirstMoleculeIndex();
        //for( unsigned int mCnt = 0; mCnt < mol->MoleculeCount(); mCnt++ ) {
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
                    tmpStr.Format("%c %i", mol->Chains()[cCnt].Name(), mol->Residues()[this->resIndex.Last()]->OriginalResIndex());
                    this->aminoAcidIndexStrings[currentRow].Add( tmpStr);
                    this->resCount++;
                } // residues
            } // secondary structures
        } // molecules
    } // chains
    
    // set the number of columns
    this->resCols = vislib::math::Min(this->resCount,
        static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value()));
    this->resCountPerRowParam.Param<param::IntParam>()->SetValue( this->resCols);
    // compute the number of rows
    this->resRows = static_cast<unsigned int>( std::ceilf(static_cast<float>(this->resCount) / static_cast<float>(this->resCols)));
    
    // write the amino acid strings
    /*
    this->aminoAcidStrings.Clear();
    this->aminoAcidStrings.AssertCapacity( this->resRows);
    unsigned int cnt = 0;
    for( unsigned int y = 0; y < this->resRows; y++ ) {
        this->aminoAcidStrings.Add("");
        for( unsigned int x = 0; x < this->resCols; x++) {
            if( cnt < mol->ResidueCount() ) {
                this->aminoAcidStrings[y].Append( this->GetAminoAcidOneLetterCode(mol->ResidueTypeNames()[mol->Residues()[cnt]->Type()]));
                cnt++;
            } else {
                continue;
            }
        }
    }
    */

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


/*
 * Try to get information about binding sites.
 */
void SequenceRenderer::getBindingSites( BindingSiteCall *site) {
    if( !site ) return;

}
