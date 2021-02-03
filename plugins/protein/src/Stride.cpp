#include "stdafx.h"
#include "Stride.h"
#include <iostream>
#include <cstdio>
#include <algorithm>


using namespace megamol::protein;
using namespace megamol::protein_calls;

#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max


Stride::Stride( MolecularDataCall *mol) :
        Successful( false )
{
    // set protein chain count to zero
    ProteinChainCnt = 0;
    // set hydrogen bond count to zero
    HydroBondCnt = 0;

    ProteinChain = ( CHAIN  **)ckalloc( MAX_CHAIN*sizeof( CHAIN*));
    HydroBond    = ( HBOND  **)ckalloc( MAXHYDRBOND*sizeof( HBOND*));
    StrideCmd    = ( COMMAND *)ckalloc( sizeof( COMMAND));

    // set default values for command variable
    DefaultCmd( StrideCmd);

    // do nothing if Protein Data Interface is not valid
    if( !mol ) return;
    
    // get chains from interface
    GetChains( mol);
    // try to compute the secondary structure
    ComputeSecondaryStructure();
	// compute the indices of the hydrogen bonds
	PostProcessHBonds(mol);
}

Stride::~Stride(void)
{
    // free variables
    int i;
    for( i = 0; i < ProteinChainCnt; ++i )
        FreeChain( ProteinChain[i] );
    free(ProteinChain);

    for( i = 0; i < HydroBondCnt; ++i )
        FreeHBond(HydroBond[i]);

    free( HydroBond );

    free( StrideCmd );
}

void Stride::FreeChain(CHAIN *Chain) {
    free( Chain->File );

    for(int i =0; i < Chain->NRes; i++)
        FreeResidue( Chain->Rsd[i] );
    free(Chain->Rsd);

    for(int i =0; i < Chain->NHelix; i++)
        free( Chain->Helix[i] );
    free( Chain->Helix );

    for(int i =0; i < Chain->NSheet; i++)
        free( Chain->Sheet[i] );
    free( Chain->Sheet );

    for(int i =0; i < Chain->NTurn; i++)
        free( Chain->Turn[i] );
    free( Chain->Turn );

    for(int i =0; i < Chain->NAssignedTurn; i++)
        free( Chain->AssignedTurn[i] );
    free( Chain->AssignedTurn );

    for(int i =0; i < Chain->NBond; i++)
        free( Chain->SSbond[i] );
    free( Chain->SSbond );

    free( Chain ); 
}

void Stride::FreeResidue(RESIDUE *r) {
    free( r->Prop );
    free( r->Inv );
    free(r);
}

void Stride::FreeHBond(HBOND *h) {
//	free( h->Dnr); // TODO: free FreeChain(Dnr->Chain)?
//	free( h->Acc); // TODO: free FreeChain(Dnr->Chain)?
    free( h );
}

void Stride::GetChains(MolecularDataCall *mol) {
    int ChainCnt;
    int cntCha, cntRes, cntAtm, idx, cnt, chain;
    int atomCount, firstAtom;
    int i;
    RESIDUE *r = 0;
    CHAIN *c = 0;
    char PdbIdent[5];
    
    strcpy( PdbIdent, "~~~~");
    
    //////////////////////////////////////////////////////
    // build chains from Molecular Data Call
    //////////////////////////////////////////////////////
    
    ProteinChainCnt = std::min( (unsigned int)mol->MoleculeCount(), (unsigned int)MAX_CHAIN);
    chain = 0;
    
    // iterate over all chains
    for( cntCha = 0; cntCha < ProteinChainCnt; ++cntCha) {
        // inititalize the chain
        InitChain( &ProteinChain[cntCha]);
        ProteinChain[cntCha]->ChainId = cntCha;
        ProteinChain[cntCha]->Id = cntCha;
        ProteinChain[cntCha]->NAtom = 0;
        // write number of residues in this chain
        ProteinChain[cntCha]->NRes = mol->Molecules()[cntCha].ResidueCount();
        // set data for all residues
        idx = mol->Molecules()[cntCha].FirstResidueIndex();
        cnt = idx + mol->Molecules()[cntCha].ResidueCount();
        for( cntRes = 0; cntRes < ProteinChain[cntCha]->NRes; ++cntRes ) {
            firstAtom = mol->Residues()[idx+cntRes]->FirstAtomIndex();
            atomCount = mol->Residues()[idx+cntRes]->AtomCount();
            ProteinChain[cntCha]->NAtom += atomCount;
            
            ProteinChain[cntCha]->Rsd[cntRes] = ( RESIDUE*)ckalloc( sizeof( RESIDUE));
        
            r = ProteinChain[cntCha]->Rsd[cntRes];
            r->NAtom = atomCount;
            r->ResNumb = cntRes;
#ifdef _WIN32
            _snprintf( r->PDB_ResNumb, RES_FIELD, "%i", cntRes);
            _snprintf( r->ResType, RES_FIELD, mol->ResidueTypeNames()[mol->Residues()[cntRes]->Type()]);
#else // _WIN32
            snprintf( r->PDB_ResNumb, RES_FIELD, "%i", cntRes);
            snprintf( r->ResType, RES_FIELD, mol->ResidueTypeNames()[mol->Residues()[cntRes]->Type()]);
#endif // _WIN32
            // set atomic values
            for( cntAtm = 0; cntAtm < atomCount; ++cntAtm ) {
#ifdef _WIN32
                _snprintf( r->AtomType[cntAtm], AT_FIELD, mol->AtomTypes()[mol->AtomTypeIndices()[cntAtm + firstAtom]].Name());
#else // _WIN32
                snprintf( r->AtomType[cntAtm], AT_FIELD, mol->AtomTypes()[mol->AtomTypeIndices()[cntAtm + firstAtom]].Name());
#endif // _WIN32
                r->Coord[cntAtm][0] = mol->AtomPositions()[3*(firstAtom+cntAtm)+0];
                r->Coord[cntAtm][1] = mol->AtomPositions()[3*(firstAtom+cntAtm)+1];
                r->Coord[cntAtm][2] = mol->AtomPositions()[3*(firstAtom+cntAtm)+2];

                r->Occupancy[cntAtm] = mol->AtomOccupancies()[firstAtom+cntAtm];
                r->TempFactor[cntAtm] = mol->AtomBFactors()[firstAtom+cntAtm];
                r->ResAtomIdx[cntAtm] = firstAtom + cntAtm;
                
                // printf( "CHAIN %5i, RESIDUE %5i (%s), ATOM %6i = (%s)\n", cntCha, cntRes, pdi->AminoAcidName( pdi->ProteinChain( cntCha).AminoAcid()[cntRes].NameIndex()).PeekBuffer(), cntAtm, pdi->AtomTypes()[pdi->ProteinAtomData()[cntAtm+firstAtom].TypeIndex()].Name().PeekBuffer() );
            }
        }
    }
        
    for( ChainCnt = 0; ChainCnt < ProteinChainCnt; ++ChainCnt ) {
        c = ProteinChain[ChainCnt];
        if( c->NRes != 0 && !FindAtom( c, c->NRes-1, "CA", &i ) )
            c->NRes--;
        strcpy( c->File, StrideCmd->InputFile);

        strcpy( c->PdbIdent, PdbIdent );
        if ( c->NSheet != -1 ) c->NSheet++;
        c->Resolution = 0.0f;
        for( i = 0; i < c->NRes; ++i ) {
            r = c->Rsd[i];
            r->Inv = ( INVOLVED * ) ckalloc ( sizeof ( INVOLVED ) );
            r->Prop = ( PROPERTY * ) ckalloc ( sizeof ( PROPERTY ) );
            r->Inv->NBondDnr = 0;
            r->Inv->NBondAcc = 0;
            r->Inv->InterchainHBonds = STRIDE_NO;
            r->Prop->Asn     = 'C';
            r->Prop->PdbAsn  = 'C';
            r->Prop->Solv    = 0.0;
            r->Prop->Phi     = 360.0;
            r->Prop->Psi     = 360.0;
        }
    }

} 

bool Stride::ComputeSecondaryStructure() {
    int Cn, ValidChain = 0;
    float **PhiPsiMapHelix, **PhiPsiMapSheet;
    int i;

    // check if at least one chain is valid
    for( Cn = 0; Cn < ProteinChainCnt; ++Cn )
        ValidChain += CheckChain( ProteinChain[Cn], StrideCmd );
    if( !ValidChain )
    {
        //die( "No valid chain in %s\n", ProteinChain[0]->File );
        printf( "No valid chain.\n");
        return false;
    }

    BackboneAngles( ProteinChain, ProteinChainCnt);
    PhiPsiMapHelix = DefaultHelixMap ( StrideCmd );
    PhiPsiMapSheet = DefaultSheetMap ( StrideCmd );

    for ( Cn = 0; Cn < ProteinChainCnt; ++Cn )
        PlaceHydrogens( ProteinChain[Cn] );

    if( ( HydroBondCnt = FindHydrogenBonds( ProteinChain, Cn, HydroBond, StrideCmd) ) == 0 )
    {
        //die( "No hydrogen bonds found in %s\n", StrideCmd->InputFile );
        printf( "No hydrogen bonds found.\n" );
        return false;
    }

	NoDoubleHBond(HydroBond, HydroBondCnt);

	/*for (int i = 0; i < HydroBondCnt; i++) {
		printf("%i DRes %i DDRes %i DDIRes %i DAt %i DDAt %i DDIAt %i\n", HydroBond[i]->Dnr->Chain->ChainId, HydroBond[i]->Dnr->D_Res, HydroBond[i]->Dnr->DD_Res, HydroBond[i]->Dnr->DDI_Res, HydroBond[i]->Dnr->D_At, HydroBond[i]->Dnr->DD_At, HydroBond[i]->Dnr->DDI_At);
		printf("%i ARes %i AARes %i AA2Res %i AAt %i AAAt %i AA2At %i\n\n", HydroBond[i]->Dnr->Chain->ChainId, HydroBond[i]->Acc->A_Res, HydroBond[i]->Acc->AA_Res, HydroBond[i]->Acc->AA2_Res, HydroBond[i]->Acc->A_At, HydroBond[i]->Acc->AA_At, HydroBond[i]->Acc->AA2_At);
	}*/


    
    DiscrPhiPsi( ProteinChain, ProteinChainCnt, StrideCmd );

    // find secondary structure (helices, sheets, turns)
    for( Cn = 0; Cn < ProteinChainCnt; ++Cn )
    {
        if( ProteinChain[Cn]->Valid )
        {
            Helix( ProteinChain, Cn, HydroBond, StrideCmd, PhiPsiMapHelix );
            for( i = 0; i < ProteinChainCnt; ++i )
            {
                if ( ProteinChain[i]->Valid )
                    Sheet( ProteinChain, Cn, i, HydroBond, StrideCmd, PhiPsiMapSheet );
            }
            BetaTurn( ProteinChain, Cn );
            GammaTurn( ProteinChain, Cn, HydroBond );
        }
    }
    
    // find disulfide bonds
    SSBond( ProteinChain, ProteinChainCnt);
    
    return true;
}

bool Stride::WriteToInterface( MolecularDataCall *mol) {
    if( mol ) {
        int Cn, i;
        char type;
        int firstRes;
        int resCnt;
        int idx = 0;
        
        std::vector<MolecularDataCall::SecStructure> sec;
        
        if ( !ExistsSecStr( ProteinChain, ProteinChainCnt ) )
            return false;
        
        for ( Cn = 0; Cn < ProteinChainCnt; ++Cn ) {
            // do nothing if the current chain is not valid
            if ( !ProteinChain[Cn]->Valid )
                continue;
            
            // set initial values for first sec struct elem
            firstRes = mol->Molecules()[Cn].FirstResidueIndex();
            resCnt = 1;
            type = ProteinChain[Cn]->Rsd[0]->Prop->Asn;
            
            for ( i = 1; i < ProteinChain[Cn]->NRes; i++ ) {
                // update values if type did not change
                if( ProteinChain[Cn]->Rsd[i]->Prop->Asn == type ) {
                    resCnt++;
                } else {
                    // write sec struct elem to vector if new elem starts
                    sec.push_back( MolecularDataCall::SecStructure());
                    sec.back().SetPosition( firstRes, resCnt);
                    if( type == 'G' || type == 'H' || type == 'I' )
                        sec.back().SetType( MolecularDataCall::SecStructure::TYPE_HELIX);
                    else if( type == 'E' )
                        sec.back().SetType( MolecularDataCall::SecStructure::TYPE_SHEET);
                    else
                        sec.back().SetType( MolecularDataCall::SecStructure::TYPE_COIL);
                    // start new sec struct elem
                    firstRes = i + mol->Molecules()[Cn].FirstResidueIndex();
                    resCnt = 1;
                    type = ProteinChain[Cn]->Rsd[i]->Prop->Asn;
                }
                
                //printf( "CHAIN %4i RES %5i ASN %c\n", Cn, i, ProteinChain[Cn]->Rsd[i]->Prop->Asn);
            }
            // write last sec struct elem to vector
            sec.push_back( MolecularDataCall::SecStructure());
            sec.back().SetPosition( firstRes, resCnt);
            if( type == 'G' || type == 'H' || type == 'I' )
                sec.back().SetType( MolecularDataCall::SecStructure::TYPE_HELIX);
            else if( type == 'E' )
                sec.back().SetType( MolecularDataCall::SecStructure::TYPE_SHEET);
            else
                sec.back().SetType( MolecularDataCall::SecStructure::TYPE_COIL);
            mol->SetMoleculeSecondaryStructure( Cn, idx, (unsigned int)sec.size() - idx);
            idx = (int)sec.size();
        }
        // handled all residues of current chain, copy sec struct to interface
		mol->SetSecondaryStructureCount((unsigned int)sec.size());
        for( i = 0; i < (int)sec.size(); ++i ) {
            mol->SetSecondaryStructure( i, sec[i]);
        }

		// set the found hydrogen bonds
		mol->SetHydrogenBonds(this->ownHydroBonds.data(), static_cast<unsigned int>(HydroBondCnt));

    } else {
        return false;
    }
    return true;
}

void Stride::DefaultCmd( COMMAND *Cmd ) {

    Cmd->SideChainHBond    = STRIDE_NO;
    Cmd->MainChainHBond    = STRIDE_YES;
    Cmd->MainChainPolarInt = STRIDE_YES;
    Cmd->UseResolution     = STRIDE_NO;
    Cmd->Truncate          = STRIDE_YES;
    Cmd->OutSeq            = STRIDE_NO;

    Cmd->EnergyType        = 'G';

    Cmd->DistCutOff        =  6.0f;
    Cmd->PhiPsiStep        =  0.0f;

    Cmd->C1_H              = -1.0f;
    Cmd->C2_H              =  1.0f;
    Cmd->C1_E              = -0.2f;
    Cmd->C2_E              =  0.2f;

    Cmd->Treshold_H1       = -230.0f;
    Cmd->Treshold_H3       =  0.12f;
    Cmd->Treshold_H4       =  0.06f;
    Cmd->Treshold_E1       = -240.0f;
    Cmd->Treshold_E2       = -310.0f;

    Cmd->MinResolution     =  0.1f;
    Cmd->MaxResolution     =  100.0f;

    Cmd->MinLength         = 0;
    Cmd->MaxLength         = MAX_RES;

    Cmd->NPixel            = 0;
    Cmd->NActive           = 0;
    Cmd->NProcessed        = 0;

    strcpy ( Cmd->MapFileHelix,"" );
    strcpy ( Cmd->MapFileSheet,"" );
    strcpy ( Cmd->Active,"" );
    strcpy ( Cmd->Processed,"" );
    strcpy ( Cmd->Cond,"" );
    strcpy ( Cmd->OutFile, "" );
    strcpy ( Cmd->InputFile, "" );

    Cmd->NActive = ( int ) strlen ( Cmd->Active );
    Cmd->NProcessed = ( int ) strlen ( Cmd->Processed );
}

int Stride::ReadPDBFile( CHAIN **Chain, int *Cn, COMMAND *Cmd ) {

    int ChainCnt, i;
    BOOLEAN First_ATOM;
    float Resolution = 0.0;
    FILE *pdb;
    BUFFER Buffer;
    char PdbIdent[5];
    RESIDUE *r;
    CHAIN *c;

    *Cn= 0;
    strcpy( PdbIdent,"~~~~" );

    if ( ! ( pdb = fopen ( Cmd->InputFile, "r") ) )
        return ( FAILURE );
    
    First_ATOM = STRIDE_YES;

    while ( fgets ( Buffer,BUFSZ,pdb ) ) {
        if ( !strncmp ( Buffer,"ENDMDL",6 ) ) {
            Process_ENDMDL ( Buffer,Chain,Cn );
            break;
        }
        else if ( !strncmp ( Buffer,"ATOM",4 ) && !Process_ATOM ( Buffer,Chain,Cn,&First_ATOM,Cmd ) )
            return ( FAILURE );
    }
    fclose ( pdb );

    for ( ChainCnt=0; ChainCnt< *Cn; ChainCnt++ ) {
        c = Chain[ChainCnt];
        if ( c->NRes != 0  && !FindAtom ( c,c->NRes,"CA",&i ) )
            c->NRes--;
        strcpy ( c->File,Cmd->InputFile );

        strcpy ( c->PdbIdent,PdbIdent );
        if ( c->NRes != 0 )  c->NRes++;
        if ( c->NSheet != -1 ) c->NSheet++;
        c->Resolution = Resolution;
        for ( i=0; i<c->NRes; i++ )
        {
            r = c->Rsd[i];
            r->Inv = ( INVOLVED * ) ckalloc ( sizeof ( INVOLVED ) );
            r->Prop = ( PROPERTY * ) ckalloc ( sizeof ( PROPERTY ) );
            r->Inv->NBondDnr = 0;
            r->Inv->NBondAcc = 0;
            r->Inv->InterchainHBonds = STRIDE_NO;
            r->Prop->Asn     = 'C';
            r->Prop->PdbAsn  = 'C';
            r->Prop->Solv    = 0.0;
            r->Prop->Phi     = 360.0;
            r->Prop->Psi     = 360.0;
        }
    }

    // ------- START DEBUG -------
    /*
    printf( "Number of Chains %i\n", *Cn);
    for( i=0; i < *Cn; ++i )
    {
        printf( "   Chain %i - #Res: %i, #Atoms: %i\n", i, Chain[i]->NRes, Chain[i]->NAtom);
    }
    */
    // ------- END DEBUG -------
    
    return ( SUCCESS );
}

void Stride::die( const char *format, ... ) {
    //void exit ( int return_code );
    va_list ptr;
    va_start ( ptr,format );
    vfprintf ( stderr,format,ptr );
    exit ( 1 );
    va_end ( ptr );
}

int Stride::CheckChain( CHAIN *Chain, COMMAND *Cmd) {
  if( Cmd->NProcessed && !ChInStr( Cmd->Processed, SpaceToDash(Chain->Id)) ) {
    Chain->Valid = STRIDE_NO;
    return(FAILURE);
  }

  if( Chain->NRes < 5 ) {
        //fprintf(stderr,"IGSTRIDE_NORED %s %c ",Chain->File,SpaceToDash(Chain->Id));
        //fprintf(stderr,"(less than 5 residues)\n");
        Chain->Valid = STRIDE_NO;
        return(FAILURE);
  }
  
  return(SUCCESS);
}

void Stride::BackboneAngles( CHAIN **Chain, int NChain ) {
    int Res, Cn;

    for ( Cn=0; Cn<NChain; Cn++ )
    {

        for ( Res=0; Res<Chain[Cn]->NRes; Res++ )
        {
            PHI ( Chain[Cn],Res );
            PSI ( Chain[Cn],Res );
        }
    }
}

float** Stride::DefaultHelixMap( COMMAND *Cmd) {
    int i;
    
    float **Map;
    static float Data[DEFNUMPIXEL][DEFNUMPIXEL] = {{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0009014423f, 0.0041898815f,
        0.0085105160f, 0.0133839026f, 0.0245425366f, 0.0407802090f, 0.0464176536f, 0.0330946408f,
        0.0134803243f, 0.0024038462f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0007370283f, 0.0077203326f, 0.0269849468f,
        0.0492307022f, 0.0621860325f, 0.0747849122f, 0.0919913873f, 0.0918549150f, 0.0617070347f,
        0.0241584498f, 0.0041428790f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0041416897f, 0.0287234355f, 0.0835687742f,
        0.1384727061f, 0.1562444866f, 0.1470608264f, 0.1360232681f, 0.1159155145f, 0.0742164999f,
        0.0290896539f, 0.0050673936f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0009375000f, 0.0156580955f, 0.0757770315f, 0.1856354773f,
        0.2785892785f, 0.2880102694f, 0.2332847565f, 0.1741978228f, 0.1281246394f, 0.0793832615f,
        0.0320557840f, 0.0058840578f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0048893229f, 0.0437000208f, 0.1617751122f, 0.3399706185f,
        0.4626395404f, 0.4418565035f, 0.3235570788f, 0.2100441158f, 0.1358627081f, 0.0776144490f,
        0.0297011137f, 0.0052390974f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0136979166f, 0.0917820632f, 0.2773087323f, 0.5047551394f,
        0.6214492917f, 0.5485223532f, 0.3655386865f, 0.2054343373f, 0.1121114418f, 0.0548815951f,
        0.0178668182f, 0.0025975490f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0246484373f, 0.1396044195f, 0.3594934344f, 0.5710113049f,
        0.6337110400f, 0.5133636594f, 0.3054708838f, 0.1402616948f, 0.0584463216f, 0.0228670351f,
        0.0058531328f, 0.0005151099f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0265885405f, 0.1365883052f, 0.3163702190f, 0.4545661211f,
        0.4628692269f, 0.3425511420f, 0.1761947423f, 0.0607788190f, 0.0158569515f, 0.0042061093f,
        0.0008107311f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0152018229f, 0.0738445148f, 0.1630392224f, 0.2269553691f,
        0.2237145752f, 0.1528334022f, 0.0652616471f, 0.0150429625f, 0.0014589608f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0035156249f, 0.0165251363f, 0.0379281938f, 0.0584417619f,
        0.0619409233f, 0.0404052660f, 0.0136552500f, 0.0016678370f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0011718750f, 0.0046875002f,
        0.0070312503f, 0.0046875002f, 0.0011718750f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0006944445f, 0.0036063762f, 0.0080820229f, 0.0101532144f, 0.0076146079f,
        0.0032324446f, 0.0006009616f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f}
    };
    
    Map = (float **)ckalloc(DEFNUMPIXEL*sizeof(float *));
    
    for( i=0; i<DEFNUMPIXEL; i++ )
        Map[i] = &(Data[i][0]);
    
    Cmd->NPixel = DEFNUMPIXEL;
    Cmd->PhiPsiStep = (float)(MAXPHIPSI - MINPHIPSI)/(float)Cmd->NPixel;
    
    return(Map);
}

float** Stride::DefaultSheetMap( COMMAND *Cmd) {
    int i;
    
    float **Map;
    static float Data[DEFNUMPIXEL][DEFNUMPIXEL] = {{
        0.2769023776f, 0.1408346891f, 0.0464910716f, 0.0073784725f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0047086575f, 0.0218229108f, 0.0569166169f,
        0.1254088134f, 0.2340224832f, 0.3511219919f, 0.4355685711f, 0.4584180117f, 0.4007356465f},{
        0.4067636132f, 0.2329865396f, 0.0927943364f, 0.0237838365f, 0.0055147060f, 0.0013786765f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0088186050f, 0.0420726910f, 0.1043856740f,
        0.2086037844f, 0.3677131534f, 0.5367187858f, 0.6412357688f, 0.6458424330f, 0.5580080152f},{
        0.4286311865f, 0.2678007782f, 0.1282834113f, 0.0529448465f, 0.0220588241f, 0.0055147060f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0086062262f, 0.0445192643f, 0.1197573245f,
        0.2487278134f, 0.4369854629f, 0.6241853237f, 0.7160459757f, 0.6829043031f, 0.5716546178f},{
        0.3639202416f, 0.2397334576f, 0.1305907220f, 0.0683420748f, 0.0330882370f, 0.0082720593f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0053559211f, 0.0328565054f, 0.1048930883f,
        0.2402425259f, 0.4295993447f, 0.6026929021f, 0.6669865251f, 0.6039550304f, 0.4841639400f},{
        0.2637948096f, 0.1723874062f, 0.0920098722f, 0.0464194641f, 0.0220588241f, 0.0055147060f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0030202419f, 0.0224239044f, 0.0804052502f,
        0.1923188865f, 0.3456886411f, 0.4811576009f, 0.5223571062f, 0.4586051404f, 0.3565762639f},{
        0.1628032923f, 0.0930610597f, 0.0400134660f, 0.0143100554f, 0.0055147060f, 0.0013786765f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0015453297f, 0.0132468110f, 0.0489843786f,
        0.1174781919f, 0.2150468081f, 0.3082944453f, 0.3439011276f, 0.3080393970f, 0.2371628135f},{
        0.0825822726f, 0.0338854715f, 0.0092895878f, 0.0012122844f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0003863324f, 0.0046614520f, 0.0186656341f,
        0.0477515720f, 0.0961741805f, 0.1546680480f, 0.1961039603f, 0.1944279373f, 0.1469529718f},{
        0.0326442868f, 0.0073916214f, 0.0008854167f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0006347656f, 0.0031504754f,
        0.0104655549f, 0.0272454955f, 0.0570511036f, 0.0941907763f, 0.1088592261f, 0.0785619915f},{
        0.0090501504f, 0.0007651417f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0007207961f, 0.0035958111f, 0.0131648667f, 0.0318824202f, 0.0425693691f, 0.0292618107f},{
        0.0013020834f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0013020834f, 0.0052083335f, 0.0078125000f, 0.0052083335f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f},{
        0.0210939310f, 0.0078523019f, 0.0013020834f, 0.0000000000f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0014204546f,
        0.0071634995f, 0.0169352461f, 0.0272206441f, 0.0357281528f, 0.0395361669f, 0.0343801714f},{
        0.1146211401f, 0.0503530800f, 0.0130920913f, 0.0015190972f, 0.0000000000f, 0.0000000000f,
        0.0000000000f, 0.0000000000f, 0.0000000000f, 0.0010016026f, 0.0046167620f, 0.0157516468f,
        0.0453012958f, 0.0937970504f, 0.1454590708f, 0.1861637682f, 0.2019522935f, 0.1764564067f}
    };
    
    Map = (float **)ckalloc(DEFNUMPIXEL*sizeof(float *));
    
    for( i=0; i<DEFNUMPIXEL; i++ )
        Map[i] = &(Data[i][0]);
    
    Cmd->NPixel = DEFNUMPIXEL;
    Cmd->PhiPsiStep = (float)(MAXPHIPSI - MINPHIPSI)/(float)Cmd->NPixel;
    
    return(Map); 
}

int Stride::PlaceHydrogens( CHAIN *Chain ) {

    int Res, i, N, C, CA, H, PlacedCnt=0;
    float Length_N_C, Length_N_CA, Length_N_H;
    RESIDUE *r, *rr;

    for ( Res=1; Res<Chain->NRes; Res++ )
    {

        r  = Chain->Rsd[Res];
        rr = Chain->Rsd[Res-1];

        if ( !strcmp ( r->ResType,"PRO" ) ) continue;

        /* Replace deiterium atoms by hydrogens */
        if( FindAtom ( Chain,Res,"D", &H ) )
            strcpy( r->AtomType[H], "H");

        if( !FindAtom ( Chain,Res,"H",&H )  && FindAtom ( Chain,Res,"N",&N )   &&
                FindAtom ( Chain,Res-1,"C",&C ) && FindAtom ( Chain,Res,"CA",&CA ) )
        {

            H = r->NAtom;

            Length_N_C   = Dist ( r->Coord[N],rr->Coord[C] );
            Length_N_CA  = Dist ( r->Coord[N],r->Coord[CA] );

            for ( i=0; i<3; i++ )
                r->Coord[H][i] = r->Coord[N][i] -
                                 ( ( rr->Coord[C][i] -  r->Coord[N][i] ) /Length_N_C +
                                   ( r->Coord[CA][i]  -  r->Coord[N][i] ) /Length_N_CA );

            Length_N_H = Dist ( r->Coord[N],r->Coord[H] );

            for ( i=0; i<3; i++ )
                r->Coord[H][i] = r->Coord[N][i] +
                                 (float)DIST_N_H* ( r->Coord[H][i]-r->Coord[N][i] ) /Length_N_H;

            strcpy ( r->AtomType[H],"H" );
            r->NAtom++;
            PlacedCnt++;
        }
    }
    return ( PlacedCnt );
}

int Stride::FindHydrogenBonds( CHAIN **Chain, int NChain, HBOND **HBond, COMMAND *Cmd ) {
    DONOR **Dnr;
    ACCEPTOR **Acc;
    int NDnr=0, NAcc=0;
    int dc, ac, ccd, cca, cc, hc=0, i;

    Dnr = ( DONOR ** ) ckalloc ( MAXDONOR*sizeof ( DONOR * ) );
    Acc = ( ACCEPTOR ** ) ckalloc ( MAXACCEPTOR*sizeof ( ACCEPTOR * ) );

    for ( cc=0; cc<NChain; cc++ )
    {
        FindDnr ( Chain[cc],Dnr,&NDnr,Cmd );
        FindAcc ( Chain[cc],Acc,&NAcc,Cmd );
    }

    BOOLEAN *BondedDonor, *BondedAcceptor;
    BondedDonor    = ( BOOLEAN * ) ckalloc ( NDnr*sizeof ( BOOLEAN ) );
    BondedAcceptor = ( BOOLEAN * ) ckalloc ( NAcc*sizeof ( BOOLEAN ) );

    for ( i=0; i<NDnr; i++ )
        BondedDonor[i] = STRIDE_NO;
    for ( i=0; i<NAcc; i++ )
        BondedAcceptor[i] = STRIDE_NO;

    for ( dc=0; dc<NDnr; dc++ )
    {

        if ( Dnr[dc]->Group != Peptide && !Cmd->SideChainHBond ) continue;

        for ( ac=0; ac<NAcc; ac++ )
        {

            if ( abs ( Acc[ac]->A_Res - Dnr[dc]->D_Res ) < 2 && Acc[ac]->Chain->Id == Dnr[dc]->Chain->Id )
                continue;

            if ( Acc[ac]->Group != Peptide && !Cmd->SideChainHBond ) continue;

            if ( hc == MAXHYDRBOND )
                die ( "Number of hydrogen bonds exceeds current limit of %d in %s\n",
                      MAXHYDRBOND,Chain[0]->File );
            HBond[hc] = ( HBOND * ) ckalloc ( sizeof ( HBOND ) );

            HBond[hc]->ExistHydrBondRose = STRIDE_NO;
            HBond[hc]->ExistHydrBondBaker = STRIDE_NO;
            HBond[hc]->ExistPolarInter = STRIDE_NO;

            if ( ( HBond[hc]->AccDonDist =
                        Dist ( Dnr[dc]->Chain->Rsd[Dnr[dc]->D_Res]->Coord[Dnr[dc]->D_At],
                               Acc[ac]->Chain->Rsd[Acc[ac]->A_Res]->Coord[Acc[ac]->A_At] ) ) <=
                    Cmd->DistCutOff )
            {


                if ( Cmd->MainChainPolarInt && Dnr[dc]->Group == Peptide &&
                        Acc[ac]->Group == Peptide && Dnr[dc]->H != ERR )
                {
                    GRID_Energy ( Acc[ac]->Chain->Rsd[Acc[ac]->AA2_Res]->Coord[Acc[ac]->AA2_At],
                                  Acc[ac]->Chain->Rsd[Acc[ac]->AA_Res]->Coord[Acc[ac]->AA_At],
                                  Acc[ac]->Chain->Rsd[Acc[ac]->A_Res]->Coord[Acc[ac]->A_At],
                                  Dnr[dc]->Chain->Rsd[Dnr[dc]->D_Res]->Coord[Dnr[dc]->H],
                                  Dnr[dc]->Chain->Rsd[Dnr[dc]->D_Res]->Coord[Dnr[dc]->D_At],
                                  Cmd,HBond[hc] );

                    if ( HBond[hc]->Energy < -10.0 &&
                            ( ( Cmd->EnergyType == 'G' && fabs ( HBond[hc]->Et ) > Eps &&
                                fabs ( HBond[hc]->Ep ) > Eps ) || Cmd->EnergyType != 'G' ) )
                        HBond[hc]->ExistPolarInter = STRIDE_YES;
                }

                if ( Cmd->MainChainHBond &&
                        ( HBond[hc]->OHDist =
                              Dist ( Dnr[dc]->Chain->Rsd[Dnr[dc]->D_Res]->Coord[Dnr[dc]->H],
                                     Acc[ac]->Chain->Rsd[Acc[ac]->A_Res]->Coord[Acc[ac]->A_At] ) ) <= 2.5 &&
                        ( HBond[hc]->AngNHO =
                              Ang ( Dnr[dc]->Chain->Rsd[Dnr[dc]->D_Res]->Coord[Dnr[dc]->D_At],
                                    Dnr[dc]->Chain->Rsd[Dnr[dc]->D_Res]->Coord[Dnr[dc]->H],
                                    Acc[ac]->Chain->Rsd[Acc[ac]->A_Res]->Coord[Acc[ac]->A_At] ) ) >= 90.0 &&
                        HBond[hc]->AngNHO <= 180.0 &&
                        ( HBond[hc]->AngCOH =
                              Ang ( Acc[ac]->Chain->Rsd[Acc[ac]->AA_Res]->Coord[Acc[ac]->AA_At],
                                    Acc[ac]->Chain->Rsd[Acc[ac]->A_Res]->Coord[Acc[ac]->A_At],
                                    Dnr[dc]->Chain->Rsd[Dnr[dc]->D_Res]->Coord[Dnr[dc]->H] ) ) >= 90.0 &&

                        HBond[hc]->AngCOH <= 180.0 )
                    HBond[hc]->ExistHydrBondBaker = STRIDE_YES;

                if ( Cmd->MainChainHBond &&
                        HBond[hc]->AccDonDist <= Dnr[dc]->HB_Radius+Acc[ac]->HB_Radius )
                {

                    HBond[hc]->AccAng =
                        Ang ( Dnr[dc]->Chain->Rsd[Dnr[dc]->D_Res]->Coord[Dnr[dc]->D_At],
                              Acc[ac]->Chain->Rsd[Acc[ac]->A_Res]->Coord[Acc[ac]->A_At],
                              Acc[ac]->Chain->Rsd[Acc[ac]->AA_Res]->Coord[Acc[ac]->AA_At] );

                    if ( ( ( Acc[ac]->Hybrid == Nsp2 || Acc[ac]->Hybrid == Osp2 ) &&
                            ( HBond[hc]->AccAng >= MINACCANG_SP2 &&
                              HBond[hc]->AccAng <= MAXACCANG_SP2 ) ) ||
                            ( ( Acc[ac]->Hybrid == Ssp3 ||  Acc[ac]->Hybrid == Osp3 ) &&
                              ( HBond[hc]->AccAng >= MINACCANG_SP3 &&
                                HBond[hc]->AccAng <= MAXACCANG_SP3 ) ) )
                    {

                        HBond[hc]->DonAng =
                            Ang ( Acc[ac]->Chain->Rsd[Acc[ac]->A_Res]->Coord[Acc[ac]->A_At],
                                  Dnr[dc]->Chain->Rsd[Dnr[dc]->D_Res]->Coord[Dnr[dc]->D_At],
                                  Dnr[dc]->Chain->Rsd[Dnr[dc]->DD_Res]->Coord[Dnr[dc]->DD_At] );

                        if ( ( ( Dnr[dc]->Hybrid == Nsp2 || Dnr[dc]->Hybrid == Osp2 ) &&
                                ( HBond[hc]->DonAng >= MINDONANG_SP2 &&
                                  HBond[hc]->DonAng <= MAXDONANG_SP2 ) ) ||
                                ( ( Dnr[dc]->Hybrid == Nsp3 || Dnr[dc]->Hybrid == Osp3 ) &&
                                  ( HBond[hc]->DonAng >= MINDONANG_SP3 &&
                                    HBond[hc]->DonAng <= MAXDONANG_SP3 ) ) )
                        {

                            if ( Dnr[dc]->Hybrid == Nsp2 || Dnr[dc]->Hybrid == Osp2 )
                            {
                                HBond[hc]->AccDonAng =
                                    fabs ( Torsion ( Dnr[dc]->Chain->Rsd[Dnr[dc]->DDI_Res]->Coord[Dnr[dc]->DDI_At],
                                                     Dnr[dc]->Chain->Rsd[Dnr[dc]->D_Res]->Coord[Dnr[dc]->D_At],
                                                     Dnr[dc]->Chain->Rsd[Dnr[dc]->DD_Res]->Coord[Dnr[dc]->DD_At],
                                                     Acc[ac]->Chain->Rsd[Acc[ac]->A_Res]->Coord[Acc[ac]->A_At] ) );

                                if ( HBond[hc]->AccDonAng > 90.0f && HBond[hc]->AccDonAng < 270.0f )
                                    HBond[hc]->AccDonAng = fabs( 180.0f - HBond[hc]->AccDonAng );

                            }

                            if ( Acc[ac]->Hybrid == Nsp2 || Acc[ac]->Hybrid == Osp2 )
                            {
                                HBond[hc]->DonAccAng =
                                    fabs ( Torsion ( Dnr[dc]->Chain->Rsd[Dnr[dc]->D_Res]->Coord[Dnr[dc]->D_At],
                                                     Acc[ac]->Chain->Rsd[Acc[ac]->A_Res]->Coord[Acc[ac]->A_At],
                                                     Acc[ac]->Chain->Rsd[Acc[ac]->AA_Res]->Coord[Acc[ac]->AA_At],
                                                     Acc[ac]->Chain->Rsd[Acc[ac]->AA2_Res]->Coord[Acc[ac]->AA2_At] ) );

                                if ( HBond[hc]->DonAccAng > 90.0f && HBond[hc]->DonAccAng < 270.0f )
                                    HBond[hc]->DonAccAng = fabs( 180.0f - HBond[hc]->DonAccAng );

                            }

                            if ( ( Dnr[dc]->Hybrid != Nsp2 && Dnr[dc]->Hybrid != Osp2 &&
                                    Acc[ac]->Hybrid != Nsp2 && Acc[ac]->Hybrid != Osp2 ) ||
                                    ( Acc[ac]->Hybrid != Nsp2 && Acc[ac]->Hybrid != Osp2 &&
                                      ( Dnr[dc]->Hybrid == Nsp2 || Dnr[dc]->Hybrid == Osp2 ) &&
                                      HBond[hc]->AccDonAng <= ACCDONANG ) ||
                                    ( Dnr[dc]->Hybrid != Nsp2 && Dnr[dc]->Hybrid != Osp2 &&
                                      ( Acc[ac]->Hybrid == Nsp2 || Acc[ac]->Hybrid == Osp2 ) &&
                                      HBond[hc]->DonAccAng <= DONACCANG ) ||
                                    ( ( Dnr[dc]->Hybrid == Nsp2 || Dnr[dc]->Hybrid == Osp2 ) &&
                                      ( Acc[ac]->Hybrid == Nsp2 || Acc[ac]->Hybrid == Osp2 ) &&
                                      HBond[hc]->AccDonAng <= ACCDONANG &&
                                      HBond[hc]->DonAccAng <= DONACCANG ) )
                                HBond[hc]->ExistHydrBondRose = STRIDE_YES;
                        }
                    }
                }

            }

            if ( ( HBond[hc]->ExistPolarInter && HBond[hc]->Energy < 0.0 )
                    || HBond[hc]->ExistHydrBondRose || HBond[hc]->ExistHydrBondBaker )
            {
                HBond[hc]->Dnr = Dnr[dc];
                HBond[hc]->Acc = Acc[ac];
                BondedDonor[dc] = STRIDE_YES;
                BondedAcceptor[ac] = STRIDE_YES;
                if ( ( ccd = FindChain ( Chain,NChain,Dnr[dc]->Chain->Id ) ) != ERR )
                {
                    if ( Chain[ccd]->Rsd[Dnr[dc]->D_Res]->Inv->NBondDnr < MAXRESDNR )
                        Chain[ccd]->Rsd[Dnr[dc]->D_Res]->Inv->
                        HBondDnr[Chain[ccd]->Rsd[Dnr[dc]->D_Res]->Inv->NBondDnr++] = hc;
                    else
                        printf ( "Residue %s %s of chain %i is involved in more than %d hydrogen bonds (%d)\n",
                                 Chain[ccd]->Rsd[Dnr[dc]->D_Res]->ResType,
                                 Chain[ccd]->Rsd[Dnr[dc]->D_Res]->PDB_ResNumb,
                                 Chain[ccd]->ChainId,
                                 MAXRESDNR,Chain[ccd]->Rsd[Dnr[dc]->D_Res]->Inv->NBondDnr );
                }
                if ( ( cca  = FindChain ( Chain,NChain,Acc[ac]->Chain->Id ) ) != ERR )
                {
                    if ( Chain[cca]->Rsd[Acc[ac]->A_Res]->Inv->NBondAcc < MAXRESACC )
                        Chain[cca]->Rsd[Acc[ac]->A_Res]->Inv->
                        HBondAcc[Chain[cca]->Rsd[Acc[ac]->A_Res]->Inv->NBondAcc++] = hc;
                    else
                        printf ( "Residue %s %s of chain %i is involved in more than %d hydrogen bonds (%d)\n",
                                 Chain[cca]->Rsd[Acc[ac]->A_Res]->ResType,
                                 Chain[cca]->Rsd[Acc[ac]->A_Res]->PDB_ResNumb,
                                 Chain[cca]->ChainId,
                                 MAXRESDNR,Chain[cca]->Rsd[Acc[ac]->A_Res]->Inv->NBondAcc );
                }
                if ( ccd != cca && ccd != ERR )
                {
                    Chain[ccd]->Rsd[Dnr[dc]->D_Res]->Inv->InterchainHBonds = STRIDE_YES;
                    Chain[cca]->Rsd[Acc[ac]->A_Res]->Inv->InterchainHBonds = STRIDE_YES;
                    if ( HBond[hc]->ExistHydrBondRose )
                    {
                        Chain[0]->NHydrBondInterchain++;
                        Chain[0]->NHydrBondTotal++;
                    }
                }
                else
                    if ( ccd == cca && ccd != ERR && HBond[hc]->ExistHydrBondRose )
                    {
                        Chain[ccd]->NHydrBond++;
                        Chain[0]->NHydrBondTotal++;
                    }
                hc++;
            }
            else
                free ( HBond[hc] );
        }
    }
    
    for ( i=0; i<NDnr; i++ )
        if ( !BondedDonor[i] )
            free ( Dnr[i] );
    for ( i=0; i<NAcc; i++ )
        if ( !BondedAcceptor[i] )
            free ( Acc[i] );

//	if ( NDnr )
        free ( BondedDonor );
//	if ( NAcc )
        free ( BondedAcceptor );

    free(Dnr);
    free(Acc);

    return ( hc );
}

int Stride::NoDoubleHBond( HBOND **HBond, int NHBond ) {

    int i, j, NExcl=0;

    for ( i=0; i<NHBond-1; i++ )
        for ( j=i+1; j<NHBond; j++ )
            if ( HBond[i]->Dnr->D_Res == HBond[j]->Dnr->D_Res &&
                    HBond[i]->Dnr->Chain->Id == HBond[j]->Dnr->Chain->Id &&
                    HBond[i]->ExistPolarInter && HBond[j]->ExistPolarInter )
            {
                if ( HBond[i]->Energy < 5.0*HBond[j]->Energy )
                {
                    HBond[j]->ExistPolarInter = STRIDE_NO;
                    NExcl++;
                }
                else
                    if ( HBond[j]->Energy < 5.0*HBond[i]->Energy )
                    {
                        HBond[i]->ExistPolarInter = STRIDE_NO;
                        NExcl++;
                    }
            }

    return ( NExcl );
}

void Stride::DiscrPhiPsi( CHAIN **Chain, int NChain, COMMAND *Cmd ) {
    int i, Res, Cn;
    RESIDUE *r;

    for ( Cn=0; Cn<NChain; Cn++ )
    {

        for ( Res=0; Res<Chain[Cn]->NRes; Res++ )
        {

            r = Chain[Cn]->Rsd[Res];

            r->Prop->PhiZn = ERR;
            r->Prop->PsiZn = ERR;

            if ( Res != 0 )
            {
                for ( i=0; i<Cmd->NPixel; i++ )
                    if ( r->Prop->Phi  >  MINPHIPSI+ ( float ) ( i ) *Cmd->PhiPsiStep &&
                            r->Prop->Phi <=  MINPHIPSI+ ( float ) ( i+1 ) *Cmd->PhiPsiStep )
                    {
                        r->Prop->PhiZn = i;
                        break;
                    }
            }

            if ( Res != Chain[Cn]->NRes-1 )
            {
                for ( i=0; i<Cmd->NPixel; i++ )
                    if ( r->Prop->Psi  >  MINPHIPSI+ ( float ) ( i ) *Cmd->PhiPsiStep &&
                            r->Prop->Psi <=  MINPHIPSI+ ( float ) ( i+1 ) *Cmd->PhiPsiStep )
                    {
                        r->Prop->PsiZn = i;
                        break;
                    }
            }

        }

        for ( Res=0; Res<Chain[Cn]->NRes; Res++ )
        {
            r = Chain[Cn]->Rsd[Res];
            if ( Res != 0 && r->Prop->PsiZn == ERR )
                r->Prop->PsiZn = Chain[Cn]->Rsd[Res-1]->Prop->PsiZn;
            if ( Res != Chain[Cn]->NRes-1 && r->Prop->PhiZn == ERR )
                r->Prop->PhiZn = Chain[Cn]->Rsd[Res+1]->Prop->PhiZn;
        }

    }
}

void Stride::Helix( CHAIN **Chain, int Cn, HBOND **HBond, COMMAND *Cmd, float **PhiPsiMap )
{
    int BondNumb, i;
    float *Prob, CONSTf;
    RESIDUE **r;

    CONSTf = 1+Cmd->C1_H;

    Prob = ( float * ) ckalloc ( MAX_RES*sizeof ( float ) );

    for ( i=0; i<Chain[Cn]->NRes; i++ )
        Prob[i] = 0.0;


    for ( i=0; i<Chain[Cn]->NRes-5; i++ )
    {

        r = &Chain[Cn]->Rsd[i];

        if ( r[0]->Prop->PhiZn != ERR && r[0]->Prop->PsiZn != ERR &&
                r[1]->Prop->PhiZn != ERR && r[1]->Prop->PsiZn != ERR &&
                r[2]->Prop->PhiZn != ERR && r[2]->Prop->PsiZn != ERR &&
                r[3]->Prop->PhiZn != ERR && r[3]->Prop->PsiZn != ERR &&
                r[4]->Prop->PhiZn != ERR && r[4]->Prop->PsiZn != ERR )
        {

            if ( ( BondNumb = FindPolInt ( HBond,r[4],r[0] ) ) != ERR )
            {
                Prob[i] = HBond[BondNumb]->Energy* ( CONSTf+Cmd->C2_H*
                                                     0.5f * ( PhiPsiMap[r[0]->Prop->PhiZn][r[0]->Prop->PsiZn]+
                                                            PhiPsiMap[r[4]->Prop->PhiZn][r[4]->Prop->PsiZn] ) );

            }
        }
    }

    for ( i=0; i<Chain[Cn]->NRes-5; i++ )
    {

        if ( Prob[i] < Cmd->Treshold_H1 && Prob[i+1] < Cmd->Treshold_H1 )
        {

            r = &Chain[Cn]->Rsd[i];

            r[1]->Prop->Asn = 'H';
            r[2]->Prop->Asn = 'H';
            r[3]->Prop->Asn = 'H';
            r[4]->Prop->Asn = 'H';
            if ( r[0]->Prop->PhiZn!= ERR && r[0]->Prop->PsiZn != ERR &&
                    PhiPsiMap[r[0]->Prop->PhiZn][r[0]->Prop->PsiZn] > Cmd->Treshold_H3 )
                r[0]->Prop->Asn = 'H';
            if ( r[5]->Prop->PhiZn != ERR && r[5]->Prop->PsiZn != ERR &&
                    PhiPsiMap[r[5]->Prop->PhiZn][r[5]->Prop->PsiZn] > Cmd->Treshold_H4 )
                r[5]->Prop->Asn = 'H';
        }
    }

    for ( i=0; i<Chain[Cn]->NRes-4; i++ )
    {

        r = &Chain[Cn]->Rsd[i];

        if ( FindBnd ( HBond,r[3],r[0] ) != ERR && FindBnd ( HBond,r[4],r[1] ) != ERR &&
                /*************************** This should be improved **************************/
                ( ( r[1]->Prop->Asn != 'H' && r[2]->Prop->Asn != 'H' ) ||
                  ( r[2]->Prop->Asn != 'H' && r[3]->Prop->Asn != 'H' ) ) )
            /******************************************************************************/
        {
            r[1]->Prop->Asn = 'G';
            r[2]->Prop->Asn = 'G';
            r[3]->Prop->Asn = 'G';
        }
    }

    for ( i=0; i<Chain[Cn]->NRes-6; i++ )
    {

        r = &Chain[Cn]->Rsd[i];

        if ( FindBnd ( HBond,r[5],r[0] ) != ERR && FindBnd ( HBond,r[6],r[1] ) != ERR &&
                r[1]->Prop->Asn == 'C' && r[2]->Prop->Asn == 'C' &&
                r[3]->Prop->Asn == 'C' && r[4]->Prop->Asn == 'C' &&
                r[5]->Prop->Asn == 'C' )
        {
            r[1]->Prop->Asn = 'I';
            r[2]->Prop->Asn = 'I';
            r[3]->Prop->Asn = 'I';
            r[4]->Prop->Asn = 'I';
            r[5]->Prop->Asn = 'I';
        }
    }

    free ( Prob );
}

void Stride::Sheet( CHAIN **Chain, int Cn1, int Cn2, HBOND **HBond, COMMAND *Cmd, float **PhiPsiMap )
{
    PATTERN **PatN, **PatP;
    RESIDUE *Res1, *Res3, *Res2, *Res4, *ResA, *ResB, *Res1m1, *Res3p1;
    int R1, R3, R2, R4, RA, RB, PatCntN = 0, PatCntP = 0, Beg;
    char *AntiPar1, *Par1, *AntiPar2, *Par2;
    int i;

    PatN = ( PATTERN ** ) ckalloc ( MAXHYDRBOND*sizeof ( PATTERN * ) );
    PatP = ( PATTERN ** ) ckalloc ( MAXHYDRBOND*sizeof ( PATTERN * ) );

    AntiPar1  = ( char * ) ckalloc ( Chain[Cn1]->NRes*sizeof ( char ) ); /* Antiparallel strands */
    Par1      = ( char * ) ckalloc ( Chain[Cn1]->NRes*sizeof ( char ) ); /* Parallel strands */
    AntiPar2  = ( char * ) ckalloc ( Chain[Cn2]->NRes*sizeof ( char ) ); /* Antiparallel strands */
    Par2      = ( char * ) ckalloc ( Chain[Cn2]->NRes*sizeof ( char ) ); /* Parallel strands */

    for ( i=0; i<Chain[Cn1]->NRes; i++ )
    {
        AntiPar1[i] = 'C';
        Par1[i] = 'C';
    }

    for ( i=0; i<Chain[Cn2]->NRes; i++ )
    {
        AntiPar2[i] = 'C';
        Par2[i] = 'C';
    }

    for ( R1=0; R1<Chain[Cn1]->NRes; R1++ )
    {

        Res1   = Chain[Cn1]->Rsd[R1];

        if ( ( !Res1->Inv->NBondDnr && !Res1->Inv->NBondAcc ) ||
                ( ( Cn1 != Cn2 ) && !Res1->Inv->InterchainHBonds ) )
            continue;

        RA     = R1+1;
        R2     = R1+2;
        Res1m1 = Chain[Cn1]->Rsd[R1-1];
        ResA   = Chain[Cn1]->Rsd[RA];
        Res2   = Chain[Cn1]->Rsd[R2];

        if ( R2 >= Chain[Cn1]->NRes ||
                Res1->Prop->PhiZn == ERR || Res1->Prop->PsiZn == ERR ||
                Res2->Prop->PhiZn == ERR || Res2->Prop->PsiZn == ERR ||
                ResA->Prop->PhiZn == ERR || ResA->Prop->PsiZn == ERR )
            continue;

        if ( Cn1 != Cn2 )
            Beg = 0;
        else
            Beg = R1+1;

        for ( R3=Beg; R3<Chain[Cn2]->NRes; R3++ )
        {

            /* Process anti-parallel strands */

            Res3   = Chain[Cn2]->Rsd[R3];

            if ( ( !Res3->Inv->NBondAcc && !Res3->Inv->NBondDnr ) ||
                    ( ( Cn1 != Cn2 ) && !Res3->Inv->InterchainHBonds ) )
                continue;

            RB     = R3-1;
            R4     = R3-2;
            Res3p1 = Chain[Cn2]->Rsd[R3+1];
            ResB   = Chain[Cn2]->Rsd[RB];
            Res4   = Chain[Cn2]->Rsd[R4];

            if ( Cn1 != Cn2 || R3 - R1 >= 3 )
                Link ( HBond,Chain,Cn1,Cn2,Res1,Res3,Res3,Res1,Res1,Res3,
                       PhiPsiMap,PatN,&PatCntN,"1331",Cmd->Treshold_E1,Cmd,0 );

            if ( R2 < Chain[Cn1]->NRes && ( ( Cn1 != Cn2 && R4 >= 0 ) || R4-R2 >=2 ) )
                Link ( HBond,Chain,Cn2,Cn1,Res3,Res1,Res2,Res4,ResB,ResA,
                       PhiPsiMap,PatN,&PatCntN,"3124",Cmd->Treshold_E1,Cmd,0 );

            if ( ( ( Cn1 != Cn2 && RB >= 0 ) || RB-R1 > 4 ) &&
                    ( RA >= Chain[Cn1]->NRes || ( Cn1 == Cn2 && R3-RA <= 4 ) ||
                      !Link ( HBond,Chain,Cn1,Cn2,Res1,Res3,Res3,ResA,NULL,Res3,
                              PhiPsiMap,PatN,&PatCntN,"133A",Cmd->Treshold_E1,Cmd,1 ) )
                    &&
                    ( R1-1 < 0 ||
                      !Link ( HBond,Chain,Cn1,Cn2,Res1m1,ResB,ResB,Res1,NULL,ResB,
                              PhiPsiMap,PatN,&PatCntN,"1-BB1",Cmd->Treshold_E1,Cmd,1 ) ) )
                Link ( HBond,Chain,Cn1,Cn2,Res1,Res3,ResB,Res1,Res1,NULL,
                       PhiPsiMap,PatN,&PatCntN,"13B1",Cmd->Treshold_E1,Cmd,0 );

            if ( ( RA < Chain[Cn1]->NRes && ( Cn1 != Cn2 || R3-RA > 4 ) ) &&
                    ( ( Cn1 == Cn2 && RB-R1 <= 4 ) || ( Cn1 != Cn2 && RB < 0 ) ||
                      !Link ( HBond,Chain,Cn1,Cn2,Res1,Res3,ResB,Res1,Res1,NULL,
                              PhiPsiMap,PatN,&PatCntN,"13B1",Cmd->Treshold_E1,Cmd,1 ) )
                    &&
                    ( R3+1 >= Chain[Cn2]->NRes ||
                      !Link ( HBond,Chain,Cn1,Cn2,ResA,Res3p1,Res3,ResA,ResA,NULL,
                              PhiPsiMap,PatN,&PatCntN,"A3+3A",Cmd->Treshold_E1,Cmd,1 ) ) )
                Link ( HBond,Chain,Cn1,Cn2,Res1,Res3,Res3,ResA,NULL,Res3,
                       PhiPsiMap,PatN,&PatCntN,"133A",Cmd->Treshold_E1,Cmd,0 );

            /* Process parallel strands */

            R4 = R3+2;
            RB = R3+1;
            ResB   = Chain[Cn2]->Rsd[RB];
            Res4   = Chain[Cn2]->Rsd[R4];

            if ( ( Cn1 == Cn2 && abs ( R3-R1 ) <= 3 ) || R4 >= Chain[Cn2]->NRes ) continue;

            if ( R2 < Chain[Cn1]->NRes && ( Cn1 != Cn2 || abs ( R2-R3 ) > 3 ) )
                Link ( HBond,Chain,Cn2,Cn1,Res3,Res1,Res2,Res3,Res3,ResA,
                       PhiPsiMap,PatP,&PatCntP,"3123",Cmd->Treshold_E2,Cmd,0 );

            if ( R4 < Chain[Cn2]->NRes && ( Cn1 != Cn2 || abs ( R4-R1 ) > 3 ) )
                Link ( HBond,Chain,Cn1,Cn2,Res1,Res3,Res4,Res1,Res1,ResB,
                       PhiPsiMap,PatP,&PatCntP,"1341",Cmd->Treshold_E2,Cmd,0 );
        }
    }

    FilterAntiPar ( PatN,PatCntN );
    FilterPar ( PatP,PatCntP );

    MergePatternsAntiPar ( PatN,PatCntN );
    MergePatternsPar ( PatP,PatCntP );

    FillAsnAntiPar ( AntiPar1,AntiPar2,Chain,Cn1,Cn2,PatN,PatCntN,Cmd );
    FillAsnPar ( Par1,Par2,Chain,Cn1,Cn2,PatP,PatCntP,Cmd );

    Bridge ( AntiPar1,AntiPar2,Chain,Cn1,Cn2,PatN,PatCntN );
    Bridge ( Par1,Par2,Chain,Cn1,Cn2,PatP,PatCntP );

    for ( i=0; i<Chain[Cn1]->NRes; i++ )
        if ( AntiPar1[i] == 'N' || Par1[i] == 'P' )
            Chain[Cn1]->Rsd[i]->Prop->Asn = 'E';
        else
            if ( AntiPar1[i] == 'B' || Par1[i] == 'B' )
                Chain[Cn1]->Rsd[i]->Prop->Asn = 'B';
            else
                if ( AntiPar1[i] == 'b' || Par1[i] == 'b' )
                    Chain[Cn1]->Rsd[i]->Prop->Asn = 'b';

    for ( i=0; i<Chain[Cn2]->NRes; i++ )
        if ( Chain[Cn2]->Rsd[i]->Prop->Asn == 'E' )
            continue;
        else
            if ( AntiPar2[i] == 'N' || Par2[i] == 'P' )
                Chain[Cn2]->Rsd[i]->Prop->Asn = 'E';
            else
                if ( AntiPar2[i] == 'B' || Par2[i] == 'B' )
                    Chain[Cn2]->Rsd[i]->Prop->Asn = 'B';
                else
                    if ( AntiPar2[i] == 'b' || Par2[i] == 'b' )
                        Chain[Cn2]->Rsd[i]->Prop->Asn = 'b';

    /*
      for( i=0; i<PatCntN; i++ )
        free(PatN[i]);
      for( i=0; i<PatCntP; i++ )
        free(PatP[i]);
    */
    free ( PatN );
    free ( PatP );
    free ( AntiPar1 );
    free ( Par1 );
    free ( AntiPar2 );
    free ( Par2 );
}

void Stride::BetaTurn( CHAIN **Chain, int Cn )
{
    int i;
    RESIDUE **r;
    TURN *t;
    int CA1, CA4, Tn;
    float Phi2, Phi3, Psi2, Psi3, Range1 = 30.0, Range2 = 45.0;
    char TurnType;

    for ( i=0; i<Chain[Cn]->NRes-4; i++ )
    {

        r = &Chain[Cn]->Rsd[i];

        if ( r[1]->Prop->Asn == 'H' || r[2]->Prop->Asn == 'H' ||
                r[1]->Prop->Asn == 'G' || r[2]->Prop->Asn == 'G' ||
                r[1]->Prop->Asn == 'I' || r[2]->Prop->Asn == 'G' ||
                !FindAtom ( Chain[Cn],i,"CA",&CA1 ) || !FindAtom ( Chain[Cn],i+3,"CA",&CA4 ) ||
                Dist ( r[0]->Coord[CA1],r[3]->Coord[CA4] ) > 7.0 )
            continue;

        Phi2 = r[1]->Prop->Phi;
        Psi2 = r[1]->Prop->Psi;
        Phi3 = r[2]->Prop->Phi;
        Psi3 = r[2]->Prop->Psi;

        if ( TurnCondition ( Phi2,-60.0,Psi2,-30,Phi3,-90.0,Psi3,0,Range1,Range2 ) )
            TurnType = '1';
        else
            if ( TurnCondition ( Phi2,60.0,Psi2,30,Phi3,90.0,Psi3,0,Range1,Range2 ) )
                TurnType = '2';
            else
                if ( TurnCondition ( Phi2,-60.0,Psi2,120,Phi3,80.0,Psi3,0,Range1,Range2 ) )
                    TurnType = '3';
                else
                    if ( TurnCondition ( Phi2,60.0,Psi2,-120,Phi3,-80.0,Psi3,0,Range1,Range2 ) )
                        TurnType = '4';
                    else
                        if ( TurnCondition ( Phi2,-60.0,Psi2,120,Phi3,-90.0,Psi3,0,Range1,Range2 ) )
                            TurnType = '5';
                        else
                            if ( TurnCondition ( Phi2,-120.0,Psi2,120,Phi3,-60.0,Psi3,0,Range1,Range2 ) )
                                TurnType = '6';
                            else
                                if ( TurnCondition ( Phi2,-60.0,Psi2,-30,Phi3,-120.0,Psi3,120,Range1,Range2 ) )
                                    TurnType = '7';
                                else
                                    TurnType = '8';

        if ( r[0]->Prop->Asn == 'C' )
            r[0]->Prop->Asn = 'T';

        if ( r[1]->Prop->Asn == 'C' )
            r[1]->Prop->Asn = 'T';

        if ( r[2]->Prop->Asn == 'C' )
            r[2]->Prop->Asn = 'T';

        if ( r[3]->Prop->Asn == 'C' )
            r[3]->Prop->Asn = 'T';

        Tn = Chain[Cn]->NAssignedTurn;
        Chain[Cn]->AssignedTurn[Tn] = ( TURN * ) ckalloc ( sizeof ( TURN ) );
        t = Chain[Cn]->AssignedTurn[Tn];
        strcpy ( t->Res1,r[0]->ResType );
        strcpy ( t->Res2,r[3]->ResType );
        strcpy ( t->PDB_ResNumb1,r[0]->PDB_ResNumb );
        strcpy ( t->PDB_ResNumb2,r[3]->PDB_ResNumb );
        t->TurnType = TurnType;
        Chain[Cn]->NAssignedTurn++;

    }
}

void Stride::GammaTurn( CHAIN **Chain, int Cn, HBOND **HBond )
{
    int i;
    RESIDUE **r;
    TURN *t;
    int Tn;
    float Phi2, Psi2;
    char TurnType, Asn;

    for ( i=0; i<Chain[Cn]->NRes-2; i++ )
    {

        r = &Chain[Cn]->Rsd[i-1];

        Asn = r[2]->Prop->Asn;

        if ( Asn == 'H' || Asn == 'T' || Asn == 'G' || Asn == 'I' ||
                FindBnd ( HBond,r[3],r[1] ) == ERR ||
                ( i > 0 && FindBnd ( HBond,r[3],r[0] ) != ERR ) ||
                ( i < Chain[Cn]->NRes-3 && FindBnd ( HBond,r[4],r[1] ) != ERR ) )
            continue;

        Phi2 = r[2]->Prop->Phi;
        Psi2 = r[2]->Prop->Psi;

        if ( Phi2 > 0.0 && Psi2 < 0.0 )
            TurnType = '@';
        else
            if ( Phi2 < 0.0 && Psi2 > 0.0 )
                TurnType = '&';
            else
                continue;

        if ( r[1]->Prop->Asn == 'C' )
            r[1]->Prop->Asn = 'T';

        if ( r[2]->Prop->Asn == 'C' )
            r[2]->Prop->Asn = 'T';

        if ( r[3]->Prop->Asn == 'C' )
            r[3]->Prop->Asn = 'T';

        Tn = Chain[Cn]->NAssignedTurn;
        Chain[Cn]->AssignedTurn[Tn] = ( TURN * ) ckalloc ( sizeof ( TURN ) );
        t = Chain[Cn]->AssignedTurn[Tn];
        strcpy ( t->Res1,r[1]->ResType );
        strcpy ( t->Res2,r[3]->ResType );
        strcpy ( t->PDB_ResNumb1,r[1]->PDB_ResNumb );
        strcpy ( t->PDB_ResNumb2,r[3]->PDB_ResNumb );
        t->TurnType = TurnType;
        Chain[Cn]->NAssignedTurn++;
    }
}

int Stride::TurnCondition( float Phi2,float Phi2S,float Psi2,float Psi2S,
                    float Phi3,float Phi3S,float Psi3,float Psi3S,
                    float Range1,float Range2 )
{
    if ( ( IN_STRIDE ( Phi2,Phi2S,Range2 ) ==STRIDE_YES && IN_STRIDE ( Psi2,Psi2S,Range1 ) ==STRIDE_YES &&
            IN_STRIDE ( Phi3,Phi3S,Range1 ) ==STRIDE_YES && IN_STRIDE ( Psi3,Psi3S,Range1 ) ==STRIDE_YES )
            ||
            ( IN_STRIDE ( Phi2,Phi2S,Range1 ) ==STRIDE_YES && IN_STRIDE ( Psi2,Psi2S,Range2 ) ==STRIDE_YES &&
              IN_STRIDE ( Phi3,Phi3S,Range1 ) ==STRIDE_YES && IN_STRIDE ( Psi3,Psi3S,Range1 ) ==STRIDE_YES )
            ||
            ( IN_STRIDE ( Phi2,Phi2S,Range1 ) ==STRIDE_YES && IN_STRIDE ( Psi2,Psi2S,Range1 ) ==STRIDE_YES &&
              IN_STRIDE ( Phi3,Phi3S,Range2 ) ==STRIDE_YES && IN_STRIDE ( Psi3,Psi3S,Range1 ) ==STRIDE_YES )
            ||
            ( IN_STRIDE ( Phi2,Phi2S,Range1 ) ==STRIDE_YES && IN_STRIDE ( Psi2,Psi2S,Range1 ) ==STRIDE_YES &&
              IN_STRIDE ( Phi3,Phi3S,Range1 ) ==STRIDE_YES && IN_STRIDE ( Psi3,Psi3S,Range2 ) ==STRIDE_YES )
       )
        return ( SUCCESS );

    return ( FAILURE );
}

int Stride::SSBond( CHAIN **Chain, int NChain )
{
    int Res1, Res2, Cn1, Cn2;
    int S1, S2, Bn, Cnt=0;

    for ( Cn1=0; Cn1<NChain; Cn1++ )
        for ( Res1=0; Res1<Chain[Cn1]->NRes; Res1++ )
        {
            if ( strcmp ( Chain[Cn1]->Rsd[Res1]->ResType,"CYS" ) )
                continue;
            for ( Cn2=Cn1; Cn2<NChain; Cn2++ )
                for ( Res2 = ( ( Cn2 == Cn1 ) ? Res1+1 : 0 ) ; Res2<Chain[Cn2]->NRes; Res2++ )
                {
                    if ( strcmp ( Chain[Cn2]->Rsd[Res2]->ResType,"CYS" ) )
                        continue;

                    if ( !ExistSSBond ( Chain,NChain,Cn1,Cn2,
                                        Chain[Cn1]->Rsd[Res1]->PDB_ResNumb,
                                        Chain[Cn2]->Rsd[Res2]->PDB_ResNumb ) &&
                            FindAtom ( Chain[Cn1],Res1,"SG",&S1 ) && FindAtom ( Chain[Cn2],Res2,"SG",&S2 ) &&
                            Dist ( Chain[Cn1]->Rsd[Res1]->Coord[S1],
                                   Chain[Cn2]->Rsd[Res2]->Coord[S2] ) <= SSDIST )
                    {
                        Bn = Chain[0]->NBond;
                        Chain[0]->SSbond[Bn] = ( SSBOND * ) ckalloc ( sizeof ( SSBOND ) );
                        strcpy ( Chain[0]->SSbond[Bn]->PDB_ResNumb1,Chain[Cn1]->Rsd[Res1]->PDB_ResNumb );
                        strcpy ( Chain[0]->SSbond[Bn]->PDB_ResNumb2,Chain[Cn2]->Rsd[Res2]->PDB_ResNumb );
                        Chain[0]->SSbond[Bn]->ChainId1 = Chain[Cn1]->Id;
                        Chain[0]->SSbond[Bn]->ChainId2 = Chain[Cn2]->Id;
                        Chain[0]->SSbond[Bn]->AsnSource = StrideX;
                        Chain[0]->NBond++;
                        Cnt++;
                    }
                }
        }

    return ( Cnt );
}

Stride::BOOLEAN Stride::ExistSSBond( CHAIN **Chain,int NChain, int Cn1,int Cn2,char *Res1,char *Res2 )
{
    int i;
    SSBOND *ptr;

    for ( i=0; i<Chain[0]->NBond; i++ )
    {
        ptr = Chain[0]->SSbond[i];
        if ( ( !strcmp ( Res1,ptr->PDB_ResNumb1 ) &&
                !strcmp ( Res2,ptr->PDB_ResNumb2 ) &&
                FindChain ( Chain,NChain,ptr->ChainId1 ) == Cn1 &&
                FindChain ( Chain,NChain,ptr->ChainId2 ) == Cn2 ) ||
                ( !strcmp ( Res2,ptr->PDB_ResNumb1 ) &&
                  !strcmp ( Res1,ptr->PDB_ResNumb2 ) &&
                  FindChain ( Chain,NChain,ptr->ChainId1 ) == Cn2 &&
                  FindChain ( Chain,NChain,ptr->ChainId2 ) == Cn1 ) )
            return ( SUCCESS );
    }

    return ( FAILURE );
}

void Stride::Report( CHAIN **Chain, int NChain, HBOND **HBond, COMMAND *Cmd )
{
    FILE *Out;

    if ( !strlen ( Cmd->OutFile ) )
        Out = stdout;
    else
        if ( ! ( Out = fopen ( Cmd->OutFile,"w" ) ) )
            die ( "Can not open output file %s\n",Cmd->OutFile );

    ReportShort ( Chain,NChain,Out,Cmd );
    ReportTurnTypes ( Chain,NChain,Out,Cmd );
    ReportSSBonds ( Chain,Out );

    if ( Out != stdout )
        fclose ( Out );

}

void Stride::ReportSSBonds( CHAIN **Chain, FILE *Out )
{
    int i;
    BUFFER Bf, Tmp;
    SSBOND *s;

    if ( !Chain[0]->NBond ) return;

    PrepareBuffer ( Bf,Chain );

    for ( i=0; i<Chain[0]->NBond; i++ )
    {
        s = Chain[0]->SSbond[i];
        sprintf ( Tmp,"LOC  Disulfide    CYS  %4s %c      CYS   %4s %c         ",
                  s->PDB_ResNumb1,SpaceToDash ( s->ChainId1 ),
                  s->PDB_ResNumb2,SpaceToDash ( s->ChainId2 ) );

        if ( s->AsnSource == Pdb )
            strcat ( Tmp,"   PDB" );
        else
            strcat ( Tmp,"STRIDE" );
        Glue ( Bf,Tmp,Out );
    }
}

void Stride::ReportTurnTypes( CHAIN **Chain, int NChain, FILE *Out, COMMAND *Cmd )
{
    int Cn, Tn;
    BUFFER Bf, Tmp;
    TURN *t;

    Tn = 0;
    for ( Cn=0; Cn<NChain; Cn++ )
        if ( Chain[Cn]->Valid )
            Tn += Chain[Cn]->NAssignedTurn;

    if ( !Tn ) return;

    PrepareBuffer ( Bf,Chain );

    for ( Cn=0; Cn<NChain; Cn++ )
    {
        if ( !Chain[Cn]->Valid )
            continue;
        for ( Tn=0; Tn<Chain[Cn]->NAssignedTurn; Tn++ )
        {
            t = Chain[Cn]->AssignedTurn[Tn];
            sprintf ( Tmp,"LOC  %-12s %3s  %4s %c      %3s   %4s %c",
                      Translate ( t->TurnType ),
                      t->Res1,
                      t->PDB_ResNumb1,
                      SpaceToDash ( Chain[Cn]->Id ),
                      t->Res2,
                      t->PDB_ResNumb2,
                      SpaceToDash ( Chain[Cn]->Id ) );

            Glue ( Bf,Tmp,Out );
        }
    }
}

void Stride::ReportShort( CHAIN **Chain, int NChain, FILE *Out, COMMAND *Cmd )
{
    int Cn, i;
    BUFFER Bf, Tmp;
    char *Asn;
    const char *StrTypes = "HGIE";
    int Bound[MAX_ASSIGN][2], NStr;

    if ( !ExistsSecStr ( Chain,NChain ) )
        return;

    PrepareBuffer ( Bf,Chain );

    for ( ; *StrTypes!= '\0'; StrTypes++ )
    {
        for ( Cn=0; Cn<NChain; Cn++ )
        {
            if ( !Chain[Cn]->Valid )
                continue;
            
            Asn = ( char * ) ckalloc ( Chain[Cn]->NRes*sizeof ( char ) );
            ExtractAsn ( Chain,Cn,Asn );
            NStr = Boundaries( Asn, Chain[Cn]->NRes, ( *StrTypes ), Bound );

            for ( i=0; i<NStr; i++ )
            {
                sprintf ( Tmp, "LOC  %-12s %3s  %4i %i      %3s   %4i %i",
                             Translate ( *StrTypes ),
                          Chain[Cn]->Rsd[Bound[i][0]]->ResType,
                          //Chain[Cn]->Rsd[Bound[i][0]]->PDB_ResNumb,
                          Chain[Cn]->Rsd[Bound[i][0]]->ResNumb,
                          //SpaceToDash ( Chain[Cn]->Id ),
                          Chain[Cn]->ChainId,
                          Chain[Cn]->Rsd[Bound[i][1]]->ResType,
                          //Chain[Cn]->Rsd[Bound[i][1]]->PDB_ResNumb,
                          Chain[Cn]->Rsd[Bound[i][1]]->ResNumb,
                          //SpaceToDash ( Chain[Cn]->Id ) );
                          Chain[Cn]->ChainId );
                Glue ( Bf,Tmp,Out );
            }

            free ( Asn );
        }
    }

}

void Stride::PrepareBuffer( BUFFER Bf, CHAIN **Chain )
{
    memset ( Bf,' ',OUTPUTWIDTH );

    strcpy ( Bf+OUTPUTWIDTH-5,Chain[0]->PdbIdent );
    Bf[OUTPUTWIDTH] = '\0';
    Bf[OUTPUTWIDTH-1]   = '\n';

}

void Stride::Glue( const char *String1, const char *String2, FILE *Out )
{
    BUFFER Bf;

    strcpy ( Bf,String1 );
    strncpy ( Bf,String2, ( int ) strlen ( String2 ) );

    fprintf ( Out,"%s",Bf );
}

void* Stride::ckalloc(size_t bytes)
{
  void *ret;
  
  if( !(ret = malloc(bytes)) ) die("Out of  memory\n");

  return ret;	
}

int Stride::Process_ENDMDL( BUFFER Buffer, CHAIN **Chain, int *ChainNumber)
{
  int CC;

  for( CC=0; CC < *ChainNumber; CC++ )
    Chain[CC]->Ter = 1;
  
  return(SUCCESS);
}

int Stride::Process_ATOM( BUFFER Buffer, CHAIN **Chain, int *ChainNumber,
                   BOOLEAN *First_ATOM, COMMAND *Cmd )
{
    char *Field[MAX_FIELD];
    BUFFER Tmp;
    int CC, NR, NA;
    static char LastRes[MAX_CHAIN][RES_FIELD];
    RESIDUE *r;

    // 'chainID' -- ??? --> not exectuted for 1RWE
    if ( Cmd->NActive && !ChInStr( Cmd->Active, SpaceToDash ( Buffer[21] ) ) )
    {
        return ( SUCCESS );
    }

    // 'altLoc' -- alternate locations will be ignored
    if ( Buffer[16] != 'A' && Buffer[16] != ' ' && Buffer[16] != '1' )
    {
        return ( SUCCESS );
    }
    
    // originaly used to set the PDB-ID for all chains (not used now)
    if ( *First_ATOM )
    {
        for ( CC=0; CC<MAX_CHAIN; CC++ )
            strcpy ( LastRes[CC],"XXXX" );
        *First_ATOM = STRIDE_NO;
    }

    // go to the chain with the correct chain number
    for ( CC=0; CC < *ChainNumber && Chain[CC]->Id != Buffer[21] ; CC++ );

    // if a new chain is reached, init chain and increment number of chains
    if ( CC == *ChainNumber )
    {
        InitChain ( &Chain[CC] );
        Chain[CC]->Id = Buffer[21];
        ( *ChainNumber ) ++;
    }
    else
        if ( Chain[CC]->Ter == 1 )
            return ( SUCCESS );

    // atom must have coordinates
    if ( Buffer[34] != '.' || Buffer[42] != '.' || Buffer[50] != '.' )
    {
        printf( "File %s has no coordinates\n", Cmd->InputFile);
        return FAILURE;
    }

    // Residue (i.e. amino acid)
    SplitString ( Buffer+22,Field,1 );
    if ( strcmp ( Field[0],LastRes[CC] ) )
    {
        if ( strcmp ( LastRes[CC],"XXXX" ) && !FindAtom ( Chain[CC],Chain[CC]->NRes,"CA",&NA ) )
        {
            free ( Chain[CC]->Rsd[Chain[CC]->NRes] );
            Chain[CC]->NRes--;
        }
        if ( strcmp ( LastRes[CC],"XXXX" ) )
        {
            Chain[CC]->NRes++;
        }
        NR = Chain[CC]->NRes;
        strcpy ( LastRes[CC],Field[0] );
        Chain[CC]->Rsd[NR] = ( RESIDUE * ) ckalloc ( sizeof ( RESIDUE ) );
        strcpy ( Chain[CC]->Rsd[NR]->PDB_ResNumb,LastRes[CC] );
        Chain[CC]->Rsd[NR]->NAtom = 0;
        SplitString ( Buffer+17,Field,1 );
        strcpy ( Chain[CC]->Rsd[NR]->ResType,Field[0] );
    }
    else
        NR = Chain[CC]->NRes;

    NA = Chain[CC]->Rsd[NR]->NAtom;

    if ( Buffer[16] != ' ' )
    {
        strcpy ( Tmp,Buffer );
        Tmp[16] = ' ';
        SplitString ( Tmp+12,Field,1 );
    }
    else
        SplitString ( Buffer+12,Field,1 );

    r = Chain[CC]->Rsd[NR];
    strcpy ( r->AtomType[NA],Field[0] );


    strcpy ( Tmp,Buffer );
    Buffer[38] = ' ';
    SplitString ( Tmp+30,Field,1 );
    r->Coord[NA][0] = (float)atof( Field[0] );

    strcpy ( Tmp,Buffer );
    Buffer[46] = ' ';
    SplitString ( Tmp+38,Field,1 );
    r->Coord[NA][1] = (float)atof( Field[0] );

    strcpy ( Tmp,Buffer );
    Buffer[54] = ' ';
    SplitString ( Tmp+46,Field,1 );
    r->Coord[NA][2] = (float)atof( Field[0] );

    if ( Buffer[57] == '.' )
    {
        strcpy ( Tmp,Buffer );
        Tmp[60] = ' ';
        SplitString ( Tmp+54,Field,1 );
        r->Occupancy[NA] = (float)atof( Field[0] );
    }
    else
        r->Occupancy[NA] = -1.00;

    SplitString ( Buffer+63,Field,1 );
    r->TempFactor[NA] = (float)atof( Field[0] );

    r->NAtom++;

    if ( r->NAtom > MAX_AT_IN_RES-1 )
    {
        printf( "File %s has too many atoms in residue %s %s %c\n",
            Cmd->InputFile,r->ResType,r->PDB_ResNumb,Chain[CC]->Id);
        return FAILURE;
    }

    return ( SUCCESS );
}

int Stride::FindAtom( CHAIN *Chain, int ResNumb, const char *Atom, int *AtNumb)
{

  for( (*AtNumb)=0; (*AtNumb)<Chain->Rsd[ResNumb]->NAtom; (*AtNumb)++ )
    if( !strcmp(Atom,Chain->Rsd[ResNumb]->AtomType[(*AtNumb)]) )
       return(SUCCESS);

  *AtNumb = ERR;
  return(FAILURE);
}

char Stride::SpaceToDash( char Id)
{
  static char NewId;

  if( Id == ' ' )
    NewId = '-';
  else
    NewId = Id;

  return(NewId);
}

Stride::BOOLEAN Stride::ChInStr( char *String, char Char)
{
  if( strchr(String,toupper(Char)) || 
      strchr(String,Char) ||
      strchr(String,tolower(Char)) )
    return(STRIDE_YES);
  
  return(STRIDE_NO);
}

void Stride::PHI( CHAIN *Chain, int Res )
{
    int C_Prev, N_Curr, CA_Curr, C_Curr;
    RESIDUE *r, *rr;

    r = Chain->Rsd[Res];
    r->Prop->Phi = 360.0;

    if ( Res == 0 )
        return;

    rr = Chain->Rsd[Res-1];

    if ( FindAtom ( Chain,Res-1,"C",&C_Prev ) && FindAtom ( Chain,Res,"N",&N_Curr )   &&
            FindAtom ( Chain,Res,"CA",&CA_Curr ) && FindAtom ( Chain,Res,"C",&C_Curr )   &&
            Dist ( rr->Coord[C_Prev],r->Coord[N_Curr] ) < BREAKDIST )
    {
        r->Prop->Phi = Torsion ( rr->Coord[C_Prev],r->Coord[N_Curr],
                                 r->Coord[CA_Curr],r->Coord[C_Curr] );
    }
}

void Stride::PSI( CHAIN *Chain, int Res )
{
    int N_Curr, CA_Curr, C_Curr, N_Next;
    RESIDUE *r, *rr;

    r = Chain->Rsd[Res];
    r->Prop->Psi = 360.0;

    if ( Res == Chain->NRes-1 )
        return;

    rr = Chain->Rsd[Res+1];

    if ( FindAtom ( Chain,Res,"N",&N_Curr ) && FindAtom ( Chain,Res,"CA",&CA_Curr ) &&
            FindAtom ( Chain,Res,"C",&C_Curr ) && FindAtom ( Chain,Res+1,"N",&N_Next ) &&
            Dist ( r->Coord[C_Curr],rr->Coord[N_Next] ) < BREAKDIST )
    {

        r->Prop->Psi = Torsion ( r->Coord[N_Curr],r->Coord[CA_Curr],
                                 r->Coord[C_Curr],rr->Coord[N_Next] );
    }
}

float Stride::Torsion( float *Coord1, float *Coord2, float *Coord3, float *Coord4 )
{
    double Comp[3][3], ScalarProd, TripleScalarProd, AbsTorsAng;
    double Perp_123[3], Perp_234[3], Len_Perp_123, Len_Perp_234;
    int i, j, k;

    /* Find the components of the three bond vectors */
    for ( i=0; i<3; i++ )
    {
        Comp[0][i] = ( double ) ( Coord2[i]-Coord1[i] );
        Comp[1][i] = ( double ) ( Coord3[i]-Coord2[i] );
        Comp[2][i] = ( double ) ( Coord4[i]-Coord3[i] );
    }

    /* Calculate vectors perpendicular to the planes 123 and 234 */
    Len_Perp_123 = 0.0; Len_Perp_234 = 0.0;
    for ( i=0; i<3; i++ )
    {
        j = ( i+1 ) %3;
        k = ( j+1 ) %3;
        Perp_123[i] = Comp[0][j]*Comp[1][k] - Comp[0][k]*Comp[1][j];
        Perp_234[i] = Comp[1][j]*Comp[2][k] - Comp[1][k]*Comp[2][j];
        Len_Perp_123 += Perp_123[i]*Perp_123[i];
        Len_Perp_234 += Perp_234[i]*Perp_234[i];
    }

    Len_Perp_123 = sqrt ( Len_Perp_123 );
    Len_Perp_234 = sqrt ( Len_Perp_234 );

    /* Normalize the vectors perpendicular to 123 and 234 */
    for ( i=0; i<3; i++ )
    {
        Perp_123[i] /= Len_Perp_123;
        Perp_234[i] /= Len_Perp_234;
    }

    /* Find the scalar product of the unit normals */
    ScalarProd = 0.0;
    for ( i=0; i<3; i++ )
        ScalarProd += Perp_123[i]*Perp_234[i];

    /* Find the absolute value of the torsion angle */
    if ( ScalarProd > 0.0 && fabs ( ScalarProd - 1.0 ) < Eps )
        ScalarProd -= Eps;
    else
        if ( ScalarProd < 0.0 && fabs ( ScalarProd + 1.0 ) < Eps )
            ScalarProd += Eps;
    AbsTorsAng = RADDEG*acos ( ScalarProd );

    /* Find the triple scalar product of the three bond vectors */
    TripleScalarProd = 0.0;
    for ( i=0; i<3; i++ )
        TripleScalarProd += Comp[0][i]*Perp_234[i];

    /* Torsion angle has the sign of the triple scalar product */
    return ( ( TripleScalarProd > 0.0 ) ? ( float ) AbsTorsAng : ( float ) ( -AbsTorsAng ) );

}

float Stride::Dist( float *Coord1, float *Coord2 )
{
    int i;
    float Dist=0;

    for ( i=0; i<3; i++ )
        Dist += ( Coord1[i]-Coord2[i] ) * ( Coord1[i]-Coord2[i] );

    return ( sqrt ( Dist ) );
}

int Stride::FindDnr( CHAIN *Chain, DONOR **Dnr, int *NDnr, COMMAND *Cmd )
{

    int Res, dc;
    char Rsd[RES_FIELD];

    dc = *NDnr;

    for ( Res=0; Res<Chain->NRes; Res++ )
    {

        strcpy ( Rsd,Chain->Rsd[Res]->ResType );

        DefineDnr ( Chain, Dnr, &dc, Res, Nsp2, Peptide, 1.90f, 0 );

        if ( !Cmd->SideChainHBond ) continue;

        if ( !strcmp ( Rsd,"TRP" ) )
            DefineDnr ( Chain, Dnr, &dc, Res, Nsp2, Trp, 1.90f, 0 );
        else if ( !strcmp ( Rsd,"ASN" ) ) DefineDnr ( Chain, Dnr, &dc, Res, Nsp2, Asn, 1.90f, 0 );
        else if ( !strcmp ( Rsd,"GLN" ) ) DefineDnr ( Chain, Dnr, &dc, Res, Nsp2, Gln, 1.90f, 0 );
        else if ( !strcmp ( Rsd,"ARG" ) )
        {
            DefineDnr ( Chain, Dnr, &dc, Res, Nsp2, Arg, 1.90f, 1 );
            DefineDnr ( Chain, Dnr, &dc, Res, Nsp2, Arg, 1.90f, 2 );
            DefineDnr ( Chain, Dnr, &dc, Res, Nsp2, Arg, 1.90f, 3 );
        }
        else if ( !strcmp ( Rsd,"HIS" ) )
        {
            DefineDnr ( Chain, Dnr, &dc, Res, Nsp2, His, 1.90f, 1 );
            DefineDnr ( Chain, Dnr, &dc, Res, Nsp2, His, 1.90f, 2 );
        }
        else if ( !strcmp ( Rsd,"LYS" ) ) DefineDnr ( Chain, Dnr, &dc, Res, Nsp3, Lys, 2.10f, 0 );
        else if ( !strcmp ( Rsd,"SER" ) ) DefineDnr ( Chain, Dnr, &dc, Res, Osp3, Ser, 1.70f, 0 );
        else if ( !strcmp ( Rsd,"THR" ) ) DefineDnr ( Chain, Dnr, &dc, Res, Osp3, Thr, 1.70f, 0 );
        else if ( !strcmp ( Rsd,"TYR" ) ) DefineDnr ( Chain, Dnr, &dc, Res, Osp2, Tyr, 1.70f, 0 );
    }

    *NDnr = dc;
    return ( dc );
}

int Stride::DefineDnr( CHAIN *Chain, DONOR **Dnr, int *dc, int Res, enum HYBRID Hybrid,
                              enum GROUP Group, float HB_Radius, int N )
{
    //Dnr[*dc] = ( DONOR * ) ckalloc ( sizeof ( DONOR ) );
    DONOR tempDonor;

    tempDonor.Chain = Chain;
    tempDonor.D_Res = Res;
    if ( Group != Peptide )
        tempDonor.DD_Res = Res;
    else
        tempDonor.DD_Res = Res-1;
    tempDonor.DDI_Res = Res;
    tempDonor.Hybrid = Hybrid;
    tempDonor.Group = Group;
    tempDonor.HB_Radius = HB_Radius;

    if ( Group == Peptide )
    {
        if ( Res != 0 )
        {
            FindAtom ( Chain,Res,"N",&tempDonor.D_At );
            FindAtom ( Chain,Res-1,"C",&tempDonor.DD_At );
        }
        else
        {
            tempDonor.D_At  = ERR;
            tempDonor.DD_At = ERR;
        }
        FindAtom ( Chain,Res,"CA",&tempDonor.DDI_At );
        FindAtom ( Chain,Res,"H",&tempDonor.H );
    }
    else if ( Group == Trp )
    {
        FindAtom ( Chain,Res,"NE1",&tempDonor.D_At );
        FindAtom ( Chain,Res,"CE2",&tempDonor.DD_At );
        FindAtom ( Chain,Res,"CD1",&tempDonor.DDI_At );
    }
    else if ( Group == Asn )
    {
        FindAtom ( Chain,Res,"ND1",&tempDonor.D_At );
        FindAtom ( Chain,Res,"CG",&tempDonor.DD_At );
        FindAtom ( Chain,Res,"CB",&tempDonor.DDI_At );
    }
    else if ( Group == Gln )
    {
        FindAtom ( Chain,Res,"NE2",&tempDonor.D_At );
        FindAtom ( Chain,Res,"CD",&tempDonor.DD_At );
        FindAtom ( Chain,Res,"CG",&tempDonor.DDI_At );
    }
    else if ( Group == Arg )
    {
        if ( N == 1 )
        {
            FindAtom ( Chain,Res,"NE",&tempDonor.D_At );
            FindAtom ( Chain,Res,"CZ",&tempDonor.DD_At );
            FindAtom ( Chain,Res,"CD",&tempDonor.DDI_At );
        }
        else
            if ( N == 2 )
            {
                FindAtom ( Chain,Res,"NH1",&tempDonor.D_At );
                FindAtom ( Chain,Res,"CZ",&tempDonor.DD_At );
                FindAtom ( Chain,Res,"NE",&tempDonor.DDI_At );
            }
            else
                if ( N == 3 )
                {
                    FindAtom ( Chain,Res,"NH2",&tempDonor.D_At );
                    FindAtom ( Chain,Res,"CZ",&tempDonor.DD_At );
                    FindAtom ( Chain,Res,"NE",&tempDonor.DDI_At );
                }
    }
    else if ( Group == His )
    {
        if ( N == 1 )
        {
            FindAtom ( Chain,Res,"ND1",&tempDonor.D_At );
            FindAtom ( Chain,Res,"CG",&tempDonor.DD_At );
            FindAtom ( Chain,Res,"CE1",&tempDonor.DDI_At );
        }
        else if ( N == 2 )
        {
            FindAtom ( Chain,Res,"NE2",&tempDonor.D_At );
            FindAtom ( Chain,Res,"CE1",&tempDonor.DD_At );
            FindAtom ( Chain,Res,"CD2",&tempDonor.DDI_At );
        }
    }
    else if ( Group == Tyr )
    {
        FindAtom ( Chain,Res,"OH",&tempDonor.D_At );
        FindAtom ( Chain,Res,"CZ",&tempDonor.DD_At );
        FindAtom ( Chain,Res,"CE1",&tempDonor.DDI_At );
    }
    else if ( Group == Lys )
    {
        FindAtom ( Chain,Res,"NZ",&tempDonor.D_At );
        FindAtom ( Chain,Res,"CE",&tempDonor.DD_At );
    }
    else if ( Group == Ser )
    {
        FindAtom ( Chain,Res,"OG",&tempDonor.D_At );
        FindAtom ( Chain,Res,"CB",&tempDonor.DD_At );
    }
    else if ( Group == Thr )
    {
        FindAtom ( Chain,Res,"OG1",&tempDonor.D_At );
        FindAtom ( Chain,Res,"CB",&tempDonor.DD_At );
    }

    if ( tempDonor.H == ERR || tempDonor.D_At   == ERR || tempDonor.DD_At  == ERR ||
            ( tempDonor.DDI_At == ERR && ( Hybrid == Nsp2 || Hybrid == Osp2 ) ) )
    {
        //free ( Dnr[*dc] );
        return ( FAILURE );
    }
    else {
        Dnr[*dc] = ( DONOR * ) ckalloc ( sizeof ( DONOR ) );
        memcpy(Dnr[*dc], &tempDonor, sizeof(DONOR));
        ( *dc ) ++;
    }
    return ( SUCCESS );
}

int Stride::FindAcc( CHAIN *Chain, ACCEPTOR **Acc, int *NAcc, COMMAND *Cmd )
{

    int Res, ac;
    char Rsd[RES_FIELD];

    ac = *NAcc;

    for ( Res=0; Res<Chain->NRes; Res++ )
    {
        strcpy ( Rsd,Chain->Rsd[Res]->ResType );

        DefineAcceptor ( Chain, Acc, &ac, Res, Osp2, Peptide, 1.60f, 0 );

        if ( !Cmd->SideChainHBond ) continue;

        if ( !strcmp ( Rsd,"HIS" ) )
        {
            DefineAcceptor ( Chain, Acc, &ac, Res, Nsp2, His, 1.60f, 0 );
            DefineAcceptor ( Chain, Acc, &ac, Res, Nsp2, His, 1.60f, 0 );
        }
        else if ( !strcmp ( Rsd,"SER" ) ) DefineAcceptor ( Chain, Acc, &ac, Res, Osp3, Ser, 1.70f, 0 );
        else if ( !strcmp ( Rsd,"THR" ) ) DefineAcceptor ( Chain, Acc, &ac, Res, Osp3, Thr, 1.70f, 0 );
        else if ( !strcmp ( Rsd,"ASN" ) ) DefineAcceptor ( Chain, Acc, &ac, Res, Osp2, Asn, 1.60f, 0 );
        else if ( !strcmp ( Rsd,"GLN" ) ) DefineAcceptor ( Chain, Acc, &ac, Res, Osp2, Gln, 1.60f, 0 );
        else if ( !strcmp ( Rsd,"ASP" ) )
        {
            DefineAcceptor ( Chain, Acc, &ac, Res, Osp2, Asp, 1.60f, 1 );
            DefineAcceptor ( Chain, Acc, &ac, Res, Osp2, Asp, 1.60f, 2 );
        }
        else if ( !strcmp ( Rsd,"GLU" ) )
        {
            DefineAcceptor ( Chain, Acc, &ac, Res, Osp2, Glu, 1.60f, 1 );
            DefineAcceptor ( Chain, Acc, &ac, Res, Osp2, Glu, 1.60f, 2 );
        }
        else if ( !strcmp ( Rsd,"TYR" ) ) DefineAcceptor ( Chain, Acc, &ac, Res, Osp2, Tyr, 1.70f, 0 );
        else if ( !strcmp ( Rsd,"MET" ) ) DefineAcceptor ( Chain, Acc, &ac, Res, Ssp3, Met, 1.95f, 0 );
        else if ( !strcmp ( Rsd,"CYS" ) ) DefineAcceptor ( Chain, Acc, &ac, Res, Ssp3, Cys, 1.70f, 0 );
    }

    *NAcc = ac;
    return ( ac );
}

int Stride::DefineAcceptor( CHAIN *Chain, ACCEPTOR **Acc, int *ac, int Res, enum HYBRID Hybrid,
                     enum GROUP Group, float HB_Radius, int N )
{
//	Acc[*ac] = ( ACCEPTOR * ) ckalloc ( sizeof ( ACCEPTOR ) );
    ACCEPTOR tempAcc;

    tempAcc.Chain = Chain;
    tempAcc.A_Res    = Res;
    tempAcc.AA_Res   = Res;
    tempAcc.AA2_Res   = Res;
    tempAcc.Hybrid    = Hybrid;
    tempAcc.Group     = Group;
    tempAcc.HB_Radius = HB_Radius;

    if ( Group == Peptide )
    {
        if ( Res != Chain->NRes-1 )
        {
            FindAtom ( Chain,Res,"O",&tempAcc.A_At );
            FindAtom ( Chain,Res,"C",&tempAcc.AA_At );
        }
        else
        {
            tempAcc.A_At = ERR;
            tempAcc.AA_At = ERR;
        }
        FindAtom ( Chain,Res,"CA",&tempAcc.AA2_At );
    }
    else if ( Group == His )
    {
        if ( N == 1 )
        {
            FindAtom ( Chain,Res,"ND1",&tempAcc.A_At );
            FindAtom ( Chain,Res,"CG",&tempAcc.AA_At );
            FindAtom ( Chain,Res,"CE1",&tempAcc.AA2_At );
        }
        else if ( N == 2 )
        {
            FindAtom ( Chain,Res,"NE2",&tempAcc.A_At );
            FindAtom ( Chain,Res,"CE1",&tempAcc.AA_At );
            FindAtom ( Chain,Res,"CD2",&tempAcc.AA2_At );
        }
    }
    else if ( Group == Asn )
    {
        FindAtom ( Chain,Res,"OD1",&tempAcc.A_At );
        FindAtom ( Chain,Res,"CG",&tempAcc.AA_At );
        FindAtom ( Chain,Res,"CB",&tempAcc.AA2_At );
    }
    else if ( Group == Gln )
    {
        FindAtom ( Chain,Res,"OE1",&tempAcc.A_At );
        FindAtom ( Chain,Res,"CD",&tempAcc.AA_At );
        FindAtom ( Chain,Res,"CG",&tempAcc.AA2_At );
    }
    else if ( Group == Asp )
    {
        if ( N == 1 )
        {
            FindAtom ( Chain,Res,"OD1",&tempAcc.A_At );
            FindAtom ( Chain,Res,"CG",&tempAcc.AA_At );
            FindAtom ( Chain,Res,"CB",&tempAcc.AA2_At );
        }
        else if ( N == 2 )
        {
            FindAtom ( Chain,Res,"ND2",&tempAcc.A_At );
            FindAtom ( Chain,Res,"CG",&tempAcc.AA_At );
            FindAtom ( Chain,Res,"CB",&tempAcc.AA2_At );
        }
    }
    else if ( Group == Glu )
    {
        if ( N == 1 )
        {
            FindAtom ( Chain,Res,"OE1",&tempAcc.A_At );
            FindAtom ( Chain,Res,"CD",&tempAcc.AA_At );
            FindAtom ( Chain,Res,"CG",&tempAcc.AA2_At );
        }
        else if ( N == 2 )
        {
            FindAtom ( Chain,Res,"NE2",&tempAcc.A_At );
            FindAtom ( Chain,Res,"CD",&tempAcc.AA_At );
            FindAtom ( Chain,Res,"CG",&tempAcc.AA2_At );
        }
    }
    else if ( Group == Tyr )
    {
        FindAtom ( Chain,Res,"OH",&tempAcc.A_At );
        FindAtom ( Chain,Res,"CZ",&tempAcc.AA_At );
        FindAtom ( Chain,Res,"CE1",&tempAcc.AA2_At );
    }
    else if ( Group == Ser )
    {
        FindAtom ( Chain,Res,"OG",&tempAcc.A_At );
        FindAtom ( Chain,Res,"CB",&tempAcc.AA_At );
    }
    else if ( Group == Thr )
    {
        FindAtom ( Chain,Res,"OG1",&tempAcc.A_At );
        FindAtom ( Chain,Res,"CB",&tempAcc.AA_At );
    }
    else if ( Group == Met )
    {
        FindAtom ( Chain,Res,"SD",&tempAcc.A_At );
        FindAtom ( Chain,Res,"CG",&tempAcc.AA_At );
    }
    else if ( Group == Cys )
    {
        FindAtom ( Chain,Res,"SG",&tempAcc.A_At );
        FindAtom ( Chain,Res,"CB",&tempAcc.AA_At );
    }

    if ( tempAcc.A_At   == ERR || tempAcc.AA_At  == ERR ||
            ( tempAcc.AA2_At == ERR && ( Hybrid == Nsp2 || Hybrid == Osp2 ) ) )
    {
        //free ( Acc[*ac] );
        return ( FAILURE );
    }
    else {
        Acc[*ac] = ( ACCEPTOR * ) ckalloc ( sizeof ( ACCEPTOR ) );
        memcpy( Acc[*ac], &tempAcc, sizeof(ACCEPTOR) );
        ( *ac ) ++;
    }
    return ( SUCCESS );
}

void Stride::GRID_Energy( float *CA2, float *C, float *O, float *H, float *N,
                                  COMMAND *Cmd, HBOND *HBond )
{
    // ****************************************************
    //   Calculate the hydrogen bond energy as defined by
    //   Boobbyer et al., 1989
    // *****************************************************/
    float ProjH[3];
    
    /***** Distance dependence ( 8-6 potential ) ****/
    if ( Cmd->Truncate && HBond->AccDonDist < RmGRID )
        HBond->AccDonDist = RmGRID;
    HBond->Er = float(CGRID/pow ( (double)HBond->AccDonDist,8.0 ) - DGRID/pow ( (double)HBond->AccDonDist,6.0 ));

    /************** Angular dependance ****************/
    /* Find projection of the hydrogen on the O-C-CA plane */
    Project4_123 ( O,C,CA2,H,ProjH );

    /* Three angles determining the direction of the hydrogen bond */
    HBond->ti = fabs ( 180.0f - Ang( ProjH, O, C ) );
    HBond->to = Ang( H, O, ProjH );
    HBond->p  = Ang( N, H, O );

    /* Calculate both angle-dependent HB energy components Et and Ep */
    if ( HBond->ti >= 0.0 && HBond->ti < 90.0 )
        HBond->Et = (float)(cos( RAD( HBond->to ) ) * ( 0.9f + 0.1f * sin( RAD( 2.0f*HBond->ti ) ) ));
    else
        if ( HBond->ti >= 90.0f && HBond->ti < 110.0f )
            HBond->Et = (float)(K1GRID*cos ( RAD ( HBond->to ) ) *
                        ( pow( ( K2GRID-pow( cos( RAD( HBond->ti ) ), 2.0 ) ), 3.0 ) ));
        else
            HBond->Et = 0.0f;

    if ( HBond->p > 90.0f && HBond->p < 270.0f )
        HBond->Ep = (float)pow( cos( RAD( HBond->p ) ), 2.0 );
    else
        HBond->Ep = 0.0f;

    /******** Full hydrogen bond energy *********************/
    HBond->Energy = 1000.0f*HBond->Er*HBond->Et*HBond->Ep;
}

float Stride::Ang( float *Coord1, float *Coord2, float *Coord3 )
{
    float Vector1[3], Vector2[3];
    double A, B, C, D;

    Vector1[0] = Coord1[0] - Coord2[0];
    Vector1[1] = Coord1[1] - Coord2[1];
    Vector1[2] = Coord1[2] - Coord2[2];

    Vector2[0] = Coord3[0] - Coord2[0];
    Vector2[1] = Coord3[1] - Coord2[1];
    Vector2[2] = Coord3[2] - Coord2[2];

    A = Vector1[0]*Vector2[0]+Vector1[1]*Vector2[1]+Vector1[2]*Vector2[2];
    B = sqrt ( Vector1[0]*Vector1[0]+Vector1[1]*Vector1[1]+Vector1[2]*Vector1[2] );
    C = sqrt ( Vector2[0]*Vector2[0]+Vector2[1]*Vector2[1]+Vector2[2]*Vector2[2] );

    D = A/ ( B*C );
    if ( D > 0.0 && fabs ( D - 1.0 ) < Eps )
        D -= Eps;
    else
        if ( D < 0.0 && fabs ( D + 1.0 ) < Eps )
            D += Eps;

    return ( ( float ) ( RADDEG*acos ( D ) ) );
}

int Stride::FindChain( CHAIN **Chain, int NChain, char ChainId)
{
  int i;

  for( i=0; i<NChain; i++ )
    if( Chain[i]->Id == ChainId )
      return(i);

  return(ERR);
}

int Stride::FindPolInt( HBOND **HBond, RESIDUE *Res1, RESIDUE *Res2 )
{
    int i, j, hb;
    INVOLVED *p1, *p2;

    p1 = Res1->Inv;
    p2 = Res2->Inv;

    if ( p1->NBondDnr && p2->NBondAcc )
    {
        for ( i=0; i<p1->NBondDnr; i++ )
        {
            hb = p1->HBondDnr[i];
            for ( j=0; j<p2->NBondAcc; j++ )
                if ( hb == p2->HBondAcc[j] && HBond[hb]->ExistPolarInter )
                    return ( hb );
        }
    }

    return ( ERR );
}

int Stride::FindBnd( HBOND **HBond, RESIDUE *Res1, RESIDUE *Res2 )
{
    int i, j, hb;
    INVOLVED *p1, *p2;

    p1 = Res1->Inv;
    p2 = Res2->Inv;

    if ( p1->NBondDnr && p2->NBondAcc )
    {
        for ( i=0; i<p1->NBondDnr; i++ )
        {
            hb = p1->HBondDnr[i];
            for ( j=0; j<p2->NBondAcc; j++ )
                if ( hb == p2->HBondAcc[j] && HBond[hb]->ExistHydrBondRose )
                    return ( hb );
        }
    }

    return ( ERR );
}

int Stride::Link( HBOND **HBond, CHAIN **Chain, int Cn1, int Cn2, RESIDUE *Res1_1,
           RESIDUE *Res1_2, RESIDUE *Res2_2, RESIDUE *Res2_1, RESIDUE *CRes1,
           RESIDUE *CRes2, float **PhiPsiMap, PATTERN **Pattern, int *NumPat,
           const char *Text, float Treshold, COMMAND *Cmd, int Test )
{
    int BondNumber1, BondNumber2, Flag = 0;
    float Prob1, Prob2, Conf, Coeff;

    if ( ( BondNumber1 = FindPolInt ( HBond,Res1_1,Res1_2 ) ) == ERR )
        return ( FAILURE );

    if ( ( BondNumber2 = FindPolInt ( HBond,Res2_2,Res2_1 ) ) == ERR )
        return ( FAILURE );

    if ( CRes1 == NULL )
    {
        if ( CRes2->Prop->PhiZn == ERR || CRes2->Prop->PsiZn == ERR )
            return ( FAILURE );
        Conf = PhiPsiMap[CRes2->Prop->PhiZn][CRes2->Prop->PsiZn];
    }
    else
        if ( CRes2 == NULL )
        {
            if ( CRes1->Prop->PhiZn == ERR || CRes1->Prop->PsiZn == ERR )
                return ( FAILURE );
            Conf = PhiPsiMap[CRes1->Prop->PhiZn][CRes1->Prop->PsiZn];
        }
        else
        {
            if ( CRes2->Prop->PhiZn == ERR || CRes2->Prop->PsiZn == ERR ||
                    CRes1->Prop->PhiZn == ERR || CRes1->Prop->PsiZn == ERR )
                return ( FAILURE );
            Conf =
                0.5f * ( PhiPsiMap[CRes1->Prop->PhiZn][CRes1->Prop->PsiZn]+
                       PhiPsiMap[CRes2->Prop->PhiZn][CRes2->Prop->PsiZn] );
        }
    Coeff = 1+Cmd->C1_E+Cmd->C2_E*Conf;
    Prob1 = HBond[BondNumber1]->Energy*Coeff;
    Prob2 = HBond[BondNumber2]->Energy*Coeff;

    if ( Prob1 < Treshold && Prob2 < Treshold )
    {

        if ( !Test )
        {
            Pattern[*NumPat] = ( PATTERN * ) ckalloc ( sizeof ( PATTERN ) );
            Pattern[*NumPat]->ExistPattern = STRIDE_YES;
            Pattern[*NumPat]->Hb1 = HBond[BondNumber1];
            Pattern[*NumPat]->Hb2 = HBond[BondNumber2];
            Pattern[*NumPat]->Nei1 = NULL;
            Pattern[*NumPat]->Nei2 = NULL;
            strcpy ( Pattern[*NumPat]->Type,Text );
            ( *NumPat ) ++;
        }
        Flag = 1;
    }

    return ( Flag );
}

void Stride::FilterAntiPar( PATTERN **Pat, int NPat )
{
    int i, j;
    int I1A, I1D, I2A, I2D, J1A, J1D, J2A, J2D;
    char I1ACn, I1DCn, I2ACn, I2DCn, J1ACn, J1DCn, J2ACn, J2DCn;

    for ( i=0; i<NPat; i++ )
    {

        if ( !Pat[i]->ExistPattern ) continue;

        Alias ( &I1D,&I1A,&I2D,&I2A,&I1DCn,&I1ACn,&I2DCn,&I2ACn,Pat[i] );

        for ( j=0; j<NPat; j++ )
        {

            if ( j == i || !Pat[j]->ExistPattern ) continue;

            Alias ( &J1D,&J1A,&J2D,&J2A,&J1DCn,&J1ACn,&J2DCn,&J2ACn,Pat[j] );

            if ( J1D == J2A && J2D == J1A && I1D != I2A && I2D != I1A &&
                    ( ( J1D == I1D && J1A == I1A ) || ( J1D == I1A && J1A == I1D ) ||
                      ( J1D == I2A && J1A == I2D ) || ( J1D == I2D && J1A == I2A ) ) ) continue;

            if ( ( ( I1D < I2A || I2D < I1A ) &&
                    ( ( J1A <= I2A && J1A >= I1D && J2D <= I2A && J2D >= I1D && J2DCn == I1DCn &&
                        J2A <= I1A && J2A >= I2D && J1D <= I1A && J1D >= I2D && J1DCn == I2DCn ) ||
                      ( J2A <= I2A && J2A >= I1D && J1D <= I2A && J1D >= I1D && J1DCn == I1DCn &&
                        J1A <= I1A && J1A >= I2D && J2D <= I1A && J2D >= I2D && J2DCn == I2DCn ) ) ) ||
                    ( ( I1D > I2A || I2D > I1A ) &&
                      ( ( J1A >= I2A && J1A <= I1D && J2D >= I2A && J2D <= I1D && J2DCn == I1DCn &&
                          J2A >= I1A && J2A <= I2D && J1D >= I1A && J1D <= I2D && J1DCn == I2DCn ) ||
                        ( J2A >= I2A && J2A <= I1D && J1D >= I2A && J1D <= I1D && J1DCn == I1DCn &&
                          J1A >= I1A && J1A <= I2D && J2D >= I1A && J2D <= I2D && J2DCn == I2DCn ) ) ) )
            {
                Pat[j]->ExistPattern = STRIDE_NO;
            }
        }
    }
}

void Stride::FilterPar( PATTERN **Pat, int NPat )
{
    int i, j;
    int I1A, I1D, I2A, I2D, J1A, J1D, J2A, J2D;
    char I1ACn, I1DCn, I2ACn, I2DCn, J1ACn, J1DCn, J2ACn, J2DCn;

    for ( i=0; i<NPat; i++ )
    {

        if ( !Pat[i]->ExistPattern ) continue;

        Alias ( &I1D,&I1A,&I2D,&I2A,&I1DCn,&I1ACn,&I2DCn,&I2ACn,Pat[i] );

        for ( j=0; j<NPat; j++ )
        {

            if ( j == i || !Pat[j]->ExistPattern ) continue;

            Alias ( &J1D,&J1A,&J2D,&J2A,&J1DCn,&J1ACn,&J2DCn,&J2ACn,Pat[j] );

            if ( ( ( I1A >= I2D && I1D >= I2A ) &&
                    ( ( J1A >= I2A && J1A <= I1D && J2D >= I2A && J2D <= I1D && J2DCn == I1DCn &&
                        J2A <= I1A && J2A >= I2D && J1D <= I1A && J1D >= I2D && J1DCn == I2DCn ) ||
                      ( J2A >= I2A && J2A <= I1D && J1D >= I2A && J1D <= I1D && J1DCn == I1DCn &&
                        J1A <= I1A && J1A >= I2D && J2D <= I1A && J2D >= I2D && J2DCn == I2DCn ) ) ) ||

                    ( I2A >= I1D && I2D >= I1A  &&
                      ( ( J1A <= I2A && J1A >= I1D && J2D <= I2A && J2D >= I1D && J2DCn == I1DCn &&
                          J2A >= I1A && J2A <= I2D && J1D >= I1A && J1D <= I2D && J1DCn == I2DCn ) ||

                        ( J2A <= I2A && J2A >= I1D && J1D <= I2A && J1D >= I1D && J1DCn == I1DCn &&
                          J1A >= I1A && J1A <= I2D && J2D >= I1A && J2D <= I2D && J2DCn == I2DCn ) ) ) )
            {
                Pat[j]->ExistPattern = STRIDE_NO;
            }
        }
    }
}

void Stride::MergePatternsAntiPar( PATTERN **Pat, int NPat )
{
    int i, j;
    int DB, DW, MinDB1, MinDB2, MinDW1, MinDW2, Min, Lnk1A, Lnk1D;
    int I1A, I1D, I2A, I2D, J1A, J1D, J2A, J2D;
    char I1ACn, I1DCn, I2ACn, I2DCn, J1ACn, J1DCn, J2ACn, J2DCn;

    for ( i=0; i<NPat; i++ )
    {

        if ( !Pat[i]->ExistPattern ) continue;

        MinDB1 = MinDB2 = MinDW1 = MinDW2 = 1000;
        Min = ERR;
        Lnk1D = Lnk1A = ERR;

        Alias ( &I1D,&I1A,&I2D,&I2A,&I1DCn,&I1ACn,&I2DCn,&I2ACn,Pat[i] );

        for ( j=0; j<NPat; j++ )
        {

            if ( i == j || !Pat[j]->ExistPattern ) continue;

            Alias ( &J1D,&J1A,&J2D,&J2A,&J1DCn,&J1ACn,&J2DCn,&J2ACn,Pat[j] );

            if ( Near ( I1D,J1D,J1A,I1A,J2A,J2D,I2A,I2D,I1DCn,J1DCn,J1ACn,I1ACn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSide ( J1A,J1D,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J2D,&Lnk1D,J2A,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( Near ( I1D,J1A,J1D,I1A,J2D,J2A,I2A,I2D,I1DCn,J1ACn,J1DCn,I1ACn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSide ( J1D,J1A,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J2A,&Lnk1D,J2D,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( Near ( I1D,J2D,J2A,I1A,J1A,J1D,I2A,I2D,I1DCn,J2DCn,J2ACn,I1ACn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSide ( J2A,J2D,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J1D,&Lnk1D,J1A,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( Near ( I1D,J2A,J2D,I1A,J1D,J1A,I2A,I2D,I1DCn,J2ACn,J2DCn,I1ACn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSide ( J2D,J2A,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J1A,&Lnk1D,J1D,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( Near ( I1A,J1D,J1A,I1D,J2A,J2D,I2D,I2A,I1ACn,J1DCn,J1ACn,I1DCn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSide ( J1A,J1D,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J2A,&Lnk1D,J2D,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( Near ( I1A,J1A,J1D,I1D,J2D,J2A,I2D,I2A,I1ACn,J1ACn,J1DCn,I1DCn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSide ( J1A,J1D,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J2D,&Lnk1D,J2A,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( Near ( I1A,J2D,J2A,I1D,J1A,J1D,I2D,I2A,I1ACn,J2DCn,J2ACn,I1DCn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSide ( J2D,J2A,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J1A,&Lnk1D,J1D,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( Near ( I1A,J2A,J2D,I1D,J1D,J1A,I2D,I2A,I1ACn,J2ACn,J2DCn,I1DCn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSide ( J2A,J2D,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J1D,&Lnk1D,J2D,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

        }

        for ( j=0; j<NPat; j++ )
        {

            if ( j == Min || j == i || !Pat[j]->ExistPattern ) continue;

            Alias ( &J1D,&J1A,&J2D,&J2A,&J1DCn,&J1ACn,&J2DCn,&J2ACn,Pat[j] );

            if ( Near ( I2D,J1D,J1A,I2A,J2A,J2D,I1A,I1D,I2DCn,J1DCn,J1ACn,I2ACn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J2A,J2D,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( Near ( I2D,J1A,J1D,I2A,J2D,J2A,I1A,I1D,I2DCn,J1ACn,J1DCn,I2ACn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J2D,J2A,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( Near ( I2D,J2D,J2A,I2A,J1A,J1D,I1A,I1D,I2DCn,J2DCn,J2ACn,I2ACn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J1A,J1D,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( Near ( I2D,J2A,J2D,I2A,J1D,J1A,I1A,I1D,I2DCn,J2ACn,J2DCn,I2ACn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J1D,J1A,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( Near ( I2A,J1D,J1A,I2D,J2A,J2D,I1D,I1A,I2ACn,J1DCn,J1ACn,I2DCn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J2D,J2A,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( Near ( I2A,J1A,J1D,I2D,J2D,J2A,I1D,I1A,I2ACn,J1ACn,J1DCn,I2DCn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J2A,J2D,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( Near ( I2A,J2D,J2A,I2D,J1A,J1D,I1D,I1A,I2ACn,J2DCn,J2ACn,I2DCn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J1D,J1A,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( Near ( I2A,J2A,J2D,I2D,J1D,J1A,I1D,I1A,I2ACn,J2ACn,J2DCn,I2DCn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J1A,J1D,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );
        }
    }
}

void Stride::MergePatternsPar( PATTERN **Pat, int NPat )
{
    int i, j;
    int DB, DW, MinDB1, MinDB2, MinDW1, MinDW2, Min, Lnk1A, Lnk1D;
    int I1A, I1D, I2A, I2D, J1A, J1D, J2A, J2D;
    char I1ACn, I1DCn, I2ACn, I2DCn, J1ACn, J1DCn, J2ACn, J2DCn;

    for ( i=0; i<NPat; i++ )
    {

        if ( !Pat[i]->ExistPattern ) continue;

        MinDB1 = MinDB2 = MinDW1 = MinDW2 = 1000;
        Min = ERR;
        Lnk1D = Lnk1A = ERR;

        Alias ( &I1D,&I1A,&I2D,&I2A,&I1DCn,&I1ACn,&I2DCn,&I2ACn,Pat[i] );

        for ( j=0; j<NPat; j++ )
        {

            if ( i == j || !Pat[j]->ExistPattern ) continue;

            Alias ( &J1D,&J1A,&J2D,&J2A,&J1DCn,&J1ACn,&J2DCn,&J2ACn,Pat[j] );

            if ( NearPar ( I1D,J1D,J1A,I1A,J2A,J2D,I2A,I2D,I1DCn,J1DCn,J1ACn,I1ACn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSidePar ( J1A,J1D,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J2D,&Lnk1D,J2A,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( NearPar ( I1D,J1A,J1D,I1A,J2D,J2A,I2A,I2D,I1DCn,J1ACn,J1DCn,I1ACn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSidePar ( J1D,J1A,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J2A,&Lnk1D,J2D,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( NearPar ( I1D,J2D,J2A,I1A,J1A,J1D,I2A,I2D,I1DCn,J2DCn,J2ACn,I1ACn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSidePar ( J2A,J2D,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J1D,&Lnk1D,J1A,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( NearPar ( I1D,J2A,J2D,I1A,J1D,J1A,I2A,I2D,I1DCn,J2ACn,J2DCn,I1ACn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSidePar ( J2D,J2A,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J1A,&Lnk1D,J1D,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( NearPar ( I1A,J1D,J1A,I1D,J2A,J2D,I2D,I2A,I1ACn,J1DCn,J1ACn,I1DCn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSidePar ( J1A,J1D,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J2A,&Lnk1D,J2D,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( NearPar ( I1A,J1A,J1D,I1D,J2D,J2A,I2D,I2A,I1ACn,J1ACn,J1DCn,I1DCn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSidePar ( J1A,J1D,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J2D,&Lnk1D,J2A,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( NearPar ( I1A,J2D,J2A,I1D,J1A,J1D,I2D,I2A,I1ACn,J2DCn,J2ACn,I1DCn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSidePar ( J2D,J2A,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J1A,&Lnk1D,J1D,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

            if ( NearPar ( I1A,J2A,J2D,I1D,J1D,J1A,I2D,I2A,I1ACn,J2ACn,J2DCn,I1DCn,&DB,&DW ) &&
                    ( ( DB < MinDB1 && DW <= MinDW1 ) || ( DB <= MinDB1 && DW < MinDW1 ) ) &&
                    RightSidePar ( J2A,J2D,I1A,I1D,I2A,I2D ) )
                JoinNeighbours ( &Lnk1A,J1D,&Lnk1D,J2D,&Pat[i]->Nei1,Pat[j],&MinDB1,DB,&MinDW1,DW,&Min,j );

        }

        for ( j=0; j<NPat; j++ )
        {

            if ( j == Min || j == i || !Pat[j]->ExistPattern ) continue;

            Alias ( &J1D,&J1A,&J2D,&J2A,&J1DCn,&J1ACn,&J2DCn,&J2ACn,Pat[j] );

            if ( NearPar ( I2D,J1D,J1A,I2A,J2A,J2D,I1A,I1D,I2DCn,J1DCn,J1ACn,I2ACn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J2A,J2D,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( NearPar ( I2D,J1A,J1D,I2A,J2D,J2A,I1A,I1D,I2DCn,J1ACn,J1DCn,I2ACn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J2D,J2A,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( NearPar ( I2D,J2D,J2A,I2A,J1A,J1D,I1A,I1D,I2DCn,J2DCn,J2ACn,I2ACn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J1A,J1D,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( NearPar ( I2D,J2A,J2D,I2A,J1D,J1A,I1A,I1D,I2DCn,J2ACn,J2DCn,I2ACn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J1D,J1A,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( NearPar ( I2A,J1D,J1A,I2D,J2A,J2D,I1D,I1A,I2ACn,J1DCn,J1ACn,I2DCn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J2D,J2A,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( NearPar ( I2A,J1A,J1D,I2D,J2D,J2A,I1D,I1A,I2ACn,J1ACn,J1DCn,I2DCn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J2A,J2D,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( NearPar ( I2A,J2D,J2A,I2D,J1A,J1D,I1D,I1A,I2ACn,J2DCn,J2ACn,I2DCn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J1D,J1A,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );

            if ( NearPar ( I2A,J2A,J2D,I2D,J1D,J1A,I1D,I1A,I2ACn,J2ACn,J2DCn,I2DCn,&DB,&DW ) &&
                    ( ( DB < MinDB2 && DW <= MinDW2 ) || ( DB <= MinDB2 && DW < MinDW2 ) ) &&
                    RightSide2 ( Lnk1A,Lnk1D,J1A,J1D,I1A,I1D,I2A,I2D ) )
                JoinNeighb ( &Pat[i]->Nei2,Pat[j],&MinDB2,DB,&MinDW2,DW );
        }
    }
}

int Stride::RightSide2( int L_A1, int L_D1, int LnkD, int LnkA, int I1A, int I1D, int I2A, int I2D )
{

    if ( ( I2A < I1D && LnkA <= I1D && LnkA <= I2A ) ||
            ( I2A > I1D && LnkA >= I1D && LnkA >= I2A ) ||
            ( I2D < I1A && LnkD <= I1A && LnkA <= I2D ) ||
            ( I2D > I1A && LnkD >= I1A && LnkD >= I2D ) )
        return ( SUCCESS );
    else
        if ( I2A == I1D && I2D == I1A )
        {
            if ( L_A1 != ERR &&
                    ( ( LnkD <= I2D && L_A1 <= I2D && LnkA >= I2A && L_D1 >= I2A ) ||
                      ( LnkD >= I2D && L_A1 >= I2D && LnkA <= I2A && L_D1 <= I2A ) ) )
                return ( FAILURE );
            else
                return ( SUCCESS );
        }

    return ( FAILURE );
}

int Stride::RightSide( int LnkA, int LnkD, int I1A, int I1D, int I2A, int I2D )
{

    if ( ( I1A == I2D && I1D == I2A ) ||
            ( I1A < I2D && LnkA <= I2D && LnkA <= I1A ) ||
            ( I1A > I2D && LnkA >= I2D && LnkA >= I1A ) ||
            ( I1D < I2A && LnkD <= I2A && LnkD <= I1D ) ||
            ( I1D > I2A && LnkD >= I2A && LnkD >= I1D ) )
        return ( SUCCESS );

    return ( FAILURE );
}

int Stride::RightSidePar( int LnkA, int LnkD, int I1A, int I1D, int I2A, int I2D )
{

    if ( ( I1A == I2D && I1D == I2A ) ||
            ( I1A < I2D && LnkA < I2D && LnkA <= I1A && I1D <= I2A && LnkD <= I2A && LnkD <= I1D ) ||
            ( I1A > I2D && LnkA > I2D && LnkA >= I1A && I1D >= I2A && LnkD >= I2A && LnkD >= I1D ) ||
            ( I1D < I2A && LnkD < I2A && LnkD <= I1D && I1A <= I2D && LnkA <= I2D && LnkA <= I1A ) ||
            ( I1D > I2A && LnkD > I2A && LnkD >= I1D && I1A >= I2D && LnkA >= I2D && LnkA >= I1A ) )
        return ( SUCCESS );

    return ( FAILURE );
}

void Stride::JoinNeighbours( int *Lnk1A, int Res1, int *Lnk1D, int Res2, PATTERN **Nei,
                      PATTERN *Pat, int *MinDB1, int DB, int *MinDW1, int DW, int *Min, int j )
{
    *Lnk1A = Res1;
    *Lnk1D = Res2;
    ( *Nei ) = Pat;
    *MinDB1 = DB;
    *MinDW1 = DW;
    *Min = j;
}

void Stride::JoinNeighb( PATTERN **Nei, PATTERN *Pat, int *MinDB2, int DB, int *MinDW2, int DW )
{
    ( *Nei ) = Pat;
    *MinDB2 = DB;
    *MinDW2 = DW;
}

int Stride::NearPar( int Res1, int Res2, int Res3, int Res4, int Res5, int Res6, int Res7, int Res8,
              char Cn1, char Cn2, char Cn3, char Cn4, int *DistBest, int *DistWorst )
{

    /*
       Res5 Res2 Res1
       Res6 Res3 Res4
    */

    int a, b, c1, d1, c, d, Nei1, Nei2;

    if ( Cn1 != Cn2 || Cn3 != Cn4 ) return ( FAILURE );

    if ( Res1 >= Res2 && Res2 >= Res5 && Res7 >= Res1 &&
            Res4 >= Res3 && Res4 >= Res6 && Res8 >= Res4 )
    {

        if ( Res5 == Res2 )
            Nei1 = Res2;
        else
            Nei1 = Res2-1;

        if ( Res1 == Res7 )
            Nei2 = Res1;
        else
            Nei2 = Res1+1;

        a = Nei2-Nei1;
        c1 = Nei2-Res5;

        if ( Res3 == Res6 )
            Nei1 = Res3;
        else
            Nei1 = Res3-1;

        if ( Res4 == Res8 )
            Nei2 = Res4;
        else
            Nei2 = Res4+1;

        b = Nei2-Nei1;
        d1 = Nei2-Res6;

    }
    else
        if ( Res1 <= Res2 && Res2 <= Res5 && Res7 <= Res1 &&
                Res4 <= Res3 && Res4 <= Res6 && Res8 <= Res4 )
        {

            if ( Res5 == Res2 )
                Nei1 = Res2;
            else
                Nei1 = Res2+1;

            if ( Res1 == Res7 )
                Nei2 = Res1;
            else
                Nei2 = Res1-1;

            a = Nei1-Nei2;
            c1 = Res1-Res7;

            if ( Res3 == Res6 )
                Nei1 = Res3;
            else
                Nei1 = Res3+1;

            if ( Res4 == Res8 )
                Nei2 = Res4;
            else
                Nei2 = Res4-1;

            b = Nei1-Nei2;
            d1 = Nei1-Res8;


        }
        else
            return ( FAILURE );

    c = Maximum ( c1,a );
    d = Maximum ( d1,b );

    if ( a >= 0 && b >= 0 && c >= 0 && d >= 0 &&
            ( ( a <= 2 && b <= 5 ) || ( a <= 5 && b <= 2 ) ) )
    {
        *DistBest  = Minimum ( a,b );
        *DistWorst = Maximum ( c,d );
        if ( *DistBest <= *DistWorst )
            return ( SUCCESS );
        else
            return ( FAILURE );
    }

    return ( FAILURE );
}

int Stride::Near( int Res1, int Res2, int Res3, int Res4, int Res5, int Res6, int Res7, int Res8,
           char Cn1, char Cn2, char Cn3, char Cn4, int *DistBest, int *DistWorst )
{

    /*
       Res5 Res2 Res1
       Res6 Res3 Res4
    */

    int a, b, c1, d1, c, d, Nei1, Nei2;


    if ( Cn1 != Cn2 || Cn3 != Cn4 ) return ( FAILURE );


    if ( Res1 >= Res2 && Res2 >= Res5 && Res7 >= Res1 &&
            Res4 <= Res3 && Res4 <= Res6 && Res8 <= Res4 )
    {

        if ( Res5 == Res2 )
            Nei1 = Res2;
        else
            Nei1 = Res2-1;

        if ( Res1 == Res7 )
            Nei2 = Res1;
        else
            Nei2 = Res1+1;

        a = Nei2-Nei1;
        c1 = Nei2-Res5;

        if ( Res3 == Res6 )
            Nei1 = Res3;
        else
            Nei1 = Res3+1;

        if ( Res4 == Res8 )
            Nei2 = Res4;
        else
            Nei2 = Res4-1;

        b = Nei1-Nei2;
        d1 = Res6-Nei2;
    }
    else
        return ( FAILURE );

    c = Maximum ( c1,a );
    d = Maximum ( d1,b );

    if ( a >= 0 && b >= 0 && c >= 0 && d >= 0 &&
            ( ( a <= 2 && b <= 5 ) || ( a <= 5 && b <= 2 ) ) )
    {
        *DistBest  = Minimum ( a,b );
        *DistWorst = Maximum ( c,d );
        if ( *DistBest <= *DistWorst )
            return ( SUCCESS );
        else
            return ( FAILURE );
    }

    return ( FAILURE );
}

void Stride::FillAsnAntiPar ( char *Asn1, char *Asn2, CHAIN **Chain, int Cn1, int Cn2,
                      PATTERN **Pat, int NPat, COMMAND *Cmd )
{
    int i, j;
    int Beg1, Beg2, End1, End2;
    int B1D, B1A, B2D, B2A, E1D, E1A, E2D, E2A;
    char B1DCn, B1ACn, B2DCn, B2ACn, E1DCn, E1ACn, E2DCn, E2ACn, Beg1Cn, Beg2Cn;
    PATTERN *CurrPat, *PrevPat;;

    for ( i=0; i<NPat; i++ )
    {

        if ( Pat[i]->Nei1 != NULL && Pat[i]->Nei2 == NULL )
            CurrPat = Pat[i]->Nei1;
        else
            if ( Pat[i]->Nei2 != NULL && Pat[i]->Nei1 == NULL )
                CurrPat = Pat[i]->Nei2;
            else
                continue;

        PrevPat = Pat[i];
        while ( CurrPat->Nei1 != NULL && CurrPat->Nei2 != NULL )
        {

            if ( ( CurrPat->Nei1->Nei1 == CurrPat || CurrPat->Nei1->Nei2 == CurrPat ) &&
                    CurrPat->Nei1 != PrevPat )
            {
                PrevPat = CurrPat;
                CurrPat = CurrPat->Nei1;
            }
            else
                if ( ( CurrPat->Nei2->Nei1 == CurrPat || CurrPat->Nei2->Nei2 == CurrPat ) &&
                        CurrPat->Nei2 != PrevPat )
                {
                    PrevPat = CurrPat;
                    CurrPat = CurrPat->Nei2;
                }
                else
                {
                    fprintf ( stdout,"Cycle Anti%s%c i = %d \n",Chain[Cn1]->File,Chain[Cn1]->Id,i );
                    break;
                }
        }

        Alias ( &B1D,&B1A,&B2D,&B2A,&B1DCn,&B1ACn,&B2DCn,&B2ACn,Pat[i] );
        Alias ( &E1D,&E1A,&E2D,&E2A,&E1DCn,&E1ACn,&E2DCn,&E2ACn,CurrPat );

        if ( ( Cn1 != Cn2 || E1D - B2A <  E2D - B2A ) &&
                ( MakeEnds ( &Beg1,B1D,B2A,&Beg1Cn,B1DCn,&End1,E2A,E1D,E2ACn,&Beg2,E2D,E1A,&Beg2Cn,E2DCn,
                             &End2,B1A,B2D,B1ACn,Pat,NPat ) ||
                  MakeEnds ( &Beg1,B1D,B2A,&Beg1Cn,B1DCn,&End1,E1D,E2A,E1DCn,&Beg2,E1A,E2D,&Beg2Cn,E1ACn,
                             &End2,B1A,B2D,B1ACn,Pat,NPat ) ) )
            ;
        else
            if ( ( Cn1 != Cn2 || E2D - B2A <  E1D - B2A ) &&
                    ( MakeEnds ( &Beg1,B1D,B2A,&Beg1Cn,B1DCn,&End1,E1A,E2D,E1ACn,&Beg2,E1D,E2A,&Beg2Cn,E1DCn,
                                 &End2,B1A,B2D,B1ACn,Pat,NPat ) ||
                      MakeEnds ( &Beg1,B1D,B2A,&Beg1Cn,B1DCn,&End1,E2D,E1A,E2DCn,&Beg2,E2A,E1D,&Beg2Cn,E2ACn,
                                 &End2,B1A,B2D,B1ACn,Pat,NPat ) ) )
                ;
            else
                if ( ( Cn1 != Cn2 || B2A - E1D < B2A - E2D ) &&
                        ( MakeEnds ( &Beg1,B1A,B2D,&Beg1Cn,B1ACn,&End1,E2D,E1A,E2DCn,&Beg2,E2A,E1D,&Beg2Cn,E2ACn,
                                     &End2,B1D,B2A,B1DCn,Pat,NPat ) ||
                          MakeEnds ( &Beg1,B1A,B2D,&Beg1Cn,B1ACn,&End1,E1A,E2D,E1ACn,&Beg2,E1D,E2A,&Beg2Cn,E1DCn,
                                     &End2,B1D,B2A,B1DCn,Pat,NPat ) ) )
                    ;
                else
                    if ( ( Cn1 != Cn2 || B2A - E2D < B2A - E1D ) &&
                            ( MakeEnds ( &Beg1,B1A,B2D,&Beg1Cn,B1ACn,&End1,E1D,E2A,E1DCn,&Beg2,E1A,E2D,&Beg2Cn,E1ACn,
                                         &End2,B1D,B2A,B1DCn,Pat,NPat ) ||
                              MakeEnds ( &Beg1,B1A,B2D,&Beg1Cn,B1ACn,&End1,E2A,E1D,E2ACn,&Beg2,E2D,E1A,&Beg2Cn,E2DCn,
                                         &End2,B1D,B2A,B1DCn,Pat,NPat ) ) )
                        ;
                    else
                        if ( ( Cn1 != Cn2 || B1D - E2A <  B2D - E2A ) &&
                                ( MakeEnds ( &Beg1,E1D,E2A,&Beg1Cn,E1DCn,&End1,B2A,B1D,B2ACn,&Beg2,B2D,B1A,&Beg2Cn,B2DCn,
                                             &End2,E1A,E2D,E1ACn,Pat,NPat ) ||
                                  MakeEnds ( &Beg1,E1D,E2A,&Beg1Cn,E1DCn,&End1,B1D,B2A,B1DCn,&Beg2,B1A,B2D,&Beg2Cn,B1ACn,
                                             &End2,E1A,E2D,E1ACn,Pat,NPat ) ) )
                            ;
                        else
                            if ( ( Cn1 != Cn2 || B2D - E2A <  B1D - E2A ) &&
                                    ( MakeEnds ( &Beg1,E1D,E2A,&Beg1Cn,E1DCn,&End1,B1A,B2D,B1ACn,&Beg2,B1D,B2A,&Beg2Cn,B1DCn,
                                                 &End2,E1A,E2D,E1ACn,Pat,NPat ) ||
                                      MakeEnds ( &Beg1,E1D,E2A,&Beg1Cn,E1DCn,&End1,B2D,B1A,B2DCn,&Beg2,B2A,B1D,&Beg2Cn,B2ACn,
                                                 &End2,E1A,E2D,E1ACn,Pat,NPat ) ) )
                                ;
                            else
                                if ( ( Cn1 != Cn2 || E2A - B1D < E2A - B2D ) &&
                                        ( MakeEnds ( &Beg1,E1A,E2D,&Beg1Cn,E1ACn,&End1,B2D,B1A,B2DCn,&Beg2,B2A,B1D,&Beg2Cn,B2ACn,
                                                     &End2,E1D,E2A,E1DCn,Pat,NPat ) ||
                                          MakeEnds ( &Beg1,E1A,E2D,&Beg1Cn,E1ACn,&End1,B1A,B2D,B1ACn,&Beg2,B1D,B2A,&Beg2Cn,B1DCn,
                                                     &End2,E1D,E2A,E1DCn,Pat,NPat ) ) )
                                    ;
                                else
                                    if ( ( Cn1 != Cn2 || E2A - B2D < E2A - B1D ) &&
                                            ( MakeEnds ( &Beg1,E1A,E2D,&Beg1Cn,E1ACn,&End1,B1D,B2A,B1DCn,&Beg2,B1A,B2D,&Beg2Cn,B1ACn,
                                                         &End2,E1D,E2A,E1DCn,Pat,NPat ) ||
                                              MakeEnds ( &Beg1,E1A,E2D,&Beg1Cn,E1ACn,&End1,B2A,B1D,B2ACn,&Beg2,B2D,B1A,&Beg2Cn,B2DCn,
                                                         &End2,E1D,E2A,E1DCn,Pat,NPat ) ) )
                                        ;
                                    else
                                    {
                                        fprintf ( stdout,"Ne tot variant.. Anti.. %s%c\n",Chain[Cn1]->File,Chain[Cn1]->Id );
                                        continue;
                                    }


        if ( Beg1Cn == Chain[Cn1]->Id )
        {
            for ( j=Beg1; j<=End1; j++ )
                Asn1[j] = 'N';
            for ( j=Beg2; j<=End2; j++ )
                Asn2[j] = 'N';
        }
        else
        {
            for ( j=Beg1; j<=End1; j++ )
                Asn2[j] = 'N';
            for ( j=Beg2; j<=End2; j++ )
                Asn1[j] = 'N';
        }

        Pat[i]->Nei1 = NULL;
        Pat[i]->Nei2 = NULL;
        CurrPat->Nei1 = NULL;
        CurrPat->Nei2 = NULL;

    }
}

void Stride::FillAsnPar( char *Asn1, char *Asn2, CHAIN **Chain, int Cn1, int Cn2,
                  PATTERN **Pat, int NPat, COMMAND *Cmd )
{
    int i, j;
    int Beg1, Beg2, End1, End2;
    int B1D, B1A, B2D, B2A, E1D, E1A, E2D, E2A;
    char B1DCn, B1ACn, B2DCn, B2ACn, E1DCn, E1ACn, E2DCn, E2ACn, Beg1Cn, Beg2Cn;
    PATTERN *CurrPat, *PrevPat;;

    for ( i=0; i<NPat; i++ )
    {

        if ( Pat[i]->Nei1 != NULL && Pat[i]->Nei2 == NULL )
            CurrPat = Pat[i]->Nei1;
        else
            if ( Pat[i]->Nei2 != NULL && Pat[i]->Nei1 == NULL )
                CurrPat = Pat[i]->Nei2;
            else
                continue;

        PrevPat = Pat[i];
        while ( CurrPat->Nei1 != NULL && CurrPat->Nei2 != NULL )
        {

            if ( ( CurrPat->Nei1->Nei1 == CurrPat || CurrPat->Nei1->Nei2 == CurrPat ) &&
                    CurrPat->Nei1 != PrevPat )
            {
                PrevPat = CurrPat;
                CurrPat = CurrPat->Nei1;
            }
            else
            {
                PrevPat = CurrPat;
                CurrPat = CurrPat->Nei2;
            }
        }

        Alias ( &B1D,&B1A,&B2D,&B2A,&B1DCn,&B1ACn,&B2DCn,&B2ACn,Pat[i] );
        Alias ( &E1D,&E1A,&E2D,&E2A,&E1DCn,&E1ACn,&E2DCn,&E2ACn,CurrPat );

        if ( ( Cn1 != Cn2 || abs ( E1D-B2A ) < abs ( E2D-B2A ) ) &&
                ( MakeEnds ( &Beg1,B1D,B2A,&Beg1Cn,B1DCn,&End1,E2A,E1D,E2ACn,&Beg2,B1A,B2D,&Beg2Cn,B1ACn,
                             &End2,E2D,E1A,E2DCn,Pat,NPat ) ||
                  MakeEnds ( &Beg1,B1D,B2A,&Beg1Cn,B1DCn,&End1,E1D,E2A,E1DCn,&Beg2,B1A,B2D,&Beg2Cn,B1ACn,
                             &End2,E1A,E2D,E1ACn,Pat,NPat ) ) )
            ;
        else
            if ( ( Cn1 != Cn2 || abs ( E2D-B2A ) < abs ( E1D-B2A ) ) &&
                    ( MakeEnds ( &Beg1,B1D,B2A,&Beg1Cn,B1DCn,&End1,E1A,E2D,E1ACn,&Beg2,B1A,B2D,&Beg2Cn,B1ACn,
                                 &End2,E1D,E2A,E1DCn,Pat,NPat ) ||
                      MakeEnds ( &Beg1,B1D,B2A,&Beg1Cn,B1DCn,&End1,E2D,E1A,E2DCn,&Beg2,B1A,B2D,&Beg2Cn,B1ACn,
                                 &End2,E2A,E1D,E2ACn,Pat,NPat ) ) )
                ;
            else
                if ( ( Cn1 != Cn2 || abs ( B2A-E1D ) < abs ( B2A-E2D ) ) &&
                        ( MakeEnds ( &Beg1,B1A,B2D,&Beg1Cn,B1ACn,&End1,E2D,E1A,E2DCn,&Beg2,B1D,B2A,&Beg2Cn,B1DCn,
                                     &End2,E2A,E1D,E2ACn,Pat,NPat ) ||
                          MakeEnds ( &Beg1,B1A,B2D,&Beg1Cn,B1ACn,&End1,E1A,E2D,E1ACn,&Beg2,B1D,B2A,&Beg2Cn,B1DCn,
                                     &End2,E1D,E2A,E1DCn,Pat,NPat ) ) )
                    ;
                else
                    if ( ( Cn1 != Cn2 || abs ( B2A-E2D ) < abs ( B2A-E1D ) ) &&
                            ( MakeEnds ( &Beg1,B1A,B2D,&Beg1Cn,B1ACn,&End1,E1D,E2A,E1DCn,&Beg2,B1D,B2A,&Beg2Cn,B1DCn,
                                         &End2,E1A,E2D,E1ACn,Pat,NPat ) ||
                              MakeEnds ( &Beg1,B1A,B2D,&Beg1Cn,B1ACn,&End1,E2A,E1D,E2ACn,&Beg2,B1D,B2A,&Beg2Cn,B1DCn,
                                         &End2,E2D,E1A,E2DCn,Pat,NPat ) ) )
                        ;
                    else
                        if ( ( Cn1 != Cn2 || abs ( B1D-E2A ) < abs ( B2D-E2A ) ) &&
                                ( MakeEnds ( &Beg1,E1D,E2A,&Beg1Cn,E1DCn,&End1,B2A,B1D,B2ACn,&Beg2,E1A,E2D,&Beg2Cn,E1ACn,
                                             &End2,B2D,B1A,B2DCn,Pat,NPat ) ||
                                  MakeEnds ( &Beg1,E1D,E2A,&Beg1Cn,E1DCn,&End1,B1D,B2A,B1DCn,&Beg2,E1A,E2D,&Beg2Cn,E1ACn,
                                             &End2,B1A,B2D,B1ACn,Pat,NPat ) ) )
                            ;
                        else
                            if ( ( Cn1 != Cn2 || abs ( B2D-E2A ) < abs ( B1D-E2A ) ) &&
                                    ( MakeEnds ( &Beg1,E1D,E2A,&Beg1Cn,E1DCn,&End1,B1A,B2D,B1ACn,&Beg2,E1A,E2D,&Beg2Cn,E1ACn,
                                                 &End2,B1D,B2A,B1DCn,Pat,NPat ) ||
                                      MakeEnds ( &Beg1,E1D,E2A,&Beg1Cn,E1DCn,&End1,B2D,B1A,B2DCn,&Beg2,E1A,E2D,&Beg2Cn,E1ACn,
                                                 &End2,B2A,B1D,B2ACn,Pat,NPat ) ) )
                                ;
                            else
                                if ( ( Cn1 != Cn2 || abs ( E2A-B1D ) < abs ( E2A-B2D ) ) &&
                                        ( MakeEnds ( &Beg1,E1A,E2D,&Beg1Cn,E1ACn,&End1,B2D,B1A,B2DCn,&Beg2,E1D,E2A,&Beg2Cn,E1DCn,
                                                     &End2,B2A,B1D,B2ACn,Pat,NPat ) ||
                                          MakeEnds ( &Beg1,E1A,E2D,&Beg1Cn,E1ACn,&End1,B1A,B2D,B1ACn,&Beg2,E1D,E2A,&Beg2Cn,E1DCn,
                                                     &End2,B1D,B2A,B1DCn,Pat,NPat ) ) )
                                    ;
                                else
                                    if ( ( Cn1 != Cn2 || abs ( E2A-B2D ) < abs ( E2A-B1D ) ) &&
                                            ( MakeEnds ( &Beg1,E1A,E2D,&Beg1Cn,E1ACn,&End1,B1D,B2A,B1DCn,&Beg2,E1D,E2A,&Beg2Cn,E1DCn,
                                                         &End2,B1A,B2D,B1ACn,Pat,NPat ) ||
                                              MakeEnds ( &Beg1,E1A,E2D,&Beg1Cn,E1ACn,&End1,B2A,B1D,B2ACn,&Beg2,E1D,E2A,&Beg2Cn,E1DCn,
                                                         &End2,B2D,B1A,B2DCn,Pat,NPat ) ) )
                                        ;
                                    else
                                    {
                                        fprintf ( stdout,"Ne tot variant.. Par %s%c\n",Chain[Cn1]->File,Chain[Cn1]->Id );
                                        continue;
                                    }

        if ( Beg1Cn == Chain[Cn1]->Id )
        {
            for ( j=Beg1; j<=End1; j++ ) Asn1[j] = 'P';
            for ( j=Beg2; j<=End2; j++ ) Asn2[j] = 'P';
        }
        else
        {
            for ( j=Beg1; j<=End1; j++ ) Asn2[j] = 'P';
            for ( j=Beg2; j<=End2; j++ ) Asn1[j] = 'P';
        }

        Pat[i]->Nei1 = NULL;
        Pat[i]->Nei2 = NULL;
        CurrPat->Nei1 = NULL;
        CurrPat->Nei2 = NULL;

    }
}

void Stride::Alias( int *D1,int *A1,int *D2,int *A2,char *D1Cn,char *A1Cn,char *D2Cn,
                char *A2Cn, PATTERN *Pat )
{
    *D1 = Pat->Hb1->Dnr->D_Res;
    *A1 = Pat->Hb1->Acc->A_Res;
    *D2 = Pat->Hb2->Dnr->D_Res;
    *A2 = Pat->Hb2->Acc->A_Res;
    *D1Cn = Pat->Hb1->Dnr->Chain->Id;
    *A1Cn = Pat->Hb1->Acc->Chain->Id;
    *D2Cn = Pat->Hb2->Dnr->Chain->Id;
    *A2Cn = Pat->Hb2->Acc->Chain->Id;
}

void Stride::Bridge( char *Asn1, char *Asn2, CHAIN **Chain, int Cn1, int Cn2, PATTERN **Pat, int NPat )
{
    int i;
    int B_Res;

    for ( i=0; i<NPat; i++ )
    {
        if ( Pat[i]->Nei1 != NULL || Pat[i]->Nei2 != NULL ) continue;

        if ( !strcmp ( Pat[i]->Type,"1331" ) &&
                ( Cn1 != Cn2 || abs ( Pat[i]->Hb1->Dnr->D_Res-Pat[i]->Hb1->Acc->A_Res ) >= 3 ) )
        {

            if ( Pat[i]->Hb1->Dnr->Chain->Id == Chain[Cn1]->Id )
            {
                if ( Asn1[Pat[i]->Hb1->Dnr->D_Res] == 'C' )
                    Asn1[Pat[i]->Hb1->Dnr->D_Res] = 'B';
                if ( Asn2[Pat[i]->Hb1->Acc->A_Res] == 'C' )
                    Asn2[Pat[i]->Hb1->Acc->A_Res] = 'B';
            }
            else
            {
                if ( Asn2[Pat[i]->Hb1->Dnr->D_Res] == 'C' )
                    Asn2[Pat[i]->Hb1->Dnr->D_Res] = 'B';
                if ( Asn1[Pat[i]->Hb1->Acc->A_Res] == 'C' )
                    Asn1[Pat[i]->Hb1->Acc->A_Res] = 'B';
            }

        }
        else
            if ( !strcmp ( Pat[i]->Type,"3124" ) &&
                    ( Cn1 != Cn2 ||
                      ( abs ( Pat[i]->Hb1->Dnr->D_Res-Pat[i]->Hb1->Acc->A_Res ) >= 2 &&
                        abs ( Pat[i]->Hb2->Dnr->D_Res-Pat[i]->Hb2->Acc->A_Res ) >= 2 ) ) )
            {

                if ( Pat[i]->Hb1->Dnr->Chain->Id == Chain[Cn1]->Id )
                {

                    if ( Pat[i]->Hb1->Dnr->D_Res > Pat[i]->Hb2->Acc->A_Res )
                        B_Res = Pat[i]->Hb1->Dnr->D_Res-1;
                    else
                        B_Res = Pat[i]->Hb1->Dnr->D_Res+1;

                    if ( Asn1[B_Res] == 'C' )
                        Asn1[B_Res] = 'B';

                    if ( Pat[i]->Hb2->Dnr->D_Res > Pat[i]->Hb1->Acc->A_Res )
                        B_Res = Pat[i]->Hb2->Dnr->D_Res-1;
                    else
                        B_Res = Pat[i]->Hb2->Dnr->D_Res+1;

                    if ( Asn2[B_Res] == 'C' )
                        Asn2[B_Res] = 'B';
                }
                else
                {
                    if ( Pat[i]->Hb1->Dnr->D_Res > Pat[i]->Hb2->Acc->A_Res )
                        B_Res = Pat[i]->Hb1->Dnr->D_Res-1;
                    else
                        B_Res = Pat[i]->Hb1->Dnr->D_Res+1;

                    if ( Asn2[B_Res] == 'C' )
                        Asn2[B_Res] = 'B';

                    if ( Pat[i]->Hb2->Dnr->D_Res > Pat[i]->Hb1->Acc->A_Res )
                        B_Res = Pat[i]->Hb2->Dnr->D_Res-1;
                    else
                        B_Res = Pat[i]->Hb2->Dnr->D_Res+1;

                    if ( Asn1[B_Res] == 'C' )
                        Asn1[B_Res] = 'B';
                }
            }
            else
                if ( ( ( !strcmp ( Pat[i]->Type,"3123" ) || !strcmp ( Pat[i]->Type,"1341" ) ) &&
                        ( Cn1 != Cn2 ||
                          ( abs ( Pat[i]->Hb1->Dnr->D_Res-Pat[i]->Hb1->Acc->A_Res ) > 3 &&
                            abs ( Pat[i]->Hb2->Dnr->D_Res-Pat[i]->Hb2->Acc->A_Res ) > 3 ) ) ) )
                {

                    if ( Pat[i]->Hb1->Dnr->Chain->Id == Chain[Cn1]->Id )
                    {

                        if ( Pat[i]->Hb1->Dnr->D_Res == Pat[i]->Hb2->Acc->A_Res )
                        {

                            if ( Asn1[Pat[i]->Hb1->Dnr->D_Res] == 'C' )
                                Asn1[Pat[i]->Hb1->Dnr->D_Res] = 'B';

                            if ( Pat[i]->Hb2->Dnr->D_Res > Pat[i]->Hb1->Acc->A_Res )
                                B_Res = Pat[i]->Hb2->Dnr->D_Res-1;
                            else
                                B_Res = Pat[i]->Hb2->Dnr->D_Res+1;

                            if ( Asn2[B_Res] == 'C' )
                                Asn2[B_Res] = 'B';
                        }
                        else
                        {
                            if ( Pat[i]->Hb2->Dnr->D_Res == Pat[i]->Hb1->Acc->A_Res )

                                if ( Asn2[Pat[i]->Hb2->Dnr->D_Res] == 'C' )
                                    Asn2[Pat[i]->Hb2->Dnr->D_Res] = 'B';

                            if ( Pat[i]->Hb1->Dnr->D_Res > Pat[i]->Hb2->Acc->A_Res )
                                B_Res = Pat[i]->Hb1->Dnr->D_Res-1;
                            else
                                B_Res = Pat[i]->Hb1->Dnr->D_Res+1;

                            if ( Asn1[B_Res] == 'C' )
                                Asn1[B_Res] = 'B';
                        }
                    }
                }
                else
                    if ( ( !strcmp ( Pat[i]->Type,"13B1" ) || !strcmp ( Pat[i]->Type,"133A" ) ) &&
                            ( Cn1 != Cn2 ||
                              ( abs ( Pat[i]->Hb1->Dnr->D_Res-Pat[i]->Hb1->Acc->A_Res ) > 4 &&
                                abs ( Pat[i]->Hb2->Dnr->D_Res-Pat[i]->Hb2->Acc->A_Res ) > 4 ) ) )
                    {

                        if ( Pat[i]->Hb1->Dnr->Chain->Id == Chain[Cn1]->Id )
                        {

                            if ( Pat[i]->Hb1->Dnr->D_Res == Pat[i]->Hb2->Acc->A_Res )
                            {

                                if ( Asn1[Pat[i]->Hb1->Dnr->D_Res] == 'C' )
                                    Asn1[Pat[i]->Hb1->Dnr->D_Res] = 'B';

                                if ( Pat[i]->Hb2->Dnr->D_Res > Pat[i]->Hb1->Acc->A_Res )
                                    B_Res = Pat[i]->Hb2->Dnr->D_Res-1;
                                else
                                    B_Res = Pat[i]->Hb2->Dnr->D_Res+1;

                                if ( Asn2[B_Res] == 'C' )
                                    Asn2[B_Res] = 'B';
                            }
                            else
                            {
                                if ( Pat[i]->Hb2->Dnr->D_Res == Pat[i]->Hb1->Acc->A_Res )

                                    if ( Asn2[Pat[i]->Hb2->Dnr->D_Res] == 'C' )
                                        Asn2[Pat[i]->Hb2->Dnr->D_Res] = 'b';

                                if ( Pat[i]->Hb1->Dnr->D_Res > Pat[i]->Hb2->Acc->A_Res )
                                    B_Res = Pat[i]->Hb1->Dnr->D_Res-1;
                                else
                                    B_Res = Pat[i]->Hb1->Dnr->D_Res+1;

                                if ( Asn1[B_Res] == 'C' )
                                    Asn1[B_Res] = 'b';
                            }
                        }
                    }
    }
}

const char* Stride::Translate( char Code )
{
    const char *Dictionary[18] =
    {
        "AlphaHelix","310Helix","PiHelix","Strand","Bridge","Coil","TurnI","TurnI'",
        "TurnII","TurnII'","TurnVIa","TurnVIb","TurnVIII","TurnIV","GammaClassic",
        "GammaInv","Turn","Unknown"
    };

    switch ( Code )
    {
        case 'H': return ( Dictionary[0] );
        case 'G': return ( Dictionary[1] );
        case 'I': return ( Dictionary[2] );
        case 'E': return ( Dictionary[3] );
        case 'B':
        case 'b':
            return ( Dictionary[4] );
        case 'C': return ( Dictionary[5] );
        case '1': return ( Dictionary[6] );
        case '2': return ( Dictionary[7] );
        case '3': return ( Dictionary[8] );
        case '4': return ( Dictionary[9] );
        case '5': return ( Dictionary[10] );
        case '6': return ( Dictionary[11] );
        case '7': return ( Dictionary[12] );
        case '8': return ( Dictionary[13] );
        case '@': return ( Dictionary[14] );
        case '&': return ( Dictionary[15] );
        case 'T': return ( Dictionary[16] );
        default:  return ( Dictionary[17] );
    }
}

Stride::BOOLEAN Stride::ExistsSecStr( CHAIN **Chain, int NChain )
{
    int i, Cn;

    for ( Cn=0; Cn<NChain; Cn++ )
        for ( i=0; i<Chain[Cn]->NRes; i++ )
            if ( Chain[Cn]->Rsd[i]->Prop->Asn != 'C' )
                return ( STRIDE_YES );

    return ( STRIDE_NO );
}

void Stride::ExtractAsn( CHAIN **Chain, int Cn, char *Asn )
{
    int Res;

    for ( Res=0; Res<Chain[Cn]->NRes; Res++ )
        Asn[Res] = Chain[Cn]->Rsd[Res]->Prop->Asn;
}

int Stride::Boundaries( char *Asn, int L, char SecondStr, int ( *Bound ) [2] )
{
    int Res;
    int NStr = 0, Flag = 0;

    for ( Res=0; Res<L; Res++ )
    {
        if ( Asn[Res] == SecondStr && Flag == 0 )
        {
            Flag = 1;
            Bound[NStr][0] = Res;
        }
        else
            if ( Asn[Res] != SecondStr && Flag == 1 )
            {
                Flag = 0;
                Bound[NStr++][1] = Res-1;
            }
    }
    return ( NStr );
}

void Stride::InitChain( CHAIN **Chain )
{
    *Chain = ( CHAIN * ) ckalloc ( sizeof ( CHAIN ) );

    ( *Chain )->NRes                = 0;
    ( *Chain )->NHelix              = 0;
    ( *Chain )->NSheet              = -1;
    ( *Chain )->NTurn               = 0;
    ( *Chain )->NAssignedTurn       = 0;
    ( *Chain )->NBond               = 0;
    ( *Chain )->NHydrBond           = 0;
    ( *Chain )->NHydrBondTotal      = 0;
    ( *Chain )->NHydrBondInterchain = 0;
    ( *Chain )->Ter                 = 0;
    ( *Chain )->Resolution          = 0.0;

    ( *Chain )->File                = ( char           * ) ckalloc ( BUFSZ*sizeof ( char ) );
    ( *Chain )->Rsd                 = ( RESIDUE       ** ) ckalloc ( MAX_RES*sizeof ( RESIDUE * ) );
    ( *Chain )->Helix               = ( HELIX         ** ) ckalloc ( MAX_HELIX*sizeof ( HELIX * ) );
    ( *Chain )->Sheet               = ( SHEET         ** ) ckalloc ( MAX_SHEET*sizeof ( SHEET * ) );
    ( *Chain )->Turn                = ( TURN          ** ) ckalloc ( MAX_TURN*sizeof ( TURN * ) );
    ( *Chain )->AssignedTurn        = ( TURN          ** ) ckalloc ( MAX_TURN*sizeof ( TURN * ) );
    ( *Chain )->SSbond              = ( SSBOND        ** ) ckalloc ( MAX_BOND*sizeof ( SSBOND * ) );

    ( *Chain )->Valid               = STRIDE_YES;
}

int Stride::SplitString( char *Buffer, char **Fields, int MaxField )
{
    int FieldCnt, SymbCnt, FieldFlag, BuffLen;
    static char LocalBuffer[BUFSZ];


    FieldCnt =0; FieldFlag = 0;
    BuffLen = ( int ) strlen ( Buffer ) - 1;

    strcpy ( LocalBuffer,Buffer );

    for ( SymbCnt=0; SymbCnt<BuffLen; SymbCnt++ )
    {
        if ( ( isspace ( LocalBuffer[SymbCnt] ) ) && FieldFlag == 0 && SymbCnt != BuffLen-1 ) continue;
        if ( ( !isspace ( LocalBuffer[SymbCnt] ) ) && FieldFlag == 1 && SymbCnt == BuffLen-1 )
        {
            LocalBuffer[SymbCnt+1] = '\0';
            return ( FieldCnt );
        }
        else
            if ( ( isspace ( LocalBuffer[SymbCnt] ) ) && FieldFlag == 1 )
            {
                LocalBuffer[SymbCnt] = '\0';
                FieldFlag = 0;
                if ( FieldCnt == MaxField ) return ( FieldCnt );
            }
            else
                if ( ( !isspace ( LocalBuffer[SymbCnt] ) ) && FieldFlag == 0 )
                {
                    FieldFlag = 1;
                    Fields[FieldCnt] = LocalBuffer+SymbCnt;
                    FieldCnt++;
                }
    }

    return ( FieldCnt );
}

void Stride::Project4_123( float *Coord1, float *Coord2, float *Coord3,
                                    float *Coord4, float *Coord_Proj4_123 )
{

    /*
                              Atom4
       Atom3                  .
       \                     .
        \                   .
         \                 .  .Proj4_123
          \               ..
          Atom2-------Atom1

    */


    float Vector21[3], Vector23[3], Vector14[3], VectorNormal_123[3];
    float Length_21 = 0.0, Length_23 = 0.0, Length_14 = 0.0, NormalLength;
    float COS_Norm_14, Proj_14_Norm;
    int i;

    for ( i=0; i<3; i++ )
    {
        Vector21[i] = Coord1[i] - Coord2[i];
        Vector23[i] = Coord3[i] - Coord2[i];
        Vector14[i] = Coord4[i] - Coord1[i];
        Length_21 += Vector21[i]*Vector21[i];
        Length_23 += Vector23[i]*Vector23[i];
        Length_14 += Vector14[i]*Vector14[i];
    }

    Length_21 = sqrt ( Length_21 );
    Length_23 = sqrt ( Length_23 );
    Length_14 = sqrt ( Length_14 );

    NormalLength = VectorProduct ( Vector21,Vector23,VectorNormal_123 );

    for ( i=0; i<3; i++ )
        VectorNormal_123[i] /= NormalLength;

    COS_Norm_14 = 0.0;

    for ( i=0; i<3; i++ )
        COS_Norm_14 += VectorNormal_123[i]*Vector14[i];

    COS_Norm_14 /= ( Length_14*NormalLength );

    if ( COS_Norm_14 < 0.0 )
    {
        COS_Norm_14 = fabs ( COS_Norm_14 );
        for ( i=0; i<3; i++ )
            VectorNormal_123[i] = -VectorNormal_123[i];
    }

    Proj_14_Norm = Length_14*COS_Norm_14;

    for ( i=0; i<3; i++ )
    {
        VectorNormal_123[i] *= Proj_14_Norm;
        Coord_Proj4_123[i] = ( Vector14[i] - VectorNormal_123[i] ) + Coord1[i];
    }
}

int Stride::MakeEnds( int *Beg1, int ResBeg1, int NeiBeg1, char *Beg1Cn, char ResBeg1Cn,
                             int *End1, int ResEnd1, int NeiEnd1, char ResEnd1Cn, int *Beg2,
                             int ResBeg2, int NeiBeg2, char *Beg2Cn, char ResBeg2Cn, int *End2,
                             int ResEnd2, int NeiEnd2, char ResEnd2Cn, PATTERN **Pat, int NPat )
{
    int i;
    int Flag1 = 0, Flag2 = 0;


    if ( ResBeg1 <= NeiBeg1 && NeiBeg1 <= NeiEnd1 && NeiEnd1 <= ResEnd1 &&
            ResBeg2 <= NeiBeg2 && NeiBeg2 <= NeiEnd2 && NeiEnd2 <= ResEnd2 &&
            ResBeg1Cn == ResEnd1Cn && ResBeg2Cn == ResEnd2Cn )
    {

        *Beg1 = ResBeg1;
        *End1 = ResEnd1;
        *Beg2 = ResBeg2;
        *End2 = ResEnd2;
        *Beg1Cn = ResBeg1Cn;
        *Beg2Cn = ResBeg2Cn;

        for ( i=0; i<NPat && ( Flag1 == 0 || Flag2 == 0 ); i++ )
        {
            if ( ( ( Pat[i]->Hb1->Dnr->D_Res == ( *Beg1 )
                     && Pat[i]->Hb1->Acc->A_Res == ( *End2 )
                     && Pat[i]->Hb1->Dnr->Chain->Id == ( *Beg1Cn )
                     && Pat[i]->Hb1->Acc->Chain->Id == ( *Beg2Cn ) )
                    ||
                    ( Pat[i]->Hb1->Acc->A_Res == ( *Beg1 )
                      && Pat[i]->Hb1->Dnr->D_Res == ( *End2 )
                      && Pat[i]->Hb1->Acc->Chain->Id == ( *Beg1Cn )
                      && Pat[i]->Hb1->Dnr->Chain->Id == ( *Beg2Cn ) ) )
                    && Pat[i]->Hb1->Dnr->D_Res == Pat[i]->Hb2->Acc->A_Res
                    && Pat[i]->Hb2->Dnr->D_Res == Pat[i]->Hb1->Acc->A_Res )
                Flag1 = 1;
            if ( ( ( Pat[i]->Hb1->Dnr->D_Res == ( *Beg2 )
                     && Pat[i]->Hb1->Acc->A_Res == ( *End1 )
                     && Pat[i]->Hb1->Dnr->Chain->Id == ( *Beg2Cn )
                     && Pat[i]->Hb1->Acc->Chain->Id == ( *Beg1Cn ) )
                    ||
                    ( Pat[i]->Hb1->Acc->A_Res == ( *Beg2 )
                      && Pat[i]->Hb1->Dnr->D_Res == ( *End1 )
                      && Pat[i]->Hb1->Acc->Chain->Id == ( *Beg2Cn )
                      && Pat[i]->Hb1->Dnr->Chain->Id == ( *Beg1Cn ) ) )
                    && Pat[i]->Hb1->Dnr->D_Res == Pat[i]->Hb2->Acc->A_Res
                    && Pat[i]->Hb2->Dnr->D_Res == Pat[i]->Hb1->Acc->A_Res )
                Flag2 = 1;
        }

        if ( !Flag1 )
        {
            if ( *Beg1 != NeiBeg1 ) ( *Beg1 ) ++;
            if ( *End2 != NeiEnd2 ) ( *End2 )--;
        }

        if ( !Flag2 )
        {
            if ( *End1 != NeiEnd1 ) ( *End1 )--;
            if ( *Beg2 != NeiBeg2 ) ( *Beg2 ) ++;
        }
        return ( SUCCESS );
    }

    return ( FAILURE );
}

float Stride::VectorProduct( float *Vector1, float *Vector2, float *Product )
{
    int i, j, k;
    float ProductLength;

    ProductLength = 0.0;

    for ( i=0; i<3; i++ )
    {
        j = ( i+1 ) %3;
        k = ( j+1 ) %3;
        Product[i] = Vector1[j]*Vector2[k] - Vector1[k]*Vector2[j];
        ProductLength += Product[i]*Product[i];
    }

    return ( sqrt ( ProductLength ) );
}

void Stride::PostProcessHBonds(megamol::protein_calls::MolecularDataCall *mol) {
	this->ownHydroBonds.resize(HydroBondCnt * 2);

	for (unsigned int bondIdx = 0; bondIdx < static_cast<unsigned int>(HydroBondCnt); bondIdx++) {
		auto bond = HydroBond[bondIdx];
		unsigned int donor = GetMoleculeIndex(bond->Dnr->Chain->ChainId, bond->Dnr->D_Res, bond->Dnr->D_At, mol);
		unsigned int acceptor = GetMoleculeIndex(bond->Acc->Chain->ChainId, bond->Acc->A_Res, bond->Acc->A_At, mol);
		this->ownHydroBonds[bondIdx * 2 + 0] = donor;
		this->ownHydroBonds[bondIdx * 2 + 1] = acceptor;
	}

	mol->SetHydrogenBonds(this->ownHydroBonds.data(), static_cast<unsigned int>(HydroBondCnt));
}

unsigned int Stride::GetMoleculeIndex(unsigned int ChainIdx, unsigned int ResidueIdx, unsigned int InternalIdx, megamol::protein_calls::MolecularDataCall *mol) {
	unsigned int firstResidue = mol->Molecules()[ChainIdx].FirstResidueIndex();
	unsigned int firstAtom = mol->Residues()[firstResidue + ResidueIdx]->FirstAtomIndex();
	return firstAtom + InternalIdx;
}