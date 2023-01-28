/*
 * Stride.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_STRIDE_H_INCLUDED
#define MEGAMOL_STRIDE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#define _USE_MATH_DEFINES

#include "protein_calls/MolecularDataCall.h"
#include "vislib/String.h"
#include "vislib/math/Vector.h"
#include <cstdio>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

namespace megamol::protein {

#define Eps 0.000001
#define Minimum(x, y) ((x) < (y) ? x : y)
#define Maximum(x, y) ((x) < (y) ? y : x)
#define IN_STRIDE(x, target, range) ((x >= (target - range)) && (x <= (target + range)))
#define RAD(x) (x) * M_PI / 180.0
#define DEG(x) (x) * 180.0 / M_PI
#define RADDEG 57.2958
#define BREAKDIST 2.5
#define SSDIST 3.0

#define SUCCESS 1
#define FAILURE 0
#define STRIDE_YES 1
#define STRIDE_NO 0
#define ERR -1

#define BUFSZ 1024
#define MAX_FIELD 50

#define MAX_AtomType 200
#define MAX_ResType 50
#define MAXNONSTAND 4.0
#define MAX_CHAIN 100
#define MAX_RES 10000
#define MAX_HELIX 100
#define MAX_SHEET 100
#define MAX_STRAND_IN_SHEET 20
#define MAX_TURN 100
#define MAX_BOND 100
#define MAX_ASSIGN 300
#define MAX_AT_IN_RES 150 // 50
#define MAX_AT_IN_HETERORES 200
#define MAXRESDNR 6
#define MAXRESACC 6

#define RES_FIELD 6
#define AT_FIELD 5

#define MAX_X 180.000
#define MAX_Y 180.000
#define MAX_Z 180.000
#define MIN_X -100.000
#define MIN_Y -100.000
#define MIN_Z -100.000
#define MAX_Occupancy 1.00
#define MIN_Occupancy 0.00
#define MAX_TempFactor 1000.00
#define MIN_TempFactor 0.00

#define OUTPUTWIDTH 80
#define MAXCONDITIONS 20

#define MAXHYDRBOND 30000
#define MAXDONOR MAX_RES
#define MAXACCEPTOR MAX_RES

#define MINPHIPSI -180.0
#define MAXPHIPSI 180.0
#define DEFNUMPIXEL 18

#define DIST_N_H 1.0
#define RmGRID 3.0
#define EmGRID -2.8
#define CGRID -3.0 * EmGRID* pow(RmGRID, 8.0)
#define DGRID -4.0 * EmGRID* pow(RmGRID, 6.0)
#define K1GRID 0.9 / pow(cos(RAD(110.0)), 6.0)
#define K2GRID pow(cos(RAD(110.0)), 2.0)

#define MINACCANG_SP2 90.0
#define MAXACCANG_SP2 180.0
#define MINACCANG_SP3 60.0
#define MAXACCANG_SP3 180.0
#define MINDONANG_SP2 90.0
#define MAXDONANG_SP2 180.0
#define MINDONANG_SP3 90.0
#define MAXDONANG_SP3 180.0
#define ACCDONANG 60.0
#define DONACCANG 90.0

class Stride {
public:
    enum ASNSOURCE { StrideX, Pdb };
    enum METHOD { XRay, NMR, Model };
    enum HYBRID { Nsp2, Nsp3, Osp2, Osp3, Ssp3 };
    enum GROUP { Peptide, Trp, Asn, Gln, Arg, His, Lys, Ser, Thr, Tyr, Asp, Glu, Met, Cys };
    enum HBONDTYPE { MM, MS, SM, SS };

    typedef char BUFFER[BUFSZ + 1];

    typedef int BOOLEAN;

    typedef struct // PROPERTY
    {
        float Phi, Psi, Omega;
        int PhiZn, PsiZn;
        float Solv;
        char Asn, PdbAsn;
    } PROPERTY;

    typedef struct // INVOLVED
    {
        int HBondDnr[MAXRESDNR];
        int HBondAcc[MAXRESACC];
        int NBondDnr, NBondAcc;
        BOOLEAN InterchainHBonds;
    } INVOLVED;

    typedef struct // RESIDUE
    {
        int NAtom;
        char PDB_ResNumb[RES_FIELD];
        int ResNumb;
        char ResType[RES_FIELD];
        char AtomType[MAX_AT_IN_RES][AT_FIELD];
        float Coord[MAX_AT_IN_RES][3];
        float Occupancy[MAX_AT_IN_RES];
        float TempFactor[MAX_AT_IN_RES];
        int ResAtomIdx[MAX_AT_IN_RES];
        PROPERTY* Prop;
        INVOLVED* Inv;
    } RESIDUE;

    typedef struct // HELIX
    {
        char Res1[RES_FIELD];
        char Res2[RES_FIELD];
        char PDB_ResNumb1[RES_FIELD], PDB_ResNumb2[RES_FIELD];
        char InsCode1, InsCode2;
        int Class;
    } HELIX;

    typedef struct // SHEET
    {
        int NStrand;
        char SheetId[RES_FIELD];
        char ResType1[MAX_STRAND_IN_SHEET][RES_FIELD];
        char ResType2[MAX_STRAND_IN_SHEET][RES_FIELD];
        char PDB_ResNumb1[MAX_STRAND_IN_SHEET][RES_FIELD];
        char PDB_ResNumb2[MAX_STRAND_IN_SHEET][RES_FIELD];
        char InsCode1[MAX_STRAND_IN_SHEET];
        char InsCode2[MAX_STRAND_IN_SHEET];
        int Sence[MAX_STRAND_IN_SHEET];
        int RegYN[MAX_STRAND_IN_SHEET];
        char AtomNameReg1[MAX_STRAND_IN_SHEET][AT_FIELD];
        char AtomNameReg2[MAX_STRAND_IN_SHEET][AT_FIELD];
        char ResTypeReg1[MAX_STRAND_IN_SHEET][RES_FIELD];
        char ResTypeReg2[MAX_STRAND_IN_SHEET][RES_FIELD];
        char PDB_ResNumbReg1[MAX_STRAND_IN_SHEET][RES_FIELD];
        char PDB_ResNumbReg2[MAX_STRAND_IN_SHEET][RES_FIELD];
        char InsCodeReg1[MAX_STRAND_IN_SHEET];
        char InsCodeReg2[MAX_STRAND_IN_SHEET];
    } SHEET;

    typedef struct // TURN
    {
        char Res1[RES_FIELD];
        char Res2[RES_FIELD];
        char PDB_ResNumb1[RES_FIELD], PDB_ResNumb2[RES_FIELD];
        char InsCode1, InsCode2;
        char TurnType;
    } TURN;

    typedef struct // SSBOND
    {
        char PDB_ResNumb1[RES_FIELD], PDB_ResNumb2[RES_FIELD];
        char InsCode1, InsCode2;
        char ChainId1, ChainId2;
        enum ASNSOURCE AsnSource;
    } SSBOND;

    typedef struct // CHAIN
    {
        int NRes, Ter;
        int NAtom, NHelix, NSheet;
        int NTurn, NAssignedTurn, NBond, NHydrBond, NHydrBondInterchain, NHydrBondTotal;
        char Id, *File;
        int ChainId;
        float Resolution;
        BOOLEAN Valid;

        RESIDUE** Rsd;
        HELIX** Helix;
        SHEET** Sheet;
        TURN** Turn;
        TURN** AssignedTurn;
        SSBOND** SSbond;
        char PdbIdent[5];

    } CHAIN;

    typedef struct // DONOR
    {
        CHAIN* Chain;
        int D_Res, DD_Res, DDI_Res;
        int D_At, DD_At, DDI_At, H;
        enum HYBRID Hybrid;
        enum GROUP Group;
        float HB_Radius;
    } DONOR;

    typedef struct // ACCEPTOR
    {
        CHAIN* Chain;
        int A_Res, AA_Res, AA2_Res;
        int A_At, AA_At, AA2_At;
        enum HYBRID Hybrid;
        enum GROUP Group;
        float HB_Radius;
    } ACCEPTOR;

    typedef struct // COMMAND
    {
        BUFFER InputFile, OutFile;
        BUFFER MapFileHelix, MapFileSheet;
        char EnergyType, Active[MAX_CHAIN + 1];
        char Processed[MAX_CHAIN + 1], Cond[MAXCONDITIONS];

        int NPixel, NActive, NProcessed;
        int MinLength, MaxLength;

        float PhiPsiStep, DistCutOff;
        float Treshold_H1, Treshold_H2, Treshold_H3, Treshold_H4;
        float Treshold_E1, Treshold_E2, Treshold_E3, Treshold_E4;
        float MinResolution, MaxResolution;
        float C1_H, C2_H, C1_E, C2_E;

        BOOLEAN SideChainHBond, MainChainHBond, MainChainPolarInt;
        BOOLEAN UseResolution, Truncate;
        BOOLEAN OutSeq;

    } COMMAND;

    typedef struct // HBOND
    {
        DONOR* Dnr;
        ACCEPTOR* Acc;
        BOOLEAN ExistPolarInter, ExistHydrBondRose, ExistHydrBondBaker;
        float Energy, Er, Et, Ep, ti, to, p;
        float AccDonDist, OHDist, AngNHO, AngCOH;
        float AccAng, DonAng, AccDonAng, DonAccAng;
    } HBOND;

    typedef struct PAT // PATTERN
    {
        HBOND *Hb1, *Hb2;
        struct PAT *Nei1, *Nei2;
        BOOLEAN ExistPattern;
        BUFFER Type;
    } PATTERN;

    Stride(megamol::protein_calls::MolecularDataCall* mol);
    virtual ~Stride();

    bool WriteToInterface(megamol::protein_calls::MolecularDataCall* mol);

protected:
    typedef struct // OWNBOND
    {
        unsigned int donor;
        unsigned int acceptor;
    } OWNBOND;

    void GetChains(megamol::protein_calls::MolecularDataCall* mol);
    bool ComputeSecondaryStructure();

    void PostProcessHBonds(megamol::protein_calls::MolecularDataCall* mol);
    unsigned int GetMoleculeIndex(unsigned int ChainIdx, unsigned int ResidueIdx, unsigned int InternalIndex,
        megamol::protein_calls::MolecularDataCall* mol);

    void DefaultCmd(COMMAND* Cmd);
    int ReadPDBFile(CHAIN** Chain, int* Cn, COMMAND* Cmd);
    void die(const char* format, ...);
    int CheckChain(CHAIN* Chain, COMMAND* Cmd);
    void BackboneAngles(CHAIN** Chain, int NChain);
    float** DefaultHelixMap(COMMAND* Cmd);
    float** DefaultSheetMap(COMMAND* Cmd);
    int PlaceHydrogens(CHAIN* Chain);
    int FindHydrogenBonds(CHAIN** Chain, int NChain, HBOND** HBond, COMMAND* Cmd);
    int NoDoubleHBond(HBOND** HBond, int NHBond);
    void DiscrPhiPsi(CHAIN** Chain, int NChain, COMMAND* Cmd);
    void Helix(CHAIN** Chain, int Cn, HBOND** HBond, COMMAND* Cmd, float** PhiPsiMap);
    void Sheet(CHAIN** Chain, int Cn1, int Cn2, HBOND** HBond, COMMAND* Cmd, float** PhiPsiMap);
    void BetaTurn(CHAIN** Chain, int Cn);
    void GammaTurn(CHAIN** Chain, int Cn, HBOND** HBond);
    int TurnCondition(float Phi2, float Phi2S, float Psi2, float Psi2S, float Phi3, float Phi3S, float Psi3,
        float Psi3S, float Range1, float Range2);
    int SSBond(CHAIN** Chain, int NChain);
    BOOLEAN ExistSSBond(CHAIN** Chain, int NChain, int Cn1, int Cn2, char* Res1, char* Res2);
    void Report(CHAIN** Chain, int NChain, HBOND** HBond, COMMAND* Cmd);
    void ReportSSBonds(CHAIN** Chain, FILE* Out);
    void ReportTurnTypes(CHAIN** Chain, int NChain, FILE* Out, COMMAND* Cmd);
    void ReportShort(CHAIN** Chain, int NChain, FILE* Out, COMMAND* Cmd);
    void PrepareBuffer(BUFFER Bf, CHAIN** Chain);
    void Glue(const char* String1, const char* String2, FILE* Out);
    void* ckalloc(size_t bytes);
    int Process_ENDMDL(BUFFER Buffer, CHAIN** Chain, int* ChainNumber);
    int Process_ATOM(BUFFER Buffer, CHAIN** Chain, int* ChainNumber, BOOLEAN* First_ATOM, COMMAND* Cmd);
    int FindAtom(CHAIN* Chain, int ResNumb, const char* Atom, int* AtNumb);
    char SpaceToDash(char Id);
    BOOLEAN ChInStr(char* String, char Char);
    void PHI(CHAIN* Chain, int Res);
    void PSI(CHAIN* Chain, int Res);
    float Torsion(float* Coord1, float* Coord2, float* Coord3, float* Coord4);
    float Dist(float* Coord1, float* Coord2);
    int FindDnr(CHAIN* Chain, DONOR** Dnr, int* NDnr, COMMAND* Cmd);
    int DefineDnr(
        CHAIN* Chain, DONOR** Dnr, int* dc, int Res, enum HYBRID Hybrid, enum GROUP Group, float HB_Radius, int N);
    int FindAcc(CHAIN* Chain, ACCEPTOR** Acc, int* NAcc, COMMAND* Cmd);
    int DefineAcceptor(
        CHAIN* Chain, ACCEPTOR** Acc, int* ac, int Res, enum HYBRID Hybrid, enum GROUP Group, float HB_Radius, int N);
    void GRID_Energy(float* CA2, float* C, float* O, float* H, float* N, COMMAND* Cmd, HBOND* HBond);
    float Ang(float* Coord1, float* Coord2, float* Coord3);
    int FindChain(CHAIN** Chain, int NChain, char ChainId);
    int FindPolInt(HBOND** HBond, RESIDUE* Res1, RESIDUE* Res2);
    int FindBnd(HBOND** HBond, RESIDUE* Res1, RESIDUE* Res2);
    int Link(HBOND** HBond, CHAIN** Chain, int Cn1, int Cn2, RESIDUE* Res1_1, RESIDUE* Res1_2, RESIDUE* Res2_2,
        RESIDUE* Res2_1, RESIDUE* CRes1, RESIDUE* CRes2, float** PhiPsiMap, PATTERN** Pattern, int* NumPat,
        const char* Text, float Treshold, COMMAND* Cmd, int Test);
    void FilterAntiPar(PATTERN** Pat, int NPat);
    void FilterPar(PATTERN** Pat, int NPat);
    void MergePatternsAntiPar(PATTERN** Pat, int NPat);
    void MergePatternsPar(PATTERN** Pat, int NPat);
    int RightSide2(int L_A1, int L_D1, int LnkD, int LnkA, int I1A, int I1D, int I2A, int I2D);
    int RightSide(int LnkA, int LnkD, int I1A, int I1D, int I2A, int I2D);
    int RightSidePar(int LnkA, int LnkD, int I1A, int I1D, int I2A, int I2D);
    void JoinNeighbours(int* Lnk1A, int Res1, int* Lnk1D, int Res2, PATTERN** Nei, PATTERN* Pat, int* MinDB1, int DB,
        int* MinDW1, int DW, int* Min, int j);
    void JoinNeighb(PATTERN** Nei, PATTERN* Pat, int* MinDB2, int DB, int* MinDW2, int DW);
    int NearPar(int Res1, int Res2, int Res3, int Res4, int Res5, int Res6, int Res7, int Res8, char Cn1, char Cn2,
        char Cn3, char Cn4, int* DistBest, int* DistWorst);
    int Near(int Res1, int Res2, int Res3, int Res4, int Res5, int Res6, int Res7, int Res8, char Cn1, char Cn2,
        char Cn3, char Cn4, int* DistBest, int* DistWorst);
    void FillAsnAntiPar(char* Asn1, char* Asn2, CHAIN** Chain, int Cn1, int Cn2, PATTERN** Pat, int NPat, COMMAND* Cmd);
    void FillAsnPar(char* Asn1, char* Asn2, CHAIN** Chain, int Cn1, int Cn2, PATTERN** Pat, int NPat, COMMAND* Cmd);
    void Alias(int* D1, int* A1, int* D2, int* A2, char* D1Cn, char* A1Cn, char* D2Cn, char* A2Cn, PATTERN* Pat);
    void Bridge(char* Asn1, char* Asn2, CHAIN** Chain, int Cn1, int Cn2, PATTERN** Pat, int NPat);
    const char* Translate(char Code);
    BOOLEAN ExistsSecStr(CHAIN** Chain, int NChain);
    void ExtractAsn(CHAIN** Chain, int Cn, char* Asn);
    int Boundaries(char* Asn, int L, char SecondStr, int (*Bound)[2]);
    void InitChain(CHAIN** Chain);
    void FreeChain(CHAIN* Chain);
    void FreeResidue(RESIDUE* r);
    void FreeHBond(HBOND* h);
    int SplitString(char* Buffer, char** Fields, int MaxField);
    void Project4_123(float* Coord1, float* Coord2, float* Coord3, float* Coord4, float* Coord_Proj4_123);
    int MakeEnds(int* Beg1, int ResBeg1, int NeiBeg1, char* Beg1Cn, char ResBeg1Cn, int* End1, int ResEnd1, int NeiEnd1,
        char ResEnd1Cn, int* Beg2, int ResBeg2, int NeiBeg2, char* Beg2Cn, char ResBeg2Cn, int* End2, int ResEnd2,
        int NeiEnd2, char ResEnd2Cn, PATTERN** Pat, int NPat);
    float VectorProduct(float* Vector1, float* Vector2, float* Product);

private:
    COMMAND* StrideCmd;
    CHAIN** ProteinChain;
    int ProteinChainCnt;
    HBOND** HydroBond;
    int HydroBondCnt;
    std::vector<unsigned int> ownHydroBonds;

    // was the computation successful?
    bool Successful;
};

} // namespace megamol::protein

#endif /* MEGAMOL_STRIDE_H_INCLUDED */
