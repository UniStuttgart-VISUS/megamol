#include "BindingSiteDataSource.h"
#include "stdafx.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/sys/ASCIIFileBuffer.h"
#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/ProteinColor.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/sysfunctions.h"
#include <math.h>


using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

/*
 * BindingSiteDataSource::BindingSiteDataSource (CTOR)
 */
BindingSiteDataSource::BindingSiteDataSource(void)
        : megamol::core::Module()
        , dataOutSlot_("dataout", "The slot providing the binding site data")
        , pdbFilenameSlot_("pdbFilename", "The PDB file containing the binding site information")
        , colorTableFileParam_("ColorTableFilename", "The filename of the color table.")
        , enzymeModeParam_("enzymeMode", "Activates the enzyme-mode, coloring only the relevant parts of the residues")
        , gxTypeFlag_("gxType", "Flag whether the protein used is a gx type or not") {

    this->pdbFilenameSlot_ << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->pdbFilenameSlot_);

    this->dataOutSlot_.SetCallback(BindingSiteCall::ClassName(),
        BindingSiteCall::FunctionName(BindingSiteCall::CallForGetData), &BindingSiteDataSource::getData);
    this->MakeSlotAvailable(&this->dataOutSlot_);

    // fill color table with default values and set the filename param
    this->colorTableFileParam_.SetParameter(
        new param::FilePathParam("colors.txt", core::param::FilePathParam::FilePathFlags_::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&this->colorTableFileParam_);
    ProteinColor::ReadColorTableFromFile(
        this->colorTableFileParam_.Param<param::FilePathParam>()->Value(), this->colorLookupTable_);

    this->enzymeModeParam_.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->enzymeModeParam_);

    this->gxTypeFlag_.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->gxTypeFlag_);
}

/*
 * BindingSiteDataSource::~BindingSiteDataSource (DTOR)
 */
BindingSiteDataSource::~BindingSiteDataSource(void) {
    this->Release();
}

/*
 * BindingSiteDataSource::create
 */
bool BindingSiteDataSource::create() {

    return true;
}

/*
 * BindingSiteDataSource::release
 */
void BindingSiteDataSource::release() {}

/*
 * BindingSiteDataSource::getData
 */
bool BindingSiteDataSource::getData(Call& call) {
    using megamol::core::utility::log::Log;

    BindingSiteCall* site = dynamic_cast<BindingSiteCall*>(&call);
    if (!site)
        return false;

    // read and update the color table, if necessary
    if (this->colorTableFileParam_.IsDirty()) {
        ProteinColor::ReadColorTableFromFile(
            this->colorTableFileParam_.Param<param::FilePathParam>()->Value(), this->colorLookupTable_);
        this->colorTableFileParam_.ResetDirty();
        this->enzymeModeParam_.ResetDirty();
        this->gxTypeFlag_.ResetDirty();
    }

    // try to load file, if necessary
    if (this->pdbFilenameSlot_.IsDirty()) {
        this->pdbFilenameSlot_.ResetDirty();
        this->loadPDBFile(this->pdbFilenameSlot_.Param<core::param::FilePathParam>()->Value().string());
    }

    // pass data to call, if available
    if (this->bindingSites_.empty()) {
        return false;
    } else {
        // site->SetDataHash( this->datahash);
        site->SetBindingSiteNames(&this->bindingSiteNames_);
        site->SetBindingSiteDescriptions(&this->bindingSiteDescription_);
        site->SetBindingSiteResNames(&this->bindingSiteResNames_);
        site->SetBindingSite(&this->bindingSites_);
        site->SetBindingSiteColors(reinterpret_cast<std::vector<glm::vec3>*>(&this->bindingSiteColors_));
        site->SetEnzymeMode(this->enzymeModeParam_.Param<param::BoolParam>()->Value());
        site->SetGXTypeFlag(this->gxTypeFlag_.Param<param::BoolParam>()->Value());
        return true;
    }
}

/*
 * BindingSiteDataSource::loadPDBFile
 */
void BindingSiteDataSource::loadPDBFile(const std::string& filename) {
    using megamol::core::utility::log::Log;

    // temp variables
    unsigned int i, j, lineCnt, bsIdx, /*resCnt,*/ cnt;
    std::string line, seqNumString, tmpBSName;
    char chainId;
    unsigned int resId;
    vislib::sys::ASCIIFileBuffer file;
    std::vector<std::string> bsEntries;
    std::vector<std::string> remarkEntries;
    SIZE_T entriesCapacity = 100;
    bsEntries.reserve(entriesCapacity);
    remarkEntries.reserve(entriesCapacity);

    // reset data
    for (i = 0; i < this->bindingSites_.size(); i++) {
        this->bindingSites_[i].clear();
        this->bindingSiteResNames_[i].clear();
    }
    this->bindingSites_.clear();
    this->bindingSites_.reserve(20);
    this->bindingSiteResNames_.clear();
    this->bindingSiteResNames_.reserve(20);
    this->bindingSiteNames_.clear();
    this->bindingSiteNames_.reserve(20);

    // try to load the file
    if (file.LoadFile(filename.c_str())) {
        // file successfully loaded, read first frame
        lineCnt = 0;
        while (lineCnt < file.Count() && !strStartsWith(line, "END")) {
            // get the current line from the file
            line = file.Line(lineCnt);
            // store all site entries
            if (strStartsWith(line, "SITE") || strStartsWith(line, "BSITE")) {
                // add site entry
                bsEntries.emplace_back(line);
            }
            // store all remark 800 entries
            if (strStartsWith(line, "REMARK 800")) {
                line = line.substr(10);
                strTrimSpaces(line);
                // add remark entry
                if (!line.empty()) {
                    remarkEntries.emplace_back(line);
                }
            }
            // next line
            lineCnt++;
        }

        // parse site entries
        for (unsigned int i = 0; i < bsEntries.size(); i++) {
            // write binding site name (check if this is the first entry)
            if (this->bindingSiteNames_.empty()) {
                this->bindingSiteNames_.emplace_back(bsEntries[i].substr(11, 4));
                strTrimSpaces(this->bindingSiteNames_.back());
                this->bindingSites_.emplace_back(std::vector<std::pair<char, unsigned int>>(10));
                this->bindingSiteResNames_.emplace_back(std::vector<std::string>(10));
                bsIdx = 0;
            } else {
                // check if next entry is still the same binding site
                tmpBSName = bsEntries[i].substr(11, 4);
                strTrimSpaces(tmpBSName);
                if (tmpBSName != bindingSiteNames_.back()) {
                    seqNumString = bsEntries[i].substr(7, 3);
                    strTrimSpaces(seqNumString);
                    if (std::stoi(seqNumString) == 1) {
                        this->bindingSiteNames_.emplace_back(bsEntries[i].substr(11, 4));
                        strTrimSpaces(this->bindingSiteNames_.back());
                        this->bindingSites_.emplace_back(std::vector<std::pair<char, unsigned int>>(10));
                        this->bindingSiteResNames_.emplace_back(std::vector<std::string>(10));
                        bsIdx++;
                    }
                }
            }

            // get number of residues
            // line = bsEntries[i].Substring( 15, 2);
            // line.TrimSpaces();
            //// regular PDB SITE entries can store a maximum of 4 residues per line
            // if (bsEntries[i].StartsWith("SITE")) {
            //    resCnt = vislib::math::Clamp(
            //        static_cast<unsigned int>(atoi(line) - bindingSites[bsIdx].Count()),
            //        0U, 4U);
            //}
            // else {
            //    resCnt = static_cast<unsigned int>(atoi(line) - bindingSites[bsIdx].Count());
            //}

            // add residues
            cnt = 0;
            // for( j = 0; j < resCnt; j++ ) {
            for (j = 0; j < 4; j++) {
                // resName
                line = bsEntries[i].substr(18 + 11 * cnt, 3);
                strTrimSpaces(line);
                if (line.empty())
                    break;
                this->bindingSiteResNames_[bsIdx].emplace_back(line);
                // chainID
                line = bsEntries[i].substr(22 + 11 * cnt, 1);
                chainId = line[0];
                // seq (res seq num)
                line = bsEntries[i].substr(23 + 11 * cnt, 4);
                strTrimSpaces(line);
                resId = static_cast<unsigned int>(std::stoi(line));
                // add binding site information
                this->bindingSites_[bsIdx].emplace_back(std::pair<char, unsigned int>(chainId, resId));
                cnt++;
            }
        }
        // get binding site descriptons and set colors
        this->bindingSiteDescription_.resize(this->bindingSiteNames_.size());
        this->bindingSiteColors_.resize(this->bindingSiteNames_.size());
        for (unsigned int i = 0; i < this->bindingSiteNames_.size(); i++) {
            this->bindingSiteDescription_[i] =
                this->ExtractBindingSiteDescripton(this->bindingSiteNames_[i], remarkEntries);
            this->bindingSiteColors_[i] = this->colorLookupTable_[i % this->colorLookupTable_.size()];
        }

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Bindings Site count: %i", bindingSiteNames_.size()); // DEBUG
    }
}


/*
 *
 */
std::string BindingSiteDataSource::ExtractBindingSiteDescripton(
    std::string bsName, std::vector<std::string> remarkArray) {
    std::string retStr("");
    for (unsigned int i = 0; i < remarkArray.size(); i++) {
        // search for binding site name
        if (strEndsWith(remarkArray[i], bsName)) {
            if ((i + 2) < remarkArray.size() && strStartsWith(remarkArray[i + 2], "SITE_DESCRIPTION:")) {
                retStr = remarkArray[i + 2].substr(17);
                strTrimSpaces(retStr);
                remarkArray.erase(remarkArray.begin() + i);
                remarkArray.erase(remarkArray.begin() + i);
                remarkArray.erase(remarkArray.begin() + i);
                return retStr;
            }
        }
    }
    return retStr;
}
