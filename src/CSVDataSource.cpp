/*
 * CSVDataSource.cpp
 *
 * Copyright (C) 2015 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CSVDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/BoolParam.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "mmcore/CoreInstance.h"
#include "vislib/StringTokeniser.h"
#include <vector>
#include <list>
#include <random>

using namespace megamol;
using namespace megamol::stdplugin;


datatools::CSVDataSource::CSVDataSource(void) : core::Module(),
        filenameSlot("filename", "The file name"),
        readNameLineSlot("readNameLine", "The first row of the data set stores the names of the columns"),
        readTypeLineSlot("readTypeLine", "The second row of the data set stores the data types of the columns"),
        clearSlot("clear", "Clears the data"),
        colSepSlot("colSep", "The column separator (Empty for autodetection)"),
        decTypeSlot("decType", "The decimal point parser format type"),
        shuffleSlot("shuffle", "Shuffle data points"),
        getDataSlot("getData", "Slot providing the data"),
        dataHash(0), columns(), values() {

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->readNameLineSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->readNameLineSlot);

    this->readTypeLineSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->readTypeLineSlot);

    this->clearSlot << new core::param::ButtonParam();
    this->clearSlot.SetUpdateCallback(&CSVDataSource::clearData);
    this->MakeSlotAvailable(&this->clearSlot);

    this->colSepSlot << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->colSepSlot);

    core::param::EnumParam *ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "Autodetect");
    ep->SetTypePair(1, "US (3.141)");
    ep->SetTypePair(2, "DE (3,141)");
    this->decTypeSlot << ep;
    this->MakeSlotAvailable(&this->decTypeSlot);
    
    this->shuffleSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->shuffleSlot);

    this->getDataSlot.SetCallback(CallFloatTableData::ClassName(), "GetData", &CSVDataSource::getDataCallback);
    this->getDataSlot.SetCallback(CallFloatTableData::ClassName(), "GetHash", &CSVDataSource::getHashCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

}

datatools::CSVDataSource::~CSVDataSource(void) {
    this->Release();
}

bool datatools::CSVDataSource::create(void) {
    // nothing to do
    return true;
}

void datatools::CSVDataSource::release(void) {
    this->columns.clear();
    this->values.clear();
}

void datatools::CSVDataSource::assertData(void) {
    if (!this->filenameSlot.IsDirty()
        && !this->readNameLineSlot.IsDirty()
        && !this->readTypeLineSlot.IsDirty()
        && !this->colSepSlot.IsDirty()
        && !this->decTypeSlot.IsDirty()) 
    {
        if (this->shuffleSlot.IsDirty()) {
            shuffleData();
            this->shuffleSlot.ResetDirty();
            this->dataHash++;
        }
        return; // nothing to do
    }
    
    this->filenameSlot.ResetDirty();
    this->readNameLineSlot.ResetDirty();
    this->readTypeLineSlot.ResetDirty();
    this->colSepSlot.ResetDirty();
    this->decTypeSlot.ResetDirty();
    this->shuffleSlot.ResetDirty();

    this->columns.clear();
    this->values.clear();

    try {
        vislib::sys::ASCIIFileBuffer file;
        if (!file.LoadFile(this->filenameSlot.Param<core::param::FilePathParam>()->Value(), vislib::sys::ASCIIFileBuffer::PARSING_LINES)) throw vislib::Exception(__FILE__, __LINE__);
        if (file.Count() < 2) throw vislib::Exception("No data in csv file", __FILE__, __LINE__);

        vislib::StringA colSep(this->colSepSlot.Param<core::param::StringParam>()->Value());

        int decType = this->decTypeSlot.Param<core::param::EnumParam>()->Value();

        if (colSep.IsEmpty()) {
            // autodetecting column-separator
            vislib::StringA l1(file[0]);
            vislib::StringA l2(file[1]);

            SIZE_T c1 = l1.Count(';');
            if ((c1 > 0) && (c1 == l2.Count(';'))) {
                colSep = ";";
            } else {
                c1 = l1.Count('\t');
                if ((c1 > 0) && (c1 == l2.Count('\t'))) {
                    colSep = "\t";
                } else {
                    c1 = l1.Count(',');
                    if ((c1 > 0) && (c1 == l2.Count(','))) {
                        colSep = ",";
                    } else {
                        c1 = l1.Count('|');
                        if ((c1 > 0) && (c1 == l2.Count('|'))) {
                            colSep = "|";
                        } else {
                            throw vislib::Exception("Failed to autodetect column separator", __FILE__, __LINE__);
                        }
                    }
                }
            }
        }

        int firstDatRow = 0;
        if (readNameLineSlot.Param<core::param::BoolParam>()->Value()) firstDatRow++;
        if (readTypeLineSlot.Param<core::param::BoolParam>()->Value()) firstDatRow++;

        if (decType == 0) {
            // autodetect decimal type
            vislib::Array<vislib::StringA> tokens(vislib::StringTokeniserA::Split(file[firstDatRow], colSep, false));
            for (SIZE_T i = 0; i < tokens.Count(); i++) {
                bool hasDot = tokens[i].Contains('.');
                bool hasComma = tokens[i].Contains(',');
                if (hasDot && !hasComma) {
                    decType = 1; // US
                    break;
                } else if (hasComma && !hasDot) {
                    decType = 2; // DE
                    break;
                }
                // not clear ... continue with next column
            }
            if (decType == 0) {
                decType = 1; // Could not autodetect decimal format type. Assume US format ...
                // throw new vislib::Exception("Failed to autodetect decimal format type", __FILE__, __LINE__);
            }
        }
        bool DEdouble = (decType == 2);

        // format is now clear ... start parsing the actual data!
        vislib::Array<vislib::StringA> dimNames;
        if (readNameLineSlot.Param<core::param::BoolParam>()->Value()) {
            dimNames = vislib::StringTokeniserA::Split(file[0], colSep, false);
        } else {
            dimNames = vislib::StringTokeniserA::Split(file[0], colSep, false);
            for (SIZE_T i = 0; i < dimNames.Count(); ++i) {
                dimNames[i].Format("Dim %d", static_cast<int>(i));
            }
        }
        this->columns.resize(dimNames.Count());
        this->values.clear();

        std::vector<std::list<vislib::StringA> > categories(dimNames.Count());

        if (readTypeLineSlot.Param<core::param::BoolParam>()->Value()) {
            vislib::Array<vislib::StringA> tokens(vislib::StringTokeniserA::Split(file[readNameLineSlot.Param<core::param::BoolParam>()->Value() ? 1 : 0], colSep, false));
            for (SIZE_T i = 0; i < dimNames.Count(); i++) {
				datatools::CallFloatTableData::ColumnType T = datatools::CallFloatTableData::ColumnType::QUANTITATIVE;
                try {
                    if (tokens.Count() > i) {
                        if (tokens[i].Equals("CATEGORICAL", true)) {
							T = datatools::CallFloatTableData::ColumnType::CATEGORICAL;
                        }
                    }
                } catch(...) {}
                this->columns[i].SetName(dimNames[i].PeekBuffer())
                    .SetType(T)
                    .SetMinimumValue(0.0f)
                    .SetMaximumValue(1.0f);
            }
        } else {
            for (SIZE_T i = 0; i < dimNames.Count(); i++) {
                this->columns[i].SetName(dimNames[i].PeekBuffer())
					.SetType(datatools::CallFloatTableData::ColumnType::QUANTITATIVE)
                    .SetMinimumValue(0.0f)
                    .SetMaximumValue(1.0f);
                //this->values[i].AssertCapacity(this->values[i].Count() + file.Count() - 1);
            }
        }

        std::vector<float> lv;
        lv.resize(this->columns.size());
        for (SIZE_T ti = 0; ti < file.Count() - firstDatRow; ti++) {
            vislib::Array<vislib::StringA> tokens(vislib::StringTokeniserA::Split(file[firstDatRow + ti], colSep, false));
            if (tokens.Count() != this->columns.size()) continue; // ignore truncated lines
            for (SIZE_T di = 0; di < this->columns.size(); di++) {
                vislib::StringA s = tokens[di];

				if (this->columns[di].Type() == datatools::CallFloatTableData::ColumnType::QUANTITATIVE) {
                    if (DEdouble) s.Replace(',', '.');
                    try {
                        lv[di] = static_cast<float>(vislib::CharTraitsA::ParseDouble(s));
                    } catch(...) {
                        throw;
                    }

				}
				else if (this->columns[di].Type() == datatools::CallFloatTableData::ColumnType::CATEGORICAL) {
                    // s.ToLowerCase(); // not sure about that.
                    auto x = std::find(categories[di].begin(), categories[di].end(), s);
                    if (x == categories[di].end()) {
                        lv[di] = static_cast<float>(categories[di].size());
                        categories[di].push_back(s);
                    } else {
                        lv[di] = static_cast<float>(std::distance(categories[di].begin(), x));
                    }
                } else {
                    assert(false);
                }
            }

            if (this->values.size() == 0) {
                for (SIZE_T di = 0; di < this->columns.size(); di++) {
                    this->columns[di].SetMinimumValue(lv[di])
                        .SetMaximumValue(lv[di]);
                }
            } else {
                for (SIZE_T di = 0; di < this->columns.size(); di++) {
                    this->columns[di].SetMinimumValue(std::min<float>(this->columns[di].MinimumValue(), lv[di]));
                    this->columns[di].SetMaximumValue(std::max<float>(this->columns[di].MaximumValue(), lv[di]));
                }
            }
            this->values.resize(this->values.size() + this->columns.size());
            ::memcpy(this->values.data() + this->values.size() - this->columns.size(), lv.data(), lv.size() * sizeof(float));
        }

        this->GetCoreInstance()->Log().WriteInfo("Highdimensional data loaded: %u dimensions; %u samples\n",
            static_cast<unsigned int>(this->columns.size()), static_cast<unsigned int>(file.Count()));

    } catch(const vislib::Exception& ex) {
        this->GetCoreInstance()->Log().WriteError("Could not load CSV: %s [%s, &d]", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        this->columns.clear();
        this->values.clear();
    } catch(...) {
        this->columns.clear();
        this->values.clear();
    }

    shuffleData();
    
    this->dataHash++;
}


void datatools::CSVDataSource::shuffleData()
{
    // Shuffle if neccessary
    if (!this->shuffleSlot.Param<core::param::BoolParam>()->Value())
        return;

    std::default_random_engine eng(static_cast<unsigned int>(dataHash));
    size_t numCols = columns.size();
    size_t numRows = values.size() / numCols;
	std::uniform_int_distribution<size_t> dist(0, numRows);
	for (size_t i = 0; i < numRows; ++i) {
		size_t idx2 = dist(eng);
		for (size_t j = 0; j < numCols; ++j)
            std::swap(values[j + i * numCols], values[j + idx2 * numCols]);
    }
}




bool datatools::CSVDataSource::getDataCallback(core::Call& caller) {
    CallFloatTableData *tfd = dynamic_cast<CallFloatTableData*>(&caller);
    if (tfd == nullptr) return false;

    this->assertData();

    tfd->SetDataHash(this->dataHash);
    if (values.size() == 0) {
        tfd->Set(0, 0, nullptr, nullptr);
    } else {
        assert((values.size() % columns.size()) == 0);
        tfd->Set(columns.size(), values.size() / columns.size(), columns.data(), values.data());
    }
    tfd->SetUnlocker(nullptr);

    return true;
}

bool datatools::CSVDataSource::getHashCallback(core::Call& caller) {
    CallFloatTableData *tfd = dynamic_cast<CallFloatTableData*>(&caller);
    if (tfd == nullptr) return false;

    this->assertData();

    tfd->SetDataHash(this->dataHash);
    tfd->SetUnlocker(nullptr);

    return true;
}

bool datatools::CSVDataSource::clearData(core::param::ParamSlot& caller) {
    this->columns.clear();
    this->values.clear();

    return true;
}
