/*
 * CSVDataSource.cpp
 *
 * Copyright (C) 2015-2016 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "floattable/CSVDataSource.h"
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
#include <map>
#include <limits>
#include <omp.h>

using namespace megamol;
using namespace megamol::stdplugin;


datatools::floattable::CSVDataSource::CSVDataSource(void) : core::Module(),
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
    
    this->shuffleSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->shuffleSlot);

    this->getDataSlot.SetCallback(CallFloatTableData::ClassName(), "GetData", &CSVDataSource::getDataCallback);
    this->getDataSlot.SetCallback(CallFloatTableData::ClassName(), "GetHash", &CSVDataSource::getHashCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

}

datatools::floattable::CSVDataSource::~CSVDataSource(void) {
    this->Release();
}

bool datatools::floattable::CSVDataSource::create(void) {
    // nothing to do
    return true;
}

void datatools::floattable::CSVDataSource::release(void) {
    this->columns.clear();
    this->values.clear();
}

void datatools::floattable::CSVDataSource::assertData(void) {
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

        // 1. first, actually load the whole file into memory (FAST!)
        //////////////////////////////////////////////////////////////////////
        if (!file.LoadFile(this->filenameSlot.Param<core::param::FilePathParam>()->Value(), vislib::sys::ASCIIFileBuffer::PARSING_LINES)) throw vislib::Exception(__FILE__, __LINE__);
        if (file.Count() < 2) throw vislib::Exception("No data in csv file", __FILE__, __LINE__);

        // 2. Now (autodetect) the column separator
        //////////////////////////////////////////////////////////////////////
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

        // 3. (Auto-detect) float style
        //////////////////////////////////////////////////////////////////////
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

        // 4. Column headers (and optionally type)
        //////////////////////////////////////////////////////////////////////
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
        bool hasCatDims = false;

        if (readTypeLineSlot.Param<core::param::BoolParam>()->Value()) {
            vislib::Array<vislib::StringA> tokens(vislib::StringTokeniserA::Split(file[readNameLineSlot.Param<core::param::BoolParam>()->Value() ? 1 : 0], colSep, false));
            for (SIZE_T i = 0; i < dimNames.Count(); i++) {
                CallFloatTableData::ColumnType T = CallFloatTableData::ColumnType::QUANTITATIVE;
                try {
                    if (tokens.Count() > i) {
                        if (tokens[i].Equals("CATEGORICAL", true)) {
                            T = CallFloatTableData::ColumnType::CATEGORICAL;
                            hasCatDims = true;
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
                    .SetType(CallFloatTableData::ColumnType::QUANTITATIVE)
                    .SetMinimumValue(0.0f)
                    .SetMaximumValue(1.0f);
                //this->values[i].AssertCapacity(this->values[i].Count() + file.Count() - 1);
            }
        }

        // 5. format is now clear ... start parsing the actual data!
        //////////////////////////////////////////////////////////////////////
        // lets assume all lines will work and parser in parallel
        size_t colCnt = static_cast<size_t>(this->columns.size());
        size_t rowCnt = static_cast<size_t>(file.Count() - firstDatRow);
        int colSepEnd = colSep.Length() - 1;

        // check for empty lines at the end
        for (; rowCnt > 0; --rowCnt) {
            const char *start = file[firstDatRow + rowCnt - 1];
            const char *end = start;
            size_t col = 0;
            while ((*end != '\0') && (col < colCnt)) {
                int colSepPos = 0;
                while ((*end != '\0') && ((*end != colSep[colSepEnd]) || (colSepEnd != colSepPos))) {
                    if (*end == colSep[colSepPos]) colSepPos++; else colSepPos = 0;
                    ++end;
                }
                col++;
            }
            if (col >= colCnt) break; // we found the last line containing a full data set
        }

        std::vector<std::map<std::string, float>> catMaps;
        int thCnt = omp_get_max_threads();
        catMaps.resize(colCnt * thCnt);

        values.resize(colCnt * rowCnt);
        bool hasInvalids = false;
#pragma omp parallel for
        for (long long idx = 0; idx < static_cast<long long>(rowCnt); ++idx) {
            int thId = omp_get_thread_num();
            const char *start = file[static_cast<size_t>(firstDatRow + idx)];
            const char *end = start;
            size_t col = 0;
            while ((*end != '\0') && (col < colCnt)) {
                std::map<std::string, float> &catMap = catMaps[thId + col * thCnt];
                int colSepPos = 0;
                while ((*end != '\0') && ((*end != colSep[colSepEnd]) || (colSepEnd != colSepPos))) {
                    if (*end == colSep[colSepPos]) colSepPos++; else colSepPos = 0;
                    ++end;
                }

                char *ez = const_cast<char*>(end - colSepEnd);
                ez = '\0';

                if (this->columns[col].Type() == CallFloatTableData::ColumnType::QUANTITATIVE) {
                    if (DEdouble) for (ez = const_cast<char*>(start); ez != end; ++ez) if (*ez == ',') *ez = '.';
                    try {
                        values[static_cast<size_t>(idx * colCnt + col)] = static_cast<float>(vislib::CharTraitsA::ParseDouble(start));
                    } catch(...) {
                        values[static_cast<size_t>(idx * colCnt + col)] = std::numeric_limits<float>::quiet_NaN();
                        hasInvalids = true;
                    }
                }
                else if (this->columns[col].Type() == CallFloatTableData::ColumnType::CATEGORICAL) {
                    assert(hasCatDims);
                    std::map<std::string, float>::iterator cmi = catMap.find(start);
                    if (cmi == catMap.end()) {
                        cmi = catMap.insert(std::pair<std::string, float>(start, static_cast<float>(thId + thCnt * catMap.size()))).first;
                    }
                    values[static_cast<size_t>(idx * colCnt + col)] = cmi->second;
                } else {
                    assert(false);
                }

                col++;
                if (*end != '\0') {
                    start = end + 1;
                    end = start;
                }
            }
            for (;col < colCnt; ++col) {
                values[static_cast<size_t>(idx * colCnt + col)] = std::numeric_limits<float>::quiet_NaN();
                hasInvalids = true;
            }
        }

        // remove invalid data if present
        if (hasInvalids) {
            this->GetCoreInstance()->Log().WriteWarn("CSV file contains invalid field:");
            size_t tr = 0;
            for (size_t r = 0; r < rowCnt; ++r) {
                bool rowInvalid = false;
                for (size_t c = 0; c < colCnt; ++c) {
                    float f = values[r * colCnt + c];
                    if (f != f) {
                        this->GetCoreInstance()->Log().WriteWarn("\tline %d, column %d", 1 + firstDatRow + r, 1 + c);
                        rowInvalid = true;
                        break;
                    }
                    values[tr * colCnt + c] = f;
                }
                if (!rowInvalid) tr++;
            }
            rowCnt = tr;
        }

        // merge category data
        if (hasCatDims) {
            for (size_t c = 0; c < colCnt; ++c) {
                if (columns[c].Type() != CallFloatTableData::ColumnType::CATEGORICAL) continue;
                std::map<int, int> catRemap;
                std::map<std::string, int> catMap;
                for (int ci = static_cast<int>(c) * thCnt; ci < static_cast<int>(c + 1) * thCnt; ++ci) {
                    for (const std::pair<std::string, float>& p : catMaps[ci]) {
                        int vi = static_cast<int>(p.second + 0.49f);
                        std::map<std::string, int>::iterator cmi = catMap.find(p.first);
                        if (cmi == catMap.end()) {
                            int nv = static_cast<int>(catMap.size());
                            catMap[p.first] = nv;
                            catRemap[vi] = nv;
                        } else {
                            catRemap[vi] = cmi->second;
                        }
                    }
                }

                for (size_t r = 0; r < rowCnt; ++r) {
                    int vi = static_cast<int>(values[r * colCnt + c] + 0.49f);
                    values[r * colCnt + c] = static_cast<float>(catRemap[vi]);
                }
            }
        }

        // collect minMax
        std::vector<float> minVals(colCnt, (std::numeric_limits<float>::max)());
        std::vector<float> maxVals(colCnt, -(std::numeric_limits<float>::max)());
        for (size_t r = 0; r < rowCnt; ++r) {
            for (size_t c = 0; c < colCnt; ++c) {
                float f = values[r * colCnt + c];
                if (f < minVals[c]) minVals[c] = f;
                if (f > maxVals[c]) maxVals[c] = f;
            }
        }
        for (size_t c = 0; c < colCnt; ++c) {
            columns[c].SetMinimumValue(minVals[c]).SetMaximumValue(maxVals[c]);
        }

        // n. all done!
        //////////////////////////////////////////////////////////////////////
        this->GetCoreInstance()->Log().WriteInfo("Highdimensional data loaded: %u dimensions; %u samples\n",
            static_cast<unsigned int>(colCnt), static_cast<unsigned int>(rowCnt));

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


void datatools::floattable::CSVDataSource::shuffleData()
{
    // Shuffle if neccessary
    if (!this->shuffleSlot.Param<core::param::BoolParam>()->Value())
        return;

    std::default_random_engine eng(static_cast<unsigned int>(dataHash));
    size_t numCols = columns.size();
    size_t numRows = values.size() / numCols;
    std::uniform_int_distribution<size_t> dist(0, numRows - 1);
    for (size_t i = 0; i < numRows; ++i) {
        size_t idx2 = dist(eng);
        for (size_t j = 0; j < numCols; ++j)
            std::swap(values[j + i * numCols], values[j + idx2 * numCols]);
    }
}




bool datatools::floattable::CSVDataSource::getDataCallback(core::Call& caller) {
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

bool datatools::floattable::CSVDataSource::getHashCallback(core::Call& caller) {
    CallFloatTableData *tfd = dynamic_cast<CallFloatTableData*>(&caller);
    if (tfd == nullptr) return false;

    this->assertData();

    tfd->SetDataHash(this->dataHash);
    tfd->SetUnlocker(nullptr);

    return true;
}

bool datatools::floattable::CSVDataSource::clearData(core::param::ParamSlot& caller) {
    this->columns.clear();
    this->values.clear();

    return true;
}
