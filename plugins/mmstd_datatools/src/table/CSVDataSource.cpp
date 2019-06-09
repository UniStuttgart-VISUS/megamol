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
#include "mmcore/param/IntParam.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "mmcore/CoreInstance.h"
#include "vislib/StringTokeniser.h"
#include <sstream>
#include <vector>
#include <list>
#include <random>
#include <map>
#include <limits>
#include <omp.h>

using namespace megamol;
using namespace megamol::stdplugin;

enum class DecimalSeparator : int {
    Unknown = 0,
    US = 1,
    DE = 2
};

template <char C>
std::istream& expect(std::istream& is) {
    if ((is >> std::ws).peek() == C) {
        is.ignore();
    } else {
        is.setstate(std::ios_base::failbit);
    }
    return is;
}

double parseValue(const char* tokenStart, const char* tokenEnd) {
    std::string token(tokenStart, tokenEnd - tokenStart);

    std::istringstream iss(token);
    iss.imbue(std::locale::classic());
    if (iss.rdbuf()->in_avail() == 0) {
        return NAN;
    }

    // Try to parse as number.
    double number;
    iss >> number;
    if (!iss.fail() && iss.eof()) {
        return number;
    }

    // Reset stream.
    iss.str(token);

    // Timestamp fractions (to be converted to milliseconds).
    unsigned short fractions[4];

    // Try to parse as timestamp (HH:mm:ss)
    iss >> fractions[0] >> expect<':'> >> fractions[1] >> expect<':'> >> fractions[2];
    if (!iss.fail() && iss.eof()) {
        return  fractions[0] * (60 * 60 * 1000) + fractions[1] * (60 * 1000) + fractions[2] * 1000;
    }

    // Try to parse as timestamp (HH:mm:ss.SSS)
    iss >> expect<'.'> >> fractions[3];
    if (!iss.fail() && iss.eof()) {
        return fractions[0] * (60 * 60 * 1000) + fractions[1] * (60 * 1000) + fractions[2] * 1000 + fractions[3];
    }

    // Bail out.
    return NAN;
}

datatools::floattable::CSVDataSource::CSVDataSource(void) : core::Module(),
filenameSlot("filename", "Filename to read from"),
skipPrefaceSlot("skipPreface", "Number of lines to skip before parsing"),
headerNamesSlot("headerNames", "Interpret the first data row as column names"),
headerTypesSlot("headerTypes", "Interpret the second data row as column types (quantitative or categorical)"),
commentPrefixSlot("commentPrefix", "Prefix that indicates a line-comment"),
clearSlot("clear", "Clears the data"),
colSepSlot("colSep", "The column separator (detected if empty)"),
decSepSlot("decSep", "The decimal point parser format type"),
shuffleSlot("shuffle", "Shuffle data points"),
getDataSlot("getData", "Slot providing the data"),
dataHash(0), columns(), values() {
    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->skipPrefaceSlot.SetParameter(new core::param::IntParam(0));
    this->MakeSlotAvailable(&this->skipPrefaceSlot);

    this->headerNamesSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->headerNamesSlot);

    this->headerTypesSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->headerTypesSlot);

    this->commentPrefixSlot.SetParameter(new core::param::StringParam(""));
    this->MakeSlotAvailable(&this->commentPrefixSlot);

    this->clearSlot << new core::param::ButtonParam();
    this->clearSlot.SetUpdateCallback(&CSVDataSource::clearData);
    this->MakeSlotAvailable(&this->clearSlot);

    this->colSepSlot << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->colSepSlot);

    core::param::EnumParam *ep = new core::param::EnumParam(0);
    ep->SetTypePair(static_cast<int>(DecimalSeparator::Unknown), "Auto");
    ep->SetTypePair(static_cast<int>(DecimalSeparator::US), "US (3.141)");
    ep->SetTypePair(static_cast<int>(DecimalSeparator::DE), "DE (3,141)");
    this->decSepSlot << ep;
    this->MakeSlotAvailable(&this->decSepSlot);

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
        && !this->skipPrefaceSlot.IsDirty()
        && !this->headerNamesSlot.IsDirty()
        && !this->headerTypesSlot.IsDirty()
        && !this->commentPrefixSlot.IsDirty()
        && !this->colSepSlot.IsDirty()
        && !this->decSepSlot.IsDirty()) {
        if (this->shuffleSlot.IsDirty()) {
            shuffleData();
            this->shuffleSlot.ResetDirty();
            this->dataHash++;
        }
        return; // nothing to do
    }

    this->filenameSlot.ResetDirty();
    this->skipPrefaceSlot.ResetDirty();
    this->headerNamesSlot.ResetDirty();
    this->headerTypesSlot.ResetDirty();
    this->commentPrefixSlot.ResetDirty();
    this->colSepSlot.ResetDirty();
    this->decSepSlot.ResetDirty();
    this->shuffleSlot.ResetDirty();

    this->columns.clear();
    this->values.clear();

	auto filename = this->filenameSlot.Param<core::param::FilePathParam>()->Value();

    try {
        vislib::sys::ASCIIFileBuffer file;

        // 1. Load the whole file into memory (FAST!)
        //////////////////////////////////////////////////////////////////////
        if (!file.LoadFile(filename, vislib::sys::ASCIIFileBuffer::PARSING_LINES)) throw vislib::Exception(__FILE__, __LINE__);
        if (file.Count() < 2) throw vislib::Exception("No data in CSV file", __FILE__, __LINE__);

        // 2. Determine the first row, column separator, and decimal point
        //////////////////////////////////////////////////////////////////////
        int firstHeaRow = this->skipPrefaceSlot.Param<core::param::IntParam>()->Value();
        int firstDatRow = this->skipPrefaceSlot.Param<core::param::IntParam>()->Value();
        if (headerNamesSlot.Param<core::param::BoolParam>()->Value()) firstDatRow++;
        if (headerTypesSlot.Param<core::param::BoolParam>()->Value()) firstDatRow++;

        auto comment = this->commentPrefixSlot.Param<core::param::StringParam>()->Value();
        if (!comment.IsEmpty()) {
                // Skip comments at the beginning of the file.
            while (firstHeaRow < file.Count()) {
                if (!vislib::StringA(file[firstHeaRow]).StartsWith(comment)) {
                    break;
                }
                firstHeaRow++;
                firstDatRow++;
            }
        }

        vislib::StringA colSep(this->colSepSlot.Param<core::param::StringParam>()->Value());
        if (colSep.IsEmpty()) {
            // Detect column separator
            const char ColSepCanidates[] = { '\t', ';', ',', '|' };
            vislib::StringA l1(file[firstHeaRow]);
            vislib::StringA l2(file[firstHeaRow]);
            for (int i = 0; i < sizeof(ColSepCanidates) / sizeof(char); ++i) {
                SIZE_T c1 = l1.Count(ColSepCanidates[i]);
                if ((c1 > 0) && (c1 == l2.Count(ColSepCanidates[i]))) {
                    colSep.Append(ColSepCanidates[i]);
                    break;
                }
            }
            if (colSep.IsEmpty()) {
                throw vislib::Exception("Failed to detect column separator", __FILE__, __LINE__);
            }
        }

        DecimalSeparator decType = static_cast<DecimalSeparator>(this->decSepSlot.Param<core::param::EnumParam>()->Value());
        if (decType == DecimalSeparator::Unknown) {
            // Detect decimal type
            vislib::Array<vislib::StringA> tokens(vislib::StringTokeniserA::Split(file[firstDatRow], colSep, false));
            for (SIZE_T i = 0; i < tokens.Count(); i++) {
                bool hasDot = tokens[i].Contains('.');
                bool hasComma = tokens[i].Contains(',');
                if (hasDot && !hasComma) {
                    decType = DecimalSeparator::US;
                    break;
                } else if (hasComma && !hasDot) {
                    decType = DecimalSeparator::DE;
                    break;
                }
            }
            if (decType == DecimalSeparator::Unknown) {
                                // Assume US format if detection failed.
                decType = DecimalSeparator::US;
            }
        }

        // 3. Table layout is now clear... determine column headers.
        //////////////////////////////////////////////////////////////////////
        vislib::Array<vislib::StringA> dimNames;
        if (headerNamesSlot.Param<core::param::BoolParam>()->Value()) {
            dimNames = vislib::StringTokeniserA::Split(file[firstHeaRow], colSep, false);
            firstHeaRow++;
        } else {
            dimNames = vislib::StringTokeniserA::Split(file[firstHeaRow], colSep, false);
            for (SIZE_T i = 0; i < dimNames.Count(); ++i) {
                dimNames[i].Format("Dim %d", static_cast<int>(i));
            }
        }
        this->columns.resize(dimNames.Count());
        this->values.clear();

        bool hasCatDims = false;
        if (headerTypesSlot.Param<core::param::BoolParam>()->Value()) {
            vislib::Array<vislib::StringA> tokens(vislib::StringTokeniserA::Split(file[firstHeaRow], colSep, false));
            for (SIZE_T i = 0; i < dimNames.Count(); i++) {
                CallFloatTableData::ColumnType type = CallFloatTableData::ColumnType::QUANTITATIVE;
                if (tokens.Count() > i && tokens[i].Equals("CATEGORICAL", true)) {
                    type = CallFloatTableData::ColumnType::CATEGORICAL;
                    hasCatDims = true;
                }
                this->columns[i].SetName(dimNames[i].PeekBuffer())
                    .SetType(type)
                    .SetMinimumValue(0.0f)
                    .SetMaximumValue(1.0f);
            }
        } else {
            for (SIZE_T i = 0; i < dimNames.Count(); i++) {
                this->columns[i].SetName(dimNames[i].PeekBuffer())
                    .SetType(CallFloatTableData::ColumnType::QUANTITATIVE)
                    .SetMinimumValue(0.0f)
                    .SetMaximumValue(1.0f);
            }
        }

        // 4. Data format is now clear... finally parse actual data
        //////////////////////////////////////////////////////////////////////
        size_t colCnt = static_cast<size_t>(this->columns.size());
        size_t rowCnt = static_cast<size_t>(file.Count() - firstDatRow);
        int colSepEnd = colSep.Length() - 1;

        // Test for empty lines at the end
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

        // Parse in parallel, assuming all lines will work
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

                if (this->columns[col].Type() == CallFloatTableData::ColumnType::QUANTITATIVE) {
                    if (decType == DecimalSeparator::DE) {
                        for (char *ez = const_cast<char*>(start); ez != end; ++ez) if (*ez == ',') *ez = '.';
                    }
                    double value = parseValue(start, end);
                    values[static_cast<size_t>(idx * colCnt + col)] = static_cast<float>(value);
                    if (std::isnan(value)) {
                        hasInvalids = true;
                    }
                } else if (this->columns[col].Type() == CallFloatTableData::ColumnType::CATEGORICAL) {
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
            for (; col < colCnt; ++col) {
                values[static_cast<size_t>(idx * colCnt + col)] = std::numeric_limits<float>::quiet_NaN();
                hasInvalids = true;
            }
        }

        // Report invalid data if present (note: do not drop data!)
        if (hasInvalids) {
            this->GetCoreInstance()->Log().WriteWarn("CSV file contains invalid data:");
            for (size_t c = 0; c < colCnt; ++c) {
                std::stringstream ss;
                bool invalidColumn = true;
                for (size_t r = 0; r < rowCnt; ++r) {
                    float value = values[r * colCnt + c];
                    if (std::isnan(value)) {
                        size_t line = 1 + firstDatRow + r;
                        ss << line << " ";
                    } else {
                        invalidColumn = false;
                    }
                }
                std::string lines = ss.str();
                if (invalidColumn) {
                    this->GetCoreInstance()->Log().WriteWarn("  lines in column %d: all", 1 + c);
                } else if (!lines.empty()) {
                    this->GetCoreInstance()->Log().WriteWarn("  lines in column %d: %s", 1 + c, lines.c_str());
                }
            }
        }

        // Merge categorical data so that all `value indices` map to one `string key`
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

        // Collect min/max
        std::vector<float> minVals(colCnt, std::numeric_limits<float>::max());
        std::vector<float> maxVals(colCnt, -std::numeric_limits<float>::max());
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

        // 5. All done... report summary
        //////////////////////////////////////////////////////////////////////
        this->GetCoreInstance()->Log().WriteInfo("Tabular data loaded: %u dimensions; %u samples\n",
            static_cast<unsigned int>(colCnt), static_cast<unsigned int>(rowCnt));

    } catch (const vislib::Exception& ex) {
        this->GetCoreInstance()->Log().WriteError("Could not load \"%s\": %s [%s, %d]", filename.PeekBuffer(), ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        this->columns.clear();
        this->values.clear();
    } catch (...) {
        this->columns.clear();
        this->values.clear();
    }

    shuffleData();

    this->dataHash++;
}

void datatools::floattable::CSVDataSource::shuffleData() {
    if (!this->shuffleSlot.Param<core::param::BoolParam>()->Value()) {
                // Do not shuffle, unless requested
        return;
    }

    std::default_random_engine eng(static_cast<unsigned int>(dataHash));
    size_t numCols = columns.size();
    size_t numRows = values.size() / numCols;
    std::uniform_int_distribution<size_t> dist(0, numRows - 1);
    for (size_t i = 0; i < numRows; ++i) {
        size_t idx2 = dist(eng);
        for (size_t j = 0; j < numCols; ++j) {
            std::swap(values[j + i * numCols], values[j + idx2 * numCols]);
        }
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

    tfd->SetFrameCount(1);
    tfd->SetFrameID(0);
    return true;
}

bool datatools::floattable::CSVDataSource::clearData(core::param::ParamSlot& caller) {
    this->columns.clear();
    this->values.clear();

    return true;
}
