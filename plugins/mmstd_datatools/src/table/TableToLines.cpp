#include "stdafx.h"
#include "TableToLines.h"

#include "mmcore/param/StringParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "vislib/sys/Log.h"
#include "vislib/sys/PerformanceCounter.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <nanoflann.hpp>

using namespace megamol::stdplugin::datatools;
using namespace megamol;


/*
 * THIS IS THE APEX OF SHIT and a non-quality copy from nanoflann/examples/utils.h
 * TODO: Replace it with a proper adapter instead of creating a copy to index data!
 */
template <typename T>
struct PointCloud
{
	struct Point
	{
		T  x,y,z;
	};

	std::vector<Point>  pts;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline T kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		if (dim == 0) return pts[idx].x;
		else if (dim == 1) return pts[idx].y;
		else return pts[idx].z;
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

/*
 * FloattableToLines::FloattableToLines
 */
TableToLines::TableToLines(void) : Module(),
        slotTF("gettransferfunction", "Connects to the transfer function module"),
        slotDeployData("linedata", "Provides the data as line data call."),
        slotCallTable("table", "table input call"),
        slotColumnR("redcolumnname", "The name of the column holding the red colour channel value."),
        slotColumnG("greencolumnname", "The name of the column holding the green colour channel value."),
        slotColumnB("bluecolumnname", "The name of the column holding the blue colour channel value."),
        slotColumnI("intensitycolumnname", "The name of the column holding the intensity colour channel value."),
        slotGlobalColor("color", "Constant sphere color."),
        slotColumnIndex("indexcolumnname", "The name of the column holding the index data."),
        slotConnectionType("connectiontype", "Type of the line connection."),
        slotColorMode("colormode", "Pass on color as RGB or intensity"),
        slotColumnX("xcolumnname", "The name of the column holding the x-coordinate."),
        slotColumnY("ycolumnname", "The name of the column holding the y-coordinate."),
        slotColumnZ("zcolumnname", "The name of the column holding the z-coordinate."),
        inputHash(0), myHash(0), columnIndex() {

    /* Register parameters. */
    core::param::FlexEnumParam *rColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnR << rColumnEp;
    this->MakeSlotAvailable(&this->slotColumnR);

    core::param::FlexEnumParam *gColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnG << gColumnEp;
    this->MakeSlotAvailable(&this->slotColumnG);

    core::param::FlexEnumParam *bColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnB << bColumnEp;
    this->MakeSlotAvailable(&this->slotColumnB);

    core::param::FlexEnumParam *iColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnI << iColumnEp;
    this->MakeSlotAvailable(&this->slotColumnI);

    this->slotGlobalColor << new megamol::core::param::StringParam(_T("white"));
    this->MakeSlotAvailable(&this->slotGlobalColor);

    core::param::EnumParam *ep = new core::param::EnumParam(2);
    ep->SetTypePair(0, "RGB");
    ep->SetTypePair(1, "Intensity");
    ep->SetTypePair(2, "global RGB");
    this->slotColorMode << ep;
    this->MakeSlotAvailable(&this->slotColorMode);

    core::param::EnumParam *ct = new core::param::EnumParam(1);
    ct->SetTypePair(0, "Grid");
    ct->SetTypePair(1, "Index");
    this->slotConnectionType << ct;
    this->MakeSlotAvailable(&this->slotConnectionType);

    core::param::FlexEnumParam *indexep= new core::param::FlexEnumParam("undef");
    this->slotColumnIndex << indexep;
    this->MakeSlotAvailable(&this->slotColumnIndex);

    core::param::FlexEnumParam *xColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnX << xColumnEp;
    this->MakeSlotAvailable(&this->slotColumnX);

    core::param::FlexEnumParam *yColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnY << yColumnEp;
    this->MakeSlotAvailable(&this->slotColumnY);

    core::param::FlexEnumParam *zColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnZ << zColumnEp;
    this->MakeSlotAvailable(&this->slotColumnZ);

    /* Register calls. */
    this->slotTF.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->slotTF);

    this->slotDeployData.SetCallback(
        geocalls::LinesDataCall::ClassName(),
        "GetData",
        &TableToLines::getLineData);
    this->slotDeployData.SetCallback(
        geocalls::LinesDataCall::ClassName(),
        "GetExtent",
        &TableToLines::getLineDataExtent);
    this->MakeSlotAvailable(&this->slotDeployData);

    this->slotCallTable.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->slotCallTable);
}


/*
 * FloattableToLines::~FloattableToLines
 */
TableToLines::~TableToLines(void) {
    this->Release();
}


/*
 * megamol::pcl::PclDataSource::create
 */
bool TableToLines::create(void) {
    bool retval = true;
    return true;
}

bool TableToLines::anythingDirty() {
    return this->slotColumnR.IsDirty()
        || this->slotColumnG.IsDirty()
        || this->slotColumnB.IsDirty()
        || this->slotColumnI.IsDirty()
        || this->slotGlobalColor.IsDirty()
        || this->slotColorMode.IsDirty()
        || this->slotColumnX.IsDirty()
        || this->slotColumnY.IsDirty()
        || this->slotColumnZ.IsDirty()
        || this->slotColumnIndex.IsDirty()
        || this->slotConnectionType.IsDirty();
}

void TableToLines::resetAllDirty() {
    this->slotColumnR.ResetDirty();
    this->slotColumnG.ResetDirty();
    this->slotColumnB.ResetDirty();
    this->slotColumnI.ResetDirty();
    this->slotGlobalColor.ResetDirty();
    this->slotColorMode.ResetDirty();
    this->slotColumnX.ResetDirty();
    this->slotColumnY.ResetDirty();
    this->slotColumnZ.ResetDirty();
    this->slotColumnIndex.ResetDirty();
    this->slotConnectionType.ResetDirty();
}

std::string TableToLines::cleanUpColumnHeader(const std::string& header) const {
    return this->cleanUpColumnHeader(vislib::TString(header.data()));
}

std::string TableToLines::cleanUpColumnHeader(const vislib::TString& header) const {
    vislib::TString h(header);
    h.TrimSpaces();
    h.ToLowerCase();
    return std::string(T2A(h.PeekBuffer()));
}

bool TableToLines::pushColumnIndex(std::vector<size_t>& cols, const vislib::TString& colName) {
    std::string c = cleanUpColumnHeader(colName);
    if (this->columnIndex.find(c) != columnIndex.end()) {
        cols.push_back(columnIndex[c]);
        return true;
    } else {
        vislib::sys::Log::DefaultLog.WriteError("unknown column '%s'", c.c_str());
        return false;
    }
}

bool TableToLines::assertData(table::TableDataCall *ft) {
    if (this->inputHash == ft->DataHash() && !anythingDirty()) return true;

    if (this->inputHash != ft->DataHash()) {
        vislib::sys::Log::DefaultLog.WriteInfo("TableToLines: Dataset changed -> Updating EnumParams\n");
        this->columnIndex.clear();

        this->slotColumnX.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnY.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnZ.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnR.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnG.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnB.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnI.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnIndex.Param<core::param::FlexEnumParam>()->ClearValues();

        for (size_t i = 0; i < ft->GetColumnsCount(); i++) {
            std::string n = std::string(this->cleanUpColumnHeader(ft->GetColumnsInfos()[i].Name()));
            columnIndex[n] = i;

            this->slotColumnX.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnY.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnZ.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnR.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnG.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnB.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnI.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnIndex.Param<core::param::FlexEnumParam>()->AddValue(n);
        }
    }

    size_t rows = ft->GetRowsCount();
    size_t cols = ft->GetColumnsCount();

    allVerts.clear();
    allColor.clear();
    lineColor.clear();
    lineIndices.clear();
    lineVerts.clear();
    lines.clear();


    allVerts.resize(rows * 3);

    stride = 3;
    switch (this->slotColorMode.Param<core::param::EnumParam>()->Value()) {
        case 0: // RGB
            stride += 3;
            allColor.reserve(rows * 3);
            break;
        case 1: // I
            stride += 1;
            allColor.reserve(rows);
            break;
        case 2: // global RGB
            break;
    }

    switch (this->slotConnectionType.Param<core::param::EnumParam>()->Value()) {
    case 0: //Grid
        break;
    case 1: //Index
        stride += 1;
        lineIndices.resize(rows);
        break;
    }



    bool retValue = true;

    std::vector<size_t> indicesToCollect;
    if (!pushColumnIndex(indicesToCollect, this->slotColumnX.Param<core::param::FlexEnumParam>()->ValueString())) {
        retValue = false;
    }
    if (!pushColumnIndex(indicesToCollect, this->slotColumnY.Param<core::param::FlexEnumParam>()->ValueString())) {
        retValue = false;
    }
    if (!pushColumnIndex(indicesToCollect, this->slotColumnZ.Param<core::param::FlexEnumParam>()->ValueString())) {
        retValue = false;
    }

    switch (this->slotConnectionType.Param<core::param::EnumParam>()->Value()) {
    case 0: //Grid
        break;
    case 1: //Index
        if (!pushColumnIndex(indicesToCollect, this->slotColumnIndex.Param<core::param::FlexEnumParam>()->ValueString())) {
            retValue = false;
        }
        break;
    }


    switch (this->slotColorMode.Param<core::param::EnumParam>()->Value()) {
        case 0: // RGB
            if (!pushColumnIndex(indicesToCollect, this->slotColumnR.Param<core::param::FlexEnumParam>()->ValueString())) {
                retValue = false;
            }
            if (!pushColumnIndex(indicesToCollect, this->slotColumnG.Param<core::param::FlexEnumParam>()->ValueString())) {
                retValue = false;
            }
            if (!pushColumnIndex(indicesToCollect, this->slotColumnB.Param<core::param::FlexEnumParam>()->ValueString())) {
                retValue = false;
            }
            break;
        case 1: // I
            if (!pushColumnIndex(indicesToCollect, this->slotColumnI.Param<core::param::FlexEnumParam>()->ValueString())) {
                retValue = false;
            } else {
                iMin = ft->GetColumnsInfos()[indicesToCollect[indicesToCollect.size() - 1]].MinimumValue();
                iMax = ft->GetColumnsInfos()[indicesToCollect[indicesToCollect.size() - 1]].MaximumValue();
            }
            break;
        case 2: // global RGB
            break;
    }

    const float *ftData = ft->GetData();
    size_t numIndices = indicesToCollect.size();

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < numIndices; j++) {
            if (j < 3) {
                allVerts[3 * i + j] = ftData[cols * i + indicesToCollect[j]];
            }
            if (j == 3) {
                if (this->slotConnectionType.Param<core::param::EnumParam>()->Value() == 1) {
                    lineIndices[i] = ftData[cols * i + indicesToCollect[j]];
                } else {
                    allColor.push_back( (ftData[cols * i + indicesToCollect[j]] - 
                        ft->GetColumnsInfos()[indicesToCollect[j]].MinimumValue())/
                        (ft->GetColumnsInfos()[indicesToCollect[j]].MaximumValue() - 
                        ft->GetColumnsInfos()[indicesToCollect[j]].MinimumValue()));
                }
            }
            if (j > 3) {
                allColor.push_back((ftData[cols * i + indicesToCollect[j]] -
                    ft->GetColumnsInfos()[indicesToCollect[j]].MinimumValue()) /
                    (ft->GetColumnsInfos()[indicesToCollect[j]].MaximumValue() -
                    ft->GetColumnsInfos()[indicesToCollect[j]].MinimumValue()));
            }
        }
    }

    unsigned int tex_size;
    std::vector<float> processedColor;
    processedColor.reserve(allColor.size() * 3);
    if (!(this->slotColorMode.Param<core::param::EnumParam>()->Value() == 2)) { // I or RGB
        core::view::CallGetTransferFunction *cgtf = this->slotTF.CallAs<core::view::CallGetTransferFunction>();
        if (cgtf != NULL && ((*cgtf)())) {
            float const* tf_tex = cgtf->GetTextureData();
            tex_size = cgtf->TextureSize();
            this->colorTransferGray(allColor, tf_tex, tex_size, processedColor, 3);
        } else {
            this->colorTransferGray(allColor, NULL, 0, processedColor, 3);
        }
    }

    // GRID
    if (this->slotConnectionType.Param<core::param::EnumParam>()->Value() == 0) {

        typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Adaptor<float, PointCloud<float>>,
            PointCloud<float>, 3> kd_tree;

        PointCloud<float> pc;
        pc.pts.resize(rows);
        for (unsigned int i = 0; i < rows; ++i) {
            pc.pts[i].x = allVerts[3 * i + 0];
            pc.pts[i].y = allVerts[3 * i + 1];
            pc.pts[i].z = allVerts[3 * i + 2];
        }

        kd_tree index(3, pc, nanoflann::KDTreeSingleIndexAdaptorParams());
        index.buildIndex();
        kd_tree::BoundingBox bbox;
        index.computeBoundingBox(bbox);

        float query_pt[3];
        std::vector<size_t> ret_index;
        std::vector<float> out_dist_sqr;

        std::vector<float> tmpVert(6);
        std::vector<float> tmpColor(6);

        for (unsigned int i = 0; i < rows; ++i) {

            size_t num_neighbors = 6;

            ret_index.clear();
            out_dist_sqr.clear();

            // check for corners(-3), edges(-2) and planes(-1)
            if (pc.pts[i].x == bbox[0].low || pc.pts[i].x == bbox[0].high) {
                num_neighbors -= 1;
            }
            if (pc.pts[i].y == bbox[1].low || pc.pts[i].y == bbox[1].high) {
                num_neighbors -= 1;
            }
            if (pc.pts[i].z == bbox[2].low || pc.pts[i].z == bbox[2].high) {
                num_neighbors -= 1;
            }

            const size_t num_results = num_neighbors + 1;
            nanoflann::KNNResultSet<float> resultSet(num_results);
            ret_index.resize(num_results);
            out_dist_sqr.resize(num_results);

            query_pt[0] = pc.pts[i].x;
            query_pt[1] = pc.pts[i].y;
            query_pt[2] = pc.pts[i].z;

            resultSet.init(ret_index.data(), out_dist_sqr.data());

            index.findNeighbors(resultSet, query_pt, nanoflann::SearchParams());
            for (unsigned int j = 0; j < num_neighbors; ++j) {
                // Vertices
                tmpVert[0] = pc.pts[i].x;
                tmpVert[1] = pc.pts[i].y;
                tmpVert[2] = pc.pts[i].z;

                tmpVert[3] = pc.pts[ret_index[j + 1]].x;
                tmpVert[4] = pc.pts[ret_index[j + 1]].y;
                tmpVert[5] = pc.pts[ret_index[j + 1]].z;

                lineVerts.push_back(tmpVert);

                // Color
                if (!(this->slotColorMode.Param<core::param::EnumParam>()->Value() == 2)) { // I or RGB

                    tmpColor[0] = processedColor[3 * i + 0];
                    tmpColor[1] = processedColor[3 * i + 1];
                    tmpColor[2] = processedColor[3 * i + 2];

                    tmpColor[3] = processedColor[3 * ret_index[j + 1] + 0];
                    tmpColor[4] = processedColor[3 * ret_index[j + 1] + 1];
                    tmpColor[5] = processedColor[3 * ret_index[j + 1] + 2];

                    lineColor.push_back(tmpColor);
                }


            } // for num_neighbors end
        } // for rows end

    } else { // INDEX
        // distinguish lines
        auto uniqueLineIndices(lineIndices);
        auto last = std::unique(uniqueLineIndices.begin(), uniqueLineIndices.end());
        uniqueLineIndices.erase(last, uniqueLineIndices.end());

        lineVerts.resize(uniqueLineIndices.size());
        lineColor.resize(uniqueLineIndices.size());
        lines.resize(uniqueLineIndices.size());

        for (unsigned int i = 0; i < uniqueLineIndices.size(); ++i) {
            for (unsigned int j = 0; j < rows; ++j) {
                if (lineIndices[j] == uniqueLineIndices[i]) {
                    // Vertices
                    lineVerts[i].push_back(allVerts[3 * j + 0]);
                    lineVerts[i].push_back(allVerts[3 * j + 1]);
                    lineVerts[i].push_back(allVerts[3 * j + 2]);

                    //Color
                    if (!(this->slotColorMode.Param<core::param::EnumParam>()->Value() == 2)) { // I or RGB
                        lineColor[i].push_back(processedColor[3 * j + 0]);
                        lineColor[i].push_back(processedColor[3 * j + 1]);
                        lineColor[i].push_back(processedColor[3 * j + 2]);
                    }
                }
            } // end for lineIndices
        } // end for uniqueLineIndices
    }


    for (size_t i = 0; i < (numIndices < 3 ? numIndices : 3); i++) {
        this->bboxMin[i] = ft->GetColumnsInfos()[indicesToCollect[i]].MinimumValue();
        this->bboxMax[i] = ft->GetColumnsInfos()[indicesToCollect[i]].MaximumValue();
    }

    this->myHash++;
    this->resetAllDirty();
    this->inputHash = ft->DataHash();
    return retValue;
}

/*
 * megamol::pcl::PclDataSource::getMultiParticleData
 */
bool TableToLines::getLineData(core::Call& call) {
    try {
        geocalls::LinesDataCall& c = dynamic_cast<
            geocalls::LinesDataCall&>(call);
        table::TableDataCall *ft = this->slotCallTable.CallAs<table::TableDataCall>();
        if (ft == NULL) return false;
        (*ft)();

        if (!assertData(ft)) return false;

        c.SetFrameCount(1);
        c.SetFrameID(0);
        c.SetDataHash(this->myHash);

        c.SetExtent(1,
            this->bboxMin[0], this->bboxMin[1], this->bboxMin[2],
            this->bboxMax[0], this->bboxMax[1], this->bboxMax[2]);

        lines.resize(lineVerts.size());

        for (size_t loop = 0; loop < lineVerts.size(); loop++) {
            if (this->slotColorMode.Param<core::param::EnumParam>()->Value() == 2) {
                unsigned char rgb[3];
                core::utility::ColourParser::FromString(this->slotGlobalColor.Param<core::param::StringParam>()->Value(), 3, rgb);
                lines[loop].Set(static_cast<unsigned int>(lineVerts[loop].size() / 3), lineVerts[loop].data(), vislib::graphics::ColourRGBAu8(rgb[0], rgb[1], rgb[2], 255));
            } else {
                lines[loop].Set(static_cast<unsigned int>(lineVerts[loop].size() / 3), lineVerts[loop].data(), lineColor[loop].data(), false);
            }
        }

        c.SetData(lines.size(), lines.data());
        c.SetUnlocker(NULL);

        return true;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteError(1, e.GetMsg());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(1, _T("Unexpected exception ")
            _T("in callback getMultiParticleData."));
        return false;
    }
}


/*
 * megamol::pcl::PclDataSource::getMultiparticleExtent
 */
bool TableToLines::getLineDataExtent(core::Call& call) {
    try {
        geocalls::LinesDataCall& c = dynamic_cast<
            geocalls::LinesDataCall&>(call);
        table::TableDataCall *ft = this->slotCallTable.CallAs<table::TableDataCall>();
        if (ft == NULL) return false;
        (*ft)();

        if (!assertData(ft)) return false;

        c.SetFrameCount(1);
        c.SetFrameID(0);
        c.SetDataHash(this->myHash);

        c.SetExtent(1,
            this->bboxMin[0], this->bboxMin[1], this->bboxMin[2],
            this->bboxMax[0], this->bboxMax[1], this->bboxMax[2]);
        c.SetUnlocker(NULL);
        return true;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteError(1, e.GetMsg());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(1, _T("Unexpected exception ")
            _T("in callback getLineDataExtend."));
        return false;
    }
}


/*
 * megamol::pcl::PclDataSource::release
 */
void TableToLines::release(void) {
}


void TableToLines::colorTransferGray(std::vector<float> &grayArray, float const* transferTable, unsigned int tableSize, std::vector<float> &rgbaArray, unsigned int target_length=3) {

    if (grayArray.size() == 0) {
        return;
    }

    float gray_max = *std::max_element(grayArray.begin(), grayArray.end());
    float gray_min = *std::min_element(grayArray.begin(), grayArray.end());

    for (auto &gray : grayArray) {
        float scaled_gray;
        if ((gray_max - gray_min) <= 1e-4f) {
            scaled_gray = 0;
        } else {
            scaled_gray = (gray - gray_min) / (gray_max - gray_min);
        }
        if (transferTable == NULL && tableSize == 0) {
            for (int i = 0; i < 3; i++) {
                rgbaArray.push_back((0.3f + scaled_gray) / 1.3f);
            }
            if (target_length == 4) {
                rgbaArray.push_back(1.0f);
            }
        } else {
            float exact_tf = (tableSize - 1) * scaled_gray;
            int floor = std::floor(exact_tf);
            float tail = exact_tf - (float)floor;
            floor *= 4;
            for (int i = 0; i < target_length; i++) {
                float colorFloor = transferTable[floor + i];
                float colorCeil = transferTable[floor + i + 4];
                float finalColor = colorFloor + (colorCeil - colorFloor)*(tail);
                rgbaArray.push_back(finalColor);
            }
        }
    }
}
