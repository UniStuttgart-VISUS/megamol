/*
 * BitmapCodecCollection.cpp
 *
 * Copyright (C) 2010 by SGrottel
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/BitmapCodecCollection.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/Exception.h"
#include "vislib/MemmappedFile.h"
#include "vislib/Path.h"
#include "vislib/SmartPtr.h"
#include "vislib/StringTokeniser.h"
#include "vislib/vislibsymbolimportexport.inl"

#include "vislib/BmpBitmapCodec.h"
#include "vislib/PpmBitmapCodec.h"


/*
 * __vl_bitmapcodeccollection_defcol
 */
VISLIB_STATICSYMBOL vislib::graphics::BitmapCodecCollection
__vl_bitmapcodeccollection_defcol
#ifndef VISLIB_SYMBOL_IMPORT
    = vislib::graphics::BitmapCodecCollection::BuildDefaultCollection()
#endif /* !VISLIB_SYMBOL_IMPORT */
    ;


/*
 * vislib::graphics::BitmapCodecCollection::BuildDefaultCollection
 */
vislib::graphics::BitmapCodecCollection
vislib::graphics::BitmapCodecCollection::BuildDefaultCollection(void) {
    BitmapCodecCollection defCol;

    //
    // Add the built-in default codecs here
    //
    defCol.AddCodec(new BmpBitmapCodec());
    defCol.AddCodec(new PpmBitmapCodec());

    defCol.codecs.Trim();
    return defCol;
}


/*
 * vislib::graphics::BitmapCodecCollection::DefaultCollection
 */
vislib::graphics::BitmapCodecCollection&
vislib::graphics::BitmapCodecCollection::DefaultCollection(void) {
    return __vl_bitmapcodeccollection_defcol;
}


/*
 * vislib::graphics::BitmapCodecCollection::BitmapCodecCollection
 */
vislib::graphics::BitmapCodecCollection::BitmapCodecCollection(void)
        : codecs() {
    // intentionally empty
}


/*
 * vislib::graphics::BitmapCodecCollection::BitmapCodecCollection
 */
vislib::graphics::BitmapCodecCollection::BitmapCodecCollection(
        const vislib::graphics::BitmapCodecCollection& src)
        : codecs(src.codecs) {
    // intentionally empty
}


/*
 * vislib::graphics::BitmapCodecCollection::~BitmapCodecCollection
 */
vislib::graphics::BitmapCodecCollection::~BitmapCodecCollection(void) {
    this->codecs.Clear();
}


/*
 * vislib::graphics::BitmapCodecCollection::LoadBitmapImage
 */
bool vislib::graphics::BitmapCodecCollection::LoadBitmapImage(
        BitmapImage& outImg, const vislib::StringA& filename) {
    vislib::sys::File f;

    if (!f.Open(filename, vislib::sys::File::READ_ONLY,
            vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        throw vislib::Exception("Unable to open image file",
            __FILE__, __LINE__);
    }

    const SIZE_T MAX_AD_SIZE = 1024;
    SIZE_T adsize = 0;
    vislib::SmartPtr<char, vislib::ArrayAllocator<char> > admem
        = new char[MAX_AD_SIZE];
    adsize = static_cast<SIZE_T>(f.Read(admem.operator->(),
        static_cast<vislib::sys::File::FileSize>(MAX_AD_SIZE)));
    f.Close();

    CodecArray codecs;
    this->selectCodecsByFilename(filename, codecs);

    for (unsigned int pass = 0; pass < 2; pass++) {

        if (pass == 1) { // now try all the other codecs
            CodecArray oldC(codecs);
            codecs.Clear();
            for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
                if (!oldC.Contains(this->codecs[i])) {
                    codecs.Add(this->codecs[i]);
                }
            }
        }

        CodecArray sure;
        CodecArray unsure;
        this->autodetecCodec(admem.operator->(), adsize, codecs,
            sure, unsure);

        for (SIZE_T i = 0; i < sure.Count(); i++) {
            sure[i]->Image() = &outImg;
            bool suc = sure[i]->Load(filename);
            sure[i]->Image() = NULL;
            if (suc) return true;
        }

        for (SIZE_T i = 0; i < unsure.Count(); i++) {
            unsure[i]->Image() = &outImg;
            bool suc = unsure[i]->Load(filename);
            unsure[i]->Image() = NULL;
            if (suc) return true;
        }

        for (SIZE_T i = 0; i < codecs.Count(); i++) {
            if (codecs[i]->CanAutoDetect()) continue;
            codecs[i]->Image() = &outImg;
            bool suc = codecs[i]->Load(filename);
            codecs[i]->Image() = NULL;
            if (suc) return true;
        }

    }

    // no suitable codec found
    return false;
}


/*
 * vislib::graphics::BitmapCodecCollection::LoadBitmapImage
 */
bool vislib::graphics::BitmapCodecCollection::LoadBitmapImage(
        BitmapImage& outImg, const vislib::StringW& filename) {
    vislib::sys::File f;

    if (!f.Open(filename, vislib::sys::File::READ_ONLY,
            vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        throw vislib::Exception("Unable to open image file",
            __FILE__, __LINE__);
    }

    const SIZE_T MAX_AD_SIZE = 1024;
    SIZE_T adsize = 0;
    vislib::SmartPtr<char, vislib::ArrayAllocator<char> > admem
        = new char[MAX_AD_SIZE];
    adsize = static_cast<SIZE_T>(f.Read(admem.operator->(),
        static_cast<vislib::sys::File::FileSize>(MAX_AD_SIZE)));
    f.Close();

    CodecArray codecs;
    this->selectCodecsByFilename(filename, codecs);

    for (unsigned int pass = 0; pass < 2; pass++) {

        if (pass == 1) { // now try all the other codecs
            CodecArray oldC(codecs);
            codecs.Clear();
            for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
                if (!oldC.Contains(this->codecs[i])) {
                    codecs.Add(this->codecs[i]);
                }
            }
        }

        CodecArray sure;
        CodecArray unsure;
        this->autodetecCodec(admem.operator->(), adsize, codecs,
            sure, unsure);

        for (SIZE_T i = 0; i < sure.Count(); i++) {
            sure[i]->Image() = &outImg;
            bool suc = sure[i]->Load(filename);
            sure[i]->Image() = NULL;
            if (suc) return true;
        }

        for (SIZE_T i = 0; i < unsure.Count(); i++) {
            unsure[i]->Image() = &outImg;
            bool suc = unsure[i]->Load(filename);
            unsure[i]->Image() = NULL;
            if (suc) return true;
        }

        for (SIZE_T i = 0; i < codecs.Count(); i++) {
            if (codecs[i]->CanAutoDetect()) continue;
            codecs[i]->Image() = &outImg;
            bool suc = codecs[i]->Load(filename);
            codecs[i]->Image() = NULL;
            if (suc) return true;
        }

    }

    // no suitable codec found
    return false;
}


/*
 * vislib::graphics::BitmapCodecCollection::LoadBitmapImage
 */
bool vislib::graphics::BitmapCodecCollection::LoadBitmapImage(
        BitmapImage& outImg, vislib::sys::File& file) {

    vislib::sys::File::FileSize filepos = file.Tell();

    const SIZE_T MAX_AD_SIZE = 1024;
    SIZE_T adsize = 0;
    char *admem = new char[MAX_AD_SIZE];
    adsize = static_cast<SIZE_T>(file.Read(admem, MAX_AD_SIZE));

    CodecArray sure;
    CodecArray unsure;
    this->autodetecCodec(admem, adsize, this->codecs, sure, unsure);

    delete[] admem;

    for (SIZE_T i = 0; i < sure.Count(); i++) {
        sure[i]->Image() = &outImg;
        file.Seek(filepos);
        bool suc = sure[i]->Load(file);
        sure[i]->Image() = NULL;
        if (suc) return true;
    }

    for (SIZE_T i = 0; i < unsure.Count(); i++) {
        unsure[i]->Image() = &outImg;
        file.Seek(filepos);
        bool suc = unsure[i]->Load(file);
        unsure[i]->Image() = NULL;
        if (suc) return true;
    }

    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        if (this->codecs[i]->CanAutoDetect()) continue;
        this->codecs[i]->Image() = &outImg;
        file.Seek(filepos);
        bool suc = this->codecs[i]->Load(file);
        this->codecs[i]->Image() = NULL;
        if (suc) return true;
    }

    return false;
}


/*
 * vislib::graphics::BitmapCodecCollection::LoadBitmapImage
 */
bool vislib::graphics::BitmapCodecCollection::LoadBitmapImage(
        BitmapImage& outImg, const void *mem, SIZE_T size) {

    CodecArray sure;
    CodecArray unsure;
    this->autodetecCodec(mem, size, this->codecs, sure, unsure);

    for (SIZE_T i = 0; i < sure.Count(); i++) {
        sure[i]->Image() = &outImg;
        bool suc = sure[i]->Load(mem, size);
        sure[i]->Image() = NULL;
        if (suc) return true;
    }

    for (SIZE_T i = 0; i < unsure.Count(); i++) {
        unsure[i]->Image() = &outImg;
        bool suc = unsure[i]->Load(mem, size);
        unsure[i]->Image() = NULL;
        if (suc) return true;
    }

    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        if (this->codecs[i]->CanAutoDetect()) continue;
        this->codecs[i]->Image() = &outImg;
        bool suc = this->codecs[i]->Load(mem, size);
        this->codecs[i]->Image() = NULL;
        if (suc) return true;
    }

    return false;
}


/*
 * vislib::graphics::BitmapCodecCollection::selectCodecsByFilename
 */
void vislib::graphics::BitmapCodecCollection::selectCodecsByFilename(
        const vislib::StringA& filename,
        vislib::graphics::BitmapCodecCollection::CodecArray& outCodecs)
        const {
    outCodecs.Clear();
    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        vislib::StringTokeniserA exts(this->codecs[i]->FileNameExtsA(), ';');
        while (exts.HasNext()) {
            if (filename.EndsWith(exts.Next())) {
                outCodecs.Add(this->codecs[i]);
                break;
            }
        }
    }
}


/*
 * vislib::graphics::BitmapCodecCollection::selectCodecsByFilename
 */
void vislib::graphics::BitmapCodecCollection::selectCodecsByFilename(
        const vislib::StringW& filename,
        vislib::graphics::BitmapCodecCollection::CodecArray& outCodecs)
        const {
    outCodecs.Clear();
    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        vislib::StringTokeniserW exts(this->codecs[i]->FileNameExtsW(), L';');
        while (exts.HasNext()) {
            if (filename.EndsWith(exts.Next())) {
                outCodecs.Add(this->codecs[i]);
                break;
            }
        }
    }
}


/*
 * vislib::graphics::BitmapCodecCollection::autodetecCodec
 */
void vislib::graphics::BitmapCodecCollection::autodetecCodec(
        const void *mem, SIZE_T size,
        const vislib::graphics::BitmapCodecCollection::CodecArray& codecs,
        vislib::graphics::BitmapCodecCollection::CodecArray& outMatchingCodecs,
        vislib::graphics::BitmapCodecCollection::CodecArray& outUnsureCodecs)
        const {
    outMatchingCodecs.Clear();
    outUnsureCodecs.Clear();
    for (SIZE_T i = 0; i < codecs.Count(); i++) {
        if (!codecs[i]->CanAutoDetect()) continue;
        int rv = codecs[i]->AutoDetect(mem, size);
        if (rv == 1) {
            outMatchingCodecs.Add(codecs[i]);
        } else if (rv == -1) {
            outUnsureCodecs.Add(codecs[i]);
        }
    }
}
