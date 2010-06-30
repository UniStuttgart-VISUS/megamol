/*
 * BitmapCodecCollection.cpp
 *
 * Copyright (C) 2010 by SGrottel
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/BitmapCodecCollection.h"
#include "vislib/Exception.h"
#include "vislib/MemmappedFile.h"
#include "vislib/StringTokeniser.h"

#include "vislib/BmpBitmapCodec.h"
#include "vislib/PpmBitmapCodec.h"


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
    static BitmapCodecCollection defCol = BuildDefaultCollection();
    return defCol;
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
    vislib::sys::MemmappedFile file;
    if (!file.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        throw vislib::Exception("Unable to open image file", __FILE__, __LINE__);
    }

    // 1.: test depending on the file name extension
    const SIZE_T MAX_AD_SIZE = 1024;
    SIZE_T adsize;
    char *admem = NULL;
    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        if (!this->codecs[i]->CanLoad()) continue;

        vislib::StringTokeniserA exts(this->codecs[i]->FileNameExtsA(), ';');
        while (exts.HasNext()) {
            if (filename.EndsWith(exts.Next())) {
                // matching file name, so try this codec

                if (this->codecs[i]->CanAutoDetect()) {
                    if (admem == NULL) {
                        admem = new char[MAX_AD_SIZE];
                        file.SeekToBegin();
                        adsize = static_cast<SIZE_T>(file.Read(admem, MAX_AD_SIZE));
                    }
                    int adr = this->codecs[i]->AutoDetect(admem, adsize);
                    if (adr == 0) break; // not loadable by this codec
                }

                bool rv = false;
                this->codecs[i]->Image() = &outImg;

                if (this->codecs[i]->CanLoadFromStream()) {
                    file.SeekToBegin();
                    rv = this->codecs[i]->Load(file);
                } else if (this->codecs[i]->CanLoadFromFile()) {
                    rv = this->codecs[i]->Load(filename);
                } else if (this->codecs[i]->CanLoadFromMemory()) {
                    vislib::RawStorage rs;
                    SIZE_T s(static_cast<SIZE_T>(file.GetSize()));
                    rs.EnforceSize(s);
                    file.SeekToBegin();
                    SIZE_T r(static_cast<SIZE_T>(file.Read(rs, s)));
                    rv = this->codecs[i]->Load(rs);
                }

                this->codecs[i]->Image() = NULL;
                if (rv) return true; // successfully loaded

                break;
            }
        }
    }
    delete[] admem;

    // 2.: test based on autodetection
    file.SeekToBegin();
    if (this->LoadBitmapImage(outImg, file)) return true;

    // 3: test for codecs only capable of loading directly from file
    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        if (this->codecs[i]->CanLoadFromFile()
                && !this->codecs[i]->CanLoadFromMemory()
                && !this->codecs[i]->CanLoadFromStream()) {
            // very esotheric codec ...
            this->codecs[i]->Image() = &outImg;
            bool rv = this->codecs[i]->Load(filename);
            this->codecs[i]->Image() = NULL;
            if (rv) return true; // we did it!?
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
    vislib::sys::MemmappedFile file;
    if (!file.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        throw vislib::Exception("Unable to open image file", __FILE__, __LINE__);
    }

    // 1.: test depending on the file name extension
    const SIZE_T MAX_AD_SIZE = 1024;
    SIZE_T adsize;
    char *admem = NULL;
    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        if (!this->codecs[i]->CanLoad()) continue;

        vislib::StringTokeniserW exts(this->codecs[i]->FileNameExtsW(), ';');
        while (exts.HasNext()) {
            if (filename.EndsWith(exts.Next())) {
                // matching file name, so try this codec

                if (this->codecs[i]->CanAutoDetect()) {
                    if (admem == NULL) {
                        admem = new char[MAX_AD_SIZE];
                        file.SeekToBegin();
                        adsize = static_cast<SIZE_T>(file.Read(admem, MAX_AD_SIZE));
                    }
                    int adr = this->codecs[i]->AutoDetect(admem, adsize);
                    if (adr == 0) break; // not loadable by this codec
                }

                bool rv = false;
                this->codecs[i]->Image() = &outImg;

                if (this->codecs[i]->CanLoadFromStream()) {
                    file.SeekToBegin();
                    rv = this->codecs[i]->Load(file);
                } else if (this->codecs[i]->CanLoadFromFile()) {
                    rv = this->codecs[i]->Load(filename);
                } else if (this->codecs[i]->CanLoadFromMemory()) {
                    vislib::RawStorage rs;
                    SIZE_T s(static_cast<SIZE_T>(file.GetSize()));
                    rs.EnforceSize(s);
                    file.SeekToBegin();
                    SIZE_T r(static_cast<SIZE_T>(file.Read(rs, s)));
                    rv = this->codecs[i]->Load(rs);
                }

                this->codecs[i]->Image() = NULL;
                if (rv) {
                    delete[] admem;
                    return true; // successfully loaded
                }

                break;
            }
        }
    }
    delete[] admem;

    // 2.: test based on autodetection
    file.SeekToBegin();
    if (this->LoadBitmapImage(outImg, file)) return true;

    // 3: test for codecs only capable of loading directly from file
    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        if (this->codecs[i]->CanLoadFromFile()
                && !this->codecs[i]->CanLoadFromMemory()
                && !this->codecs[i]->CanLoadFromStream()) {
            // very esotheric codec ...
            this->codecs[i]->Image() = &outImg;
            bool rv = this->codecs[i]->Load(filename);
            this->codecs[i]->Image() = NULL;
            if (rv) return true; // we did it!?
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

    // 1: test auto-detecting codecs
    const SIZE_T MAX_AD_SIZE = 1024;
    SIZE_T adsize;
    char *admem = NULL;
    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        if (!this->codecs[i]->CanAutoDetect()) continue;
        if (!this->codecs[i]->CanLoadFromStream()
            && !this->codecs[i]->CanLoadFromMemory()) continue;

        if (admem == NULL) {
            admem = new char[MAX_AD_SIZE];
            file.Seek(filepos);
            adsize = static_cast<SIZE_T>(file.Read(admem, MAX_AD_SIZE));
        }
        int adr = this->codecs[i]->AutoDetect(admem, adsize);
        if (adr == 0) continue; // not loadable by this codec

        bool rv = false;
        this->codecs[i]->Image() = &outImg;

        if (this->codecs[i]->CanLoadFromStream()) {
            file.Seek(filepos);
            rv = this->codecs[i]->Load(file);
        } else if (this->codecs[i]->CanLoadFromMemory()) {
            vislib::RawStorage rs;
            SIZE_T s(static_cast<SIZE_T>(file.GetSize() - filepos));
            rs.EnforceSize(s);
            file.Seek(filepos);
            SIZE_T r(static_cast<SIZE_T>(file.Read(rs, s)));
            rv = this->codecs[i]->Load(rs);
        }

        this->codecs[i]->Image() = NULL;
        if (rv) {
            delete[] admem;
            return true; // successfully loaded
        }
    }
    delete[] admem;

    // 2: try codecs without auto-detection
    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        if (this->codecs[i]->CanAutoDetect()) continue;
        if (!this->codecs[i]->CanLoadFromStream()
            && !this->codecs[i]->CanLoadFromMemory()) continue;

        bool rv = false;
        this->codecs[i]->Image() = &outImg;

        if (this->codecs[i]->CanLoadFromStream()) {
            file.Seek(filepos);
            rv = this->codecs[i]->Load(file);
        } else if (this->codecs[i]->CanLoadFromMemory()) {
            vislib::RawStorage rs;
            SIZE_T s(static_cast<SIZE_T>(file.GetSize() - filepos));
            rs.EnforceSize(s);
            file.Seek(filepos);
            SIZE_T r(static_cast<SIZE_T>(file.Read(rs, s)));
            rv = this->codecs[i]->Load(rs);
        }

        this->codecs[i]->Image() = NULL;
        if (rv) return true; // successfully loaded
    }

    return false;
}


/*
 * vislib::graphics::BitmapCodecCollection::LoadBitmapImage
 */
bool vislib::graphics::BitmapCodecCollection::LoadBitmapImage(
        BitmapImage& outImg, const void *mem, SIZE_T size) {

    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        if (!this->codecs[i]->CanAutoDetect()) continue;
        if (!this->codecs[i]->CanLoadFromMemory()) continue;
        int adr = this->codecs[i]->AutoDetect(mem, size);
        if (adr == 0) continue;
        bool rv = false;
        this->codecs[i]->Image() = &outImg;
        rv = this->codecs[i]->Load(mem, size);
        this->codecs[i]->Image() = NULL;
        if (rv) return true; // successfully loaded
    }

    for (SIZE_T i = 0; i < this->codecs.Count(); i++) {
        if (this->codecs[i]->CanAutoDetect()) continue;
        if (!this->codecs[i]->CanLoadFromMemory()) continue;
        bool rv = false;
        this->codecs[i]->Image() = &outImg;
        rv = this->codecs[i]->Load(mem, size);
        this->codecs[i]->Image() = NULL;
        if (rv) return true; // successfully loaded
    }

    return false;
}
