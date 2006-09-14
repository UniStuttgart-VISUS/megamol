/*
 * testfile.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testfile.h"

#include <iostream>

#include "testhelper.h"
#include "vislib/BufferedFile.h"
#include "vislib/File.h"
#include "vislib/IOException.h"


using namespace vislib::sys;


static void runTests(File& f1) {
    const File::FileSize TEXT_SIZE = 26;
    char writeBuffer[TEXT_SIZE + 1] = "abcdefghijklmnopqrstuvwxyz";
    char readBuffer[TEXT_SIZE];
    
    AssertFalse("\"horst.txt\" does not exist", File::Exists("horst.txt"));
    AssertFalse("L\"horst.txt\" does not exist", File::Exists(L"horst.txt"));

    AssertFalse("Cannot open not existing file", f1.Open("horst.txt", 
        File::READ_WRITE, File::SHARE_READWRITE, File::OPEN_ONLY));
    AssertFalse("Cannot open not existing file", f1.Open(L"horst.txt", 
        File::READ_WRITE, File::SHARE_READWRITE, File::OPEN_ONLY));

    AssertTrue("Can create file", f1.Open("horst.txt", File::READ_WRITE, 
        File::SHARE_READWRITE, File::OPEN_CREATE));
    AssertTrue("File is open now", f1.IsOpen());
    f1.Close();
    AssertFalse("Closing file", f1.IsOpen());

    AssertFalse("CREATE_ONLY cannot overwrite existing file", f1.Open(
        "horst.txt", File::READ_WRITE, File::SHARE_READWRITE, 
        File::CREATE_ONLY));
    AssertTrue("OPEN_ONLY can open existing file", f1.Open("horst.txt", 
        File::WRITE_ONLY, File::SHARE_READ, File::OPEN_ONLY));
    AssertTrue("File is open now", f1.IsOpen());

    AssertEqual("Writing to file", f1.Write(writeBuffer, TEXT_SIZE), TEXT_SIZE);
    AssertTrue("Is EOF", f1.IsEOF());

    AssertEqual("Seek to begin", f1.SeekToBegin(), 
        static_cast<File::FileSize>(0));
    AssertEqual("Is at begin", f1.Tell(), static_cast<File::FileSize>(0));
    AssertFalse("Is not EOF", f1.IsEOF());

    try {
        f1.Read(readBuffer, TEXT_SIZE);
        AssertTrue("Cannot read on WRITE_ONLY file", false);
    } catch (IOException e) {
        AssertTrue("Cannot read on WRITE_ONLY file", true);
        std::cout << e.GetMsgA() << std::endl;
    }

    f1.Close();
    AssertFalse("Closing file", f1.IsOpen());

    AssertTrue("Open file for reading", f1.Open("horst.txt", 
        File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY));
    AssertEqual("Is at begin of file", f1.Tell(), 
        static_cast<File::FileSize>(0));
    AssertNotEqual("Seek to end", f1.SeekToEnd(), 
        static_cast<File::FileSize>(0));
    AssertTrue("Is at end", f1.IsEOF());

    AssertEqual("Seek to 2", f1.Seek(2, File::BEGIN), 
        static_cast<File::FileSize>(2));
    AssertEqual("Is at 2", f1.Tell(), static_cast<File::FileSize>(2));
    AssertEqual("Reading from 2", f1.Read(readBuffer, 1),
        static_cast<File::FileSize>(1));
    AssertEqual("Read correct character", readBuffer[0], 'c');
    
    AssertEqual("File size", f1.GetSize(), TEXT_SIZE);

    f1.Close();
    AssertFalse("Closing file", f1.IsOpen());

    AssertTrue("Renaming file", File::Rename("horst.txt", "hugo.txt"));
    AssertFalse("Old file does not exist", File::Exists("horst.txt"));
    AssertTrue("New file exists", File::Exists("hugo.txt"));

    AssertTrue("Opening file for read/write", f1.Open(L"hugo.txt", 
        File::READ_WRITE, File::SHARE_READ, File::OPEN_ONLY));
    AssertEqual("Reading 9 characters", f1.Read(readBuffer, 9),
        static_cast<File::FileSize>(9));
    AssertEqual("Read correct characters", 
        ::memcmp(readBuffer, writeBuffer, 9), 0);
    AssertEqual("Seek to 2", f1.Seek(2, File::BEGIN), 
        static_cast<File::FileSize>(2));
    AssertEqual("Reading 9 characters", f1.Read(readBuffer, 9),
        static_cast<File::FileSize>(9));
    AssertEqual("Read correct characters", 
        ::memcmp(readBuffer, writeBuffer + 2, 9), 0);

    f1.Close();
    AssertFalse("Closing file", f1.IsOpen());

    File::Delete("hugo.txt");
    AssertFalse("\"hugo.txt\" was deleted", File::Exists("hugo.txt"));
}


void TestFile(void) {
    ::TestBaseFile();
    ::TestBufferedFile();
}


void TestBaseFile(void) {
    File f1;
    std::cout << std::endl << "Tests for File" << std::endl;
    ::runTests(f1);
}


void TestBufferedFile(void) {
    BufferedFile f1;
    std::cout << std::endl << "Tests for BufferedFile" << std::endl;
    ::runTests(f1);

    f1.SetBufferSize(8);
    std::cout << std::endl << "Tests for BufferedFile, buffer size 8" << std::endl;
    ::runTests(f1);
}