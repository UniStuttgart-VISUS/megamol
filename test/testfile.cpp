/*
 * testfile.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testfile.h"

#include <iostream>

#include "testhelper.h"
#include "vislib/BufferedFile.h"
#include "vislib/error.h"
#include "vislib/File.h"
#include "vislib/Path.h"
#include "vislib/IOException.h"
#include "vislib/SystemMessage.h"
#include "vislib/sysfunctions.h"


using namespace vislib::sys;


static void runTests(File& f1) {
    const File::FileSize TEXT_SIZE = 26;
    char writeBuffer[TEXT_SIZE + 1] = "abcdefghijklmnopqrstuvwxyz";
    char readBuffer[TEXT_SIZE];

    std::cout << "Clean up possible old trash ..." << std::endl;
    ::remove("horst.txt");
    ::remove("hugo.txt");
    
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
    
    AssertEqual("Seek to begin", f1.SeekToBegin(), 
        static_cast<File::FileSize>(0));
    AssertEqual("Reading 7 characters", f1.Read(readBuffer, 7),
        static_cast<File::FileSize>(7));
    AssertEqual("Read correct characters", 
        ::memcmp(readBuffer, writeBuffer, 7), 0);
    AssertEqual("Reading additionally 5 characters", f1.Read(readBuffer, 5),
        static_cast<File::FileSize>(5));
    AssertEqual("Read correct characters", 
        ::memcmp(readBuffer, writeBuffer + 7, 5), 0);

    AssertEqual("Seek to 1", f1.Seek(1, File::BEGIN), 
        static_cast<File::FileSize>(1));
    AssertEqual("Writing 5 characters to file", f1.Write(writeBuffer, 3), 
        static_cast<File::FileSize>(3));
    AssertEqual("Reading after new characters ", f1.Read(readBuffer, 1),
        static_cast<File::FileSize>(1));
    AssertEqual("Read 'e' after new characters", readBuffer[0], 'e');
    AssertEqual("Seek to begin", f1.SeekToBegin(), 
        static_cast<File::FileSize>(0));
    AssertEqual("Reading 5 characters", f1.Read(readBuffer, 5),
        static_cast<File::FileSize>(5));
    AssertEqual("New file @0 = 'a'", readBuffer[0], 'a');
    AssertEqual("New file @1 = 'a'", readBuffer[1], 'a');
    AssertEqual("New file @2 = 'b'", readBuffer[2], 'b');
    AssertEqual("New file @3 = 'c'", readBuffer[3], 'c');
    AssertEqual("New file @4 = 'e'", readBuffer[4], 'e');

    AssertEqual("Seek to begin", f1.SeekToBegin(), 
        static_cast<File::FileSize>(0));
    AssertEqual("Writing 9 characters to file", f1.Write(writeBuffer, 9), 
        static_cast<File::FileSize>(9));

    AssertEqual("Seek to begin", f1.SeekToBegin(), 
        static_cast<File::FileSize>(0));
    AssertEqual("Writing 5 characters to file", f1.Write(writeBuffer, 5), 
        static_cast<File::FileSize>(5));
    AssertEqual("Reading 4 after new characters ", f1.Read(readBuffer, 4),
        static_cast<File::FileSize>(4));
    AssertEqual("Read correct characters", 
        ::memcmp(readBuffer, writeBuffer + 5, 4), 0);

    f1.Close();
    AssertFalse("Closing file", f1.IsOpen());

    AssertTrue("Open file for reading", f1.Open("hugo.txt", 
        File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY));
    AssertEqual("File size", f1.GetSize(), static_cast<File::FileSize>(26)); 
    f1.Close();
    AssertFalse("Closing file", f1.IsOpen());

    File::Delete("hugo.txt");
    AssertFalse("\"hugo.txt\" was deleted", File::Exists("hugo.txt"));
}


void TestFile(void) {
    try {
        ::TestBaseFile();
        ::TestBufferedFile();
    } catch (IOException e) {
        std::cout << e.GetMsgA() << std::endl;
    }
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


void TestPath(void) {
    using namespace vislib;
    using namespace vislib::sys;
    using namespace std;

    try {
        cout << "Working directory \"" << static_cast<const char *>(Path::GetCurrentDirectoryA()) << "\"" << endl;
        cout << Path::Resolve("horst\\") << endl;

#ifdef _WIN32
        AssertEqual("Canonicalise \"horst\\..\\hugo\"", Path::Canonicalise("horst\\..\\hugo"), StringA("\\hugo"));
        AssertEqual("Canonicalise \"\\horst\\..\\hugo\"", Path::Canonicalise("\\horst\\..\\hugo"), StringA("\\hugo"));
        AssertEqual("Canonicalise \"\\..\\horst\\..\\hugo\"", Path::Canonicalise("\\..\\horst\\..\\hugo"), StringA("\\hugo"));
        AssertEqual("Canonicalise \"\\..\\horst\\..\\..\\hugo\"", Path::Canonicalise("\\..\\horst\\..\\..\\hugo"), StringA("\\hugo"));
        AssertEqual("Canonicalise \"horst\\.\\hugo\"", Path::Canonicalise("horst\\.\\hugo"), StringA("horst\\hugo"));
        AssertEqual("Canonicalise \"horst\\\\hugo\"", Path::Canonicalise("horst\\\\hugo"), StringA("horst\\hugo"));
        AssertEqual("Canonicalise \"horst\\\\\\hugo\"", Path::Canonicalise("horst\\\\\\hugo"), StringA("horst\\hugo"));
        AssertEqual("Canonicalise \"\\horst\\hugo\"", Path::Canonicalise("\\horst\\hugo"), StringA("\\horst\\hugo"));
        AssertEqual("Canonicalise \"\\\\horst\\hugo\"", Path::Canonicalise("\\\\horst\\hugo"), StringA("\\\\horst\\hugo"));
        AssertEqual("Canonicalise \"\\\\\\horst\\hugo\"", Path::Canonicalise("\\\\\\horst\\hugo"), StringA("\\\\horst\\hugo"));
#else /* _WIN32 */
        AssertEqual("Canonicalise \"horst/../hugo\"", Path::Canonicalise("horst/../hugo"), StringA("/hugo"));
        AssertEqual("Canonicalise \"/horst/../hugo\"", Path::Canonicalise("/horst/../hugo"), StringA("/hugo"));
        AssertEqual("Canonicalise \"/../horst/../hugo\"", Path::Canonicalise("/../horst/../hugo"), StringA("/hugo"));
        AssertEqual("Canonicalise \"/../horst/../../hugo\"", Path::Canonicalise("/../horst/../../hugo"), StringA("/hugo"));
        AssertEqual("Canonicalise \"horst/./hugo\"", Path::Canonicalise("horst/./hugo"), StringA("horst/hugo"));
        AssertEqual("Canonicalise \"horst//hugo\"", Path::Canonicalise("horst//hugo"), StringA("horst/hugo"));
        AssertEqual("Canonicalise \"horst///hugo\"", Path::Canonicalise("horst///hugo"), StringA("horst/hugo"));

#endif /* _WIN32 */
        
    } catch (SystemException e) {
        cout << e.GetMsgA() << endl;
    }
}
