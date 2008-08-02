/*
 * testfile.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testfile.h"

#include <iostream>

#include "testhelper.h"
#include "vislib/BufferedFile.h"
#include "vislib/MemmappedFile.h"
#include "vislib/error.h"
#include "vislib/File.h"
#include "vislib/Path.h"
#include "vislib/IOException.h"
#include "vislib/SystemMessage.h"
#include "vislib/sysfunctions.h"
#include "vislib/SystemInformation.h"
#include "vislib/PerformanceCounter.h"

#ifdef _WIN32
#pragma warning ( disable : 4996 )
#define SNPRINTF _snprintf
#define FORMATINT64 "%I64u"
#else /* _WIN32 */
#define SNPRINTF snprintf
#define FORMATINT64 "%lu"
#endif /* _WIN32 */

using namespace vislib::sys;

//static const File::FileSize BIGFILE_SIZE = 5368708992;
//static const File::FileSize BIGFILE_LASTVAL = 671088623;
static const File::FileSize BIGFILE_SIZE = 300000;
//static const File::FileSize BIGFILE_LASTVAL = 37499;
static const File::FileSize BIGFILE_LASTVAL = (BIGFILE_SIZE / 8) - 1;

static const char fname[] = "bigfile.bin";
static File::FileSize BUF_SIZE = SystemInformation::AllocationGranularity() * 2;

static void generateBigOne(File& f1) {
	File::FileSize dataLeft = BIGFILE_SIZE;
	File::FileSize splitPos = 0, val = 0, *tmp;
	char *buf = new char[static_cast<size_t>(BUF_SIZE)];
	vislib::sys::PerformanceCounter *p = new vislib::sys::PerformanceCounter();

	File::FileSize perf, ptotal = 0, numWritten;
	float minrate = 100000000.0f;
	float maxrate = 0.0f;
	float rate;

	if (File::Exists(fname)) {
		return;
	}
	AssertTrue("Can generate file", f1.Open(fname, File::WRITE_ONLY, File::SHARE_READWRITE, File::CREATE_ONLY));
	AssertTrue("AllocationGranularity is a multiple of 8", SystemInformation::AllocationGranularity() % 8 == 0);

	try {
		while(dataLeft > 0) {
			for (tmp = (File::FileSize*)buf; tmp < (File::FileSize*)(buf + BUF_SIZE); tmp++) {
				*tmp = val++;
			}
			p->SetMark();
			if (dataLeft < BUF_SIZE) {
				f1.Write(buf, dataLeft);
				numWritten = dataLeft;
				dataLeft = 0;
			} else {
				f1.Write(buf, splitPos);
				f1.Write(buf + splitPos, BUF_SIZE - splitPos);
				dataLeft -= BUF_SIZE;
				numWritten = BUF_SIZE;
			}
			perf = p->Difference();
			ptotal += perf;
			rate = (float)numWritten/1024.0f / perf;
			if (rate < minrate) minrate = rate;
			if (rate > maxrate) maxrate = rate;
			splitPos = (++splitPos) % BUF_SIZE;
		}
		AssertTrue("Generate big testfile", true);
		SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "Linear writing: %03.1f MB/s min, %03.1f MB/s max, %03.1f MB/s average\n", minrate, maxrate, f1.GetSize() / 1024.0f / ptotal);
		std::cout << buf;
		f1.Close();
	} catch (IOException e) {
		AssertTrue("Generate big testfile", false);
		std::cout << e.GetMsgA() << std::endl;
	}
	f1.Open(fname, File::READ_ONLY, File::SHARE_READWRITE, File::OPEN_ONLY);
	SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "File size = "FORMATINT64, BIGFILE_SIZE);
	AssertEqual(buf, BIGFILE_SIZE, f1.GetSize());
	f1.Close();
}

static File::FileSize checkContent(char *buf, File::FileSize pos, File::FileSize chunk) {
	char *offset;
	File::FileSize i, ret;

	for (offset = buf, i = pos / 8; offset < buf + chunk; offset += 8, i++) {
        ret = *(reinterpret_cast<File::FileSize*>(offset)); // for better debugging
		if (ret != i) {
			AssertTrue("File contents comparison", false);
		}
		ret = i;
	}
	return ret;
}

static void testBigOne(File& f1) {
	vislib::sys::PerformanceCounter *p = new vislib::sys::PerformanceCounter();
	char *buf = new char[static_cast<size_t>(BUF_SIZE)];
	File::FileSize numRead, perf, ptotal, pos, lastval;
	float minrate = 100000000.0f;
	float maxrate = 0.0f;
	float rate;

	f1.Open(fname, File::READ_ONLY, File::SHARE_EXCLUSIVE, File::OPEN_ONLY);

	pos = ptotal = 0;
	try {
		p->SetMark();
		while((numRead = f1.Read(buf, BUF_SIZE)) == BUF_SIZE) {
			perf = p->Difference();
			ptotal += perf;
			rate = (float)numRead/1024.0f / perf;
			if (rate < minrate) minrate=rate;
			if (rate > maxrate) maxrate=rate;

			lastval = checkContent(buf, pos, numRead);
			pos += numRead;

			p->SetMark();
		}
		if (numRead > 0) {
			perf = p->Difference();
			ptotal += perf;
			rate = (float)numRead/1024.0f / perf;
			if (rate < minrate) minrate=rate;
			if (rate > maxrate) maxrate=rate;
			lastval = checkContent(buf, pos, numRead);
			pos += numRead;
		}
	} catch (IOException e) {
		SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "Reading successful ("FORMATINT64")", pos);
		AssertTrue(buf, false);
	}
	SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "Values consistent up to pos "FORMATINT64, BIGFILE_SIZE);
	AssertEqual(buf, pos, BIGFILE_SIZE);
	SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "Last value = "FORMATINT64, BIGFILE_LASTVAL);
	AssertEqual(buf, lastval, BIGFILE_LASTVAL);
	SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "Linear reading: %03.1f MB/s min, %03.1f MB/s max, %03.1f MB/s average\n", minrate, maxrate, f1.GetSize()/1024.0f / ptotal);
	std::cout << buf;
	f1.Close();
    delete[] buf;
}

static void removeBigOne(void) {
    AssertTrue("Delete test file", vislib::sys::File::Delete(fname));
}

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

    try {
        f1.Write(readBuffer, TEXT_SIZE);
        AssertTrue("Cannot write on READ_ONLY file", false);
    } catch (IOException e) {
        AssertTrue("Cannot write on READ_ONLY file", true);
        std::cout << e.GetMsgA() << std::endl;
    }

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
    AssertEqual("Writing 3 characters to file", f1.Write(writeBuffer, 3), 
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

	AssertTrue("All tests complete", 1);
}


void TestFile(void) {
    AssertEqual("TestFileSize % 8 == 0", static_cast<INT64>(BIGFILE_SIZE) % static_cast<INT64>(8), static_cast<INT64>(0));

    try {
        ::TestBaseFile();
        ::TestBufferedFile();
		::TestMemmappedFile();
    } catch (IOException e) {
        std::cout << e.GetMsgA() << std::endl;
    }
}


void TestBaseFile(void) {
    File f1;
    std::cout << std::endl << "Tests for File" << std::endl;
    ::runTests(f1);

	::generateBigOne(f1);
	::testBigOne(f1);
    ::removeBigOne();
}


void TestBufferedFile(void) {
    BufferedFile f1;
    std::cout << std::endl << "Tests for BufferedFile" << std::endl;
    ::runTests(f1);

    f1.SetBufferSize(8);
    std::cout << std::endl << "Tests for BufferedFile, buffer size 8" << std::endl;
    ::runTests(f1);

	::generateBigOne(f1);
	::testBigOne(f1);	
    ::removeBigOne();
}

void TestMemmappedFile(void) {
	MemmappedFile f1;
	std::cout << std::endl << "Tests for MemmappedFile" << std::endl;
	::runTests(f1);
	
	::generateBigOne(f1);
	char *buf = new char[static_cast<size_t>(BUF_SIZE)];

	SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "Testing with views of "FORMATINT64" (machine default)\n", f1.GetViewSize());
	std::cout << buf;
	::testBigOne(f1);

	f1.SetViewSize(8 * 1024 * 1024);
	SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "Testing with views of "FORMATINT64"\n", f1.GetViewSize());
	std::cout << buf;
	::testBigOne(f1);
    ::removeBigOne();
}


void TestPath(void) {
    using namespace vislib;
    using namespace vislib::sys;
    using namespace std;

    try {
        cout << "Working directory \"" << static_cast<const char *>(Path::GetCurrentDirectoryA()) << "\"" << endl;

        cout << "Resolve \"~\" " << Path::Resolve("~") << endl;
        cout << "Resolve \"~/\" " << Path::Resolve("~/") << endl;
        cout << "Resolve \"~/heinz\" " << Path::Resolve("~/heinz") << endl;
        cout << "Resolve \"~heinz\" " << Path::Resolve("~heinz") << endl;
        cout << "Resolve \"./~\" " << Path::Resolve("./~") << endl;

        cout << "Resolve \"horst\" " << Path::Resolve("horst") << endl;
        cout << "Resolve \"/horst\" " << Path::Resolve("/horst") << endl;
        cout << "Resolve \"/horst/\" " << Path::Resolve("/horst/") << endl;
        cout << "Resolve \"//horst/\" " << Path::Resolve("//horst/") << endl;

#ifdef _WIN32
        cout << "Resolve \"horst\" " << Path::Resolve("horst") << endl;
        cout << "Resolve \"\\horst\" " << Path::Resolve("\\horst") << endl;
        cout << "Resolve \"C:\\horst\" " << Path::Resolve("C:\\horst") << endl;
        cout << "Resolve \"\\horst\\\" " << Path::Resolve("\\horst\\") << endl;
        cout << "Resolve \"\\\\horst\\\" " << Path::Resolve("\\\\horst\\") << endl;
        cout << "Resolve \"~\\\" " << Path::Resolve("~\\") << endl;
        cout << "Resolve \"~\\heinz\" " << Path::Resolve("~\\heinz") << endl;
        cout << "Resolve \"~heinz\" " << Path::Resolve("~heinz") << endl;
        cout << "Resolve \".\\~\" " << Path::Resolve(".\\~") << endl;

        AssertTrue("Path \"\\hugo\" is relative", Path::IsRelative("\\hugo"));
        AssertTrue("Path \"\\\\hugo\" is absolute", Path::IsAbsolute("\\\\hugo"));

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

        // testing compare methods
#ifdef _WIN32
        AssertTrue("Compare \"C:\\horst\\\" == \"C:\\..\\HoRsT\\\"", Path::Compare<vislib::CharTraitsA>("C:\\horst\\", "C:\\..\\HoRsT\\"));
        AssertTrue("Compare \"\\horst\\\" == \"\\..\\HoRsT\\\"", Path::Compare<vislib::CharTraitsA>("\\horst\\", "\\..\\HoRsT\\"));
        AssertFalse("Compare \"Path::Resolve(hugo)\" != \"\\hugo\"", Path::Compare<vislib::CharTraitsA>(Path::Resolve("hugo"), "\\hugo"));
        AssertTrue("Compare \"Path::Resolve(hugo)\" == \"hugo\"", Path::Compare<vislib::CharTraitsA>(Path::Resolve("hugo"), "hugo"));

        AssertTrue("Compare \"C:\\horst\\\" == \"C:\\..\\HoRsT\\\"", Path::Compare<vislib::CharTraitsW>(L"C:\\horst\\", L"C:\\..\\HoRsT\\"));
        AssertTrue("Compare \"\\horst\\\" == \"\\..\\HoRsT\\\"", Path::Compare<vislib::CharTraitsW>(L"\\horst\\", L"\\..\\HoRsT\\"));
        AssertFalse("Compare \"Path::Resolve(hugo)\" != \"\\hugo\"", Path::Compare<vislib::CharTraitsW>(Path::Resolve(L"hugo"), L"\\hugo"));
        AssertTrue("Compare \"Path::Resolve(hugo)\" == \"hugo\"", Path::Compare<vislib::CharTraitsW>(Path::Resolve(L"hugo"), L"hugo"));
#else /* _WIN32 */
        AssertTrue("Compare \"/horst/\" == \"/../horst/\"", Path::Compare<vislib::CharTraitsA>("/horst/", "/../horst/"));
        AssertFalse("Compare \"/horst\" != \"/Horst\"", Path::Compare<vislib::CharTraitsA>("/horst", "/Horst"));
        AssertFalse("Compare \"Path::Resolve(hugo)\" != \"/hugo\"", Path::Compare<vislib::CharTraitsA>(Path::Resolve("hugo"), "/hugo"));
        AssertTrue("Compare \"Path::Resolve(hugo)\" == \"hugo\"", Path::Compare<vislib::CharTraitsA>(Path::Resolve("hugo"), "hugo"));

        AssertTrue("Compare \"/horst/\" == \"/../horst/\"", Path::Compare<vislib::CharTraitsW>(L"/horst/", L"/../horst/"));
        AssertFalse("Compare \"/horst\" != \"/Horst\"", Path::Compare<vislib::CharTraitsW>(L"/horst", L"/Horst"));
        AssertFalse("Compare \"Path::Resolve(hugo)\" != \"/hugo\"", Path::Compare<vislib::CharTraitsW>(Path::Resolve(L"hugo"), L"/hugo"));
        AssertTrue("Compare \"Path::Resolve(hugo)\" == \"hugo\"", Path::Compare<vislib::CharTraitsW>(Path::Resolve(L"hugo"), L"hugo"));
#endif /* _WIN32 */
        
    } catch (SystemException e) {
        cout << e.GetMsgA() << endl;
    }

#ifdef _WIN32
    vislib::StringA p = vislib::sys::Path::FindExecutablePath("cmd.exe");
    cout << "Executable found at: " << p.PeekBuffer() << endl;
    p = vislib::sys::Path::FindExecutablePath("notepad.exe");
    cout << "Executable found at: " << p.PeekBuffer() << endl;
    p = vislib::sys::Path::FindExecutablePath("iexplore.exe");
    cout << "Executable found at: " << p.PeekBuffer() << endl;
    p = vislib::sys::Path::FindExecutablePath("calc.exe");
    cout << "Executable found at: " << p.PeekBuffer() << endl;
    p = vislib::sys::Path::FindExecutablePath("subwcrev.exe");
    cout << "Executable found at: " << p.PeekBuffer() << endl;
    p = vislib::sys::Path::FindExecutablePath("test.exe");
    cout << "Executable found at: " << p.PeekBuffer() << endl;
    vislib::StringW w = vislib::sys::Path::FindExecutablePath(L"cmd.exe");
    cout << "Executable found at: " << vislib::StringA(w).PeekBuffer() << endl;
    w = vislib::sys::Path::FindExecutablePath(L"notepad.exe");
    cout << "Executable found at: " << vislib::StringA(w).PeekBuffer() << endl;
    w = vislib::sys::Path::FindExecutablePath(L"iexplore.exe");
    cout << "Executable found at: " << vislib::StringA(w).PeekBuffer() << endl;
    w = vislib::sys::Path::FindExecutablePath(L"calc.exe");
    cout << "Executable found at: " << vislib::StringA(w).PeekBuffer() << endl;
    w = vislib::sys::Path::FindExecutablePath(L"subwcrev.exe");
    cout << "Executable found at: " << vislib::StringA(w).PeekBuffer() << endl;
    w = vislib::sys::Path::FindExecutablePath(L"test.exe");
    cout << "Executable found at: " << vislib::StringA(w).PeekBuffer() << endl;
#else /* _WIN32 */
    vislib::StringA p = vislib::sys::Path::FindExecutablePath("xterm");
    cout << "Executable found at: " << p.PeekBuffer() << endl;
    p = vislib::sys::Path::FindExecutablePath("bash");
    cout << "Executable found at: " << p.PeekBuffer() << endl;
#endif /* _WIN32 */

}
