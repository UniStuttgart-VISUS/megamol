/*
 * testdate.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testdate.h"
#include "testhelper.h"

#include <iostream>

#include "vislib/DateTime.h"


static void DateConversionTest(const INT year, const INT month, const INT day,
        const INT hour, const INT minute, const INT second) {
    INT outYear, outMonth, outDay, outHour, outMinute, outSecond;

    std::cout << "Testing " << day << "." << month << "." << year << " "
        << hour << ":" << minute << ":" << second << " ..." << std::endl;
    vislib::sys::DateTime date(year, month, day, hour, minute, second);
    date.Get(outYear, outMonth, outDay, outHour, outMinute, outSecond);
    std::cout << "Got " << outDay << "." << outMonth << "." << outYear << " "
        << outHour << ":" << outMinute << ":" << outSecond << "." << std::endl;
    AssertEqual("Set and get year", year, outYear);
    AssertEqual("Set and get month", month, outMonth);
    AssertEqual("Set and get day", day, outDay);
    AssertEqual("Set and get hour", hour, outHour);
    AssertEqual("Set and get minute", minute, outMinute);
    AssertEqual("Set and get second", second, outSecond);
}

static void DateValueTest(const INT year, const INT month, const INT day,
        const INT hour, const INT minute, const INT second, 
        const INT64 expectedValue) {
    std::cout << "Testing " << day << "." << month << "." << year << " "
        << hour << ":" << minute << ":" << second << ", expected value is "
        << expectedValue << " ..." << std::endl;
    vislib::sys::DateTime date(year, month, day, hour, minute, second);
    std::cout << "Internal representation is " << date.GetValue() << "." << std::endl;
    AssertEqual("Internal date representation", date.GetValue(), expectedValue);
}


/*
 * ::TestDateTime
 */
void TestDateTime(void)  {
    vislib::sys::DateTime dateTime;
    INT year, month, day, hour, minute, second;

    ::EnableAssertSuccessOutput(false);

    ::DateValueTest(1, 1, 1, 0, 0, 0, INT64(0));
    ::DateValueTest(1, 1, 1, 0, 0, 1, INT64(1000));
    ::DateValueTest(1, 1, 1, 0, 1, 0, INT64(60) * 1000);
    ::DateValueTest(1, 1, 1, 1, 0, 0, INT64(60) * 60 * 1000);
    ::DateValueTest(1, 1, 2, 0, 0, 0, INT64(24) * 60 * 60 * 1000);
    ::DateValueTest(1, 2, 1, 0, 0, 0, INT64(31) * 24 * 60 * 60 * 1000);
    ::DateValueTest(1, 2, 2, 0, 0, 0, INT64(31 + 1) * 24 * 60 * 60 * 1000);
    ::DateValueTest(1, 12, 31, 0, 0, 0, INT64(364) * 24 * 60 * 60 * 1000);
    ::DateValueTest(1, 12, 31, 23, 59, 59, INT64(365) * 24 * 60 * 60 * 1000 - 1000);
    ::DateValueTest(2, 1, 1, 0, 0, 0, INT64(365) * 24 * 60 * 60 * 1000);
    ::DateValueTest(3, 1, 1, 0, 0, 0, INT64(2) * 365 * 24 * 60 * 60 * 1000);
    ::DateValueTest(4, 1, 1, 0, 0, 0, INT64(3) * 365 * 24 * 60 * 60 * 1000);
    ::DateValueTest(4, 2, 28, 0, 0, 0, (INT64(3 * 365) + 31 + 27) * 24 * 60 * 60 * 1000);
    ::DateValueTest(4, 2, 29, 0, 0, 0, (INT64(3 * 365) + 31 + 28) * 24 * 60 * 60 * 1000);
    ::DateValueTest(4, 3, 1, 0, 0, 0, (INT64(3 * 365) + 31 + 29) * 24 * 60 * 60 * 1000);
    ::DateValueTest(-1, 12, 31, 23, 59, 59, INT64(-1000));
    ::DateValueTest(-1, 12, 31, 0, 0, 0, INT64(-1) * 24 * 60 * 60 * 1000);
    ::DateValueTest(-1, 1, 1, 0, 0, 0, INT64(-1) * 365 * 24 * 60 * 60 * 1000);
    ::DateValueTest(-1, 1, 1, 0, 0, 1, INT64(-1) * 365 * 24 * 60 * 60 * 1000 + 1000);
    ::DateValueTest(-1, 1, 1, 0, 1, 0, INT64(-1) * 365 * 24 * 60 * 60 * 1000 + 60 * 1000);
    ::DateValueTest(-1, 1, 1, 1, 0, 0, INT64(-1) * 365 * 24 * 60 * 60 * 1000 + 60 * 60 * 1000);
    ::DateValueTest(2, -1, 1, 0, 0, 0, INT64(365 - 31) * 24 * 60 * 60 * 1000);
    ::DateValueTest(2, 1, -1, 0, 0, 0, INT64(365 - 1) * 24 * 60 * 60 * 1000);
    ::DateValueTest(2, -12, 1, 0, 0, 0, INT64(0));
    ::DateValueTest(1, -1, 1, 0, 0, 0, INT64(-1) * 31 * 24 * 60 * 60 * 1000);
    ::DateValueTest(-1, 13, 1, 0, 0, 0, INT64(0));
    ::DateValueTest(1, 13, 1, 0, 0, 0, INT64(365) * 24 * 60 * 60 * 1000);
    ::DateConversionTest(1, 1, 1, 0, 0, 0);
    ::DateConversionTest(1, 1, 1, 0, 0, 1);
    ::DateConversionTest(1, 1, 1, 0, 1, 1);
    ::DateConversionTest(1, 1, 1, 1, 1, 1);
    ::DateConversionTest(3, 1, 1, 0, 0, 0);
    ::DateConversionTest(4, 1, 1, 0, 0, 0);
    ::DateConversionTest(5, 1, 1, 0, 0, 0);
    ::DateConversionTest(99, 1, 1, 0, 0, 0);
    ::DateConversionTest(100, 1, 1, 0, 0, 0);
    ::DateConversionTest(101, 1, 1, 0, 0, 0);
    ::DateConversionTest(399, 1, 1, 0, 0, 0);
    ::DateConversionTest(400, 1, 1, 0, 0, 0);
    ::DateConversionTest(401, 1, 1, 0, 0, 0);
    ::DateConversionTest(1700, 1, 1, 0, 0, 0);
    ::DateConversionTest(1700, 1, 1, 12, 34, 56);
    ::DateConversionTest(1704, 1, 1, 0, 0, 0);
    ::DateConversionTest(1704, 2, 29, 0, 0, 0);
    ::DateConversionTest(1704, 3, 1, 0, 0, 0);
    ::DateConversionTest(1704, 3, 2, 0, 0, 0);
    ::DateConversionTest(1705, 1, 1, 0, 0, 0);
    ::DateConversionTest(1705, 2, 28, 0, 0, 0);
    ::DateConversionTest(1705, 3, 1, 0, 0, 0);
    ::DateConversionTest(1705, 3, 2, 0, 0, 0);
    ::DateConversionTest(1708, 1, 1, 0, 0, 0);
    ::DateConversionTest(1708, 3, 1, 0, 0, 0);
    ::DateConversionTest(1709, 1, 1, 0, 0, 0);
    ::DateConversionTest(1709, 3, 1, 0, 0, 0);
    ::DateConversionTest(1900, 1, 1, 5, 32, 35);
    ::DateConversionTest(1900, 2, 1, 5, 32, 35);
    ::DateConversionTest(1900, 3, 1, 5, 32, 35);
    ::DateConversionTest(1904, 3, 1, 5, 32, 35);
    ::DateConversionTest(1999, 3, 5, 1, 4, 55);
    ::DateConversionTest(2000, 1, 4, 12, 54, 22);
    ::DateConversionTest(2000, 2, 29, 12, 0, 0);
    ::DateConversionTest(2000, 3, 1, 5, 22, 33);
    ::DateConversionTest(2000, 12, 31, 5, 22, 33);
    ::DateConversionTest(2000, 12, 3, 0, 32, 35);
    ::DateConversionTest(2004, 1, 1, 5, 32, 35);
    ::DateConversionTest(2004, 2, 1, 5, 32, 35);
    ::DateConversionTest(2004, 3, 1, 5, 32, 35);

    ::DateConversionTest(-1, 1, 1, 0, 0, 0);
    ::DateConversionTest(-1, 1, 1, 0, 0, 1);
    ::DateConversionTest(-3, 1, 1, 0, 0, 0);
    ::DateConversionTest(-4, 1, 1, 0, 0, 0);
    ::DateConversionTest(-5, 1, 1, 0, 0, 0);
    ::DateConversionTest(-99, 1, 1, 0, 0, 0);
    ::DateConversionTest(-100, 1, 1, 0, 0, 0);
    ::DateConversionTest(-101, 1, 1, 0, 0, 0);
    ::DateConversionTest(-399, 1, 1, 0, 0, 0);
    ::DateConversionTest(-400, 1, 1, 0, 0, 0);
    ::DateConversionTest(-401, 1, 1, 0, 0, 0);
    ::DateConversionTest(-1700, 1, 1, 0, 0, 0);
    ::DateConversionTest(-1700, 1, 1, 12, 34, 56);
    ::DateConversionTest(-2004, 1, 1, 5, 32, 35);
    ::DateConversionTest(-2004, 2, 1, 5, 1, 00);
    ::DateConversionTest(-2004, 3, 1, 5, 00, 35);
    ::DateConversionTest(-2004, 12, 31, 0, 0, 1);
    ::DateConversionTest(-2005, 1, 1, 5, 32, 35);
    ::DateConversionTest(-2005, 2, 1, 5, 1, 00);
    ::DateConversionTest(-2005, 3, 1, 5, 00, 35);
    ::DateConversionTest(-2005, 12, 31, 0, 0, 0);
    ::DateConversionTest(-2005, 12, 31, 0, 0, 1);
    ::DateConversionTest(-1, 12, 31, 23, 59, 59);


    //time_t unixTimeStamp = ::time(NULL);
    //vislib::sys::DateTime unixDateTime(unixTimeStamp);
    //::AssertEqual("Unix timestamp", unixTimeStamp, static_cast<time_t>(unixDateTime));

#ifdef _WIN32
    //SYSTEMTIME systemTime;
    //::GetSystemTime(&systemTime);
    //vislib::sys::DateTime systemTimeDateTime(systemTime);
    //::AssertEqual("SYSTEMTIME", systemTime, static_cast<SYSTEMTIME>(systemTimeDateTime));

    //FILETIME fileTime;
    //::SystemTimeToFileTime(&systemTime, &fileTime);
    //vislib::sys::DateTime fileTimeDateTime(fileTime);
    //::AssertEqual("SYSTEMTIME", fileTime, static_cast<FILETIME>(fileTimeDateTime));

#endif /* _WIN32 */

    //dateTime.Set(0, 0, 0, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year 0 converted to 1", year, 1);
    //::AssertEqual("Month 0 converted to 1", month, 1);
    //::AssertEqual("Day 0 converted to 1", day, 1);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);

    //dateTime.Set(1, 13, 1, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year 1 rolled to 1", year,2);
    //::AssertEqual("Month 13 rolled to 1", month, 1);
    //::AssertEqual("Day 1 remains 1", day, 1);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);

    //dateTime.Set(0, 13, 1, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year 0 rolled to 2", year, 2);
    //::AssertEqual("Month 13 rolled to 1", month, 1);
    //::AssertEqual("Day 1 remains 1", day, 1);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);

    //dateTime.Set(0, 14, 1, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year 0 rolled to 2", year, 2);
    //::AssertEqual("Month 14 rolled to 2", month, 2);
    //::AssertEqual("Day 1 remains 1", day, 1);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);

    //dateTime.Set(1, 14, 30, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year 1 rolled to 2", year, 2);
    //::AssertEqual("Month 14 rolled to 3", month, 3);
    //::AssertEqual("Day 30 rolled to 2", day, 2);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);

    //dateTime.Set(1, 2, -1, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year remains 1", year, 1);
    //::AssertEqual("Month 2 rolled to 1", month, 1);
    //::AssertEqual("Day -1 rolled to 31", day, 31);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);

    //dateTime.Set(1999, 3, -2, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year remains 1999", year, 1999);
    //::AssertEqual("Month 3 rolled to 2", month, 2);
    //::AssertEqual("Day -2 rolled to 27", day, 27);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);

    //dateTime.Set(1999, -1, 1, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year rolled to 1998", year, 1998);
    //::AssertEqual("Month -1 rolled to 12", month, 12);
    //::AssertEqual("Day remains 1", day, 1);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);

    //dateTime.Set(1999, -12, 1, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year rolled to 1998", year, 1998);
    //::AssertEqual("Month -12 rolled to 1", month, 1);
    //::AssertEqual("Day remains 1", day, 1);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);

    //dateTime.Set(1999, -13, 1, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year rolled to 1997", year, 1997);
    //::AssertEqual("Month -13 rolled to 12", month, 12);
    //::AssertEqual("Day remains 1", day, 1);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);

    //dateTime.Set(-1, 13, 1, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year rolled to 1", year, 1);
    //::AssertEqual("Month 13 rolled to 1", month, 1);
    //::AssertEqual("Day remains 1", day, 1);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);

    //dateTime.Set(1, -1, 1, 0, 0, 0);
    //dateTime.Get(year, month, day, hour, minute, second);
    //::AssertEqual("Year rolled to -1", year, -1);
    //::AssertEqual("Month -1 rolled to 12", month, 12);
    //::AssertEqual("Day remains 1", day, 1);
    //::AssertEqual("Hour remains 0", hour, 0);
    //::AssertEqual("Minute remains 0", minute, 0);
    //::AssertEqual("Second remains 0", second, 0);
}
