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
    //AssertEqual("Set and get day", day, outDay);
    AssertEqual("Set and get hour", hour, outHour);
    AssertEqual("Set and get minute", minute, outMinute);
    AssertEqual("Set and get second", second, outSecond);
}

static void DateValueTest(const INT year, const INT month, const INT day,
        const INT hour, const INT minute, const INT second, const INT millis,
        const INT64 expectedValue) {
    std::cout << "Testing " << day << "." << month << "." << year << " "
        << hour << ":" << minute << ":" << second << "." << millis 
        << ", expected value is " << expectedValue << " ..." << std::endl;
    vislib::sys::DateTime date(year, month, day, hour, minute, second, millis);
    std::cout << "Internal representation is " << date.GetValue() << "." << std::endl;
    AssertEqual("Internal date representation", date.GetValue(), expectedValue);
}


static void TestSpan(void) {
    using vislib::sys::DateTimeSpan;
    DateTimeSpan ts1, ts2;

    ::AssertEqual("Value of MILLISECONDS_PER_SECOND.", DateTimeSpan::MILLISECONDS_PER_SECOND, (INT64) 1000);
    ::AssertEqual("Value of MILLISECONDS_PER_MINUTE.", DateTimeSpan::MILLISECONDS_PER_MINUTE, (INT64) 60 * 1000);
    ::AssertEqual("Value of MILLISECONDS_PER_HOUR.", DateTimeSpan::MILLISECONDS_PER_HOUR, (INT64) 60 * 60 * 1000);
    ::AssertEqual("Value of MILLISECONDS_PER_DAY.", DateTimeSpan::MILLISECONDS_PER_DAY, (INT64) 24 * 60 * 60 * 1000);
    ::AssertEqual("Empty time span is 0 millis.", (INT64) DateTimeSpan::EMPTY, (INT64) 0);
    ::AssertEqual("Value of ONE_MILLISECOND.", (INT64) DateTimeSpan::ONE_MILLISECOND, (INT64) 1);
    ::AssertEqual("Value of ONE_SECOND.", (INT64) DateTimeSpan::ONE_SECOND, (INT64) 1000);
    ::AssertEqual("Value of ONE_MINUTE.", (INT64) DateTimeSpan::ONE_MINUTE, (INT64) 60 * 1000);
    ::AssertEqual("Value of ONE_HOUR.", (INT64) DateTimeSpan::ONE_HOUR, (INT64) 60 * 60 * 1000);
    ::AssertEqual("Value of ONE_DAY.", (INT64) DateTimeSpan::ONE_DAY, (INT64) 24 * 60 * 60 * 1000);

    ::AssertEqual("Initialisation with default ctor.", (INT64) DateTimeSpan(), (INT64) 0);
    ::AssertEqual("Initialisation with millis.", (INT64) DateTimeSpan(10), (INT64) 10);
    ::AssertEqual("Initialisation with all members except millis.", (INT64) DateTimeSpan(1, 1, 1, 1), 
        1 * DateTimeSpan::MILLISECONDS_PER_DAY 
        + 1 * DateTimeSpan::MILLISECONDS_PER_HOUR 
        + 1 * DateTimeSpan::MILLISECONDS_PER_MINUTE
        + 1 * DateTimeSpan::MILLISECONDS_PER_SECOND);
    ::AssertEqual("Initialisation with all members.", (INT64) DateTimeSpan(1, 1, 1, 1, 1),
        1 * DateTimeSpan::MILLISECONDS_PER_DAY 
        + 1 * DateTimeSpan::MILLISECONDS_PER_HOUR 
        + 1 * DateTimeSpan::MILLISECONDS_PER_MINUTE
        + 1 * DateTimeSpan::MILLISECONDS_PER_SECOND 
        + 1);

    ts1 = DateTimeSpan(1, 1, 1, 1, 1);
    ::AssertEqual("ToStringA", ts1.ToStringA(), vislib::StringA("1:01:01:01.0001"));
    ::AssertEqual("ToStringW", ts1.ToStringW(), vislib::StringW(L"1:01:01:01.0001"));
    ::AssertFalse("Test for equality returns false.", ts1 == ts2);
    ::AssertTrue("Test for inequality returns true.", ts1 != ts2);
    ::AssertEqual("GetDays().", ts1.GetDays(), (INT64) 1);
    ::AssertEqual("GetHours().", ts1.GetHours(), (INT64) 1);
    ::AssertEqual("GetMinutes().", ts1.GetMinutes(), (INT64) 1);
    ::AssertEqual("GetSeconds().", ts1.GetSeconds(), (INT64) 1);
    ::AssertEqual("GetMilliseconds().", ts1.GetMilliseconds(), (INT64) 1);

    ts2 = ts1;
    ::AssertEqual("Assignment succeeds.", (INT64) ts1, (INT64) ts2);
    
    ::AssertTrue("Test for equality returns true.", ts1 == ts2);
    ::AssertFalse("Test for inequality returns false.", ts1 != ts2);

    ts1 = ts2 + DateTimeSpan::ONE_MILLISECOND;
    ::AssertEqual("Addition.", (INT64) ts1, (INT64) ts2 + 1);

    ts1 = ts2 - DateTimeSpan::ONE_MILLISECOND;
    ::AssertEqual("Subtraction.", (INT64) ts1, (INT64) ts2 - 1);

    ts1 = ts2;
    ts1 += DateTimeSpan::ONE_MILLISECOND;
    ::AssertEqual("Addition assigment.", (INT64) ts1, (INT64) ts2 + 1);
    ::AssertTrue("operator >.", ts1 > ts2);
    ::AssertTrue("operator >=.", ts1 >= ts2);
    ::AssertFalse("operator <.", ts1 < ts2);
    ::AssertFalse("operator <=.", ts1 <= ts2);

    ts1 = ts2;
    ts1 -= DateTimeSpan::ONE_MILLISECOND;
    ::AssertEqual("Subtraction assigment.", (INT64) ts1, (INT64) ts2 - 1);
    ::AssertFalse("operator >.", ts1 > ts2);
    ::AssertFalse("operator >=.", ts1 >= ts2);
    ::AssertTrue("operator <.", ts1 < ts2);
    ::AssertTrue("operator <=.", ts1 <= ts2);

    ts1 = ts2;
    ::AssertFalse("operator > with equal value.", ts1 > ts2);
    ::AssertTrue("operator >= with equal value.", ts1 >= ts2);
    ::AssertFalse("operator < with equal value.", ts1 < ts2);
    ::AssertTrue("operator <= with equal value.", ts1 <= ts2);

    ::AssertEqual("Negation.", (INT64) -ts1, -((INT64) ts1));

    ts2.Set(10, 0, 0, 0, 0);
    ::AssertEqual("GetDays() == 10.", ts2.GetDays(), (INT64) 10);
    ::AssertEqual("GetHours() == 0.", ts2.GetHours(), (INT64) 0);
    ::AssertEqual("GetMinutes() == 0.", ts2.GetMinutes(), (INT64) 0);
    ::AssertEqual("GetSeconds() == 0.", ts2.GetSeconds(), (INT64) 0);
    ::AssertEqual("GetMilliseconds() == 0.", ts2.GetMilliseconds(), (INT64) 0);

    ts2.Set(-10, 0, 0, 0, 0);
    ::AssertEqual("GetDays() == -10.", ts2.GetDays(), (INT64) -10);
    ::AssertEqual("GetHours() == 0.", ts2.GetHours(), (INT64) 0);
    ::AssertEqual("GetMinutes() == 0.", ts2.GetMinutes(), (INT64) 0);
    ::AssertEqual("GetSeconds() == 0.", ts2.GetSeconds(), (INT64) 0);
    ::AssertEqual("GetMilliseconds() == 0.", ts2.GetMilliseconds(), (INT64) 0);

    ts2.Set(-10, 1, 0, 0, 0);
    ::AssertEqual("GetDays() == -9.", ts2.GetDays(), (INT64) -9);
    ::AssertEqual("GetHours() == -23.", ts2.GetHours(), (INT64) -23);
    ::AssertEqual("GetMinutes() == 0.", ts2.GetMinutes(), (INT64) 0);
    ::AssertEqual("GetSeconds() == 0.", ts2.GetSeconds(), (INT64) 0);
    ::AssertEqual("GetMilliseconds() == 0.", ts2.GetMilliseconds(), (INT64) 0);

    ts2.Set(0, 0, 0, -1, 0);
    ::AssertEqual("GetDays() == 0.", ts2.GetDays(), (INT64) 0);
    ::AssertEqual("GetHours() == 0.", ts2.GetHours(), (INT64) 0);
    ::AssertEqual("GetMinutes() == 0.", ts2.GetMinutes(), (INT64) 0);
    ::AssertEqual("GetSeconds() == -1.", ts2.GetSeconds(), (INT64) -1);
    ::AssertEqual("GetMilliseconds() == 0.", ts2.GetMilliseconds(), (INT64) 0);

    ts2.Set(0, 0, 0, 0, 1001);
    ::AssertEqual("GetDays() == 0.", ts2.GetDays(), (INT64) 0);
    ::AssertEqual("GetHours() == 0.", ts2.GetHours(), (INT64) 0);
    ::AssertEqual("GetMinutes() == 0.", ts2.GetMinutes(), (INT64) 0);
    ::AssertEqual("GetSeconds() == 1.", ts2.GetSeconds(), (INT64) 1);
    ::AssertEqual("GetMilliseconds() == 1.", ts2.GetMilliseconds(), (INT64) 1);

    ts2.Set(0, 0, 0, 0, -1001);
    ::AssertEqual("GetDays() == 0.", ts2.GetDays(), (INT64) 0);
    ::AssertEqual("GetHours() == 0.", ts2.GetHours(), (INT64) 0);
    ::AssertEqual("GetMinutes() == 0.", ts2.GetMinutes(), (INT64) 0);
    ::AssertEqual("GetSeconds() == -1.", ts2.GetSeconds(), (INT64) -1);
    ::AssertEqual("GetMilliseconds() == -1.", ts2.GetMilliseconds(), (INT64) -1);
}


static void TestTime(void) {
    using vislib::sys::DateTime;
    using vislib::sys::DateTimeSpan;
    DateTime dateTime;
    INT year, month, day, hour, minute, second, millis;
    time_t unixTimeStamp;

    ::AssertEqual("DateTime::EMPTY internal data.", DateTime::EMPTY.GetValue(), INT64(0));

    ::DateValueTest(0, 0, 0, 0, 0, 0, 0, INT64(0));
    ::DateValueTest(0, 0, 0, 0, 0, 0, 1, INT64(1));
    ::DateValueTest(0, 1, 1, 0, 0, 0, 0, INT64(0));
    ::DateValueTest(0, 1, 1, 0, 0, 0, 1, INT64(1));
    ::DateValueTest(0, 1, 1, 0, 0, 0, 999, INT64(999));
    ::DateValueTest(0, 1, 1, 0, 0, 0, 1000, INT64(1000));
    ::DateValueTest(0, 1, 1, 0, 0, 0, 1001, INT64(1001));
    ::DateValueTest(0, 1, 1, 0, 0, 1, 0, INT64(1000));
    ::DateValueTest(0, 1, 1, 0, 1, 0, 0, INT64(60) * 1000);
    ::DateValueTest(0, 1, 1, 1, 0, 0, 0, INT64(60) * 60 * 1000);
    ::DateValueTest(0, 1, 2, 0, 0, 0, 0, INT64(24) * 60 * 60 * 1000);
    ::DateValueTest(-1, 12, 31, 23, 59, 59, 1000, INT64(0));
    ::DateValueTest(-1, 12, 31, 23, 59, 59, 999, INT64(-1));
    ::DateValueTest(-1, 12, 31, 23, 59, 59, 0, INT64(-1000));
    ::DateValueTest(-1, 12, 31, 23, 59, 0, 0, INT64(-60) * 1000);
    ::DateValueTest(-1, 12, 31, 23, 0, 0, 0, INT64(-60) * 60 * 1000);
    ::DateValueTest(-1, 12, 31, 0, 0, 0, 0, INT64(-24) * 60 * 60 * 1000);

    ::DateValueTest(0, 1, 31, 0, 0, 0, 0, INT64(30) * 24 * 60 * 60 * 1000);
    ::DateValueTest(0, 1, 31, 0, 0, 0, 1, INT64(30) * 24 * 60 * 60 * 1000 + 1);
    ::DateValueTest(0, 1, 31, 0, 0, 1, 0, INT64(30) * 24 * 60 * 60 * 1000 + 1000);
    ::DateValueTest(0, 1, 31, 0, 1, 0, 0, INT64(30) * 24 * 60 * 60 * 1000 + 60 * 1000);
    ::DateValueTest(0, 1, 31, 1, 0, 0, 0, INT64(30) * 24 * 60 * 60 * 1000 + 60 * 60 * 1000);
    ::DateValueTest(0, 1, 31, 24, 0, 0, 0, INT64(30) * 24 * 60 * 60 * 1000 + 24 * 60 * 60 * 1000);
    ::DateValueTest(0, 2, 1, 0, 0, 0, 0, INT64(31) * 24 * 60 * 60 * 1000);
    ::DateValueTest(0, 2, 28, 0, 0, 0, 0, INT64(27 + 31) * 24 * 60 * 60 * 1000);
    ::DateValueTest(0, 2, 29, 0, 0, 0, 0, INT64(28 + 31) * 24 * 60 * 60 * 1000);
    ::DateValueTest(0, 12, 31, 0, 0, 0, 0, INT64(365) * 24 * 60 * 60 * 1000);

    ::DateValueTest(1, 1, 1, 0, 0, 0, 0, INT64(366) * 24 * 60 * 60 * 1000);
    ::DateValueTest(2, 1, 1, 0, 0, 0, 0, INT64(365 + 366) * 24 * 60 * 60 * 1000);
    ::DateValueTest(3, 1, 1, 0, 0, 0, 0, INT64(2 * 365 + 366) * 24 * 60 * 60 * 1000);
    ::DateValueTest(4, 1, 1, 0, 0, 0, 0, INT64(3 * 365 + 366) * 24 * 60 * 60 * 1000);
    ::DateValueTest(5, 1, 1, 0, 0, 0, 0, INT64(3 * 365 + 2 * 366) * 24 * 60 * 60 * 1000);

    ::DateConversionTest(0, 1, 1, 0, 0, 0);
    ::DateConversionTest(0, 1, 2, 0, 0, 0);
    ::DateConversionTest(0, 1, 31, 0, 0, 0);
    ::DateConversionTest(0, 2, 2, 0, 0, 0);
    ::DateConversionTest(0, 2, 28, 0, 0, 0);
    ::DateConversionTest(0, 2, 29, 0, 0, 0);
    ::DateConversionTest(0, 3, 1, 0, 0, 0);
    ::DateConversionTest(0, 12, 31, 0, 0, 0);

    for (INT y = 1900; y <= 1904; y++) {
        for (INT m = 1; m <= 12; m++) {
            for (INT d = 1; d <= 31; d++) {
                if ((m == 2) && ((d > 28) || (DateTime::IsLeapYear(y) && (d > 29)))) {
                    break;
                }
                if (((m < 8) && (m % 2 == 0)) || ((m >= 8) && (m % 2 == 1))) {
                    break;
                }
                ::DateConversionTest(y, m, d, 0, 0, 0);
            }
        }
    }

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

    //::DateConversionTest(-1, 1, 1, 0, 0, 0);
    //::DateConversionTest(-1, 2, 1, 0, 0, 0);
    //::DateConversionTest(-1, 3, 1, 0, 0, 0);
    //::DateConversionTest(-1, 1, 1, 0, 0, 1);
    //::DateConversionTest(-3, 1, 1, 0, 0, 0);
    //::DateConversionTest(-4, 1, 1, 0, 0, 0);
    //::DateConversionTest(-4, 1, 1, 0, 0, 1);
    //::DateConversionTest(-4, 2, 29, 0, 0, 0);
    //::DateConversionTest(-4, 3, 1, 0, 0, 0);
    //::DateConversionTest(-5, 1, 1, 0, 0, 0);
    //::DateConversionTest(-99, 1, 1, 0, 0, 0);
    //::DateConversionTest(-100, 1, 1, 0, 0, 0);
    //::DateConversionTest(-101, 1, 1, 0, 0, 0);
    //::DateConversionTest(-399, 1, 1, 0, 0, 0);
    //::DateConversionTest(-400, 1, 1, 0, 0, 0);
    //::DateConversionTest(-401, 1, 1, 0, 0, 0);
    //::DateConversionTest(-404, 1, 1, 0, 0, 0);
    //::DateConversionTest(-405 , 1, 1, 0, 0, 0);
    //::DateConversionTest(-799, 1, 1, 0, 0, 0);
    //::DateConversionTest(-800, 1, 1, 0, 0, 0);
    //::DateConversionTest(-800, 2, 28, 0, 0, 0);
    //::DateConversionTest(-800, 2, 29, 0, 0, 0);
    //::DateConversionTest(-800, 3, 1, 0, 0, 0);
    //::DateConversionTest(-801, 1, 1, 0, 0, 0);
    //::DateConversionTest(-1700, 1, 1, 0, 0, 0);
    //::DateConversionTest(-1700, 1, 1, 12, 34, 56);
    //::DateConversionTest(-1704, 1, 1, 5, 32, 35);
    //::DateConversionTest(-1704, 2, 1, 5, 1, 00);
    //::DateConversionTest(-1704, 3, 1, 5, 00, 35);
    //::DateConversionTest(-1704, 12, 31, 0, 0, 1);
    //::DateConversionTest(-2004, 1, 1, 5, 32, 35);
    //::DateConversionTest(-2004, 2, 1, 5, 1, 00);
    //::DateConversionTest(-2004, 3, 1, 5, 00, 35);
    //::DateConversionTest(-2004, 12, 31, 0, 0, 1);
    //::DateConversionTest(-2005, 1, 1, 5, 32, 35);
    //::DateConversionTest(-2005, 2, 1, 5, 1, 00);
    //::DateConversionTest(-2005, 3, 1, 5, 00, 35);
    //::DateConversionTest(-2005, 12, 31, 0, 0, 0);
    //::DateConversionTest(-2005, 12, 31, 0, 0, 1);
    //::DateConversionTest(-1, 12, 31, 23, 59, 59);

//    unixTimeStamp = ::time(NULL);
//    vislib::sys::DateTime unixDateTime(unixTimeStamp);
//    ::AssertEqual("Unix timestamp", unixTimeStamp, static_cast<time_t>(unixDateTime));
//
//#ifdef _WIN32
//    //SYSTEMTIME systemTime;
//    //::GetSystemTime(&systemTime);
//    //vislib::sys::DateTime systemTimeDateTime(systemTime);
//    //::AssertEqual("SYSTEMTIME", systemTime, static_cast<SYSTEMTIME>(systemTimeDateTime));
//
//    //FILETIME fileTime;
//    //::SystemTimeToFileTime(&systemTime, &fileTime);
//    //vislib::sys::DateTime fileTimeDateTime(fileTime);
//    //::AssertEqual("SYSTEMTIME", fileTime, static_cast<FILETIME>(fileTimeDateTime));
//
//#endif /* _WIN32 */
//
//    dateTime.Set(0, 0, 0, 0, 0, 0);
//    dateTime.Get(year, month, day, hour, minute, second);
//    ::AssertEqual("Year 0 remains", year, 0);
//    ::AssertEqual("Month 0 converted to 1", month, 1);
//    ::AssertEqual("Day 0 converted to 1", day, 1);
//    ::AssertEqual("Hour remains 0", hour, 0);
//    ::AssertEqual("Minute remains 0", minute, 0);
//    ::AssertEqual("Second remains 0", second, 0);
//
//    dateTime.Set(1, 13, 1, 0, 0, 0);
//    dateTime.Get(year, month, day, hour, minute, second);
//    ::AssertEqual("Year 1 rolled to 1", year,2);
//    ::AssertEqual("Month 13 rolled to 1", month, 1);
//    ::AssertEqual("Day 1 remains 1", day, 1);
//    ::AssertEqual("Hour remains 0", hour, 0);
//    ::AssertEqual("Minute remains 0", minute, 0);
//    ::AssertEqual("Second remains 0", second, 0);
//
//    dateTime.Set(0, 13, 1, 0, 0, 0);
//    dateTime.Get(year, month, day, hour, minute, second);
//    ::AssertEqual("Year 0 rolled to 1", year, 1);
//    ::AssertEqual("Month 13 rolled to 1", month, 1);
//    ::AssertEqual("Day 1 remains 1", day, 1);
//    ::AssertEqual("Hour remains 0", hour, 0);
//    ::AssertEqual("Minute remains 0", minute, 0);
//    ::AssertEqual("Second remains 0", second, 0);
//
//    dateTime.Set(0, 14, 1, 0, 0, 0);
//    dateTime.Get(year, month, day, hour, minute, second);
//    ::AssertEqual("Year 0 rolled to 1", year, 1);
//    ::AssertEqual("Month 14 rolled to 2", month, 2);
//    ::AssertEqual("Day 1 remains 1", day, 1);
//    ::AssertEqual("Hour remains 0", hour, 0);
//    ::AssertEqual("Minute remains 0", minute, 0);
//    ::AssertEqual("Second remains 0", second, 0);
//
//    dateTime.Set(1, 14, 30, 0, 0, 0);
//    dateTime.Get(year, month, day, hour, minute, second);
//    ::AssertEqual("Year 1 rolled to 2", year, 2);
//    ::AssertEqual("Month 14 rolled to 3", month, 3);
//    ::AssertEqual("Day 30 rolled to 2", day, 2);
//    ::AssertEqual("Hour remains 0", hour, 0);
//    ::AssertEqual("Minute remains 0", minute, 0);
//    ::AssertEqual("Second remains 0", second, 0);
//
//    dateTime.Set(1, 2, -1, 0, 0, 0);
//    dateTime.Get(year, month, day, hour, minute, second);
//    ::AssertEqual("Year remains 1", year, 1);
//    ::AssertEqual("Month 2 rolled to 1", month, 1);
//    ::AssertEqual("Day -1 rolled to 31", day, 31);
//    ::AssertEqual("Hour remains 0", hour, 0);
//    ::AssertEqual("Minute remains 0", minute, 0);
//    ::AssertEqual("Second remains 0", second, 0);
//
//    dateTime.Set(1999, 3, -2, 0, 0, 0);
//    dateTime.Get(year, month, day, hour, minute, second);
//    ::AssertEqual("Year remains 1999", year, 1999);
//    ::AssertEqual("Month 3 rolled to 2", month, 2);
//    ::AssertEqual("Day -2 rolled to 27", day, 27);
//    ::AssertEqual("Hour remains 0", hour, 0);
//    ::AssertEqual("Minute remains 0", minute, 0);
//    ::AssertEqual("Second remains 0", second, 0);
//
//    dateTime.Set(1999, -1, 1, 0, 0, 0);
//    dateTime.Get(year, month, day, hour, minute, second);
//    ::AssertEqual("Year rolled to 1998", year, 1998);
//    ::AssertEqual("Month -1 rolled to 12", month, 12);
//    ::AssertEqual("Day remains 1", day, 1);
//    ::AssertEqual("Hour remains 0", hour, 0);
//    ::AssertEqual("Minute remains 0", minute, 0);
//    ::AssertEqual("Second remains 0", second, 0);
//
//    dateTime.Set(1999, -12, 1, 0, 0, 0);
//    dateTime.Get(year, month, day, hour, minute, second);
//    ::AssertEqual("Year rolled to 1998", year, 1998);
//    ::AssertEqual("Month -12 rolled to 1", month, 1);
//    ::AssertEqual("Day remains 1", day, 1);
//    ::AssertEqual("Hour remains 0", hour, 0);
//    ::AssertEqual("Minute remains 0", minute, 0);
//    ::AssertEqual("Second remains 0", second, 0);
//
//    dateTime.Set(1999, -13, 1, 0, 0, 0);
//    dateTime.Get(year, month, day, hour, minute, second);
//    ::AssertEqual("Year rolled to 1997", year, 1997);
//    ::AssertEqual("Month -13 rolled to 12", month, 12);
//    ::AssertEqual("Day remains 1", day, 1);
//    ::AssertEqual("Hour remains 0", hour, 0);
//    ::AssertEqual("Minute remains 0", minute, 0);
//    ::AssertEqual("Second remains 0", second, 0);
//
//    //dateTime.Set(-1, 13, 1, 0, 0, 0);
//    //dateTime.Get(year, month, day, hour, minute, second);
//    //::AssertEqual("Year rolled to 1", year, 1);
//    //::AssertEqual("Month 13 rolled to 1", month, 1);
//    //::AssertEqual("Day remains 1", day, 1);
//    //::AssertEqual("Hour remains 0", hour, 0);
//    //::AssertEqual("Minute remains 0", minute, 0);
//    //::AssertEqual("Second remains 0", second, 0);
//
//    //dateTime.Set(1, -1, 1, 0, 0, 0);
//    //dateTime.Get(year, month, day, hour, minute, second);
//    //::AssertEqual("Year rolled to -1", year, -1);
//    //::AssertEqual("Month -1 rolled to 12", month, 12);
//    //::AssertEqual("Day remains 1", day, 1);
//    //::AssertEqual("Hour remains 0", hour, 0);
//    //::AssertEqual("Minute remains 0", minute, 0);
//    //::AssertEqual("Second remains 0", second, 0);

    dateTime.Set(1, 1, 1, 0, 0, 0);
    dateTime += DateTimeSpan(0, 0, 0, 0, 1);
    ::AssertEqual("Add 1 ms", dateTime, DateTime(1, 1, 1, 0, 0, 0, 1));

    dateTime.Set(1, 1, 1, 0, 0, 0, 1);
    dateTime -= DateTimeSpan(0, 0, 0, 0, 1);
    ::AssertEqual("Subtract 1 ms", dateTime, DateTime(1, 1, 1, 0, 0, 0, 0));
}


/*
 * ::TestDateTime
 */
void TestDateTime(void)  {
    ::EnableAssertSuccessOutput(false);
    ::TestSpan();
    ::TestTime();
}