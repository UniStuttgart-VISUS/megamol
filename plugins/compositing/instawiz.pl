#!/usr/bin/perl
#
# MegaMol� Plugin Instantiation Wizard
# Copyright 2010-2015 by MegaMol Team
# Alle Rechte vorbehalten.
#
use strict;
use warnings 'all';

# dependencies
my $HAVE_MD5 = 1;
eval "use Digest::MD5 qw(md5_hex)";
$HAVE_MD5 = 0  if ($@);
my $HAVE_CWD = 1;
eval "use Cwd";
$HAVE_CWD = 0 if ($@);

# constants
my $ERR_ALREADY_INSTA = "plugin already instantiated, go away.";
my $SRCGUID = "B5FFC5D2-5CF9-4D34-A1C1-C61D7FAC0AB8";
my $ZEROGUID = "00000000-0000-0000-0000-000000000000";

# variables
my $ok = 0;
my $filename;
my $temp;
my $guid;
my $fn;

# subs
sub autoEuthanize($) {
    print shift;
    print "\n";
    exit(1);
}

sub genGUID($) {
    my $seed = shift;
    my $md5 = uc md5_hex ($seed);
    my @octets = $md5 =~ /(.{2})/g;
    
    substr $octets[6], 0, 1, '4'; # GUID Version 4
    substr $octets[8], 0, 1, 'A'; # draft-leach-uuids-guids-01.txt GUID variant 
    my $GUID = "@octets[0..3]-@octets[4..5]-@octets[6..7]-@octets[8..9]-@octets[10..15]";
    
    $GUID =~ s/ //g;
    return $GUID;
}

sub checkGUID($) {
    my $guid = shift;
    return ($guid =~ /^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$/i);
}

sub IsInputNo($) {
    my $input = shift;
    return ($input eq "0" or $input =~ /^\s*false\s*$/i or $input =~ /^\s*off\s*$/i or $input =~ /^\s*no\s*$/i
            or $input =~ /^\s*f\s*$/i or $input =~ /^\s*n\s*$/i);
}

sub slurpFile($) {
    my $nl = $/;
    undef $/;
    my $file = shift;
    open VICTIM, $file or autoEuthanize("could not read $file.");
    my $temp = <VICTIM>;
    close VICTIM;
    $/ = $nl;
    return $temp;
}

sub writeFile($$) {
    my $file = shift;
    my $stuff = shift;
    open VICTIM, ">$file" or autoEuthanize("could not write $file.");
    print VICTIM $stuff;
    close VICTIM;
}

sub renameFile($$) {
    my $oldName = shift;
    my $newName = shift;
    print "$oldName -> $newName: ";
    if (rename $oldName, $newName) {
        print "ok\n";
    } elsif (-e $newName) {
        print "ok(magic)\n";
    } else {
        autoEuthanize("FAILED.");
    }
}

# greeting
print "\n";
print "MegaMol(TM) Plugin Instantiation Wizard\n";
print "Copyright 2010-2015 by MegaMol Team\n";
print "Alle Rechte vorbehalten.\n\n";

# ask for parameters
#  - filename
if ($HAVE_CWD) {
    $filename = getcwd();
}
$filename =~ s/.*[\/\\]//;
do {
    $filename =~ s/[^0-9a-zA-Z_]/_/g;
    print "Input the plugin filename [$filename]: ";
    chomp($temp = <STDIN>);
    if ($temp ne "") {
        $filename = $temp;
    }
    if (not $filename =~ /^[a-zA-Z][0-9a-zA-Z_]*$/) {
        print "ERROR: filename is invalid\n";
        print "       A plugin filename must start with a character and must only contain characters, numbers and underscores.\n";
    }
} while (not $filename =~ /^[a-zA-Z][0-9a-zA-Z_]*$/);
#  - guid
if ($HAVE_MD5) {
    $guid = genGUID($filename);
} else {
    $guid = $ZEROGUID;
}
print "Input a well-formed GUID for your project [$guid]: ";
chomp($temp = <STDIN>);
if ($temp ne "") {
    $guid = $temp;
}
if (not checkGUID($guid)) {
    autoEuthanize("GUID is ill-formed, you nub.");
}

# paranoia confirmation
print "generating plugin \"$filename\" with GUID \"$guid\". OK? [ /n]: ";
chomp($temp = <STDIN>);
if (IsInputNo($temp)) {
    autoEuthanize("aborting.");
}

# Now perform the instantiation
#  - source code files
renameFile("include/MegaMolPlugin", "include/$filename");

$fn = "include/$filename/$filename.h";
renameFile("include/$filename/MegaMolPlugin.h", $fn);
$temp = slurpFile($fn);
$temp =~ s/MegaMolPlugin/$filename/g;
$temp =~ s/MEGAMOLPLUGIN/\U$filename\E/g;
writeFile($fn, $temp);

$fn = "src/$filename.cpp";
renameFile("src/MegaMolPlugin.cpp", $fn);
$temp = slurpFile($fn);
$temp =~ s/MegaMolPlugin/$filename/g;
$temp =~ s/MEGAMOLPLUGIN/\U$filename\E/g;
writeFile($fn, $temp);

$fn = "src/stdafx.h";
$temp = slurpFile($fn);
$temp =~ s/MEGAMOLPLUGIN/\U$filename\E/g;
writeFile($fn, $temp);

#  - Cmake files
$fn = "CMakeLists.txt";
$temp = slurpFile($fn);
$temp =~ s/MegaMolPlugin/$filename/g;
writeFile($fn, $temp);

# Completed
print("\n== Instantiation complete ==\n\n");
print("You should do now:\n");
print("  * delete instawiz.pl\n");
print("  * Commit changes to git\n");
print("  * Start implementing\n");
print("\n");
