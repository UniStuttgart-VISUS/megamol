#!/usr/bin/perl
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

# greeting
print "\n";
print "MegaMol(TM) Plugin Instantiation Wizard\n";
print "Copyright 2010 by VISUS (Universitaet Stuttgart)\n";
print "Alle Rechte vorbehalten.\n\n";

# anti-idiot test
open SOLUTION, "MegaMolPlugin.sln" or autoEuthanize($ERR_ALREADY_INSTA);
while(<SOLUTION>) {
    if (/$SRCGUID/) {
        $ok = 1;
        last;
    }
}
close SOLUTION;
autoEuthanize($ERR_ALREADY_INSTA) unless $ok == 1;

# ask for parameters
if ($HAVE_CWD) {
    $filename = getcwd();
}
$filename =~ s/.*[\/\\]//;
print "Input the plugin filename [$filename]: ";
chomp($temp = <STDIN>);
if ($temp ne "") {
    $filename = $temp;
}
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
print "generating plugin $filename with GUID $guid. OK? [ /n]: ";
chomp($temp = <STDIN>);
if (IsInputNo($temp)) {
    autoEuthanize("aborting.");
}

# action jackson
foreach $temp (glob "MegaMolPlugin.*") {
    $temp =~ /.*\.(\S+)$/;
    my $ext = $1;
    print "$temp -> $filename.$ext: ";
    if (rename $temp, "$filename.$ext") {
        print "ok\n";
    } else {
        autoEuthanize("FAILED.");
    }
}

$temp = slurpFile("$filename.sln");
$temp =~ s/MegaMolPlugin/$filename/g;
$temp =~ s/$SRCGUID/$guid/g;
writeFile("$filename.sln", $temp);

$temp = slurpFile("$filename.vcproj");
$temp =~ s/Name="MegaMolPlugin"/Name="$filename"/g;
$temp =~ s/RootNamespace="MegaMolPlugin"/RootNamespace="$filename"/g;
$temp =~ s/$SRCGUID/$guid/g;
$temp =~ s/MEGAMOLPLUGIN_EXPORTS/\U$filename\E_EXPORTS/g;
$temp =~ s/OutputFile="\$\(OutDir\)\\Template\$\(BitsD\).win\$\(Bits\).mmplg"/OutputFile="\$\(OutDir\)\\$filename\$\(BitsD\).win\$\(Bits\).mmplg"/g;
$temp =~ s/MegaMolPlugin/$filename/g;
writeFile("$filename.vcproj", $temp);

$temp = slurpFile("Makefile");
$temp =~ s/TargetName := Template/TargetName := $filename/g;
$temp =~ s/PluginTemplate/$filename/;
writeFile("Makefile", $temp);

$temp = slurpFile("$filename.h");
$temp =~ s/MegaMolPlugin/$filename/g;
$temp =~ s/MEGAMOLPLUGIN/\U$filename\E/g;
writeFile("$filename.h", $temp);

$temp = slurpFile("$filename.cpp");
$temp =~ s/MegaMolPlugin/$filename/g;
$temp =~ s/MEGAMOLPLUGIN/\U$filename\E/g;
$temp =~ s/PluginTemplate/$filename/;
writeFile("$filename.cpp", $temp);
