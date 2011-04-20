#!/usr/bin/perl
#
# configure.win.pl
# MegaMol Plugin
#
# Copyright (C) 2008-2010 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#
use Cwd qw{abs_path};
use strict;
use warnings 'all';
my $incpath = abs_path($0);
$incpath =~ s/\/[^\/]+$//;
push @INC, "$incpath/configperl";
require configperl;

my ($a, $b, $c);
my @pps = ();
my @fps = ();
my @cfps = ();
my @sps = ();

$a = PathParameter->new();
    $a->id("outbin");
    $a->description("Path to the global \"bin\" output directory");
    $a->placeholder("%outbin%");
    $a->autoDetect(0);
    $a->value("../bin");
    push @pps, $a;

$a = PathParameter->new();
    $a->id("mmcore");
    $a->description("Path to the MegaMol core directory");
    $a->placeholder("%mmcorePath%");
    $a->markerFile("api/MegaMolCore.h\$");
    $a->relativeLocation("../");
    $a->autoDetect(1);
    push @pps, $a;
$a = PathParameter->new();
    $a->id("vislib");
    $a->description("Path to the vislib directory");
    $a->placeholder("%vislib%");
    $a->markerFile("vislib.sln\$");
    $a->relativeLocation("./");
    $a->autoDetect(1);
    push @pps, $a;

$b = FlagParameter->new();
    $b->id("withVrpn");
    $b->description("Enable the use of the VRPN library");
    $b->placeholder("%withVrpn%");
    $b->value(0);
    push @fps, $b;
$a = PathParameter->new();
    $a->id("vrpnpath");
    $a->description("Path to the VRPN directory");
    $a->placeholder("%vrpnpath%");
    $a->markerFile("vrpn_Connection.h\$");
    $a->relativeLocation("../");
    $a->autoDetect(1);
    $a->dependencyFlagID("withVrpn");
    $a->dependencyDisabledValue("");
    push @pps, $a;

$c = ConfigFilePair->new();
    $c->inFile("ExtLibs.vsprops.input");
    $c->outFile("ExtLibs.vsprops");
    push @cfps, $c;

VISUS::configperl::Configure("MegaMol(TM) Plugin Configuration for Windows", ".megamol.plg.win.cache", \@pps, \@fps, \@cfps, \@sps);

