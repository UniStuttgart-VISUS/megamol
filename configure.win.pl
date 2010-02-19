#
# configure.win.pl
# MegaMol Console
#
# Copyright (C) 2008-2010 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#
use Cwd qw{abs_path};
my $incpath = abs_path($0);
$incpath =~ s/\/[^\/]+$//;
push @INC, "$incpath/configperl";
require "configperl.inc";

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
    $a->id("vislib");
    $a->description("Path to the vislib directory");
    $a->placeholder("%vislib%");
    $a->markerFile("vislib.sln\$");
    $a->relativeLocation("./");
    $a->autoDetect(1);
    push @pps, $a;
$a = PathParameter->new();
    $a->id("visglut");
    $a->description("Path to the visglut directory");
    $a->placeholder("%visglutPath%");
    $a->markerFile("include/visglut.h\$");
    $a->relativeLocation("../");
    $a->autoDetect(1);
    push @pps, $a;
$a = PathParameter->new();
    $a->id("mmcore");
    $a->description("Path to the MegaMol core directory");
    $a->placeholder("%mmcorePath%");
    $a->markerFile("api/MegaMolCore.h\$");
    $a->relativeLocation("../");
    $a->autoDetect(1);
    push @pps, $a;
$b = FlagParameter->new();
    $b->id("withTweakbar");
    $b->description("Enable the use of the AntTweakBar library");
    $b->placeholder("%withTweakbar%");
    $b->value(0);
    push @fps, $b;
$a = PathParameter->new();
    $a->id("tweakbarpath");
    $a->description("Path to the AntTweakBar directory");
    $a->placeholder("%tweakbarpath%");
    $a->markerFile("AntTweakBar.h\$");
    $a->relativeLocation("../");
    $a->autoDetect(1);
    $a->dependencyFlagID("withTweakbar");
    $a->dependencyDisabledValue("");
    push @pps, $a;

$c = ConfigFilePair->new();
    $c->inFile("ExtLibs.vsprops.input");
    $c->outFile("ExtLibs.vsprops");
    push @cfps, $c;
$c = ConfigFilePair->new();
    $c->inFile("ExtLibTw.vsprops.input");
    $c->outFile("ExtLibTw.vsprops");
    push @cfps, $c;

VISUS::configperl::Configure("MegaMol(TM) Console Configuration for Windows", ".megamol.console.win.cache", \@pps, \@fps, \@cfps, \@sps);

