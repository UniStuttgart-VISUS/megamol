#!/usr/bin/perl
#
# configure.win.pl
# MegaMol Plugin
#
# Copyright (C) 2008-2011 by VISUS (Universitaet Stuttgart).
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
    $a->directorySeparator("\\");
    $a->enforceTrailingDirectorySeparator(1);
    push @pps, $a;

$a = PathParameter->new();
    $a->id("mmcore");
    $a->description("Path to the MegaMol core directory");
    $a->placeholder("%mmcorePath%");
    $a->markerFile("include/mmcore/CoreInstance.h");
    $a->relativeLocation("../../");
    $a->autoDetect(1);
    $a->directorySeparator("\\");
    $a->enforceTrailingDirectorySeparator(1);
    push @pps, $a;

$a = PathParameter->new();
    $a->id("vislib");
    $a->description("Path to the vislib directory");
    $a->placeholder("%vislib%");
    $a->markerFile("vislib.sln");
    $a->relativeLocation("./");
    $a->autoDetect(1);
    $a->directorySeparator("\\");
    $a->enforceTrailingDirectorySeparator(1);
    push @pps, $a;

$a = PathParameter->new();
    $a->id("mmstd_trisoup");
    $a->description("Path to the mmstd_trisoup directory");
    $a->placeholder("%mmstd_trisoup%");
    $a->markerFile("mmstd_trisoup.sln");
    $a->relativeLocation("./");
    $a->autoDetect(1);
    $a->directorySeparator("\\");
    $a->enforceTrailingDirectorySeparator(1);
    push @pps, $a;

$a = PathParameter->new();
    $a->id("ospray");
    $a->description("Path to the ospray directory");
    $a->placeholder("%ospray%");
    $a->markerFile("ospray.dll");
    $a->relativeLocation("../");
    $a->autoDetect(1);
    $a->directorySeparator("\\");
    $a->enforceTrailingDirectorySeparator(1);
    push @pps, $a;

$a = PathParameter->new();
    $a->id("protein_calls");
    $a->description("Path to the protein_calls directory");
    $a->placeholder("%protein_calls%");
    $a->markerFile("protein_calls.sln");
    $a->relativeLocation("../");
    $a->autoDetect(1);
    $a->directorySeparator("\\");
    $a->enforceTrailingDirectorySeparator(1);
    push @pps, $a;

$a = PathParameter->new();
$a->id("tbb");
$a->description("Path to the tbb directory");
$a->placeholder("%tbb%");
$a->markerFile("tbb.h");
$a->relativeLocation("../../");
$a->autoDetect(1);
$a->directorySeparator("\\");
$a->enforceTrailingDirectorySeparator(1);
push @pps, $a;

$c = ConfigFilePair->new();
    $c->inFile("ExtLibs.props.input");
    $c->outFile("ExtLibs.props");
    push @cfps, $c;

VISUS::configperl::Configure("MegaMol(TM) Plugin Configuration for Windows", ".megamol.plg.win.cache", \@pps, \@fps, \@cfps, \@sps, \@ARGV);

