#!/usr/bin/perl
#
# configure.win.pl
# MegaMol Core
#
# Copyright (C) 2008-2015 by VISUS (Universitaet Stuttgart).
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
    $a->id("expat");
    $a->description("Path to the expat directory");
    $a->placeholder("%expatPath%");
    $a->markerFile("expat.h");
    $a->relativeLocation("../");
    $a->autoDetect(1);
    $a->directorySeparator("\\");
    $a->enforceTrailingDirectorySeparator(1);
    push @pps, $a;
$a = FlagParameter->new();
    $a->id("buildd3d");
    $a->placeholder("buildd3d");
    $a->description("enable Direct3D support - requires DXSDK with shader compiler!");
    $a->value(0);
    push @fps, $a;

$c = ConfigFilePair->new();
    $c->inFile("ExtLibs.props.input");
    $c->outFile("ExtLibs.props");
    push @cfps, $c;
$c = ConfigFilePair->new();
    $c->inFile("D3D.props.input");
    $c->outFile("D3D.props");
    push @cfps, $c;

VISUS::configperl::Configure("MegaMol(TM) Core Configuration for Windows", ".megamol.core.win.cache", \@pps, \@fps, \@cfps, \@sps, \@ARGV);

