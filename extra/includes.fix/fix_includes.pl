#!/usr/bin/perl
#
# build_include_table.pl
# VISlib 2
#
# Copyright (C) 2015 by Sebastian Grottel
# Alle Rechte vorbehalten. All rights reserved
#
use strict;
use File::Find;
use File::Basename;
require "include_table.inc";

my $vislib_includes = get_vislib_includes();

sub fix_includes_in_file {
	return unless -f;
	my $filename = $File::Find::name;

	my $name;
	my $dir;
	my $ext;
	($name,$dir,$ext) = fileparse($filename,'\..*');
	if (($ext ne ".cpp") && ($ext ne ".h")) {
		print "  Skipping $filename\n";
		return;
	}

	print "Working on $filename\n";
	
	local $/=undef;
	open (my $fh, $filename) or die "Couldn't read file '$filename': $!";
	my $content = <$fh>;
	close $fh;
	
	while(my($k, $v) = each $vislib_includes) {
		#if ($content =~ m/vislib[\/\\]$k/) {
		#	print "FOUND: ";
		#}
		#print "|" . $k . "|\n";
		$content =~ s|^(\s*#\s*include\s*)["<]vislib[/\\]$k[">]([^\n]*$)|$1"$v/$k"$3|smg;
	}
	  
	#print $content;
    open (my $fh, '>', $filename) or die "Couldn't write file '$filename': $!";
	print $fh $content;
	close $fh;
}

find(\&fix_includes_in_file, shift);
