#!/usr/bin/perl
#
# build_include_table.pl
# MegaMol Core
#
# Copyright (C) 2015 by Sebastian Grottel
# Alle Rechte vorbehalten. All rights reserved
#
use strict;
use File::Find;
use File::Basename;

#
# Get vislib include table data
#
require "include_table.inc";
my @mmcore_includes = get_mmcore_includes();


#
# Fix all include commands in one file using '$File::Find::name'
#
sub fix_includes_in_file {
	# Don't work on folders
	return unless -f;

	my $filename = $File::Find::name;
	my $name;
	my $dir;
	my $ext;
	($name,$dir,$ext) = fileparse($filename,'\..*');

	# do not descend into '.svn' subdirectories
	return unless ($dir !~ /\/.svn\//);

	# do not descend into '.git' subdirectories
	return unless ($dir !~ /\/.git\//);

	# only work on specific file types (could be optimized, but I don't care)
	if (($ext ne ".cpp") && ($ext ne ".h") && ($ext ne ".cu") && ($ext ne ".cuh")) {
		print "  Skipping $filename\n";
		return;
	}

	print "Working on $filename\n";
	
	# Read whole file into string '$content'
	local $/=undef;
	open (my $fh, $filename) or die "Couldn't read file '$filename': $!";
	my $content = <$fh>;
	close $fh;
	
	# Replace all includes to vislib files with updated include directives
	for my $k (@mmcore_includes) {
		#if ($content =~ m|^(\s*#\s*include\s*)["<]$k[">]([^\n]*$)|smg) {
		#	print "FOUND: ";
		#}
		#print "|" . $k . "|\n";
		$content =~ s|^(\s*#\s*include\s*)["<]$k[">]([^\n]*$)|$1"mmcore/$k"$3|smg;
	}
	  
	# output updated content to the file again
	# Maybe that didn't even change anything, but I don't care
	#print $content;
	open (my $fh, '>', $filename) or die "Couldn't write file '$filename': $!";
	print $fh $content;
	close $fh;
}

# Go through all files/subdirectories specified by the first command line argument
find(\&fix_includes_in_file, shift);
