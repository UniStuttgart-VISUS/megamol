#
# configure.pl
# VISlib GlutInclude
#
# Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#
use strict;
use File::Copy;


#
# Test input for boolean answer
#
sub IsInputYes {
    my $input = shift;
    return ($input eq "1" or $input =~ /^\s*true\s*$/i or $input =~ /^\s*on\s*$/i or $input =~ /^\s*yes\s*$/i
            or $input =~ /^\s*t\s*$/i or $input =~ /^\s*y\s*$/i);
}


#
# 'Copy' a file and replaces one placeholder text
#
sub copyEditFile {
    my $fromFile = shift;
    my $toFile = shift;
    my $placeholder = shift;
    my $text = shift;
    
    if (open IN, $fromFile) {
        if (open OUT, ">".$toFile) {
            while(<IN>) {
                s/$placeholder/$text/;
                print OUT;
            }
            close OUT;
        } else {
            print "could not write to " . $toFile . "\n";
        }
        close IN;
    } else {
        print "could not read from " . $fromFile . "\n";
    }
}


#
# Configure for Windows
#
sub configWindows {
    my $glutType;
    my $visGlutPath;
    my $confirm;

    $glutType = 0;
    while (($glutType < 1) || ($glutType > 2)) {
        print "\nSelect the type of Glut you want to use for Windows:\n";
        print "\t1) system defined Glut\n";
        print "\t2) VISglut\n";
        print "Select: ";
        chomp($glutType = <STDIN>);
    }

    if ($glutType == 1) {
        print "Configuring for system defined Glut\n";
    } elsif ($glutType == 2) {
        print "Enter Path to VISglut: ";
        chomp($visGlutPath = <STDIN>);
        $visGlutPath =~ s/\\/\//g;
        $visGlutPath =~ s/\/$//;
        print "Configuring for VISglut at \"$visGlutPath\"\n";
    }

    print "Proceed? [y/n]: ";
    chomp($confirm = <STDIN>);
    if (IsInputYes($confirm)) {
        if ($glutType == 1) {
            copy("glutInclude.system.h", "glutInclude.win.h");
        } elsif ($glutType == 2) {
            copyEditFile("glutInclude.visglut.h", "glutInclude.win.h", "%visglutPath%", $visGlutPath);
        }
    }
}


#
# Configure for Linux
#
sub configLinux {
    my $glutType;
    my $visGlutPath;
    my $confirm;

    $glutType = 0;
    while (($glutType < 1) || ($glutType > 2)) {
        print "\nSelect the type of Glut you want to use for Linux:\n";
        print "\t1) system defined Glut\n";
        print "\t2) VISglut\n";
        print "Select: ";
        chomp($glutType = <STDIN>);
    }

    if ($glutType == 1) {
        print "Configuring for system defined Glut\n";
    } elsif ($glutType == 2) {
        print "Enter Path to VISglut: ";
        chomp($visGlutPath = <STDIN>);
        $visGlutPath =~ s/\/$//;
        print "Configuring for VISglut at \"$visGlutPath\"\n";
    }

    print "Proceed? [y/n]: ";
    chomp($confirm = <STDIN>);
    if (IsInputYes($confirm)) {
        if ($glutType == 1) {
            copy("glutInclude.system.h", "glutInclude.lin.h");
            copy("glutInclude.system.mk", "glutInclude.lin.mk");
        } elsif ($glutType == 2) {
            copyEditFile("glutInclude.visglut.h", "glutInclude.lin.h", "%visglutPath%", $visGlutPath);
            copyEditFile("glutInclude.visglut.mk", "glutInclude.lin.mk", "%visglutPath%", $visGlutPath);
        }
    }
}


#
# Ask User if he wants to configure for Windows
#
sub askConfigWindows {
    my $input;
    print "\nDo you want to configure the Windows Glut Include? [y/n]: ";
    chomp($input = <STDIN>);
    if (IsInputYes($input)) {
        configWindows();
    } else {
        print "\n";
    }
}


#
# Ask User if he wants to configure for Linux
#
sub askConfigLinux {
    my $input;
    print "\nDo you want to configure the Linux Glut Include? [y/n]: ";
    chomp($input = <STDIN>);
    if (IsInputYes($input)) {
        configLinux();
    } else {
        print "\n";
    }
}


#
# Main Code
#
if ($^O eq "MSWin32") {
    configWindows();
    askConfigLinux();
} elsif ($^O eq "linux") {
    configLinux();
    askConfigWindows();
} else {
    askConfigWindows();
    askConfigLinux();
}
