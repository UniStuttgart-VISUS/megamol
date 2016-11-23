package VISUS::configperl;
#
# configperl.inc
# Copyright (c) 2008-2013, VISUS
# (Visualization Research Center, Universitaet Stuttgart)
# All rights reserved.
#
# See: LICENCE.TXT or
# https://svn.vis.uni-stuttgart.de/utilities/configperl/LICENCE.TXT
#
use strict;
use warnings 'all';
use Config;
use Class::Struct;
use File::Spec::Functions;
use Cwd qw{abs_path cwd};

my $haveThreads;

BEGIN {
    my $incpath = abs_path($0);
    $incpath =~ s/\/[^\/]+$//;
    chdir $incpath;
    #print "begin chdirs to " . $incpath . "\n";
    if((eval {require threads}) && (eval {require Time::HiRes})) {
        #print "found threads\n";
        $haveThreads = 1;
        use searcher;
    } else {
        #print "no threads!\n";
        $haveThreads = 0;
    }
}

our %headers;

struct PathParameter => [
    id => '$',
    description => '$',
    placeholder => '$',
    autoDetect => '$',
    markerFile => '$',
    relativeLocation => '$',
    value => '$',
    dependencyFlagID => '$',
    dependencyDisabledValue => '$',
    directorySeparator => '$',
    enforceTrailingDirectorySeparator => '$'
];

sub PathParameter::Serialize {
    my $self = shift;
    my $OUT = shift;
    print $OUT $self->id . "\n";
    print $OUT $self->value . "\n";
    return;
}

sub PathParameter::Deserialize {
    my $self = shift;
    my $IN = shift;
    my $temp;
    chomp($temp = <$IN>);
    $self->id($temp);
    chomp($temp = <$IN>);
    $self->value($temp);
    return;
}

sub PathParameter::CopyFrom {
    my $self = shift;
    my $other = shift;
    if ($other->value gt "") {
            $self->value($other->value);
    }
    return;
}

struct FlagParameter => [
    id => '$',
    description => '$',
    placeholder => '$',
    value => '$'
];

sub FlagParameter::Serialize {
    my $self = shift;
    my $OUT = shift;
    print $OUT $self->id . "\n";
    print $OUT $self->value . "\n";
    return;
}

sub FlagParameter::Deserialize {
    my $self = shift;
    my $IN = shift;
    my $temp;
    chomp($temp = <$IN>);
    $self->id($temp);
    chomp($temp = <$IN>);
    $self->value($temp);
    return;
}

sub FlagParameter::CopyFrom {
    my $self = shift;
    my $other = shift;
    if ($other->value gt "") {
        $self->value($other->value);
    }
    return;
}

struct ConfigFilePair => [
    inFile => '$',
    outFile => '$'
];

struct StringPair => [
    placeholder => '$',
    value => '$'
];

sub SaveCache {
    $#_ == 3 || die "you must not call SaveCache without 4 parameters";
    my $configcache = shift;
    my @pps = @{shift()};
    my @fps = @{shift()};
    #my @cfps = @{shift()};
    
    open (my $OUT, ">", "$configcache") or die "cannot write cache $configcache";
    foreach my $pp (@pps) {
        print $OUT "#PATHPARAMETER\n";
        $pp->Serialize($OUT);
    }
    foreach my $fp (@fps) {
        print $OUT "#FLAGPARAMETER\n";
        $fp->Serialize($OUT);
    }
    close $OUT;
    return;
}

sub LoadCache {
    $#_ == 3 || die "you must not call LoadCache without 4 parameters";
    my $configcache = shift;
    my @pps = @{shift()};
    my @fps = @{shift()};
    my @cfps = @{shift()};
    my $pp;
    my $fp;
    my @newpps = ();
    my @newfps = ();
    my @newcfps = ();

    if (open my $IN, "<", "$configcache") {
        while(<$IN>) {
            chomp;
            if (/^#PATHPARAMETER$/) {
                $pp = PathParameter->new();
                $pp->Deserialize($IN);
                push @newpps, $pp;
            }
            if (/^#FLAGPARAMETER$/) {
                $fp = FlagParameter->new();
                $fp->Deserialize($IN);
                push @newfps, $fp;
            }
        }
        close $IN;
        foreach my $pp (@newpps) {
            foreach my $oldpp (@pps) {
                if ($oldpp->id eq $pp->id) {
                    $oldpp->CopyFrom($pp);
                }
            }
        }
        foreach my $fp (@newfps) {
            foreach my $oldfp (@fps) {
                if ($oldfp->id eq $fp->id) {
                    $oldfp->CopyFrom($fp);
                }
            }
        }
    } else {
        # no cache, huh
    }
    return;
}

sub IsInputNo {
    my $input = shift;
    return ($input eq "0" or $input =~ /^\s*false\s*$/i or $input =~ /^\s*off\s*$/i or $input =~ /^\s*no\s*$/i
                    or $input =~ /^\s*f\s*$/i or $input =~ /^\s*n\s*$/i);
}

sub IsInputEmpty {
    my $input = shift;
    if (defined $input) {
        chomp($input);
        return $input =~ /^\s*$/;
    }
    #print "input $input empty\n";
    return 1;
}

sub DoExit {
    if ($haveThreads) {
        VISUS::configperl::searcher::stopSearch();
    }
    exit 0;
    #return;
}

sub CheckCommands {
    my $input = shift;
    if ($input =~ /^\s*q\s*$/) {
        DoExit();
        #return;
    }
    if ($input =~ /^\+(.*)$/) {
        my $d = $1;
        if (-d $d) {
            VISUS::configperl::searcher::addSearchFolder($d, 1);
        } else {
            print "$d is not a directory!\n";
        }
        print "\n";
        return 0;
    }
    return 1;
}

sub CheckPPCommands {
    chomp(my $input = shift);
    my $pp = shift;
    #print "checking $input for $pp\n";
    if ($input =~ /^\s*\?\s*$/) {
        print "\nParameter  : " . $pp->id . "\n";
        print "Description: " . $pp->description . "\n";
        if (defined $pp->autoDetect && $pp->autoDetect) {
            if ($haveThreads) {
                print " " x 13 . "- will be searched for -\n";
                #print "Suggestions: ";
                #VISUS::configperl::searcher::printResults($pp->markerFile);
                #print "none so far.\n";
            }
        }
        print "\n";
        VISUS::configperl::searcher::printSearchFolders();
        return 0;
    }
    if ($input =~ /^\s*(\d+)\s*$/) {
        #print "you wanted $1\n";
        my $r = VISUS::configperl::searcher::getResult($pp->markerFile, $1);
        if ($r gt "") {
            $pp->value($r);
            return 2;
        } else {
            return 0;
        }
    }
    return CheckCommands($input);
}

sub CheckFPCommands {
    chomp(my $input = shift);
    my $fp = shift;
    if ($input =~ /^\s*\?\s*$/) {
        print "\nParameter  : " . $fp->id . "\n";
        print "Description: " . $fp->description . "\n";
        print "\n";
        return 0;
    }
    return CheckCommands($input);
}

sub AddSearchFolder {
    if ($haveThreads) {
        foreach my $a (@_) {
            VISUS::configperl::searcher::addSearchFolder($a);
        }
    }
    return;
}

sub PostProcessPP {
    my $pp = shift;
    if (defined $pp->directorySeparator) {
        my $x = $pp->value;
        my $y = $pp->directorySeparator;
        $x =~ s/[\/\\]/$y/g;
        $pp->value($x);
    }
    if ($pp->enforceTrailingDirectorySeparator) {
        my $x = $pp->value;
        my $y = $pp->directorySeparator;
        if (!defined $y) {
            # warning! perl idiom ("goatse"):
            # enforce list context, then count entries
            my $countFw = () = $x =~ /\//g;
            my $countBw = () = $x =~ /\\/g;
            $y = ($countBw > $countFw) ? "\\" : "/";
            $x =~ s/([^\/\\])$/$1$y/;
        } else {
            my $z = "\\" . $y;
            $x =~ s/([^$z])$/$1$y/;
        }
        $pp->value($x);
    }
    return;
}

sub Configure {
    ($#_ > 4) && ($#_ < 7) || die "you must not call Configure without 6 parameters";
    my $title = shift;
    my $configcache = shift;
    my @pps = @{shift()};
    my @fps = @{shift()};
    my @cfps = @{shift()};
    my @sps = @{shift()};
    my $input = "";
    my $done;
    my $fullauto = 0;
    my $useConfig = "";
    my %cmdlineOptions = ();
    my $something = shift();
    my $anythingMissing = 0;
    
    if (!defined $something) {
        $something = 0;
    }
    if ($something == 1 || $something == 0) {
        $fullauto = $something;
    } else {
        my @av = @{$something};
        if ((grep {$_ eq "fullauto"} @av) || (defined $ENV{'CONFIGPERL_FULLAUTO'})) {
            $fullauto = 1;
        }
    
        for (my $a = 0; $a <= $#av; $a++) {
            if ($av[$a] eq "-useconfig") {
                if ($#av > $a) {
                    $useConfig = $av[$a + 1];
                } else {
                    die "you must supply an additional argument for -useconfig";
                }
                last;
            }
        }
        
        foreach my $opt (split /;/,$useConfig) {
            my ($o, $v) = split /=/, $opt;
            $cmdlineOptions{$o} = $v;
        }
    }
    
    # start background search
    if ($haveThreads) {
        foreach my $pp (@pps) {
            if (defined $pp->autoDetect && $pp->autoDetect) {
                $headers{$pp->markerFile} = $pp->relativeLocation;
            }
        }
        VISUS::configperl::searcher::startSearch($fullauto);
    }

    # slurp cache, deserialize defaults from last use.
    if (!$fullauto) {
        LoadCache($configcache, \@pps, \@fps, \@cfps);
    }
            
    print "\n\n    $title\n    Copyright (c) 2008-2013 by Universitaet Stuttgart (VISUS)\n    All Rights reserved.\n\n\n";
    
    print "Input option values. Special commands:\n'?' for help,\n'+' to push a path onto the search stack,\n'q' to quit.\n\n";
    
    foreach my $fp (@fps) {
        $done = 0;
        while (! $done) {
            if (defined $cmdlineOptions{$fp->id}) {
                $fp->value($cmdlineOptions{$fp->id});
                $done = 1;
            } else {
                $anythingMissing = 1;
                print $fp->description . "\n";
                print "Enable " . $fp->id . " [" . $fp->value . "]: ";
                if ($fullauto) {
                    $input = $fp->value;
                    print "$input\n";
                } else {
                    chomp($input = <STDIN>);
                }
                if (!IsInputEmpty($input)) {
                    if (CheckFPCommands($input, $fp)) {
                        if (IsInputNo($input)) {
                            $fp->value(0);
                        } else {
                            $fp->value(1);
                        }
                        $done = 1;
                    }
                } else {
                    $done = 1;
                }
            }
        }
    }
    
    # wait for the searcher to finish
    if ($fullauto) {
        VISUS::configperl::searcher::waitForSearch();
    }
    
    foreach my $pp (@pps) {
        $done = 0;
        if ($pp->markerFile && $pp->markerFile =~ /\$$/) {
            die "ERROR: Markerfiles with trailing \\\$ are no longer supported!\n";
        }
        
        if (defined $pp->dependencyFlagID) {
            my $found = 0;
            my $val;
            foreach my $fp (@fps) {
                if ($fp->id eq $pp->dependencyFlagID) {
                    $found = 1;
                    $val = $fp->value;
                    last;
                }
            }
            if (! $found) {
                die "\nWarning: $pp->id depends on $pp->dependencyFlagID, which does not exist!\n";
            } else {
                if ($val == 0) {
                    next;
                }
            }
        }
        while (! $done) {
            if (defined $cmdlineOptions{$pp->id}) {
                $pp->value($cmdlineOptions{$pp->id});
                $done = 1;
            } else {
                print "Path for " . $pp->id;
                if ($pp->markerFile) {
                    print " (type a number or supply the full path):\n    ";
                    VISUS::configperl::searcher::printResults($pp->markerFile);
                } else {
                    print "\n";
                }
                $anythingMissing = 1;
                if (defined $pp->value && $pp->value gt "") {
                    #print "Path for " . $pp->id . " [" . $pp->value . "]: ";
                    print "[" . $pp->value . "]: ";
                } else {
                    #print "Path for " . $pp->id . ": ";
                    print "[]: ";
                }
                if ($fullauto) {
                    if (defined $pp->markerFile) {
                        #print "getting value for " . $pp->markerFile . "\n";
                        $input = VISUS::configperl::searcher::getResult($pp->markerFile, 0);
                    } else {
                        $input = $pp->value;
                    }
                    print "$input\n";
                } else {
                    chomp($input = <STDIN>);
                }
                #print ("input was: $input\n");
                if (!IsInputEmpty($input)) {
                    my $c = CheckPPCommands($input, $pp);
                    if ($c == 1) {
                        $pp->value($input);
                        $done = 1;
                    } else {
                        if ($c == 2) {
                            $done = 1;
                        } else {
                            # uhm..
                        }
                    }
                } else {
                    $done = 1;
                }
            }
        }
    }
    
    #canonicalization emulation
    foreach my $pp (@pps) {
        # make those paths absolute that are not. or something.
        # grade A wildness is given.
        my $isAlreadyAbsolute = 0;
        my $isUNC = 0;
        
        if (! defined($pp->value)) {
            $pp->value("");
        }
        if ($pp->value eq "") {
            # no touchy
            $isAlreadyAbsolute = 1;
            $isUNC = 1;
        }
        
        # UNC paths
        if ($pp->value =~ /^\\\\/) {
            $isAlreadyAbsolute = 1;
            $isUNC = 1;
        }
        # win absolute paths
        if ($pp->value =~ /^\w:/ or $pp->value =~ /^\\/) {
            $isAlreadyAbsolute = 1;
        }
        # unix absolute paths
        if ($pp->value =~ /^\//) {
            $isAlreadyAbsolute = 1;
        }
        if ($isAlreadyAbsolute == 0) {
            $pp->value(canonpath(cwd() . "/" . $pp->value));
        } else {
            if ($isUNC == 0) {
                $pp->value(canonpath($pp->value));
            }
        }
        #if ($isUNC == 0) {
        #    my $temp = $pp->value;
        #    $temp =~ s/\\/\//g;
        #    $pp->value($temp);
        #}
    }
    
    foreach my $pp (@pps) {
        PostProcessPP($pp);
    }
    
    print "\n\n\nSummary\n-------\n";
    foreach my $pp (@pps) {
        print $pp->id . ": ";

        if (defined $pp->dependencyFlagID) {
            foreach my $fp (@fps) {
                if ($fp->id eq $pp->dependencyFlagID) {
                    if($fp->value == 0) {
                        print $pp->dependencyDisabledValue;
                    } else {
                        print $pp->value;
                    }
                    last;
                }
            }
        } else {
            print $pp->value;
        }
        print "\n";
    }
    foreach my $fp (@fps) {
        print $fp->id . ": " . ($fp->value? "On" : "Off") . "\n";
    }
    
    print "\nProceed generating the configuration? ";
    if (!$fullauto && $anythingMissing) {
        chomp($input = <STDIN>);
        if (IsInputNo($input)) {
            DoExit();
            return;
        }
        SaveCache($configcache, \@pps, \@fps, \@cfps);
    } else {
        print "y\n";
    }
        
    foreach my $cfp (@cfps) {
        my $from;
        my $to;
        if (open my $IN, "<", $cfp->inFile) {
            if (open my $OUT, ">", $cfp->outFile) {
                while(<$IN>) {
                    foreach my $pp (@pps) {
                        $from = $pp->placeholder;
                        if (defined $pp->dependencyFlagID) {
                            foreach my $fp (@fps) {
                                if ($fp->id eq $pp->dependencyFlagID) {
                                    if ($fp->value == 1) {
                                        $to = $pp->value;
                                    } else {
                                        $to = $pp->dependencyDisabledValue;
                                    }
                                    last;
                                }
                            }
                        } else {
                            $to = $pp->value;
                        }
                        s/$from/$to/g;
                    }
                    foreach my $fp (@fps) {
                        $from = $fp->placeholder;
                        $to = $fp->value;
                        if ($fp->value == 1) {
                            s/$from(.)([^\1]*?)\1([^\1]*?)\1/$2/xg;
                        } else {
                            s/$from(.)([^\1]*?)\1([^\1]*?)\1/$3/xg;
                        }
                    }
                    foreach my $sp (@sps) {
                        $from = $sp->placeholder;
                        $to = $sp->value;
                        s/$from/$to/g;
                    }
                    print $OUT $_;
                }
                close $OUT;
            } else {
                print "could not write to " . $cfp->outFile . "\n";
            }
            close $IN;
        } else {
            print "could not read from " . $cfp->inFile . "\n";
        }
    }
    
    DoExit();
}

return 1;