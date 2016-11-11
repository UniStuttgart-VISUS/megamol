#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
# 
# #############################################################################
# PYTHON Script: UncertaintyInputData.py
# #############################################################################
# 
# Description:
#     
#    <todo> ...
# 
# 
# 
# TODO:
#     - Output file as parameter?
#     - Write Descriptions
#     - For missing PDB-files on STRIDE and DSSP web server (requested via PDB-ID):
#           - Upload PDB file to get data ...
#           - ... or execute local program?
#     - Support for OSX/macOS?
# 
# -----------------------------------------------------------------------------
# Example:
# 
# > python3.5 UncertaintyInputData.py <PDB-ID> -d
# 
# -----------------------------------------------------------------------------
# Usage:
# 
# > python3.5 UncertaintyInputData.py -h
# 
# usage: UncertaintyInputData.py [-h] [-d] PDBId
#
# Generate Input Data File for MegaMol Protein Uncertainty Plugin.
# 
# positional arguments:
#   PDBId        Specify a PDB ID in the four letter code
# 
# optional arguments:
#   -h, --help   show this help message and exit
#   -d, --debug  Flag to enable debug output
# 
# -----------------------------------------------------------------------------
# 
# #############################################################################


# #############################################################################
# Imports

import sys
import os

print('VERSION: {0}'.format(sys.version))
			
import subprocess
import logging
import argparse
import json
import time
from datetime import date
            
try: # Import urllib for Python3.x
    import urllib.request
    import urllib.parse
except Exception as Error:
    print('WARNING: Import of \"urllib.request\" and \"urllib.parse\" for python 3.x failed with error:')
    print('>>>      {0}'.format(Error))
    print('>>>      Trying to import \"urllib\" for python 2.x ...')
    try: # Import urllib for Python 2.x
        import urllib
    except Exception as Error:
        print('ERROR:   Import of \"urllib\" failed with error:')
        print('>>>      {0}'.format(Error))
        print('>>>      Current python version is NOT supported: {0}'.format(sys.version))
        exit()
    else: 
        func_urlretrieve = urllib.urlretrieve
        func_urlencode   = urllib.urlencode
        func_urlopen     = urllib.urlopen     
else:
    func_urlretrieve = urllib.request.urlretrieve
    func_urlencode   = urllib.parse.urlencode
    func_urlopen     = urllib.request.urlopen
    
    

# #############################################################################
# Class: UncertaintyInputData

class UncertaintyInputData:

    # #########################################################################
    # Function: __init__
    # 
    # Description:
    #     Function ...
    #     
    # Parameters:
    #     Param_PDBId      = An PDB ID.
    #     Param_Debug      = Debug flag. When TRUE you get additional debug output.
    #     Param_ScriptPath = Need path to script location if current execution 
    #                        path is different.
    # Return value:
    #     -
    # 
    # #########################################################################
    
    def __init__(self, Param_PDBId = 'undefined', Param_Debug = False, Param_ScriptPath='.'):
           
        self.Param_PDBId      = Param_PDBId
        self.Param_Debug      = Param_Debug
        self.Param_ScriptPath = Param_ScriptPath      
            
        # Checking if Param_PDBId is a valid PDB ID
        if (len(self.Param_PDBId) != 4) or (not self.Param_PDBId[0].isdigit()):
            try:
                raise NameError(' \"{0}\" is no valid PDB ID.'.format(self.Param_PDBId))
            except Exception as Error:          
                logging.error('  Checking PDB ID failed with error:')
                logging.error('>>>   {0}'.format(Error))
                return
            
        # Change path if script is executed from other location
        if os.path.isdir(os.path.split(self.Param_ScriptPath)[0]):
            os.chdir(os.path.split(self.Param_ScriptPath)[0])
            
        # Enable logging
        if self.Param_Debug:
            LogLevel = logging.DEBUG
        else:
            LogLevel = logging.INFO
        logging.basicConfig(stream=sys.stderr, level=LogLevel)
        
        logging.debug('  Start logging for class \"{0}\"'.format(self.__class__.__name__))
        

    
    # #########################################################################
    # Function: GenerateDataFile
    # 
    # Description:
    #     Main function ...
    # 
    # Parameters:
    #     - 
    # 
    # Return value:
    #     Status as logical value (bool)
    # 
    # #########################################################################
     
    def GenerateDataFile(self):

        # Class variables    
        PDBId             = self.Param_PDBId.lower()                                # PDB id from arg
        PDBFileName       = (PDBId + '.pdb')                                        # PDB file name
        CacheLocation     = os.path.normcase('cache')                               # cache location of for output files
        PDBFile           = os.path.join(CacheLocation, PDBFileName)                # PDB file path 
           
        ProgLocation      = os.path.normcase('prog')                                # location of program files        
        STRIDEProgName    = 'stride'                                                # STRIDE program name 
        STRIDEOutFile     = os.path.join(CacheLocation, (PDBId + '.stride'))        # STRIDE output file path
        DSSPProgName      = 'dssp'                                                  # DSSP program name        
        DSSPOutFile       = os.path.join(CacheLocation, (PDBId + '.dssp'))          # DSSP output file path
        OUTFile           = os.path.join(CacheLocation, (PDBId + '.uid'))           # Result Outfile: uid = uncertainty input data 
        PDBUrl            = 'http://files.rcsb.org/download/' + PDBFileName        # PDB file download url
        # PDBUrl            = 'https://files.rcsb.org/download/' + PDBFileName        # PDB file download url
     
        
        # Choosing 64bit or 32bit version of program files
        if(sys.maxsize > 2**32): # 64 bit
            STRIDEProgName += '64'
            DSSPProgName += '64'
        else: # 32 bit
            STRIDEProgName += '32'
            DSSPProgName += '32'
        
        # Choosing os dependend program files
        logging.debug('  Choosing right program files for OS \"{0}\"'.format(sys.platform))
        if sys.platform.startswith('win'):  
            STRIDEProg    = os.path.join(ProgLocation, (STRIDEProgName + '.exe'))   # STRIDE program path and added executable ending for windows
            DSSPProg      = os.path.join(ProgLocation, (DSSPProgName + '.exe'))     # DSSP program path and added executable ending for windows
        elif sys.platform.startswith('linux'):
            STRIDEProg    = os.path.join(ProgLocation, (STRIDEProgName))            # STRIDE program path
            DSSPProg      = os.path.join(ProgLocation, (DSSPProgName))              # DSSP program path
        else:
            try:
                raise OSError('  \"{0}\" is no supported OS'.format(sys.platform))
            except Exception as Error:          
                logging.error('  Checking OS failed with error:')
                logging.error('>>>   {0}'.format(Error))
                return False

        # Create missing cache directory
        if not os.path.isdir(CacheLocation):
            logging.debug('  Cache directory not found \"{0}\"'.format(CacheLocation))      
            os.makedirs(CacheLocation)
            logging.info('  Created new cache directory \"{0}\"'.format(CacheLocation))
            
        # Check if PDB-file exists locally
        logging.debug('  Check if PDB-file exists')
        if (not os.path.isfile(PDBFile)):
            logging.debug('  No file \"{0}\" found'.format(PDBFile))  
            logging.info('  Requesting PDB-file \"{0}\" from \"{1}\"'.format(PDBFile, PDBUrl))
            try:
                # Otherwise download the PDB file   
                func_urlretrieve(PDBUrl, PDBFile)
            except Exception as Error:         
                logging.error('  Receiving PDB file from web server failed with error:')
                logging.error('>>>   {0}'.format(Error))
                logging.error('  Failed to create \"{0}\"'.format(PDBFile))
                return False # PDB FILE IS REQUIRED ...
            else:
                logging.info('  Successfully created file \"{0}\"'.format(PDBFile))
        else:
            logging.info('  Found file \"{0}\"'.format(PDBFile))

            
        # Check if STRIDE output file alredy exists  
        logging.debug('  Check if output file of \"{0}\" for \"{1}\" exists'.format(STRIDEProgName, PDBFileName))  
        if (not os.path.isfile(STRIDEOutFile)):
            logging.debug('  Found no output file') 
            # First: Try to get STRIDE file from web server
            if (not self.GetDataFromSTRIDEServer(PDBId, STRIDEOutFile)):
                # Second: Call program STRIDE to generate output file for specified PDB-Id
                if (not self.GetDataFromProgram(STRIDEProg, '', PDBFile, '-f', STRIDEOutFile)):
                    try:
                        raise AssertionError('  Failed to create \"{0}\"'.format(STRIDEOutFile))
                    except Exception as Error:          
                        logging.error('  Receiving STRIDE file from web server and program failed with error:')
                        logging.error('>>>   {0}'.format(Error))
        else:
            logging.info('  Found file \"{0}\"'.format(STRIDEOutFile))
    
    
        # Check if DSSP output file alredy exists
        logging.debug('  Check if output file of \"{0}\" for \"{1}\"exists'.format(DSSPProgName, PDBFileName))  
        if (not os.path.isfile(DSSPOutFile)):    
            logging.debug('  Found no output file')     
            # First: Try to get DSSP file from web server
            if (not self.GetDataFromDSSPServer(PDBId, DSSPOutFile)):
                # Second: Call program DSSP to generate output file for specified PDB-Id
                if (not self.GetDataFromProgram(DSSPProg, '-i', PDBFile, '-o', DSSPOutFile)):
                    try:
                        raise AssertionError('  Failed to create \"{0}\"'.format(DSSPOutFile))
                    except Exception as Error:          
                        logging.error('  Receiving DSSP file from web server and program failed with error:')
                        logging.error('>>>   {0}'.format(Error))
        else:
            logging.info('  Found file \"{0}\"'.format(DSSPOutFile))

             
        # Generate output file
        return self.ParseDataAndCreateOutputFile(PDBFile, STRIDEOutFile, DSSPOutFile, OUTFile)
            

        
    # #########################################################################
    # Function: GetDataFromSTRIDEServer
    # 
    # Description:
    #     Function ...
    #     
    # Parameters:
    #   Param_PDBId   = PDB ID
    #   Param_Outfile = STRIDE output file name including file path
    # 
    # Return value:
    #     Status as logical value (bool)    
    # 
    # #########################################################################

    def GetDataFromSTRIDEServer(self, Param_PDBId, Param_Outfile):
            
        ServerUrl  = 'http://webclu.bio.wzw.tum.de/cgi-bin/stride/stridecgi.py'
        POSTData   = func_urlencode({'pdbid': Param_PDBId, 'action': 'compute'})
        
        logging.info('  Requesting STRIDE file for PDB-id \"{0}\" from web server \"{1}\"'.format(Param_PDBId,ServerUrl))
        try:
            # Request data from server
            STRIDEServerData = func_urlopen(ServerUrl, data=(POSTData.encode()))
        # Catch errors
        except Exception as Error:
            logging.error('  Receiving STRIDE file from web server failed with error:')
            logging.error('>>>   {0}'.format(Error))
            return False
            
        # STRIDEDataString = str(STRIDEServerData.read())
        STRIDEDataString = str(STRIDEServerData.read(), 'utf-8')
        
        # Check for errors in generated html file:
        ErrorIndex = STRIDEDataString.find('error')
        if ErrorIndex > 0:
            EndIndex = STRIDEDataString.find('</li>', ErrorIndex)
            logging.error('  Receiving STRIDE file from web server failed with error:')
            logging.error('>>>   {0}'.format(STRIDEDataString[(ErrorIndex+7):EndIndex]))
            return False
            
        # Creating STRIDE output file
        try:
            STRIDEFile = open(Param_Outfile, 'w')
        except Exception as Error:
            logging.error('  Creating STRIDE output file failed with error:')
            logging.error('>>>   {0}'.format(Error))
            return False
            
        # Writing data to STRIDE output file
        STRIDEFile.write(STRIDEDataString) # ! Have to convert from binary to string.
        STRIDEFile.close()
        logging.info('  Successfully created file \"{0}\"'.format(Param_Outfile))
        return True
        
        
        
    # #########################################################################
    # Function: GetDataFromDSSPServer
    # 
    # Description:
    #     Function ...
    #     
    # Parameters:
    #   Param_PDBId   = PDB ID
    #   Param_Outfile = DSSP output file name including file path
    # 
    # Return value:
    #     Status as logical value (bool)        
    # 
    # #########################################################################

    def GetDataFromDSSPServer(self, Param_PDBId, Param_Outfile):
        
        # ------------------------------------------------------
        # REMARK:
        #     Following code is based on examples provided here: 
        #     http://www.cmbi.ru.nl/xssp/api/examples
        # ------------------------------------------------------
        
        ServerUrl  = 'http://www.cmbi.umcn.nl/xssp/'
        POSTData   = func_urlencode({'data': Param_PDBId}) # ! Name of POST value has to be 'data'.
        
        logging.info('  Requesting DSSP file for PDB id \"{0}\" from web server \"{1}\"'.format(Param_PDBId,ServerUrl))
        GetIdUrl   = 'api/create/pdb_id/dssp/'      
        try:
            # Request job id from server
            DSSPServerData = func_urlopen(ServerUrl+GetIdUrl, data=(POSTData.encode()))
        except Exception as Error:
            logging.error('  Receiving job id from DSSP server failed with error:')
            logging.error('>>>   {0}'.format(Error))
            return False
        
        # DSSPTempData = str(DSSPServerData.read()) # ! Have to convert from binary to string.
        DSSPTempData = str(DSSPServerData.read(), 'utf-8') # ! Have to convert from binary to string.
        JobId = json.loads(DSSPTempData)['id']
        logging.debug('  Receiving Job ID was successfull \"{0}\"'.format(JobId))
            
        # Loop until the job running on the server has finished
        Ready = False
        while not Ready:

            # Check the status of the running job
            logging.debug('  Waiting until the job has finished')         
            GetStatusUrl = 'api/status/pdb_id/dssp/{0}/'.format(JobId)
            try:
                DSSPServerData = func_urlopen(ServerUrl+GetStatusUrl)
            except Exception as Error:
                logging.error('  Receiving status from DSSP server failed with error:')
                logging.error('>>>   {0}'.format(Error))
                return False

            DSSPTempData = DSSPServerData.read().decode('utf-8')
            Status = json.loads(DSSPTempData)['status']
            logging.debug('  Job status is: \"{0}\"'.format(Status))

            # If the status equals SUCCESS, exit out of the loop.
            # If the status equals either FAILURE or REVOKED, an exception is raised.
            # Otherwise, wait for 1 second and start at the beginning of the loop again.
            if Status == 'SUCCESS':
                Ready = True
            elif Status in ['FAILURE', 'REVOKED']:
                try: 
                    raise ConnectionError(json.loads(DSSPTempData)['message'])
                except Exception as Error:
                    logging.error('  DSSP server status error:')
                    logging.error('>>>   {0}'.format(Error))
                    return False                    
            else:
                time.sleep(1)
        else:
            # Requests the result of the job
            logging.debug('  Requesting result data from DSSP server')
            GetResultUrl = 'api/result/pdb_id/dssp/{0}/'.format(JobId)
            try:
                DSSPServerData = func_urlopen(ServerUrl+GetResultUrl)
            except Exception as Error:
                logging.error('  Receiving result data from DSSP server failed with error:')
                logging.error('>>>   {0}'.format(Error))
                return False
                
            DSSPTmpData = DSSPServerData.read().decode('utf-8')
            DSSPResultData = json.loads(DSSPTmpData)['result']

        # Creating DSSP output file
        try:
            DSSPFile = open(Param_Outfile, 'w')
        except Exception as Error:
            logging.error('  Creating DSSP output file failed with error:')
            logging.error('>>>   {0}'.format(Error))
            return False
            
        # Writing data to DSSP output file
        DSSPFile.write(DSSPResultData)
        DSSPFile.close()
        logging.info('  Successfully created file \"{0}\"'.format(Param_Outfile))        
        return True
              
        
        
    # #########################################################################
    # Function: GetDataFromProgram
    # 
    # Description:
    #     Function ...
    #     
    # Parameters:
    #   Param_ProgFile = Program file name including program path
    #   Param_FlagIn   = Program parameter flag for input file.
    #   Param_PDBFile  = PDB ID file name including file path
    #   Param_FalgOut  = Program parameter flag for output file.
    #   Param_Outfile  = Program outfile name including file path
    # 
    # Return value:
    #     Status as logical value (bool) 
    #        
    # #########################################################################
    
    def GetDataFromProgram(self, Param_ProgFile, Param_FlagIn, Param_PDBFile, Param_FalgOut, Param_Outfile):

        # Split program name from rest of path
        ProgName = os.path.split(Param_ProgFile)[1]

        logging.info('  Calling program \"{0}\"'.format(ProgName))       
        try:           
            # Call program 
            subprocess.call([Param_ProgFile, Param_FlagIn+Param_PDBFile, Param_FalgOut+Param_Outfile])
        # Catch errors            
        except Exception as Error:
            tb = sys.exc_info()[1]
            logging.error('  Calling program \"{0}\" failed with error:'.format(Param_ProgFile))
            logging.error('>>>   {0}'.format(Error))
            return False
        else: 
            logging.info('  Successfully created file \"{0}\"'.format(Param_Outfile))
            return True



    # #########################################################################
    # Function: ParseDataAndCreateOutputFile
    # 
    # Description:
    #     Function ...
    #     
    # Parameters:
    #   Param_PDBFile    = PDB ID file name including file path
    #   Param_STRIDEFile = STRIDE file name including file path
    #   Param_DSSPFile   = DSSP file name including file path
    #   Param_OutFile    = Output file name  
    # 
    # Return value:
    #     Status as logical value (bool)    
    # 
    # #########################################################################
        
    def ParseDataAndCreateOutputFile(self, Param_PDBFile, Param_STRIDEFile, Param_DSSPFile, Param_OutFile):
        
        # Init list buffer of output file lines
        OutFileBuffer =    [('REMARK | PDB-ID: '+(Param_OutFile[-8:-4]).upper())]     # line 0
        OutFileBuffer.append('REMARK | Date:   '+date.today().strftime('%d.%m.%Y')) # line 1
        OutFileBuffer.append('REMARK |')                                            # line 2
        OutFileBuffer.append('REMARK |')                                            # line 3
        OutFileBuffer.append('INFO   |')                                            # line 4
        OutFileBuffer.append('COLUMN |')                                            # line 5
        OutFileBuffer.append('COLUMN |')                                            # line 6     
        OutFileBuffer.append('REMARK |')                                            # line 7
        LineOffset = 8  # Buffer line offset for next 'empty' line index (index starting with 0)

        # Parsing PDB file ----------------------------------------------------
        logging.debug('  Opening PDB file')
        logging.info('  Parsing PDB file ...')
        
        # Declared here because used in PDB parsing and STRIDE and DSSP parsing
        FirstAADict = {}                                                                       # Dictionary with chain ID as index and index of first amino-acid in 'ATOM' as data
        
        if os.path.isfile(Param_PDBFile):
            try:
                PDBFile = open(Param_PDBFile, 'r')
            except Exception as Error:
                logging.error('  Opening PDB file failed with error:')
                logging.error('>>>   {0}'.format(Error))
                return False
           
            # Init list buffer with PDB header
            OutFileBuffer[2] += ('---------------------------------------------------------------------------------------------------------------------------|')
            OutFileBuffer[3] += ('---PDB --------------------------------------------------------------------------------------------------------------------|')
            OutFileBuffer[4] += ('-Nr----|--AA--|-ChainID-|-Index----|-Structure-|-ID---|-Helix-Class--|-Count-|-Sense-|-Start-AA-|-Start-#-|-End-AA-|-End-#-|')                                                                                                       # line 5
            OutFileBuffer[7] += ('-------|------|---------|----------|-----------|------|--------------|-------|-------|----------|---------|--------|-------|')
            # Length of columns:    7       6       9        10         11          6        14           7        7        10         9         8        7
                        
            # REMARK on missing amino-acids:
            #     Those amino-acids considered in STRIDE and DSSP are those who are listed in the ATOM section of the PDB file.
            #     Additionally for each amino-acid chain described in a PDB file via the ATOM entries there can be missing amino-acids.
            #     Those amino-acids are listed in the REMARK 465 section of the PDB file.
            #     The amino-acid list  given in the SEQRES section of a PDB file contains all amino-acids including the missing one.
            
            # REMARK on indices of the amino-acids:
            #     In STRIDE and DSSP the amino-acid numbering always starts with 1 and has nothing to do with indices given in the correspionding PDB file.
            #     Here (in the PDB column 'Nr' of the output file) there is used a general numbering of the amino-acids starting with 1 too. 
            #     This numbering is made with the list of the amino-acids given in the SEQRES section of the PDB file and also consideres themissing amino-acids.
            #     Independent of that for every amino-acid (per chain) there is an index assigned in the PDB file's ATOM section which can start with any number.
            #     These indices are consistent with those assigned to the missing ones in the REMARK 465 section.
                                    
            # First: Reading PDB file for getting the right amino-acid indices
            logging.debug('  Reading PDB file for getting the right amino-acid indices')
            LineOffset    = 8                                                                  # Buffer line offset      
            AANumber      = LineOffset                                                         # Numbering of amno-acids starting with 1 (has nothing to do with the PDB indexing!)
            GetMissing    = False                                                              # True if right PDB-code is read
            MissingAADict = {}                                                                 # Dictionary with chain ID as index and index of missing amino-acids as data
            EndOfFile     = False   
            while not EndOfFile:
                FileLine = PDBFile.readline()
                if not FileLine:                                                               # Empty string is end of file
                    EndOfFile = True
                else:
                # Parsing file lines ...   
                    if (FileLine[0:10] == 'REMARK 465'):                                       # PDB code for missing residues
                        if GetMissing:
                            if FileLine[19:20] not in MissingAADict:                           # Chain ID not in ...
                                MissingAADict[FileLine[19:20]] = [int(FileLine[21:26])]        # MissingAADict[chain ID] = index of missing amino-acid
                            else:
                                MissingAADict[FileLine[19:20]].append(int(FileLine[21:26]))
                            
                        elif ('M RES C SSSEQI' in FileLine[13:27]):                            # Next line(s) contain data about missing amino-acid(s)
                            GetMissing = True
                    
                    elif (FileLine[0:6] == 'SEQRES'):                                          # Contains ALL amino-acids including the missing one in the ATOM section
                        
                        FileLineList = FileLine.split() 
                        for x in range(4, len(FileLineList)):                                  # In FileLineList index 4 to end contain amino-acids in 3-letter-code
                            if AANumber >= (len(OutFileBuffer)):
                                OutFileBuffer.append('DATA   |')                            
                                                                                               # Writing an new buffer line for every new amino-acid: Number - amino-acid 3-letter-code - chain ID
                            OutFileBuffer[AANumber] += (' '+str(AANumber+1-LineOffset).rjust(5)+' |  '+FileLineList[x]+' |       '+FileLineList[2]+' |') 
                            AANumber += 1
                            
                    elif (FileLine[0:4] == 'ATOM'):                                            # Get index and amino-acid of first ATOM in each chain
                        if FileLine[21:22] not in FirstAADict:                                 # if chain ID of ATOM (= first amino-acid) not in ....                           
                                FirstAADict[FileLine[21:22]] = int(FileLine[23:26])            # New entry in dict with starting index of first ATOM in chain                       
                    
            # Sorting indices of amino-acids
            logging.debug('  Sorting indices of amino-acids') 
            ChainIndex = 0                                                                     # Index for not missing amino-acids
            MissingChainIndex = 0                                                              # Index for MissingAADict
            ChainID = ''
            for x in range(LineOffset, len(OutFileBuffer)):                                    # Buffer already contains an entry for every amino-acid
                
                # Reset offset for new chain
                if(ChainID != OutFileBuffer[x][30:31]):                                        # if chain ID != chain ID in buffer
                    ChainID = OutFileBuffer[x][30:31]
                    ChainIndex = 0
                    MissingChainIndex = 0
                    
                AAIndex = FirstAADict[ChainID]+ChainIndex                                      # Final amino-acid index = starting index of first ATOM in corresponding chain + incrementing index in chain
                MissingFlag = ' '                                                              # Flag indicating if amino-acid is 'missing'
                if ChainID in MissingAADict:                                                   # if there is a missing amino-acid for this chain
                    if (MissingChainIndex) < len(MissingAADict[ChainID]):                      # First look as long through missing amino-acid dict ...
                        if MissingAADict[ChainID][MissingChainIndex] < FirstAADict[ChainID]:   # ... until all missing amino-acids with smaller indices than the first amino-acid in ATOM are listed 
                            AAIndex = MissingAADict[ChainID][MissingChainIndex]
                            MissingFlag = 'M'
                            MissingChainIndex += 1
                            ChainIndex -= 1                                                    # Reset Index to 0 (-> is incremented at the end of the loop)
                    if (AAIndex in MissingAADict[ChainID]):                                    # Mark further missing amino-acids coming after starting index of first amino-acid in ATOM
                        MissingFlag = 'M'
            
                OutFileBuffer[x] += (' '+MissingFlag+' '+str(AAIndex).rjust(6)+' |')           # Writing flag and amino-acid index to corresponding buffer line   
                ChainIndex += 1
                
            # Lookup table for column width of PDB file entries
            #   - First Index of substring indices -1 because here index starts from 0 
            #   - Second Index +1 because range() is exclusive last index in range
            #   - Structure  ID    H-Class   Count   Sense  Start-AA  Start-#   End-AA    End-#  
            #   - #: 6       3       2       5/2       2       3        4        3        4   
            CWh = [[0,6], [7,10], [38,40], [71,76],          [15,18], [21,25], [27,30], [33,37]]  # For helix
            CWs = [[0,6], [7,10],          [14,16], [38,40], [17,20], [22,26], [28,31], [33,37]]  # For sheet   
            
            # Second: Read file for assigning the secondary structure to the right amino-acid indices
            logging.debug('  Reading PDB file for assigning the secondary structure to the right amino-acid indices')
            PDBFile.seek(0)                                                                    # Reset file iterator               
            EndOfFile = False           
            while not EndOfFile:
                FileLine = PDBFile.readline()
                if not FileLine:                                                               # Empty string is end of file
                    EndOfFile = True
                else:
                # Parsing file line ...   
                    PDBCode = FileLine[0:5]      
                    if (PDBCode == 'HELIX'):
                        LineOffset = 8   
                        # Determining starting index of HELIX in chain
                        for x in range(LineOffset, len(OutFileBuffer)):
                            if (OutFileBuffer[x][30:31] == FileLine[19:20]):                   # Chain ID == chain ID HELIX
                                if (int(OutFileBuffer[x][36:42])== int(FileLine[CWh[5][0]:CWh[5][1]])): # Amino-acid number of chain  == amino-acid start number of HELIX in chain               
                                    LineOffset = x
                                    break                                 
                                    
                        AARange = int(FileLine[CWh[7][0]:CWh[7][1]]) - int(FileLine[CWh[5][0]:CWh[5][1]]) # Index range of HELIX = ending index of HELIX - starting index of HELIX  
                        for x in range(LineOffset, LineOffset+AARange+1):                      # +1 because range() is excluding last index                     
                            OutFileBuffer[x] += ('    '+FileLine[CWh[0][0]:CWh[0][1]]+' |  '+FileLine[CWh[1][0]:CWh[1][1]]+' |           '+
                                                 FileLine[CWh[2][0]:CWh[2][1]]+' | '+FileLine[CWh[3][0]:CWh[3][1]]+' |       |      '+
                                                 FileLine[CWh[4][0]:CWh[4][1]]+' |    '+FileLine[CWh[5][0]:CWh[5][1]]+' |    '+
                                                 FileLine[CWh[6][0]:CWh[6][1]]+' |  '+FileLine[CWh[7][0]:CWh[7][1]]+' |')

                    elif (PDBCode == 'SHEET'):
                        LineOffset = 8 
                        # Determining starting index of SHEET in chain
                        for x in range(LineOffset, len(OutFileBuffer)):
                            if (OutFileBuffer[x][30:31] == FileLine[21:22]):                   # Chain ID == chain ID SHEET
                                if (int(OutFileBuffer[x][36:42]) == int(FileLine[CWs[5][0]:CWs[5][1]])): # Amino-acid number of chain  == amino-acid start number of SHEET in chain              
                                    LineOffset = x
                                    break

                        AARange = int(FileLine[CWs[7][0]:CWs[7][1]]) - int(FileLine[CWs[5][0]:CWs[5][1]])        
                        for x in range(LineOffset, LineOffset+AARange+1):                      # +1 because range() is excluding last index                     
                            # IGNORING if same strand belongs to different sheets -> only one SHEET ID is assigned
                            if (len(OutFileBuffer[x]) < len(OutFileBuffer[2])-1):              # Determined by chacking if buffer line is alredy 'filled'                
                                OutFileBuffer[x] += ('    '+FileLine[CWs[0][0]:CWs[0][1]]+' |  '+FileLine[CWs[1][0]:CWs[1][1]]+' |              |    '+
                                                     FileLine[CWs[2][0]:CWs[2][1]]+' |    '+FileLine[CWs[3][0]:CWs[3][1]]+' |      '+
                                                     FileLine[CWs[4][0]:CWs[4][1]]+' |    '+FileLine[CWs[5][0]:CWs[5][1]]+' |    '+
                                                     FileLine[CWs[6][0]:CWs[6][1]]+' |  '+FileLine[CWs[7][0]:CWs[7][1]]+' |')
                    elif (PDBCode == 'ATOM '):                                                 # Stop reading file when PDB code ATOM is read ...ATOM comes after HELIX and SHEET
                        break 
                        
            # Filling empty lines ...
            LineOffset = 8                                                                     # Buffer line offset
            for x in range(LineOffset, len(OutFileBuffer)):
                LengthDiff = len(OutFileBuffer[2]) - len(OutFileBuffer[x])                     # Length of line 2 as reference 
                if LengthDiff > 0:
                    OutFileBuffer[x] += ('           |      |              |       |       |          |         |        |       |')
                        
            PDBFile.close()   
            logging.debug('  Completed to parse file \"{0}\"'.format(Param_PDBFile))  
        else:
            logging.warn('  Didn\'t find file \"{0}\"'.format(Param_PDBFile))    
            return False                                                                       # This case should be handles earlier ... !
        
        
        # Parsing STRIDE file -------------------------------------------------
        logging.debug('  Opening STRIDE file')
        logging.info('  Parsing STRIDE file ...')
        if os.path.isfile(Param_STRIDEFile):
            try:
                STRIDEFile = open(Param_STRIDEFile, 'r')
            except Exception as Error:
                logging.error('  Opening STRIDE file failed with error:')
                logging.error('>>>   {0}'.format(Error))
                return False

            # Init list buffer with STRIDE header
            OutFileBuffer[2] += ('------------------------------------------------------------------------|') 
            OutFileBuffer[3] += ('---STRIDE --------------------------------------------------------------|') 
            OutFileBuffer[4] += ('-Nr----|-AA---|-ChainID-|--Structure------|-Phi-----|-Psi-----|-Area----|') 
            OutFileBuffer[7] += ('-------|------|---------|-----------------|---------|---------|---------|')  
            # Length of columns:     7       6       9        17                 9         9         9
            
            # ColumnWidth of STRIDE file entries
            #  - First Index of substrings -1 
            #  - Code     AA    ChainID   Nr      Structure  Phi      Psi      saa
            #  - #: 3      3       1      5         15       7        7        7
            CW = [[0,3], [5,8], [9,10], [15,20], [24,39], [42,49], [52,59], [62,69]] 
            
            LineOffset = 8                                                                     # ...
            ChainID = ''                                                                       # ...
            EndOfFile = False              
            while not EndOfFile:
                FileLine = STRIDEFile.readline()
                if not FileLine:                                                               # Empty string is end of file
                    EndOfFile = True
                else:
                # Parsing file line ...                
                    if (FileLine[CW[0][0]:CW[0][1]] == 'ASG'):                                                      
                                
                        # Skip in chain as 'missing' flagged amino-acids
                        while (OutFileBuffer[LineOffset][34:35] == 'M'):                       # ...
                            LineOffset += 1
                            
                        # Search for matching chain ID                    
                        if (OutFileBuffer[LineOffset][30:31] != ChainID) :                     # PDB Chain ID != STRIDE chain ID  
                            ChainID = FileLine[9:10]
                            LineOffset = 8                                                     # Reset LineOffset and search for next chain ID from the beginning
                            for x in range(LineOffset, len(OutFileBuffer)):                    # Start search from the beginning
                                if (OutFileBuffer[x][30:31] == ChainID) :                      # PDB Chain ID == STRIDE chain ID
                                    LineOffset = x
                                    break
                            # ... and starting index of first amino-acid in ATOM 
                            if (int(OutFileBuffer[LineOffset][36:42]) != FirstAADict[ChainID]):                            
                                for y in range(x, len(OutFileBuffer)):
                                    if (int(OutFileBuffer[y][36:42]) == FirstAADict[ChainID]):  # PDB amino-acid index == Index of first amino-acid in ATOM 
                                        LineOffset = y
                                        break
                             
                        OutFileBuffer[LineOffset] += (' '+FileLine[CW[3][0]:CW[3][1]]+' |  '+FileLine[CW[1][0]:CW[1][1]]+' |       '+
                                                  FileLine[CW[2][0]:CW[2][1]]+' | '+FileLine[CW[4][0]:CW[4][1]]+' | '+
                                                  FileLine[CW[5][0]:CW[5][1]]+' | '+FileLine[CW[6][0]:CW[6][1]]+' | '+
                                                  FileLine[CW[7][0]:CW[7][1]]+' |')
                        LineOffset += 1                     
                            
            # Fill empty lines ...
            LineOffset = 8                                                                     # Only buffer line offset
            for x in range(LineOffset, len(OutFileBuffer)):
                LengthDiff = len(OutFileBuffer[2]) - len(OutFileBuffer[x])                     # Length of line 2 as reference 
                if LengthDiff > 0:
                    OutFileBuffer[x] += ('       |      |         |                 |         |         |         |') 
                        
            STRIDEFile.close()   
            logging.debug('  Completed to parse file \"{0}\"'.format(Param_STRIDEFile))  
        else:
            logging.warn('  Didn\'t find file \"{0}\"'.format(Param_STRIDEFile))    
            # return False - NOT: Because missing STRIDE file is not essential ...
        
        
        # Parsing DSSP file ---------------------------------------------------
        logging.debug('  Opening DSSP file')
        logging.info('  Parsing DSSP file ...')        
        if os.path.isfile(Param_DSSPFile):
            try:
                DSSPFile = open(Param_DSSPFile, 'r')
            except Exception as Error:
                logging.error('  Opening DSSP file failed with error:')
                logging.error('>>>   {0}'.format(Error))
                return False

            # Init list buffer with DSSP header
            OutFileBuffer[2] += ('--------------------------------------------------------------------------------------------------------------------------------|') 
            OutFileBuffer[3] += ('---DSSP ------------------------------------------------------------------------------------------------------------------------|')
            OutFileBuffer[4] += ('-Nr----|-AA-|-ChainID-|-Structure-|-BP1--|-BP2--|-ACC--|-TCO-----|-KAPPA--|-ALPHA--|-PHI----|-PSI----|-X-CA---|-Y-CA---|-Z-CA---|')
            OutFileBuffer[7] += ('-------|----|---------|-----------|------|------|------|---------|--------|--------|--------|--------|--------|--------|--------|')
            # Length of columns:   7        4       9       11         6       6       6       9        8        8        8        8        8         8        8
            
            # ColumnWidth of DSSP file entries
            # - First Index of substring indices: -1 | Second Index of substring indices: +1
            # - Number  Amino.   ChainID  Struc.   BP1      BP2      AC C      TCO      KAPPA    ALPHA     PHI        PSI        X-CA       Y-CA       Z-CA
            # - #:  5        1        1        9        4       4          4       7        6        6        6           6         6          6          6
            CW = [[0,5], [13,14], [11,12], [16,25], [25,29], [30,34], [34,38], [84,91], [91,97], [97,103], [103,109], [109,115], [116,122], [123,129], [130,136]] 

            # Not used any more ...
            # AmoniAciodOneLetter = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'B': 'ASX', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'Z': 'GLX', 'G': 'GLY', 'H': 'HIS',
            #                        'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
            
            LineOffset = 8
            ChainID = ''              
            EndOfFile = False
            StartParsing = False
            while not EndOfFile:
                FileLine = DSSPFile.readline()
                if not FileLine:                                                               # Empty string is end of file
                    EndOfFile = True
                elif '#' in FileLine[0:5]:                                                     # Start parsing when line with '#' occures 
                    StartParsing = True                 
                # Parsing file line ...                     
                elif StartParsing:
                    if (('!' or '*') not in FileLine):                                         # Skip lines indicating chain break or dicontinuity
                    
                        # Skip other as 'missing' flagged amino-acids 
                        while (OutFileBuffer[LineOffset][34:35] == 'M'):                       # ...
                            LineOffset += 1
                                                
                        # Search for matching chain ID                    
                        if (OutFileBuffer[LineOffset][30:31] != ChainID) :                     # PDB Chain ID != DSSP chain ID  
                            ChainID = FileLine[11:12]
                            LineOffset = 8                                                     # Reset LineOffset and search for next chain ID from the beginning
                            for x in range(LineOffset, len(OutFileBuffer)):                    # Start search from the beginning
                                if (OutFileBuffer[x][30:31] == ChainID) :                      # PDB Chain ID == DSSP chain ID
                                    LineOffset = x
                                    break
                            # ... and starting index of first amino-acid in ATOM 
                            if (int(OutFileBuffer[LineOffset][36:42]) != FirstAADict[ChainID]):                            
                                for y in range(x, len(OutFileBuffer)):
                                    if (int(OutFileBuffer[y][36:42]) == FirstAADict[ChainID]):  # PDB amino-acid index == Index of first amino-acid in ATOM 
                                        LineOffset = y
                                        break
                                        
                        # "A blank in the DSSP secondary structure determination stands for loop or irregular. 
                        #  Loops and irregular elements are often, very incorrectly, called "random coil" or "coil". 
                        #  Many programs, including the PDBFINDER, replace this blank by a C 
                        #  (doing undue justice to the historical artefactual naming of loops and irregular stretches) 
                        #  because one never knows if a blank means loop or no-output, or something-went-wrong."
                        
                        # if FileLine[CW[3][0]:(CW[3][0]+1)].isspace():
                        #     FileLine = FileLine[:CW[3][0]]+'C'+FileLine[(CW[3][0]+1):]
                            
                        OutFileBuffer[LineOffset] += (' '+FileLine[CW[0][0]:CW[0][1]]+' |  '+FileLine[CW[1][0]:CW[1][1]]+' |       '+FileLine[CW[2][0]:CW[2][1]]+' | '+
                                                  FileLine[CW[3][0]:CW[3][1]]+' | '+FileLine[CW[4][0]:CW[4][1]]+' | '+FileLine[CW[5][0]:CW[5][1]]+' | '+
                                                  FileLine[CW[6][0]:CW[6][1]]+' | '+FileLine[CW[7][0]:CW[7][1]]+' | '+FileLine[CW[8][0]:CW[8][1]]+' | '+
                                                  FileLine[CW[9][0]:CW[9][1]]+' | '+FileLine[CW[10][0]:CW[10][1]]+' | '+FileLine[CW[11][0]:CW[11][1]]+' | '+
                                                  FileLine[CW[12][0]:CW[12][1]]+' | '+FileLine[CW[13][0]:CW[13][1]]+' | '+FileLine[CW[14][0]:CW[14][1]]+' |')
                        LineOffset += 1
          
            # Fill empty lines ...
            LineOffset = 8 
            for x in range(LineOffset, len(OutFileBuffer)):
                LengthDiff = len(OutFileBuffer[2]) - len(OutFileBuffer[x]) 
                if LengthDiff > 0:
                    OutFileBuffer[x] += ('       |    |         |           |      |      |      |         |        |        |        |        |        |        |        |')
                        
            DSSPFile.close()   
            logging.debug('  Completed to parse file \"{0}\"'.format(Param_DSSPFile))  
        else:
            logging.warn('  Didn\'t find file \"{0}\"'.format(Param_DSSPFile))    
            # return False - NOT: Because missing DSSP file is not essential ...
        
        
        # Adding column numbers to buffer lines 5 and 6 - Index is starting with 0!
        logging.debug('  Writing column numbers to buffer lines 5 and 6')
        for x in range(0, (len(OutFileBuffer[2])-8)):                                          # Length of buffer line 2 is reference and starting
            OutFileBuffer[5] += str(x%10)                                                      # Repeating digits 0-9
            if (x == 0):
                OutFileBuffer[6] += '          '
            else:
                if (x%10 == 0):                                                                # x modulo 10, skip 0
                    OutFileBuffer[6] += str(x)                                                 # ...
                    if x < 100:
                        OutFileBuffer[6] += '        '
                    else:
                        OutFileBuffer[6] += '       '
        
        # Creating output file
        logging.debug('  Creating output file \"{0}\"'.format(Param_OutFile))
        try:
            OutFile = open(Param_OutFile, 'w')
        except Exception as Error:
            logging.error('  Creating output file failed with error:')
            logging.error('>>>   {0}'.format(Error))
            return False
            
        # Append EOL to all buffered lines and write buffer to file
        logging.debug('  Adding line endings, file END tag and writing buffer to output file')
        OutFileBuffer.append('END')
        for BufferLine in OutFileBuffer:
            BufferLine += '\n'                                                                 # Line ending
            OutFile.write(BufferLine)   
        
        # Close output file
        OutFile.close()
        logging.info('  Successfully created file \"{0}\"'.format(Param_OutFile))
        return True
        

       
    # #########################################################################
    # Function: __del__
    # 
    # Description:
    #     Function ...
    #     
    # Parameters:
    #     -
    # Return value:
    #     -
    #     
    # #########################################################################
    
    def __del__(self):
        logging.debug('  Class \"{0}\" deleted'.format(self.__class__.__name__))
        logging.shutdown()
          
    
    
# #############################################################################
# MAIN

if __name__ == "__main__":
    
    # Argument Parser
    parser = argparse.ArgumentParser(prog='UncertaintyInputData.py',description='Generate Input Data File for MegaMol Protein Uncertainty Plugin.')
    
    # required argument which passes the PDB Id:
    parser.add_argument('PDBId', action='store', help='Specify a PDB ID in the four letter code')
    # optional argument defining if logging level is debug or info (default):
    parser.add_argument('-d', '--debug', action='store_true', help='Flag to enable debug output')
    args = parser.parse_args()

    
    myUID = UncertaintyInputData(args.PDBId, args.debug, sys.argv[0])
    myUID.GenerateDataFile()
    del myUID
    
    

# #############################################################################
# EOF
