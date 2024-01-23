# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 09:40:48 2022

@author: oaklin keefe

This is Level 00 pipeline: taking raw .dat files and turning them into a .txt file                                                                                          
First we read in the files and make sure there are enough input lines to process, and if so
then we run reg-exs** to make sure the lines don't have any errors, before we save as a text file
with commas separating the columns so that we can later read it in as a .csv file without any
trouble.
**your raw input may have a different regular expression (reg-ex), and if so, those lines would
need to be changed.

Input file location: 
    this is the folder path of the folder containing the raw data
Input files:
    *.dat files (1 file per 20 min period per instrument-port)

Output file location: 
    code_pipeline/Level1_errorLinesRemoved
Output files:
    *.txt files (1 file per 20 min period per instrument-port)
    plus sub-folder
    
"""
#%%

import os
import natsort
import re
import datetime


print('done with imports')

    
#%%

filepath= r"Z:\combined_analysis\OaklinCopyMNode\spring_deployment\mNode" #folder where mini-node data is stored
start=datetime.datetime.now()
for root, dirnames, filenames in os.walk(filepath): #this is for looping through files that are in a folder inside another folder
    for filename in natsort.natsorted(filenames):
        file = os.path.join(root, filename)
        filename_only = filename[:-4]
        newFileName = str(filename_only)+".txt"

        path_save = r"Z:\combined_analysis\OaklinCopyMNode\code_pipeline\Level1_errorLinesRemoved/" #folder where output files will be saved 
        
        
        if filename.startswith("mNode_Port1"):    
            # first test the file is big enough
            if (os.path.getsize(file) >= 1420000): #1388 kb 20 minute file =~1420850
                regex = r"^\s{1,}([-]?\d{1,}[.]?\d*)\s{1,}([-]?\d*[.]?\d*)\s{1,}([-]?\d*[.]?\d*)\s{1,}([-]?\d*[.]?\d*)\s{1}(\S{2,})\s{1,}(\S{1,})$"
                textfile = open(file, 'r')
                matches = []
                for line in textfile:
                    matches.append(re.match(regex,line))
                textfile.close()
                
                lines = []
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    for match in matches:
                        if match is None: #if there are no matches to the reg-ex, make the line all NaNs
                            print(r"NaN,NaN,NaN,NaN,NaN,NaN", file=myFile)
                        else: #if the line matches the reg ex, write it as a string joined by commas
                            new_line = ','.join(match.groups())
                            print(new_line, file=myFile)
           
            #if file isn't big enough, make all the line's NaN
            else:
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    print(r"Nan,Nan,NaN,NaN,NaN,NaN", file=myFile)
            # print(COUNT)•

        if filename.startswith("mNode_Port2"):            
            if (os.path.getsize(file) >= 1420000): #1388 kb ~1420850                
                regex = r"^\s{1,}([-]?\d{1,}[.]?\d*)\s{1,}([-]?\d*[.]?\d*)\s{1,}([-]?\d*[.]?\d*)\s{1,}([-]?\d*[.]?\d*)\s{1}(\S{2,})\s{1,}(\S{1,})$"
                textfile = open(file, 'r')
                matches = []                
                for line in textfile:
                    matches.append(re.match(regex,line))
                textfile.close()

                lines = []
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    for match in matches:
                        if match is None:
                            print(r"NaN,NaN,NaN,NaN,NaN,NaN", file=myFile)
                        else:
                            new_line = ','.join(match.groups())
                            print(new_line, file=myFile)
            else:
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    print(r"Nan,Nan,NaN,NaN,NaN,NaN", file=myFile)

        if filename.startswith("mNode_Port3"):            
            if (os.path.getsize(file) >= 1420000): #1388 kb ~1420850
                regex = r"^\s{1,}([-]?\d{1,}[.]?\d*)\s{1,}([-]?\d*[.]?\d*)\s{1,}([-]?\d*[.]?\d*)\s{1,}([-]?\d*[.]?\d*)\s{1}(\S{2,})\s{1,}(\S{1,})$"
                textfile = open(file, 'r')
                matches = []                
                for line in textfile:
                    matches.append(re.match(regex,line))
                textfile.close()

                lines = []
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    for match in matches:
                        if match is None:
                            print(r"NaN,NaN,NaN,NaN,NaN,NaN", file=myFile)
                        else:
                            new_line = ','.join(match.groups())
                            print(new_line, file=myFile)
            else:
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    print(r"Nan,Nan,NaN,NaN,NaN,NaN", file=myFile)
        
        if filename.startswith("mNode_Port4"):            
            if (os.path.getsize(file) >= 950000): #679 kb ~958961
                regex4 = r"^.{1}([-+]?\d*)[,]([-+]?\d*)[,]([-+]?\d{1,}[.]?\d*)[,]([-+]?\d*[.]?\d*)[,]([-+]?\d*[.]?\d*)[,]([-+]?\d*[.]?\d*)[,].{1}(\S{1,})$"
                textfile = open(file, 'r')
                matches = []                
                for line in textfile:
                    matches.append(re.match(regex4,line))
                textfile.close()

                lines = []
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    for match in matches:
                        if match is None:
                            print(r"NaN,NaN,NaN,NaN,NaN,NaN,NaN", file=myFile)
                        else:
                            new_line = ','.join(match.groups())
                            print(new_line, file=myFile)
            else:
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    print(r"Nan,Nan,NaN,NaN,NaN,NaN,NaN", file=myFile)
        
          #FALL Deployment (09-11 2022)
        if filename.startswith("mNode_Port5"):
            if (os.path.getsize(file) >= 83000): #82 kb ~83374                
                #regex for fall deployment port 5
                regex = r'^["]{1}((\d{4})\-(0?[1-9]|1[012])\-(0?[1-9]|[12][0-9]|3[01]))\s{1}((?:(?:([01]?\d|2[0-3]):)?([0-5]?\d):)?([0-5]?\d))["]{1}[,]{1}(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}(\d{3,4})[,]{1}(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)$'
                textfile = open(file, 'r')
                matches = []                
                for line in textfile:
                    matches.append(re.match(regex,line))
                textfile.close()

                lines = []
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    for match in matches:
                        if match is None:
                            print(r"NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN", file=myFile)
                        else:
                            new_line = ','.join(match.groups())
                            print(new_line, file=myFile)
            else:
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    print(r"NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN", file=myFile)
                        
        # SPRING deployment (04-06 2022)
        if filename.startswith("mNode_Port5"):
            if (os.path.getsize(file) >= 83000): #82 kb ~83374                
                #regex for spring deployment port 5 (different from fall deployment port 5)
                regex = r'^(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}(\d{3,4})[,]{1}(\d*[.]?\d*)[,]{1}(\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)[,]{1}([-]?\d*[.]?\d*)$'
                textfile = open(file, 'r')
                matches = []                
                for line in textfile:
                    matches.append(re.match(regex,line))
                textfile.close()

                lines = []
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    for match in matches:
                        if match is None:
                            print(r"NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN", file=myFile)
                        else:
                            new_line = ','.join(match.groups())
                            print(new_line, file=myFile)
            else:
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    print(r"NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN", file=myFile)
                        
        if filename.startswith("mNode_Port6"):            
            if (os.path.getsize(file) >= 1030000): #1013 kb ~1036761                
                regex = r"^.{4}(\d{1})(\d{3,4}.{1}\d{1,})$"
                textfile = open(file, 'r')
                matches = []
                for line in textfile:
                    matches.append(re.match(regex,line))
                textfile.close()

                newFileName = str(filename_only)+".txt"
                lines = []
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    for match in matches:
                        if match is None:
                            print(r"NaN,NaN", file=myFile)
                        else:
                            new_line = ','.join(match.groups())
                            print(new_line, file=myFile)
            else:
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    print(r"NaN,NaN", file=myFile)
                     
                        
        if filename.startswith("mNode_Port7"):            
            if (os.path.getsize(file) >= 200000): #241 kb ~246116                            
                regex = r"^[r](\d{1}.{1}\d{1,3}).{1}[a](\d{2}).{1}[q](\d{1,3})$"
                textfile = open(file, 'r')
                matches = []
                # reg = re.compile(regex)
                for line in textfile:
                    matches.append(re.match(regex,line))
                textfile.close()

                newFileName = str(filename_only)+".txt"
                lines = []
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    for match in matches:
                        if match is None:
                            print(r"NaN,NaN,NaN", file=myFile)
                        else:
                            new_line = ','.join(match.groups())
                            print(new_line, file=myFile)
            else:                
                with open(os.path.join(path_save,newFileName), "w") as myFile:
                    print(r"NaN,NaN,NaN", file=myFile)
            
end=datetime.datetime.now()
#%%
# printing start and end to tell you how long it takes to run
print(start)
print(end)
print('done with this section')
# can un-comment the following code if using windows, to have a beep signal the end of the code
# import winsound
# duration = 3000  # milliseconds
# freq = 440  # Hz
# winsound.Beep(freq, duration)






