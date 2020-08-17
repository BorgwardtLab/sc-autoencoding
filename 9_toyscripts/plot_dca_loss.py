# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:41:26 2020

@author: Mike Toreno II
"""



import sys
import os
import argparse



import matplotlib.pyplot as plt






try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass




parser = argparse.ArgumentParser(description = "calculate PCAs")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../0_runner_bashscripts/")
parser.add_argument("-p","--outputplot_dir", help="out directory", default = "../outputs/sca/dca/")

args = parser.parse_args() #required


input_dir = args.input_dir
outputplot_dir = args.outputplot_dir



def findall(target, string):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = string.find(target)
    while i != -1:
        yield i
        i = string.find(target, i+1)






with open(input_dir + "log_run_autoencoder_DCA.log", "r") as file:
    content = file.read()

indexes = findall(string = content, target = "ms/step - loss: ")
indexes = list(indexes)



losses = []
val_losses = []

losses_str = []
val_losses_str = []



for index in indexes:
    buffer1 = 16
    buffer2 = 35
    
    loss = content[index + buffer1 : index + buffer1 + 6]
    val_loss = content[index + buffer2 : index + buffer2 + 6]
    
    
    losses_str.append(loss)
    val_losses_str.append(val_loss)    
    
    loss = float(loss)
    val_loss = float(val_loss)

    losses.append(loss)
    val_losses.append(val_loss)



# print
with open(outputplot_dir + "losses.tsv", "w") as outfile:
    outfile.write("\t".join(losses_str))
    outfile.write("\n")
    outfile.write("\t".join(val_losses_str))




# %%
# draw

epochs = range(1, len(losses)+1)


plt.figure()

plt.plot(epochs, losses, 'r')
plt.plot(epochs, val_losses, 'g')
plt.legend(['loss', 'validation loss'])

plt.xlabel("epoch number")
plt.ylabel("loss")

plt.ylim(bottom = 0)

plt.show()
plt.savefig(outputplot_dir + "losses.png")








