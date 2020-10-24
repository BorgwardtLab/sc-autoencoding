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
parser.add_argument("-i","--input_file", help="input directory", default = "../0_runner_bashscripts/")
parser.add_argument("-p","--outputplot_dir", help="out directory", default = "../outputs/sca/dca/")

args = parser.parse_args() #required


input_file = args.input_file
outputplot_dir = args.outputplot_dir



def findall(target, string):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = string.find(target)
    while i != -1:
        yield i
        i = string.find(target, i+1)









# %% Start of the program



with open(input_file, "r") as file:
    content = file.read()

indexes = findall(string = content, target = "ms/step - loss: ")
indexes = list(indexes)






losses = []
val_losses = []

losses_str = []
val_losses_str = []





################ interlude to find the very first
firstindex = content.find(" - loss: ")
firstloss = content[firstindex + 9:firstindex + 15]

losses_str.append(firstloss)
losses.append(float(firstloss))
val_losses_str.append("None")
################ end of interlude



# %%



epoch_number_str = []

epoch_number_str.append("0")

e = 1

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
    
    
    epoch_number_str.append(str(e))
    e = e+1





epochs = range(0, len(losses))
val_epochs = range(1, len(val_losses)+1)




# print
with open(outputplot_dir + "losses.tsv", "w") as outfile:
    outfile.write("\t".join(epoch_number_str))
    outfile.write("\n")
    outfile.write("\t".join(losses_str))
    outfile.write("\n")
    outfile.write("\t".join(val_losses_str))


    
    


# %%
# draw


plt.figure()

plt.plot(epochs, losses, linestyle='-', marker='.', color='b')
plt.plot(val_epochs, val_losses, linestyle='-', marker='.', color='g')
plt.legend(['loss', 'validation loss'])

plt.xlabel("epoch number")
plt.ylabel("loss")


plt.ylim(bottom = 0)

plt.show()
plt.savefig(outputplot_dir + "losses.png")








