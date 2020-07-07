## --------------------------------------------------------------------------------------------------------------------------------------------
library(Seurat)
args = commandArgs(trailingOnly=TRUE)


input_dir = "../inputs/raw_input_combined/filtered_matrices_mex/hg19/"
output_dir = "../inputs/preprocessed_data/"
outputplot_dir = "../outputs/preprocessed_data/"

min_nfeature = 200
max_nfeature = 1750
max_percMT = 5

num_features = 2000



## --------------------------------------------------------------------------------------------------------------------------------------------

if (length(args)!=0 && length(args) != 7){
  
  print("error: When calling the script with arguments, ALL arguments must be supplied: ")
  
  print("Input directory, output directory, plot directory, min number of detected genes, max number of detected genes, max allowed percentage of mitochondrial genes, number of features to select")
  
  print("Calling the script without arguments corresponds to:")
  
  print("Rscript sca_preprocessing.R ../inputs/raw_input_combined/filtered_matrices_mex/hg19/ ../inputs/preprocessed_data/ ../outputs/preprocessed_data/ 200 1750 5 2000")
  
  
  print("you can also call the script with number of features to select = 0, which will skip the step and keep all genes (memory intensive)")

  
  quit(save="no")
  
} else if (length(args) == 7) {
  
  input_dir = args[1]
  output_dir = args[2]
  outputplot_dir = args[3]
  
  min_nfeature = as.integer(args[4])
  max_nfeature = as.integer(args[5])
  max_percMT = as.integer(args[6])
  
  num_features = as.integer(args[7])
  
}



## --------------------------------------------------------------------------------------------------------------------------------------------
dir.create(path = output_dir, showWarnings = TRUE, recursive = TRUE)
dir.create(path = outputplot_dir, showWarnings = TRUE, recursive = TRUE)


## --------------------------------------------------------------------------------------------------------------------------------------------

#####################################################################
### Read in Data

format(Sys.time(), "%X> Reading in Data...")



counts = Read10X(data.dir = input_dir, gene.colum = 1)
# large matrix with ROWS = GENES and columns = cells

seurat = CreateSeuratObject(counts = counts, project = "scAutoencoder")





## --------------------------------------------------------------------------------------------------------------------------------------------
#####################################################################
### Quality 

tempstring = format(Sys.time(), "%X> Doing QC")
print(paste(tempstring, " with ", 
            as.integer(min_nfeature), " min / ", 
            as.integer(max_nfeature), " max detected genes, max ", 
            as.integer(max_percMT), " percent MT genes...", 
            sep = ""))




# drop cells with too few genes detected. (not sequenced deep enough)
# drop cells with too many genes detected (multiplets)
# drop cells with high mitochondrial transcript percentage

# For mitochondrials
seurat[["percent.mt"]] = PercentageFeatureSet(seurat, pattern = "^MT[-\\.]")
# calculate the percentage of a subset, when the gene name is starting with (= "^") "MT-", "MT\" or "MT."
# gets added to the meta-data of the whole object [[ ]]


## --------------------------------------------------------------------------------------------------------------------------------------------
# inspect distributions to find "normal" values

filepath = paste(outputplot_dir, "pre_vulcano.png", sep = "")
png(file= filepath)
VlnPlot(seurat, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
dev.off()


filepath = paste(outputplot_dir, "pre_vulcano2.png", sep = "")
png(file= filepath)
VlnPlot(seurat, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3, pt.size=0)
dev.off()


library(patchwork)
plot1 <- FeatureScatter(seurat, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(seurat, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")


filepath = paste(outputplot_dir, "pre_correlationPlots.png", sep = "")
png(file= filepath, width = 980)
plot1 + plot2
dev.off()



## --------------------------------------------------------------------------------------------------------------------------------------------
## CutOff at reasonable values (defined in argparse)

seurat <- subset(seurat, subset = nFeature_RNA > min_nfeature & nFeature_RNA < max_nfeature & percent.mt < max_percMT)



## --------------------------------------------------------------------------------------------------------------------------------------------
#####################################################################
### Normalization

format(Sys.time(), "%X> Normalizing for captured RNA...")


seurat <- NormalizeData(seurat)



## --------------------------------------------------------------------------------------------------------------------------------------------
#####################################################################
### Feature Selection

if (num_features > 0) {
  
  tempstring = format(Sys.time(), "%X> Doing feature selection...")
  print(paste(tempstring, " with ", as.integer(num_features), " features...", sep = ""))
  
  
  seurat <- FindVariableFeatures(seurat, nfeatures = num_features)
  
  
  
  ## Plot the variable features
  
  top_features <- head(VariableFeatures(seurat), 20)
  plot1 <- VariableFeaturePlot(seurat)
  plot2 <- LabelPoints(plot = plot1, points = top_features, repel = TRUE)
  
  filepath = paste(outputplot_dir, "VariableFeatures.png", sep = "")
  png(file= filepath, width = 980)
  plot1 + plot2
  dev.off()
  
} else {
  print(format(Sys.time(), "%X> Feature Selection was skipped."))
}



## --------------------------------------------------------------------------------------------------------------------------------------------
#####################################################################
### Scaling / Centering Data

format(Sys.time(), "%X> Scaling / Centering Data...")

seurat <- ScaleData(seurat)



## --------------------------------------------------------------------------------------------------------------------------------------------
#####################################################################
### Generating Output

format(Sys.time(), "%X> Generating Data...")


finaldata = as.data.frame(GetAssayData(object = seurat, slot = 'scale.data'))

genes = rownames(finaldata)
barcodes = colnames(finaldata)





write.table(finaldata, file = paste0(output_dir, "matrix.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
write.table(barcodes, file = paste0(output_dir, "barcodes.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)

# writing genes would only return one column (here 1, ensemblID). This section ensures that the genes output is in the same format as the input
genesdf = as.data.frame(genes)
fullgenes = read.table(file = paste0(input_dir, "genes.tsv"), sep = '\t', header = FALSE)
colnames(fullgenes) = c("genes", "genename")
completegenes = merge(genesdf, fullgenes)
write.table(completegenes, file = paste0(output_dir, "genes.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)



# copy the label file as well (remove the labels for cells that were cut)
ori_barcodes = read.table(file = paste0(input_dir, "barcodes.tsv"), sep = '\t', header = FALSE)
labels = read.table(file = paste0(input_dir, "celltype_labels.tsv"), sep = '\t', header = FALSE)
dictionary = cbind(ori_barcodes, labels)
colnames(dictionary) = c("barcodes", "celltype label")

barcodesdf = as.data.frame(barcodes)
merged = merge(barcodesdf, dictionary, sort = FALSE)
write.table(merged[,2], file = paste0(output_dir, "celltype_labels.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)




## --------------------------------------------------------------------------------------------------------------------------------------------
format(Sys.time(), "%X> sca_preprocessing terminated successfully")

