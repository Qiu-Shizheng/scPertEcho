"""Invented features and measurements for a reproducible numerical example."""
import numpy as np


def demo_inputs():
    genes = np.array(["gene-1","gene-2","gene-3","gene-4"])
    rows = np.array(["candidate-A","candidate-B"])
    targets = np.sin(np.arange(256).reshape(2, 128) / 13).astype("float32")
    feature = np.cos(np.arange(512).reshape(4, 128) / 21).astype("float32")
    inp = dict(row_id=rows,gene_id=genes,target_features=targets,gene_features=feature,
               view=np.zeros(2,dtype="int64"),control_anchors=np.zeros((2,512),dtype="float32"),
               gene_index=np.array([0,1,2,3]))
    ref = dict(row_id=rows,gene_id=genes,response_mean=np.zeros((2,4),dtype="float32"),
               response_sd=np.full((2,4),.8,dtype="float32"),prior_de_log_odds=np.full((2,4),-1.8,dtype="float32"))
    rec = dict(gene_id=genes,target_row=rows,distance=np.array([.2,.3],dtype="float32"),
               rna_eligible=np.array([True,True]),de_eligible=np.array([True,True]),
               rna_z=np.array([[.5,0,-1,np.nan],[-.4,-.2,.6,1]],dtype="float32"),
               de_label=np.array([[1,0,0,np.nan],[0,1,0,1]],dtype="float32"))
    return inp,ref,rec
