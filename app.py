import streamlit as st
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import muon as mu
import matplotlib.pyplot as plt
import matplotlib
import importlib.util

# Set the page layout
st.set_page_config(layout="wide")

# Title
st.title("Data Analysis of Single-cell RNA-seq Experiment")
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

# Create columns for layout
col1, col2, col3 = st.columns([2, 2, 2])

# Column 1: Project explanation
with col1:
    st.markdown("""
        <h4>Project Overview</h4>
        <ol style="font-size:10px;">
            <li>Read gene expression matrix to determine cell and gene count.</li>
            <li>Plot 1 to visualize gene distribution across cells.</li>
            <li>Plot 2 to assess gene spread, filtering out mitochondrial RNA for quality control.</li>
            <li>Filter cells based on threshold values.</li>
            <li>Plot 3 to analyze variation in gene expression across cells.</li>
            <li>Select features (e.g., gene expressions between 0.02 and 4) for further analysis.</li>
            <li>Plot 4 to perform PCA for dimensionality reduction.</li>
            <li>Use Leiden Algorithm to cluster cells based on gene expression patterns presented in Plot 5</li>
        </ol>
    """, unsafe_allow_html=True)

# Column 2: File upload interface
with col2:
    st.markdown("<h4>Upload file</h4>", unsafe_allow_html=True)
    h5_file = st.file_uploader("Upload Filtered Feature Barcode Matrix (HDF5)", type="h5")
    st.markdown("""
    <div style="background-color:#F7BF6F; padding: 5px; border-radius: 5px; display: inline-block;">
        <p style="font-size: 2px; text-align:center; color:black;"> 
            <h6>The dataset used was <a href="https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets/1.0.0/pbmc_granulocyte_sorted_10k">PBMC from a healthy donor - granulocytes removed through cell sorting (10k)</a></h6><br>
            <ul>
                <li>Click <a href="https://cf.10xgenomics.com/samples/cell-arc/1.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5">here</a> to download the (.h5) file and use to analyze results.<br></li>
                <li>Click on my <a href="https://github.com/sania8/Data-Analysis-of-Single-cell-RNA-seq-Experiment.git">GitHub profile</a> to understand better.</li>
            </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Column 3: Results display
with col3:
    st.markdown("<h4>Results</h4>", unsafe_allow_html=True)

    if h5_file:
        try:
            # Save uploaded file
            with open("temp_filtered_feature_bc_matrix.h5", 'wb') as f:
                f.write(h5_file.getvalue())

            mdata = mu.read_10x_h5("temp_filtered_feature_bc_matrix.h5")
            mdata.var_names_make_unique()

            rna = mdata.mod['rna']

            rna.var['mt'] = rna.var_names.str.startswith('MT-')
            sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

            # Plot 1 - Before filtering
            st.markdown("<b>Plot before filtering out cells (Plot 1)</b>", unsafe_allow_html=True)
            fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
            sc.pl.violin(rna, 'n_genes_by_counts', jitter=0.4, ax=axs1[0], show=False)
            sc.pl.violin(rna, 'total_counts', jitter=0.4, ax=axs1[1], show=False)
            sc.pl.violin(rna, 'pct_counts_mt', jitter=0.4, ax=axs1[2], show=False)
            st.pyplot(fig1)

            # Filtering
            mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: x >= 3)
            mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= 200) & (x < 5000))
            mu.pp.filter_obs(rna, 'total_counts', lambda x: x < 15000)
            mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x < 20)

            # Normalize and log
            sc.pp.normalize_total(rna, target_sum=1e4)
            sc.pp.log1p(rna)

            # Plot 2 - After filtering
            st.markdown("<b>Plot after filtering out mitochondrial cells (Plot 2)</b>", unsafe_allow_html=True)
            fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
            sc.pl.violin(rna, 'n_genes_by_counts', jitter=0.4, ax=axs2[0], show=False)
            sc.pl.violin(rna, 'total_counts', jitter=0.4, ax=axs2[1], show=False)
            sc.pl.violin(rna, 'pct_counts_mt', jitter=0.4, ax=axs2[2], show=False)
            st.pyplot(fig2)

           # Plot 3: Highly Variable Genes
            sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
            st.markdown("<b>Highly Variable Genes (Plot 3)</b>", unsafe_allow_html=True)
            sc.pl.highly_variable_genes(rna, show=False)
            st.pyplot(plt.gcf())

            # Plot 4: PCA
            st.markdown("<b>PCA: Dimensionality Reduction (Plot 4)</b>", unsafe_allow_html=True)
            sc.tl.pca(rna, svd_solver='arpack')
            sc.pl.pca(rna, color=['CD2', 'CD79A', 'KLF4', 'IRF8'], show=False)
            st.pyplot(plt.gcf())

            # Clustering
            st.markdown("<b>Cell Clustering with UMAP + Leiden (Plot 5)</b>", unsafe_allow_html=True)

            if importlib.util.find_spec("leidenalg") is None:
                st.error("Leiden algorithm is not installed. Run `pip install leidenalg` and restart the app.")
            else:
                sc.pl.pca_variance_ratio(rna, log=True, show=False)
                sc.pp.neighbors(rna, n_neighbors=10, n_pcs=20)
                sc.tl.leiden(rna, resolution=0.5)
                sc.tl.umap(rna, spread=1., min_dist=0.5, random_state=11)
                fig5, ax5 = plt.subplots()
                sc.pl.umap(rna, color="leiden", legend_loc="on data", ax=ax5, show=False)
                st.pyplot(fig5)

                st.success("Analysis completed and results displayed.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload the required HDF5 file to proceed.")
