# Use a miniconda base image
FROM continuumio/miniconda3

# Copy the environment file into the container
COPY environment.yml /tmp/environment.yml

# Create the conda environment from the file
RUN conda env create -f /tmp/environment.yml && conda clean -a

# Activate the conda environment
ENV PATH /opt/conda/envs/torch/bin:$PATH

# Set a default command to open a bash shell with the environment activated
CMD ["/bin/bash"]
