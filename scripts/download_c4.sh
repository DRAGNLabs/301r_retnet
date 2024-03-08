# Load the git-lfs module for downloading large repository files
module load git-lfs

# Clone the dataset repository from HF.
# This will only download the repo with stub files, not the actual data.
# (The full repo is 13TB, so we don't want to download it all at once.)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4

# Move into directory
cd c4

# Download the actual data
# You can use this to specify what files exactly you want to download
# By default, it is only downloading the English "en" files (~330GB)
git lfs pull --include="en/*"
