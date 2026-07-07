---
title: "Going Offline: Installing EPI2ME Workflows by Hand on WSL2"
date: March 2025
description: A step-by-step guide to manually installing and configuring EPI2ME workflows for fully offline execution on WSL2.
---
### Introduction

If you've ever tried running EPI2ME on an air-gapped machine, you've likely hit the same wall I did: the app just won't start.

In many organizations, lab machines have no internet access at all, a deliberate security measure to protect sensitive data and instruments. That makes fully offline operation of EPI2ME essential for running long-read sequencing analyses reliably in these environments.

EPI2ME Desktop is primarily designed to operate in an online environment, often requiring network connectivity for tasks such as checking for updates, validating licenses, and communicating with local services like Docker or WSL. While the core analysis—such as running Nextflow—can be performed offline, the Windows graphical user interface (GUI) provided by the EPI2ME Desktop application may fail to initialize if it cannot access external servers or locate expected local network paths from a standard installation.

To ensure robust offline functionality, exporting and importing your WSL environment is the most reliable method for transferring a fully configured system. This approach preserves all build tools, Nextflow, and downloaded Docker or Apptainer (formerly Singularity) images. For offline use, I recommend utilizing open-source Apptainer instead of Docker to cache bioinformatics software images, as it offers greater flexibility in an offline setting. Note that with this method, analysis will be conducted exclusively via the command line for Nextflow runs.

The following plan focuses on installing and configuring all necessary components within your online WSL Ubuntu instance prior to exporting it for offline use.

---

### Phase 1: Installing and Setting Up the WSL 2 Instance on Windows

To begin, open PowerShell in administrator mode by right-clicking and selecting "Run as administrator." Next, enter the `wsl --install` command and restart your computer to complete the installation. You can view a list of available online distributions using the appropriate command. For additional details, refer to [Microsoft's install guide](https://learn.microsoft.com/en-us/windows/wsl/install).

```powershell
wsl --list --online
wsl --install Ubuntu-24.04
```

To run a specific WSL distribution from within PowerShell without changing your default distribution, use the command:

```powershell
wsl --distribution Ubuntu-24.04
# or
wsl -d Ubuntu-24.04
```
---

### Phase 2: Setting Up and Configuring EPI2ME Software Companions on the Online WSL 2 Instance

This phase assumes your Windows 11 host and WSL 2 instance have full internet access.

#### Step 1: Installing Build Tools and Dependencies

First, ensure all necessary libraries for compiling tools like Apptainer and handling compressed data are installed:

```bash
# Update package lists
sudo apt update
# Install general build tools, autoconf, libtool, compression headers, etc
sudo apt install -y build-essential autoconf libtool zlib1g-dev libbz2-dev git wget pkg-config
```

#### Step 2: Installing Java

Since Nextflow depends on Java to operate, we'll begin by installing Java. The process is streamlined with SDKMAN, which should be set up first:

```bash
# Install SDKMAN
curl -s https://get.sdkman.io | bash
# List available Java flavors and versions
sdk list java
# Install most recent Java JRE Temurin (required by Nextflow) and make it as your default Java:
sdk install java 25.0.1-tem
# Verify installation
java --version
```

#### Step 3: Installing NextFlow

Nextflow is the workflow manager utilized by EPI2ME to automate their workflows. You can also use Nextflow to develop your own custom bioinformatics pipelines, providing flexibility and scalability for complex analyses.

```bash
# Install Nextflow (download and place in PATH)
wget -qO- https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
# Verify installation
nextflow info
```

#### Step 4: Installing Apptainer

The most reliable and recommended way to install Apptainer is by building it from source. This ensures you have the latest version and that all necessary dependencies are properly managed. Below are the steps to install Apptainer from source on a Debian or Ubuntu-based system. Since Apptainer is written in Go, you’ll need to have a recent version of Go installed.

*During the configuration step (`./mconfig`), you might encounter errors related to missing header files. If this occurs, install the required development libraries by running `sudo apt install <devlib>` as indicated by the output of the configuration process.*

The following steps install Apptainer to `/usr/local/bin`.

```bash
# Install dependencies (you may have one or more of them already installed on your system)
sudo apt-get update
sudo apt install -y libssl-dev uuid-dev libgpgme-dev squashfs-tools libseccomp-dev libsubid-dev libfuse3-dev libsubid-dev

# Install latest stable Go release (find the package at [https://go.dev/dl/](https://go.dev/dl/))
wget https://go.dev/dl/go1.25.4.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.25.4.linux-amd64.tar.gz

# Configure PATH for Go
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
go version

# Download and Compile Apptainer
cd /tmp
git clone https://github.com/apptainer/apptainer.git
cd apptainer
./mconfig
make -C builddir
sudo make -C builddir install

# Verify installation
apptainer --version

# Remove Apptainer source code
sudo rm -rf /tmp/apptainer
```

#### Step 5: Configuring NextFlow for Offline Local Execution

<br>

##### Setting up Environment Variables:

**NXF_OFFLINE**

To configure NextFlow for offline execution, enable the `NXF_OFFLINE` environment variable by adding the following line to your terminal’s startup file (such as `~/.bashrc`). This ensures NextFlow operates in offline mode each time you start a new terminal session.

```bash
echo "export NXF_OFFLINE='true'" >> ~/.bashrc
```

You can also manually add `export NXF_OFFLINE="true"` to your `~/.bashrc` file by editing it directly.

**NXF_ASSETS**

When you use the `nextflow pull` command to retrieve a workflow, NextFlow automatically downloads the workflow project into the `${HOME}/.nextflow/assets` directory. It is recommended that this directory resides on a high-capacity storage drive. If necessary, you can change its location by setting the `NXF_ASSETS` environment variable. For this setup, however, we will proceed with the default configuration.

```bash
# Specify location for workflows on your high-capacity drive
echo "export NXF_ASSETS='/home/epi2me/nf_workflows'" >> ~/.bashrc
```

**APPTAINER_CACHEDIR** and **APPTAINER_DISABLE_CACHE**

To prevent Apptainer from downloading cache files when using the pull command to download SIF files (see below), configure the following two environment variables in your `~/.bashrc` file:

```bash
export APPTAINER_CACHEDIR=/dev/null
export APPTAINER_DISABLE_CACHE=1
source ~/.bashrc
```

You could also delete `~/.apptainer/cache` files by running the following command on shell:

```bash
apptainer clean cache
```

##### Setting up NextFlow Configuration Files

To efficiently manage and organize your Nextflow executions, it is best practice to define and utilize three types of configuration files (*Global level*, *Workflow level*, and *Local project level*).

**1. Global level**

The global configuration file is found or can be created at `~/.nextflow/config`. This file is mainly used to enable Apptainer and disable Docker, specify the directory for Apptainer container images (SIF files), and set the default executor to local for offline runs. With this configuration, Nextflow operates entirely offline, utilizes local SIF containers through Apptainer, and avoids connecting to Docker or remote repositories. Below is an example configuration you can follow:

*Groovy (`~/.nextflow/config`):*

```groovy
// $HOME/.nextflow/config (Global Configuration)

// Enable Apptainer and disable docker globally
apptainer.enabled = true
docker.enabled = false

// Set the base directory for Apptainer images (SIF files)
apptainer.libraryDir = '/home/epi2me/containers'

// Optional: Prevent Nextflow from trying to pull images from remote registries
// The user will need to manage the local SIF files manually.
apptainer.autoPull = false

// Ensure automatic mounting of host paths if your Apptainer installation supports it
apptainer.autoMounts = true

// Set default executor to 'local' for offline runs (adjust if using an HPC cluster with Apptainer support)
process.executor = 'local'

// Disables Nextflow's ability to automatically connect to and download remote project repositories
nextflow.scm = false
nextflow.enable_update = false
```

**2. Workflow level**

Each workflow contains its own configuration file (`nextflow.config`). It is best not to modify this file, as updates to the workflow may overwrite any changes you make. Instead, use this file to gather information necessary for downloading compatible container images (see details below).

**3. Local project level**

In this section, we enhance the flexibility of Nextflow configuration for individual local projects. Each project should have its own configuration file. Below is an example of a configuration file specifically designed for the `wf-bacterial-genomes` workflow, optimized for offline use.

*Groovy (Project Config):*

```groovy
/* Local project level configuration for offline local execution using Apptainer */

// Directory to store temporary intermediate files essential for resuming the run.
// You can override it with nextflow commandline option (-w path/to/nf_work).
workDir = "${launchDir}/../nf_work"

// Sets NextFlow local parameters.
params {
    // Results directory.
    out_dir = "${launchDir}/../results"

    // SHAs for container filenames
    // Copy from the workflow's <nextflow.config> file.
    wf {
        common_sha = "sha72f3517dd994984e0e2da0b97cb3f23f8540be4b"
        container_sha = "shaae5767292fede67faaa109d507be17647b117b81"
        container_sha_prokka = "sha08669655982fbef7c750c7895e97e100196c4967"
        container_sha_medaka = "sha447c70a639b8bcf17dc49b51e74dfcde6474837b"
        container_sha_seqsero = "sha96053b39b281e404cf1cf9c4684fa7dbc4e2761d"
        container_sha_resfinder = "sha024ca117d06e35c5e4116bcb2cd6235b96916fcf"
        container_sha_mlst = "sha8f8f258761d66116deb5b58b4204e5b7783a0c90"
    }
}

// Defines run profile.
profiles {
    offline_local {
        process {
            executor = 'local'
            // Maximum resource defaults depending on your machine's available resources and workflow requirements
            // process.cpus = 7
            // process.memory = '6 GB'

            // Disables Docker and Singularity and enforces Apptainer usage.
            docker.enabled = false
            singularity.enabled = false
            apptainer.enabled = true
            apptainer.autoMounts = true
            // Overrides container pull behavior.
            container.cacheDir = "${HOME}/containers"

            // Sets paths to local containers (SIF files).
            withLabel:wf_common {
                container = "${container.cacheDir}/wf-common_${params.wf.common_sha}.sif"
            }
            withLabel:wfbacterialgenomes {
                container = "${container.cacheDir}/wf-bacterial-genomes_${params.wf.container_sha}.sif"
                // memory = '6 GB'
            }
            withLabel:prokka {
                container = "${container.cacheDir}/prokka_${params.wf.container_sha_prokka}.sif"
            }
            withLabel:medaka {
                container = "${container.cacheDir}/medaka_${params.wf.container_sha_medaka}.sif"
                // memory = '6 GB'
            }
            withLabel:amr {
                container = "${container.cacheDir}/resfinder_${params.wf.container_sha_resfinder}.sif"
            }
            withLabel:mlst {
                container = "${container.cacheDir}/mlst_${params.wf.container_sha_mlst}.sif"
            }
            withLabel:seqsero2 {
                container = "${container.cacheDir}/seqsero2_${params.wf.container_sha_seqsero}.sif"
            }
        }
    }
}

// Configures output files in Results directory.
timeline {
    enabled = true
    file = "${params.out_dir}/execution/timeline.html"
    overwrite = true
}
report {
    enabled = true
    file = "${params.out_dir}/execution/report.html"
    overwrite = true
}
trace {
    enabled = true
    file = "${params.out_dir}/execution/trace.txt"
    overwrite = true
}
```


##### **What Happens Under the Hood When You Run the Pipeline**

NextFlow loads and merges configuration settings in the following order, from lowest to highest priority. Any setting defined later will override a setting defined earlier.

| Priority | Location | File | What it configures |
| :--- | :--- | :--- | :--- |
| 1 (Lowest) | User Home Directory | `~/.nextflow/config` | **Global Settings:** This is where you should define your system-wide `apptainer.enabled = true` and the permanent `apptainer.cacheDir` for your SIF files. |
| 2 | Workflow Directory | `nextflow.config` (pipeline default) | **Pipeline Defaults:** This defines the default parameters and the container specifications used by the workflow (like the SHA for each tool). |
| 3 | Launch Directory | `config/nextflow.config` (your project custom file) | **Project Overrides:** This is where you customize default parameters (e.g., `--out_dir`) and define your custom profiles. |
| 4 (Highest) | Command Line | `-params-file`, `--param`, `-profile` | **Runtime Overrides:** This includes the profiles you explicitly activate, which override everything else. |

<br>

NextFlow begins by loading your global configuration file, `~/.nextflow/config` (Priority 1). This file activates Apptainer support, enabling the use of a local SIF cache and offline mode for NextFlow.

Next, NextFlow reads the workflow's default configuration (Priority 2), followed by your custom project-specific configuration file, `config/nextflow.config` (Priority 3). Any parameters or settings you define here will override the defaults, allowing you to tailor the workflow for your project’s needs.

Finally, at the highest priority, NextFlow applies any explicitly activated profile, such as `offline_local` (Priority 4). This profile can specify details like CPU and memory requirements or set the executor to ‘local’, and its settings will override any conflicting options from lower-priority configuration files.

##### **An Overview of the NextFlow Filesystem**

Here is a brief, clean explanation of the important directories NextFlow uses or generates during a run.

| Original Directory/File | Created By | Purpose | Configuration Variable / Environment Variable |
| :--- | :--- | :--- | :--- |
| `work/` | Nextflow | Stores temporary task files and scripts; delete after verifying execution to save space. For better performance, set `workDir` to a local SSD or scratch space (for HPC) with faster I/O performance if available. | `workDir` (in `nextflow.config`); `-w` (command line flag); `NXF_WORK_DIR` (environment variable) |
| `.nextflow/` | Nextflow | Internal metadata, resume cache, and runtime state files. | Not directly configurable (Location tied to Launch Directory) |
| `nextflow.log` | Nextflow | Full run log, including debug, info, and error messages. | `-log` (command line flag) |
| `results/` | Pipeline | Final published outputs (User-facing results). | `params.out_dir` (workflow parameter) |
| `timeline.html` | Optional | Timeline of tasks (Performance & visualization). | `timeline.file` (in `nextflow.config`) |
| `report.html` | Optional | Resource usage and summary (Audit and documentation). | `report.file` (in `nextflow.config`) |
| `trace.txt` | Optional | CSV of all task metadata (Benchmarking & reproducibility). | `trace.file` (in `nextflow.config`) |

<br>

#### Step 6: Project Setup Using EPI2ME Workflows

This is the most critical step for offline readiness.

##### **Downloading Workflows**

Use the `nextflow pull` command to cache the latest workflow code locally. For example, we can download the bacterial genome assembly workflow (`wf-bacterial-genomes`) from epi2me-labs GitHub using the following command. As explained above, by default, NextFlow would download workflows and store them in `${HOME}/.nextflow/assets`.


```bash
nextflow pull epi2me-labs/wf-bacterial-genomes
```

##### **Downloading Container Images**

Use Apptainer to download and store the required Docker images as local SIF files. Make sure to update the image tag so it matches the exact version needed for your workflow. Repeat this process for each workflow you intend to run. For consistency and reproducibility, retrieve the container tags directly from the workflow’s `nextflow.config` file, as described above.

I recommend using the following simple Bash script to pull all necessary images into your `${HOME}/containers` directory. For example, run this script from your `project/scripts` directory to download necessary containers for `wf-bacterial-genomes` workflow. You may edit this file for other workflows.

**`wf-bg_containers.sh`**
```bash
#!/usr/bin/bash

# Pull necessary container images for wf-bacterial-genomes workflow.
# Variables are copied from most recent workflow/nextflow.config

# Create containers directory
mkdir -p ${HOME}/containers
containers_dir="${HOME}/containers"

# Container tags
common_sha="sha72f3517dd994984e0e2da0b97cb3f23f8540be4b"
container_sha="shaae5767292fede67faaa109d507be17647b117b81"
container_sha_prokka="sha08669655982fbef7c750c7895e97e100196c4967"
container_sha_medaka="sha447c70a639b8bcf17dc49b51e74dfcde6474837b"
container_sha_seqsero="sha96053b39b281e404cf1cf9c4684fa7dbc4e2761d"
container_sha_resfinder="sha024ca117d06e35c5e4116bcb2cd6235b96916fcf"
container_sha_mlst="sha8f8f258761d66116deb5b58b4204e5b7783a0c90"

# Containers paths
wf_common="ontresearch/wf-common"
wfbacterialgenomes="ontresearch/wf-bacterial-genomes"
prokka="ontresearch/prokka"
medaka="ontresearch/medaka"
seqsero="ontresearch/seqsero2"
amr="ontresearch/resfinder"
mlst="ontresearch/mlst"

# Download containers from DockerHub into local containers directory
apptainer pull --force --dir ${containers_dir} docker://${wf_common}:${common_sha}
apptainer pull --force --dir ${containers_dir} docker://${wfbacterialgenomes}:${container_sha}
apptainer pull --force --dir ${containers_dir} docker://${prokka}:${container_sha_prokka}
apptainer pull --force --dir ${containers_dir} docker://${medaka}:${container_sha_medaka}
apptainer pull --force --dir ${containers_dir} docker://${seqsero}:${container_sha_seqsero}
apptainer pull --force --dir ${containers_dir} docker://${amr}:${container_sha_resfinder}
apptainer pull --force --dir ${containers_dir} docker://${mlst}:${container_sha_mlst}

# Print Done
echo "Done!"
```

##### **Setting up Project Directory**

Setting up a clearly organized project directory is crucial for effective bioinformatics work. By structuring your project with separate folders for raw data, results, scripts, and documentation, you can easily find files, replicate analyses, and collaborate with colleagues. Some common suggestions include:

* `data/` – stores both raw and processed datasets
* `meta/` – stores metadata
* `scripts/` – contains analysis scripts and workflow manager (NextFlow) files
* `results/` – holds output files and figures
* `docs/` – includes documentation and notes
* `config/` – keeps workflow configuration files
* `logs/` – saves log files from pipeline operations
* `nf_work/` – used for temporary task files and scripts generated by NextFlow or similar tools (can be deleted later after successful run to free up space)

Following this organization makes your analyses smoother, supports reproducible research, and facilitates teamwork.

##### **Accessing Windows Files on WSL 2 Instance**

WSL automatically mounts all your Windows drives, making them available as standard Linux directories under the `/mnt/` folder. When you run your EPI2ME workflow from within your imported WSL terminal, you must use the WSL path to reference your input files.

Let's say your raw data is located on your Windows host at: `D:/Sequencing/Data/run1/pod5_files`. Your NextFlow command inside the WSL terminal would be:

```bash
nextflow run /path/to/main.nf \
    -profile offline_local \
    --fastq /mnt/d/Sequencing/Data/run1/pod5_files
```

However, note that while you can run NextFlow directly on files located under `/mnt/c/` or `/mnt/d/`, performance is significantly faster if you copy the input files into the Linux file system before starting the run.

##### **Executing Runs**

Running bioinformatics workflows through Bash scripts, rather than issuing commands directly in the terminal, provides substantial benefits for documentation, debugging, and reproducibility. By scripting your workflow, you create a comprehensive record of each step, making it much easier to track modifications, share procedures with collaborators, and revisit your analyses later. If any issues occur during execution, the script serves as a transparent log that can be systematically reviewed and debugged, helping to ensure that your results are both reproducible and verifiable.

Below is an example Bash script designed to run the `wf-bacterial-genomes` workflow on demonstration data downloaded from the EPI2ME GitHub repository.

**`run.sh`**
```bash
#!/usr/bin/bash

nextflow -c ../config/nextflow.config \
     -log ../logs/nextflow.log \
     run ${HOME}/.nextflow/assets/epi2me-labs/wf-bacterial-genomes/main.nf \
     -profile offline_local \
     --sample_sheet ../meta/isolates_sample_sheet.csv \
     --fastq ../data/isolates_fastq \
     --isolates true
```

##### **Removing Unnecessary Files**

Below is a reliable list of files and directories you can safely remove after a successful NextFlow run to free up disk space—while ensuring your final results remain intact.

| Directory/File | Safe to Delete? | Notes |
| :--- | :--- | :--- |
| `nf_work/` | Yes | Largest cleanup benefit |
| `.nextflow.log` | Yes | Run logs |
| `.nextflow/` | Yes (if don’t intend to resume run) | Removes resume ability |
| `~/.apptainer/cache/` | Yes (if have not set environment variables, as explained above) | Fully safe; only used for pulls |

To free up space on your system, you can execute the following cleanup script (source `cleanup.sh`):

**`cleanup.sh`**
```bash
#!/usr/bin/bash

# nextflow/ directory
rm -rf $(pwd)/.nextflow/

# nf_work/ directory
rm -rf $(pwd)/../nf_work/*

# logs/
rm -rf $(pwd)/../logs/*
```

---

### Phase 3: Exporting and Importing for Offline Machine
After successful completion of above steps, it is now time to export your WSL 2 instance into a portable disk or drive and import it to your offline machine by following the below steps.

#### Step 1: Exporting the WSL Instance

Shut down the WSL instance and export it from your Windows PowerShell:

```powershell
wsl --shutdown
wsl --export <DistroName> D:/wsl_backups/ubuntu_offline.tar
```
Replace `<DistroName>` (usually `Ubuntu` or `Ubuntu-XX.XX`) with the actual name.

#### Step 2: Importing the New Offline Machine

On the new machine, install WSL 2 and then import the distribution:

1.  Transfer the `ubuntu-XX.XX_offline.tar` file to the new machine using an external hard drive or a USB stick with at least 20 GB capacity.
2.  Import the distribution to your preferred location:

```powershell
# wsl --import <NewDistroName> <InstallLocation> <PathToTarFile>
wsl --import epi2me D:/wsl_distros D:/wsl_backups/ubuntu_offline.tar
```

Start your imported Ubuntu instance:

```powershell
wsl -d epi2me
```

#### Step 3: Transferring Future Workflow and Container Files

After importing the initial WSL 2 instance onto the offline machine, you can download other workflows and the necessary container images from the internet (as previously outlined) and transfer them to `$HOME/.nextflow/assets` and `$HOME/containers`, respectively. This approach eliminates the need to export and import the entire WSL instance each time you want to add a new workflow to your system and saves a lot of time for you.

---

### Appendix 1: Enabling GPU Access (Optional)

To take advantage of GPU acceleration within your WSL 2 environment, you may need to install the NVIDIA Container Toolkit. This is especially recommended if your workflows require GPU resources. Begin by following the official NVIDIA instructions to add the appropriate repository and install the toolkit, which includes `nvidia-container-cli`. Make sure to run these steps inside your WSL instance:

#### Installing NVIDIA Container Toolkit (Optional)

If you plan to use a GPU, install the toolkit (including `nvidia-container-cli`) inside the WSL instance:

```bash
# Follow the official NVIDIA steps for adding the repository and installing the toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list <<EOF
deb signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg https://nvidia.github.io/libnvidia-container/$distribution/EOF
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

After completing these steps, refer to the official documentation for detailed instructions on installing and configuring the toolkit to ensure seamless GPU access for your containers and workflows.
