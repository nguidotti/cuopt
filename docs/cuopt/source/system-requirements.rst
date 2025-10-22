===================
System Requirements
===================

Dependencies are installed automatically when using the pip and Conda installation methods. However, users would still need to make sure the system meets the minimum requirements.

.. dropdown:: Minimum Requirements

   * System Architecture:
       - x86-64
       - ARM64

   * GPU:
      - Volta architecture or better (Compute Capability >=7.0)

   * CPU:
      - 4+ cores

   * System Memory:
      - 16+ GB RAM

   * NVMe SSD Storage:
      - 100+ GB free space

   * CUDA:
      - 12.0+

   * Python:
      - >= 3.10.* and <= 3.13.*

   * NVIDIA drivers:
      - 525.60.13+ (Linux)
      - 527.41+ (Windows)

   * OS:
      - Linux distributions with glibc>=2.28 (released in August 2018):
         * Arch Linux (minimum version 2018-08-02)
         * Debian (minimum version 10.0)
         * Fedora (minimum version 29)
         * Linux Mint (minimum version 20)
         * Rocky Linux / Alma Linux / RHEL (minimum version 8)
         * Ubuntu (minimum version 20.04)
         * Windows 11 with WSL2

   * CUDA & NVIDIA Driver combinations:
      - CUDA 12.0 with Driver 525.60.13+
      - CUDA 12.2 with Driver 535.86.10+
      - CUDA 12.5 with Driver 555.42.06+
      - CUDA 12.9 with Driver 570.42.01+
      - CUDA 13.0 with Driver 580.65.06+

.. dropdown:: Recommended Requirements for Best Performance

   * System Architecture:
       - x86-64
       - ARM64

   * GPU:
      - NVIDIA H100 SXM (compute capability >= 9.0)

   * CPU:
      - 32+ cores

   * System Memory:
      - 64+ GB RAM

   * NVMe SSD Storage:
      - 100+ GB free space

   * CUDA:
      - 13.0

   * Latest NVIDIA drivers (580.65.06+)

   * OS:
      - Linux distributions with glibc>=2.28 (released in August 2018):
         * Arch Linux (minimum version 2018-08-02)
         * Debian (minimum version 10.0)
         * Fedora (minimum version 29)
         * Linux Mint (minimum version 20)
         * Rocky Linux / Alma Linux / RHEL (minimum version 8)

The above configuration will provide optimal performance for large-scale optimization problems.


Container
---------

* `nvidia-container-toolkit <https://docs.nvidia.com/ai-enterprise/deployment/vmware/latest/docker.html>`_ needs to be installed


Thin-client for Self-Hosted
----------------------------

* OS: Linux

* System Architecture:
   - x86-64
   - ARM64

* Python >= 3.10.x <= 3.13.x
