# Environment

- Captured at: 2026-07-07T10:22:14+08:00
- Hostname: autodl-container-7e7145a2ed-ff8a114c
- Git commit: ddf5241
- Git branch: main

## OS
```text
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
```

## CPU
```text
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      52 bits physical, 57 bits virtual
Byte Order:                         Little Endian
CPU(s):                             128
On-line CPU(s) list:                0-127
Vendor ID:                          GenuineIntel
Model name:                         Intel(R) Xeon(R) Gold 6430
CPU family:                         6
Model:                              143
Thread(s) per core:                 2
Core(s) per socket:                 32
Socket(s):                          2
Stepping:                           8
CPU max MHz:                        3400.0000
CPU min MHz:                        800.0000
BogoMIPS:                           4200.00
Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 invpcid_single intel_ppin cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk pconfig arch_lbr amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
Virtualization:                     VT-x
L1d cache:                          3 MiB (64 instances)
L1i cache:                          2 MiB (64 instances)
L2 cache:                           128 MiB (64 instances)
L3 cache:                           120 MiB (2 instances)
NUMA node(s):                       2
NUMA node0 CPU(s):                  0-31,64-95
```

## Memory
```text
               total        used        free      shared  buff/cache   available
Mem:           1.0Ti        28Gi       218Gi       4.2Gi       760Gi       968Gi
Swap:             0B          0B          0B
```

## Disk
```text
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G  1.4G   29G   5% /
/dev/md0        7.0T  6.6T  465G  94% /autodl-pub
AutoFS:fs1       10T  4.4T  5.7T  44% /autodl-pub/data
tmpfs            64M     0   64M   0% /dev
shm              60G     0   60G   0% /dev/shm
/dev/sda2       438G   16G  400G   4% /usr/bin/nvidia-smi
tmpfs           504G   12K  504G   1% /proc/driver/nvidia
tmpfs           504G  4.0K  504G   1% /etc/nvidia/nvidia-application-profiles-rc.d
tmpfs           504G     0  504G   0% /proc/asound
tmpfs           504G     0  504G   0% /proc/acpi
tmpfs           504G     0  504G   0% /proc/scsi
tmpfs           504G     0  504G   0% /sys/firmware
```

## NVIDIA SMI
```text
Tue Jul  7 10:22:14 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.76.05              Driver Version: 580.76.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:5A:00.0 Off |                  Off |
| 30%   31C    P8             18W /  450W |       0MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## NVIDIA GPU Query
```text
name, memory.total [MiB], driver_version, compute_cap, power.limit [W]
NVIDIA GeForce RTX 4090, 24564 MiB, 580.76.05, 8.9, 450.00 W
```

## NVCC
```text
nvcc not found
```

## Python / PyTorch
```text
python: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:46:43) [GCC 11.2.0]
torch: 2.8.0+cu128
torch cuda available: True
torch cuda version: 12.8
cudnn version: 91002
device count: 1
device 0: NVIDIA GeForce RTX 4090
  capability: 8.9
  total memory: 23.52 GB
```
